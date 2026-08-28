# -*- coding: utf-8 -*-
"""Supervised-learning training loop (task t5) with curriculum stages.

Captain supplement: three-stage curriculum learning.
  Stage 1: mode="simple" features (hand+dora+basic, first 17 channels of the
      full feature tensor), discard head only -> tsumogiri baseline accuracy.
  Stage 2: mode="full" features (49 ch), multi-task loss (discard CE + binary
      CE on riichi/chow/pon/kan where the human had that choice).

Data path (performance directive):
  --shards DIR : load uint8 training shards from src/model/preprocess.py
      (zstd npz, docs/observation_schema.md §5.1); shard-order shuffle per
      epoch + in-shard permutation + background shard prefetch; real
      mini-batches (fp16 autocast + GradScaler); samples/s logged.
  --records F  : fallback streaming JSONL (batch=1, smoke tests / tiny runs).

Measured throughput (RTX 5070, 2026-08-25, full model 50 blocks / 256 ch):
  preprocess.py : 19.2k samples/s (10 workers; target >= 3k)
  train stage1  : 5.6k samples/s  (batch 512, 17 ch)
  train stage2  : 5.6k samples/s  (batch 1024, 49 ch + 4 binary heads)

Usage examples:
  # smoke test on synthetic data (streaming fallback)
  python train_sl.py --records data/synthetic_records.jsonl --limit 500 \
      --epochs 1 --out data/models/smoke --log-dir logs --device cpu
  # real run: stage1 simple baseline then full multi-task on shards
  python train_sl.py --shards data/processed/tenhou/shards --batch-size 128 \
      --max-steps 200000
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# allow direct execution: python model/train_sl.py from anywhere
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.features import (  # noqa: E402
    BINARY_HEADS, build_features, channel_names, discard_mask,
    feature_channels, group_summary, label_kind, tile_to_kind,
)
from model.model import MultiHeadRiichiNet, get_device  # noqa: E402
from model.preprocess import NO_LABEL, load_shard  # noqa: E402

SIMPLE_CHANNELS = 17  # mode="simple" = first 17 channels of the full tensor


# ---------------------------------------------------------------------------
# data sources
# ---------------------------------------------------------------------------

class RecordStream:
    """Streaming JSONL reader with a shuffle buffer (fallback path)."""

    def __init__(self, path: str, shuffle: bool, seed: int,
                 buffer: int = 20000, limit: int = 0):
        self.path = Path(path)
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.buffer = buffer
        self.limit = limit
        if not self.path.exists():
            raise FileNotFoundError(
                f"records file not found: {path} (data pipeline output expected "
                f"at data/processed/tenhou/records.jsonl)")

    def __iter__(self):
        buf = []
        seen = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                if self.limit and seen >= self.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                seen += 1
                buf.append(json.loads(line))
                if len(buf) >= self.buffer:
                    idx = self.rng.randrange(len(buf))
                    buf[idx], buf[-1] = buf[-1], buf[idx]
                    yield buf.pop()
        while buf:
            idx = self.rng.randrange(len(buf))
            buf[idx], buf[-1] = buf[-1], buf[idx]
            yield buf.pop()


def encode_record(rec: dict, mode: str, device: torch.device):
    """record -> (features (1,C,34,1), masks, targets) tensors (streaming path).

    targets["discard"] is None when the human did NOT discard: the discard
    head is only supervised on real discards (call samples would otherwise
    hit a masked/illegal target kind and produce inf loss).
    """
    feats = torch.from_numpy(build_features(rec, mode=mode)).unsqueeze(0).to(device)
    masks = {"discard": torch.from_numpy(discard_mask(rec)).unsqueeze(0).to(device)}
    if rec["label"].get("type") in ("discard", "riichi"):
        targets = {"discard": torch.tensor([label_kind(rec)], dtype=torch.long, device=device)}
    else:
        targets = {"discard": None}
    cands = {}
    la = rec.get("legal_actions") or {}
    lab = rec["label"]
    for head in BINARY_HEADS:
        legal = la.get(head)
        if not legal:
            continue
        targets[head] = torch.tensor(
            [1 if lab.get("type") == head else 0], dtype=torch.long, device=device)
        if lab.get("type") == head:
            tiles = [lab["tile"]] if head == "riichi" else lab.get("tiles", [])
        else:
            first = legal[0]
            tiles = first.get("tiles", []) if isinstance(first, dict) else [first]
        kind = tile_to_kind(tiles[0]) if tiles else 0
        onehot = torch.zeros(1, 34, device=device)
        onehot[0, kind] = 1.0
        cands[head] = onehot
    ev = rec.get("last_event") or {}
    drawn = None
    if ev.get("type") in ("draw", "tsumo", "dahai") and "tile" in ev:
        drawn = tile_to_kind(ev["tile"])
    return feats, masks, targets, cands, drawn


class ShardDataset:
    """uint8 shard loader: shard-order shuffle per epoch + in-shard permutation.

    Batches: (x (B,C,34,1), dmask (B,34) bool, yd (B,) long, yd_valid (B,),
    drawn (B,) long, heads: {h: (y (B,), valid (B,), cand_onehot (B,34))})
    """

    def __init__(self, shards_dir: str, mode: str, batch_size: int,
                 seed: int, device: torch.device, limit: int = 0):
        self.paths = sorted(glob.glob(str(Path(shards_dir) / "shard-*.npz.zst")))
        if not self.paths:
            raise FileNotFoundError(
                f"no shards in {shards_dir} (run src/model/preprocess.py first)")
        self.mode = mode
        self.n_channels = feature_channels(mode)
        self.bs = batch_size
        self.seed = seed
        self.device = device
        self.limit = limit
        self.shard_rng = random.Random(seed)
        self.perm_rng = np.random.default_rng(seed)

    def batches(self):
        """Endless generator: shuffled shards, permuted samples, mini-batches.

        Per shard, all fields are permuted (numpy) and moved to the device ONCE;
        each batch is then just tensor slicing (views). The next shard is
        loaded+converted on a background thread while the current one trains.
        """
        import threading

        def load_one(path, out):
            d = load_shard(path)
            n = d["feat"].shape[0]
            perm = np.random.default_rng().permutation(n)
            for k in d:
                if k == "scale":
                    continue
                d[k] = d[k][perm]
            out["t"] = self._shard_to_tensors(d, n)

        while True:
            paths = self.paths[:]
            self.shard_rng.shuffle(paths)
            remaining = self.limit if self.limit > 0 else None  # None = unlimited
            prefetch = None
            th = None
            for i, p in enumerate(paths):
                if remaining is not None and remaining <= 0:
                    break
                if prefetch is None:
                    holder = {}
                    load_one(p, holder)
                    t = holder["t"]
                else:
                    t = prefetch
                    prefetch = None
                if i + 1 < len(paths):
                    holder = {}
                    th = threading.Thread(target=load_one,
                                          args=(paths[i + 1], holder), daemon=True)
                    th.start()
                n = t["feat"].shape[0]
                if remaining is not None:
                    n = min(n, remaining)
                    remaining -= n
                for lo in range(0, n, self.bs):
                    yield self._slice_batch(t, lo, min(lo + self.bs, n))
                if th is not None:
                    th.join()
                    prefetch = holder["t"]
                    th = None

    def _shard_to_tensors(self, d, n):
        # keep the uint8 features on the device and decode per batch:
        # a decoded fp32 full-mode shard (~122ch) would be ~1.7GB — decoding
        # lazily bounds VRAM with the prefetch double-buffer
        self.scale_t = torch.from_numpy(
            np.ascontiguousarray(d["scale"][:self.n_channels])
        ).float().to(self.device).view(1, -1, 1)
        out = {
            "feat": torch.from_numpy(
                np.ascontiguousarray(d["feat"][:n, :self.n_channels])
            ).to(self.device),  # uint8 (n,C,34)
            "mask": torch.from_numpy(
                np.ascontiguousarray(d["dmask"][:n])).bool().to(self.device),
            "yd": torch.from_numpy(
                np.ascontiguousarray(d["discard"][:n].copy())).long().to(self.device),
            "yd_valid": torch.from_numpy(
                np.ascontiguousarray(d["discard"][:n] != NO_LABEL)).to(self.device),
            "drawn": torch.from_numpy(
                np.ascontiguousarray(d["drawn"][:n].copy())).long().to(self.device),
            "heads": {},
        }
        for h in BINARY_HEADS:
            y = d[f"{h}_y"][:n]
            v = y != NO_LABEL
            c = d[f"{h}_cand"][:n]
            onehot = np.zeros((n, 34), dtype=np.float32)
            rows = np.where(v & (c != NO_LABEL))[0]
            if len(rows):
                onehot[rows, c[rows].astype(np.int64)] = 1.0
            out["heads"][h] = (
                torch.from_numpy(np.ascontiguousarray(y.copy())).long().to(self.device),
                torch.from_numpy(np.ascontiguousarray(v)).to(self.device),
                torch.from_numpy(np.ascontiguousarray(onehot)).to(self.device))
        return out

    def _slice_batch(self, t, lo, hi):
        sl = slice(lo, hi)
        x = (t["feat"][sl].float() * self.scale_t).unsqueeze(-1)  # decode uint8
        heads = {h: (t["heads"][h][0][sl], t["heads"][h][1][sl],
                     t["heads"][h][2][sl]) for h in BINARY_HEADS}
        return (x, t["mask"][sl], t["yd"][sl], t["yd_valid"][sl],
                t["drawn"][sl], heads)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class Metrics:
    def __init__(self):
        self.n = 0
        self.correct1 = 0
        self.correct5 = 0
        self.tsumo_n = 0
        self.tsumo_hit = 0
        self.head_hits = {h: 0 for h in BINARY_HEADS}
        self.head_n = {h: 0 for h in BINARY_HEADS}
        self.loss_sum = 0.0
        self.loss_n = 0

    def update_batch(self, logits, yd, yd_valid, drawn, heads, loss: float, n: int):
        self.loss_sum += loss * n
        self.loss_n += n
        if yd_valid.any():
            pred = logits["discard"][yd_valid]
            y = yd[yd_valid]
            p1 = pred.argmax(dim=1)
            self.n += int(yd_valid.sum().item())
            self.correct1 += int((p1 == y).sum().item())
            top5 = pred.topk(5, dim=1).indices
            self.correct5 += int((top5 == y.view(-1, 1)).any(dim=1).sum().item())
            dr = drawn[yd_valid]
            same = (y == dr) & (dr != NO_LABEL)
            self.tsumo_n += int(same.sum().item())
            self.tsumo_hit += int((same & (p1 == dr)).sum().item())
        for h in BINARY_HEADS:
            if h in heads and h in logits:
                y, v, _cand = heads[h]
                if v.any():
                    hit = int((logits[h][v].argmax(dim=1) == y[v]).sum().item())
                    self.head_hits[h] += hit
                    self.head_n[h] += int(v.sum().item())

    def summary(self) -> dict:
        out = {
            "loss": self.loss_sum / max(self.loss_n, 1),
            "discard_top1": self.correct1 / max(self.n, 1),
            "discard_top5": self.correct5 / max(self.n, 1),
            "tsumogiri_acc": self.tsumo_hit / self.tsumo_n if self.tsumo_n else None,
            "tsumogiri_frac": self.tsumo_n / max(self.n, 1),
        }
        for h in BINARY_HEADS:
            if self.head_n[h]:
                out[f"{h}_acc"] = self.head_hits[h] / self.head_n[h]
        return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def run_step(model, opt, scaler, batch, args, steps, total_steps, warmup):
    x, mask, yd, yd_valid, drawn, heads = batch
    cands = {h: heads[h][2] for h in model.binary_heads}
    amp = scaler.is_enabled()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        logits = model(x, masks={"discard": mask}, candidates=cands)
        loss = 0.0
        if yd_valid.any():
            loss = loss + F.cross_entropy(logits["discard"][yd_valid], yd[yd_valid])
        for h in model.binary_heads:
            y, v, _c = heads[h]
            if v.any():
                loss = loss + F.cross_entropy(logits[h][v], y[v])
    if not isinstance(loss, torch.Tensor):
        return logits, yd, yd_valid, drawn, heads, None, steps
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    scaler.step(opt)
    scaler.update()
    steps += 1
    if steps <= warmup:
        lr = args.lr * steps / warmup
    else:
        prog = min((steps - warmup) / max(total_steps - warmup, 1), 1.0)
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * prog))
    for g in opt.param_groups:
        g["lr"] = lr
    return logits, yd, yd_valid, drawn, heads, loss, steps


def apply_transfer(model: torch.nn.Module, ckpt_state: dict, src_in: int):
    """Warm-start a stage-2 net from a stage-1 checkpoint.

    - All keys with identical shapes (trunk blocks, discard/value heads) are
      copied verbatim.
    - The stem conv is extended in_channels -> full: the first src_in input
      rows are copied, the new rows are zero-initialized (new channels start
      neutral, so the trunk behaves exactly like stage-1 until they get
      gradients).
    - Binary heads (absent in stage-1) keep their random init.
    """
    sd = ckpt_state.get("model", ckpt_state)
    own = model.state_dict()
    for k, v in own.items():
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            own[k] = sd[k].clone()
    sk = "stem.weight"
    if sk in sd and sk in own and sd[sk].shape[1] < own[sk].shape[1]:
        ow = own[sk].clone()
        n_in = sd[sk].shape[1]
        ow[:, :n_in, :] = sd[sk]
        ow[:, n_in:, :] = 0.0  # new channels start neutral
        own[sk] = ow
    model.load_state_dict(own)


def train_stage(args, stage: int, mode: str, device: torch.device, start_ts: float):
    n_channels = feature_channels(mode)
    binary = BINARY_HEADS if stage >= 2 else ()
    model = MultiHeadRiichiNet(
        n_channels, channels=args.channels, n_blocks=args.blocks, binary_heads=binary)
    if args.transfer_from and stage >= 2 and Path(args.transfer_from).exists():
        ck = torch.load(args.transfer_from, map_location="cpu")
        sd = ck.get("model", ck)
        src_in = sd["stem.weight"].shape[1] if "stem.weight" in sd else 17
        apply_transfer(model, ck, src_in)
        print(f"[stage{stage} {mode}] transfer from {args.transfer_from} applied "
              f"(stem {src_in} -> {n_channels} input channels)", flush=True)
    model.to(device)
    n_params = model.num_parameters()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    if args.shards:
        source = ShardDataset(args.shards, mode, args.batch_size, args.seed,
                              device, limit=args.limit)
    else:
        source = RecordStream(args.records, shuffle=True, seed=args.seed,
                              limit=args.limit)
    log_path = Path(args.log_dir) / f"train_sl_stage{stage}_{mode}.jsonl"
    ckpt_dir = Path(args.out) / f"stage{stage}_{mode}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    total_steps = max(args.max_steps, 1)
    warmup = max(1, int(total_steps * 0.05))
    steps = 0
    n_samples = 0
    best_top1 = 0.0
    metrics = Metrics()
    since_eval = 0

    print(f"[stage{stage} mode={mode}] channels={n_channels} "
          f"groups={group_summary(mode)} params={n_params/1e6:.2f}M "
          f"device={device} heads={binary or ('discard',)} "
          f"source={'shards:' + str(args.shards) if args.shards else args.records}",
          flush=True)

    epoch = 0
    cur_lr = args.lr
    while steps < total_steps and epoch < args.epochs:
        if args.shards:
            seen = 0
            for batch in source.batches():
                logits, yd, yd_valid, drawn, heads, loss, steps =                     run_step(model, opt, scaler, batch, args, steps, total_steps, warmup)
                cur_lr = opt.param_groups[0]["lr"]
                bs = x_size(batch)
                n_samples += bs
                since_eval += bs
                if loss is not None:
                    metrics.update_batch(logits, yd, yd_valid, drawn, heads,
                                         loss.item(), bs)
                if since_eval >= args.eval_every:
                    best_top1 = _eval(args, stage, mode, epoch, n_samples, metrics,
                                      best_top1, model, ckpt_dir, log_path, start_ts,
                                      lr=cur_lr)
                    metrics = Metrics()
                    since_eval = 0
                seen += bs
                if steps >= total_steps or (args.limit and seen >= args.limit):
                    break
            epoch += 1
        else:
            for rec in source:
                feats, masks, targets, cands, drawn = encode_record(rec, mode, device)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=scaler.is_enabled()):
                    logits = model(feats, masks=masks, candidates=cands)
                    loss = 0.0
                    if targets["discard"] is not None:
                        loss = loss + F.cross_entropy(logits["discard"], targets["discard"])
                    for head in binary:
                        if head in targets:
                            loss = loss + F.cross_entropy(logits[head], targets[head])
                if isinstance(loss, torch.Tensor):
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    scaler.step(opt)
                    scaler.update()
                    steps += 1
                    if steps <= warmup:
                        lr = args.lr * steps / warmup
                    else:
                        prog = min((steps - warmup) / max(total_steps - warmup, 1), 1.0)
                        lr = args.lr * 0.5 * (1 + math.cos(math.pi * prog))
                    for g in opt.param_groups:
                        g["lr"] = lr
                # streaming metrics via the batched API (batch of 1)
                yd = targets["discard"]
                if yd is not None:
                    yd_valid = torch.ones(1, dtype=torch.bool, device=device)
                else:
                    yd_valid = torch.zeros(1, dtype=torch.bool, device=device)
                    yd = torch.zeros(1, dtype=torch.long, device=device)
                drawn_t = torch.tensor([drawn if drawn is not None else NO_LABEL],
                                       dtype=torch.long, device=device)
                heads_b = {}
                for h in binary:
                    if h in targets:
                        cand = cands.get(h)
                        heads_b[h] = (targets[h],
                                      torch.ones(1, dtype=torch.bool, device=device),
                                      cand if cand is not None else torch.zeros(1, 34, device=device))
                metrics.update_batch(logits, yd, yd_valid, drawn_t, heads_b,
                                     loss.item() if isinstance(loss, torch.Tensor) else 0.0, 1)
                n_samples += 1
                since_eval += 1
                cur_lr = opt.param_groups[0]["lr"]
                if since_eval >= args.eval_every:
                    best_top1 = _eval(args, stage, mode, epoch, n_samples, metrics,
                                      best_top1, model, ckpt_dir, log_path, start_ts,
                                      lr=cur_lr)
                    metrics = Metrics()
                    since_eval = 0
                if steps >= total_steps:
                    break
            epoch += 1
    best_top1 = _eval(args, stage, mode, epoch - 1, n_samples, metrics, best_top1,
                      model, ckpt_dir, log_path, start_ts, force=True, lr=cur_lr)

    torch.save({"model": model.state_dict(), "config": vars(args),
                "mode": mode, "stage": stage, "channels": channel_names(mode),
                "steps": steps, "top1": best_top1},
               ckpt_dir / "last.pt")
    cfg_path = ckpt_dir / "config.json"
    cfg_path.write_text(json.dumps({**vars(args), "mode": mode, "stage": stage,
                                    "channels": channel_names(mode)},
                                   indent=2, default=str), encoding="utf-8")
    print(f"[stage{stage} {mode}] done. best_top1={best_top1:.4f} "
          f"ckpt={ckpt_dir} cfg={cfg_path}", flush=True)
    return {"stage": stage, "mode": mode, "best_top1": best_top1,
            "channels": n_channels, "params": n_params, "ckpt": str(ckpt_dir)}


def x_size(batch):
    return batch[0].shape[0]


def _eval(args, stage, mode, epoch, n_samples, metrics, best_top1,
          model, ckpt_dir, log_path, start_ts, force=False, lr=None):
    if metrics.loss_n == 0:
        return best_top1  # empty window (e.g. final flush after a reset)
    summ = metrics.summary()
    dt = time.time() - start_ts
    summ.update({"stage": stage, "mode": mode, "epoch": epoch,
                 "step": n_samples, "samples": n_samples,
                 "samples_per_sec": n_samples / max(dt, 1e-6)})
    if lr is not None:
        summ["lr"] = lr
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summ) + "\n")
    top1 = summ["discard_top1"]
    tsu = summ.get("tsumogiri_acc")
    tsu_s = f"tsumo={tsu:.3f}" if tsu is not None else "tsumo=n/a"
    print(f"[stage{stage} {mode}] samples={n_samples} loss={summ['loss']:.4f} "
          f"top1={top1:.4f} top5={summ['discard_top5']:.4f} {tsu_s} "
          f"{summ['samples_per_sec']:.0f}/s", flush=True)
    if top1 > best_top1 and (not force):
        torch.save({"model": model.state_dict(), "config": vars(args),
                    "mode": mode, "stage": stage, "channels": channel_names(mode),
                    "samples": n_samples, "top1": top1},
                   ckpt_dir / "best.pt")
        return top1
    return best_top1


def main():
    ap = argparse.ArgumentParser(description="SL training with curriculum stages")
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--records", default=None,
                           help="records.jsonl (streaming fallback)")
    src_group.add_argument("--shards", default=None,
                           help="dir with preprocessed shard-*.npz.zst")
    ap.add_argument("--out", default="data/models/sl", help="checkpoint dir")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--mode", default=None, choices=[None, "simple", "full"],
                    help="force one mode instead of curriculum")
    ap.add_argument("--curriculum", action="store_true",
                    help="stage1 simple then stage2 full (captain supplement)")
    ap.add_argument("--stage1", action="store_true", help="run stage 1 (simple)")
    ap.add_argument("--stage2", action="store_true", help="run stage 2 (full)")
    ap.add_argument("--transfer-from", default=None,
                    help="stage1 checkpoint (last.pt) to warm-start stage2 via "
                         "stem-extension: copy matching weights; stem input rows "
                         "for new channels are zero-initialized")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1024,
                    help="512-1024 measured best on RTX 5070 (fp16)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--blocks", type=int, default=50,
                    help="residual blocks (use ~20 for quick stage-1 baseline)")
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap samples per stage (0 = all)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="cap optimizer steps per stage (0 = derive from limit)")
    ap.add_argument("--eval-every", type=int, default=10000,
                    help="eval interval in samples")
    ap.add_argument("--amp", action="store_true", default=True,
                    help="fp16 autocast on CUDA (--no-amp disables)")
    ap.add_argument("--no-amp", action="store_false", dest="amp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device("cuda" if args.device == "auto" else args.device)
    if args.max_steps == 0:
        args.max_steps = max(args.limit // max(args.batch_size, 1), 1) if args.limit else 200000

    if args.mode is not None:
        stages = [(1 if args.mode == "simple" else 2, args.mode)]
    elif args.stage1 and not args.stage2:
        stages = [(1, "simple")]
    elif args.stage2 and not args.stage1:
        stages = [(2, "full")]
    else:
        # default: curriculum per captain supplement (simple tsumogiri
        # baseline first, then full multi-task)
        stages = [(1, "simple"), (2, "full")]

    t0 = time.time()
    results = []
    for stage, mode in stages:
        results.append(train_stage(args, stage, mode, device, t0))
    print(json.dumps({"curriculum_results": results}, indent=2, default=str),
          flush=True)


if __name__ == "__main__":
    main()
