# -*- coding: utf-8 -*-
"""Offline tensorization of records.jsonl -> uint8 training shards (zstd).

Captain directive: performance is a hard constraint. Streaming JSON + online
feature encoding tops out at ~90 samples/s; this module preprocesses once with
multiprocessing and lets train_sl.py load ready-made shards.

Design (docs/observation_schema.md §5.1):
- Input : records.jsonl / records-*.jsonl (schema v1.1)
- Output: {out}/shard-{seq:06d}.npz.zst  (+ manifest.json, channel_scale.npy)
    feat  : uint8 (N, 49, 34)  full-mode channels (simple = first 17, order
                                guaranteed by features.CHANNELS registry)
    scale : float32 (49,)      per-channel max/255; decode x = feat * scale
    dmask : uint8 (N, 34)      legal discard kind mask (0/1)
    discard: uint8 (N,)        discard label kind (255 = no discard label)
    drawn : uint8 (N,)         drawn tile kind (255 = not a draw decision)
    {head}_y  : uint8 (N,)     0/1 for riichi/chow/pon/kan (255 = head n/a)
    {head}_cand: uint8 (N,)    candidate kind (255 = n/a)
- Fixed per-channel scales (no second pass): hand/discard/meld counts max 4,
  dora counts max 5, everything else max 1 (one-hot or normalized scalar).
- Multiprocessing: workers quantize to uint8 directly (spawn-safe module-level
  worker), imap_unordered; shard assembly and zstd write in the main process.

Usage:
  python src/model/preprocess.py --records data/processed/tenhou/records.jsonl
      --out data/processed/tenhou/shards --shard-size 65536 --workers 10
      [--limit 1000000]
"""
from __future__ import annotations

import argparse
import gzip
import glob
import io
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import zstandard

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.features import (  # noqa: E402
    BINARY_HEADS, build_features, channel_names, channel_scale_max,
    discard_mask, feature_channels, label_kind, tile_to_kind,
)

N_KINDS = 34
NO_LABEL = 255
DRAW_EVENTS = ("draw", "tsumo", "dahai")
_SHARD_SUFFIX = ".npz.zst"


# per-channel quantization max lives in features.channel_scale_max
# (single source of truth; all channels are 0/1 or [0,1] except the two
# count-valued dora channels: indicators max 5, dora_kinds max 4)
channel_scales = channel_scale_max


def _quantize(feat: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """float32 (N,C,34) -> uint8; decode: feat * scale[None,:,None]."""
    return np.clip(np.round(feat * (255.0 / scale[None, :, None])), 0, 255).astype(np.uint8)


def fnv1a64(s: str) -> int:
    """FNV-1a 64-bit hash of a game_id (partition-leak audit, see §5.1)."""
    h = 0xcbf29ce484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def encode_batch(chunk):
    """Encode one chunk of records -> quantized uint8 dict (pool worker)."""
    names = channel_names("full")
    scale = channel_scales(names)
    n = len(chunk)
    c = len(names)
    feat = np.zeros((n, c, N_KINDS), dtype=np.float32)
    dmask = np.zeros((n, N_KINDS), dtype=np.uint8)
    discard = np.full(n, NO_LABEL, dtype=np.uint8)
    drawn = np.full(n, NO_LABEL, dtype=np.uint8)
    heads = {h: np.full(n, NO_LABEL, dtype=np.uint8) for h in BINARY_HEADS}
    cands = {h: np.full(n, NO_LABEL, dtype=np.uint8) for h in BINARY_HEADS}
    gid = np.zeros(n, dtype=np.uint64)  # fnv1a64(game_id) for partition audit

    bad = 0
    for i, rec in enumerate(chunk):
        try:
            if rec.get("game_id"):
                gid[i] = fnv1a64(rec["game_id"])
            feat[i] = build_features(rec, mode="full")[:, :, 0]
            dmask[i] = discard_mask(rec)
            lab = rec["label"]
            ltype = lab.get("type")
            if ltype in ("discard", "riichi"):
                discard[i] = label_kind(rec)
            la = rec.get("legal_actions") or {}
            for h in BINARY_HEADS:
                legal = la.get(h)
                if not legal:
                    continue
                heads[h][i] = 1 if ltype == h else 0
                if ltype == h:
                    tiles = [lab["tile"]] if h == "riichi" else lab.get("tiles", [])
                elif isinstance(legal, list):
                    first = legal[0]
                    tiles = first.get("tiles", []) if isinstance(first, dict) else [first]
                else:
                    tiles = []  # bool decision heads carry no candidate tiles
                cands[h][i] = tile_to_kind(tiles[0]) if tiles else NO_LABEL
            ev = rec.get("last_event") or {}
            if ev.get("type") in DRAW_EVENTS and "tile" in ev:
                drawn[i] = tile_to_kind(ev["tile"])
        except Exception as e:  # noqa: BLE001 — a bad record must NEVER kill
            # the pool worker (worker death silently wedges the pool); keep
            # the sample as zeros / NO_LABEL so training ignores it
            bad += 1
            print(f"BAD_RECORD game={rec.get('game_id')} "
                  f"{type(e).__name__}: {e}", flush=True)
    if bad:
        print(f"chunk: {bad}/{len(chunk)} bad records zeroed", flush=True)

    out = {"feat": _quantize(feat, scale), "dmask": dmask, "discard": discard,
           "drawn": drawn, "gid": gid}
    for h in BINARY_HEADS:
        out[f"{h}_y"] = heads[h]
        out[f"{h}_cand"] = cands[h]
    return out


def _open_maybe_gz(p):
    return gzip.open(p, "rt", encoding="utf-8") if str(p).endswith(".gz") \
        else open(p, "r", encoding="utf-8")


def _read_jsonl_chunks(paths, chunk_lines, limit=0, game_ids=None):
    """Yield lists of parsed records, chunked, preserving file order.

    game_ids: optional set of allowed game_id values (train-partition filter;
    eval-holdout games must never enter SL training).
    """
    seen = 0
    skipped = 0
    for p in paths:
        try:
            with _open_maybe_gz(p) as fh:
                buf = []
                for line in fh:
                    if limit and seen >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if game_ids is not None and rec.get("game_id") not in game_ids:
                        continue
                    seen += 1
                    buf.append(rec)
                    if len(buf) >= chunk_lines:
                        yield buf
                        buf = []
                if buf:
                    yield buf
        except Exception as e:  # noqa: BLE001 — mid-write corpus files
            # can raise EOFError/OSError/UnicodeDecodeError/...; any uncaught
            # exception here would silently kill the pool's task-handler
            # thread and hang imap forever — always skip and continue
            skipped += 1
            print(f"SKIP {p}: {type(e).__name__}: {e} "
                  f"(likely mid-write; will re-process later)", flush=True)
        if limit and seen >= limit:
            break
    if skipped:
        print(f"skipped {skipped} unreadable files (mid-write)", flush=True)


def _merge_into(acc: dict, part: dict) -> None:
    for k, v in part.items():
        acc[k].append(v)
    acc["_n"] += len(part["feat"])


def _empty_acc(n_samples: int):
    acc = {"feat": [], "dmask": [], "discard": [], "drawn": [], "gid": []}
    for h in BINARY_HEADS:
        acc[f"{h}_y"] = []
        acc[f"{h}_cand"] = []
    acc["_n"] = n_samples
    return acc


def _concat_acc(acc: dict, scale: np.ndarray) -> dict:
    out = {}
    for k in list(acc):
        if k == "_n":
            continue
        out[k] = np.concatenate(acc[k], axis=0) if acc[k] else np.zeros((0,))
    out["scale"] = (scale / 255.0).astype(np.float32)  # decode: x = feat * scale
    return out


def write_shard(out_path: Path, arrays: dict) -> int:
    """Write one zstd-compressed npz shard; returns uncompressed bytes."""
    bio = io.BytesIO()
    np.savez(bio, **arrays)  # plain npz; zstd below is the only compression
    raw = bio.getbuffer().nbytes
    cctx = zstandard.ZstdCompressor(level=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(cctx.compress(bio.getvalue()))
    return raw


def load_shard(path) -> dict:
    """Read one shard -> dict of numpy arrays (uint8 + scale float32)."""
    with open(path, "rb") as fh:
        data = zstandard.ZstdDecompressor().decompress(fh.read())
    bio = io.BytesIO(data)
    z = np.load(bio)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description="records.jsonl(.gz) -> uint8 training shards")
    ap.add_argument("--records", nargs="*", default=None,
                    help="one or more records.jsonl(.gz) files")
    ap.add_argument("--dir", default=None, help="dir to glob --pattern in")
    ap.add_argument("--pattern", default="records-*.jsonl.gz",
                    help="glob for sharded jsonl (gzip supported)")
    ap.add_argument("--game-ids-file", default=None,
                    help="whitelist file of game_id per line (train partition; "
                         "eval-holdout games are excluded — captain directive)")
    ap.add_argument("--out", default="data/processed/tenhou/shards")
    ap.add_argument("--shard-size", type=int, default=65536, help="samples per shard")
    ap.add_argument("--chunk", type=int, default=4096, help="records per worker chunk")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="cap total samples (0=all)")
    args = ap.parse_args()

    if args.records:
        paths = list(args.records)
    elif args.dir:
        paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    else:
        paths = sorted(glob.glob("data/processed/tenhou/records-*.jsonl.gz"))
    if not paths:
        raise SystemExit("no records files found (--records or --dir/--pattern)")
    print(f"records files: {len(paths)} e.g. {paths[0]}", flush=True)

    game_ids = None
    if args.game_ids_file:
        with open(args.game_ids_file, "r", encoding="utf-8") as fh:
            game_ids = {ln.strip() for ln in fh if ln.strip()}
        print(f"game filter: {len(game_ids)} allowed game_ids", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # single-writer lock: two concurrent runs would interleave shard files
    lock = out_dir / ".preprocess.lock"
    if lock.exists():
        raise SystemExit(
            f"lock exists: {lock} — another preprocess is writing to {out_dir}; "
            f"remove the lock after confirming that run finished")
    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        _run_locked(args, paths, game_ids, out_dir)
    finally:
        lock.unlink(missing_ok=True)


def _run_locked(args, paths, game_ids, out_dir):
    names = channel_names("full")
    scale = channel_scales(names)
    np.save(out_dir / "channel_scale.npy", (scale / 255.0).astype(np.float32))
    print(f"channels={len(names)} shard_size={args.shard_size} "
          f"workers={args.workers} chunk={args.chunk}", flush=True)

    t0 = time.time()
    n_samples = 0
    n_shards = 0
    n_raw = 0
    nchunks = 0
    printed_shards = 0
    acc = _empty_acc(0)
    pool = Pool(args.workers)
    try:
        # imap_unordered over the chunk generator: main process parses JSON
        # while workers encode previous chunks (pipelined).
        for part in pool.imap_unordered(
                encode_batch,
                _read_jsonl_chunks(paths, args.chunk, args.limit, game_ids)):
            _merge_into(acc, part)
            n_samples += len(part["feat"])
            # flush full shards (acc holds lists of part-arrays; concat once)
            while acc["_n"] >= args.shard_size:
                cat = _concat_acc(acc, scale)
                split = {k: v[:args.shard_size] for k, v in cat.items()}
                rest_n = cat["feat"].shape[0] - args.shard_size
                for k in list(acc):
                    if k == "_n":
                        continue
                    acc[k] = [cat[k][args.shard_size:]]
                acc["_n"] = rest_n
                shard_path = out_dir / f"shard-{n_shards:06d}{_SHARD_SUFFIX}"
                n_raw += write_shard(shard_path, split)
                n_shards += 1
            nchunks += 1
            if nchunks % 8 == 0 or n_shards > printed_shards:
                printed_shards = n_shards
                dt = time.time() - t0
                print(f"processed {n_samples} samples, {n_shards} shards, "
                      f"{n_samples / max(dt, 1e-6):.0f} samples/s", flush=True)
    finally:
        pool.close()
        pool.join()

    if acc["_n"] > 0:
        shard_path = out_dir / f"shard-{n_shards:06d}{_SHARD_SUFFIX}"
        n_raw += write_shard(shard_path, _concat_acc(acc, scale))
        n_shards += 1

    dt = time.time() - t0
    manifest = {
        "n_shards": n_shards, "n_samples": n_samples,
        "channels": names, "shard_size": args.shard_size,
        "source_files": paths, "created_at": time.time(),
        "scale_path": str(out_dir / "channel_scale.npy"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE: {n_samples} samples -> {n_shards} shards in {dt:.1f}s "
          f"({n_samples / max(dt, 1e-6):.0f} samples/s, "
          f"uncompressed {n_raw / 2**30:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
