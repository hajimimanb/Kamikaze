# -*- coding: utf-8 -*-
"""Transfer-train the NEW decision heads (ron/tsumo/kyushu) on top of the
stage-2 full checkpoint, on 1/4 of the corpus re-extracted with the extended
labels. The shared trunk and the original heads stay FROZEN; only the new
heads receive gradients (transfer learning per user request).

Usage: python model/train_sl_transfer.py --shards <tensors dir> --ckpt <stage2 best.pt>
"""
import argparse, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.features import BINARY_HEADS, feature_channels
from model.model import MultiHeadRiichiNet, get_device
from model.train_sl import ShardDataset, apply_transfer

NEW_HEADS = ("ron", "tsumo", "kyushu")

def freeze_old(model):
    kept = 0
    for name, p in model.named_parameters():
        keep = any(name.startswith("binary." + h + ".") for h in NEW_HEADS)
        p.requires_grad = keep
        kept += int(keep)
    print("trainable params: %d (new heads only: %s)" % (kept, ", ".join(NEW_HEADS)),
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True, help="preprocessed 1/4 tensors dir")
    ap.add_argument("--ckpt", required=True, help="stage-2 best.pt")
    ap.add_argument("--out", default="checkpoints/sl/transfer")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--blocks", type=int, default=50)
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = epochs x shards/bs")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="amp")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = get_device()
    n_channels = feature_channels("full")
    model = MultiHeadRiichiNet(n_channels, channels=args.channels,
                               n_blocks=args.blocks, binary_heads=BINARY_HEADS)
    ck = torch.load(args.ckpt, map_location="cpu")
    apply_transfer(model, ck, src_in=0)  # shape-matched keys copied; new heads random
    freeze_old(model)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    loader = ShardDataset(args.shards, "full", args.batch_size, args.seed, device)
    n_shards = len(loader.paths)
    samples_per_epoch = n_shards * 65536
    steps_per_epoch = max(1, samples_per_epoch // args.batch_size)
    total = args.max_steps or (args.epochs * steps_per_epoch)
    warmup = min(500, total // 20)
    print("shards=%d steps/epoch=%d total=%d warmup=%d device=%s"
          % (n_shards, steps_per_epoch, total, warmup, device), flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    steps = 0
    epoch = 0
    run_loss = 0.0
    run_n = 0
    t0 = time.time()
    best = {h: 0.0 for h in NEW_HEADS}
    acc_buf = {}  # head -> [hits, n, pos_hits, pos_n]
    for batch in loader.batches():
        if steps >= total:
            break
        # warmup -> cosine over the remaining steps
        if steps < warmup:
            lr = args.lr * (steps + 1) / warmup
        else:
            prog = min((steps - warmup) / max(total - warmup, 1), 1.0)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = lr
        x, mask, yd, yd_valid, drawn, heads = batch
        cands = {h: heads[h][2] for h in model.binary_heads}
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=args.amp and device.type == "cuda"):
            logits = model(x, masks={"discard": mask}, candidates=cands)
            loss = None
            for h in NEW_HEADS:
                y, v, _c = heads[h]
                if v.any():
                    term = F.cross_entropy(logits[h][v], y[v])
                    loss = term if loss is None else loss + term
        if loss is None:
            continue
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()
        steps += 1
        run_loss += float(loss.detach())
        run_n += 1
        with torch.no_grad():
            dm = yd_valid
            if bool(dm.any()):
                pd = logits["discard"].argmax(1)[dm]
                b = acc_buf.setdefault("discard", [0, 0, 0, 0])
                b[0] += int((pd == yd[dm]).sum()); b[1] += int(dm.sum())
            for h in NEW_HEADS:
                y, v, _c = heads[h]
                if bool(v.any()):
                    pr = logits[h][v].argmax(1)
                    correct = pr == y[v]
                    yv = y[v]
                    pos = yv == 1
                    b = acc_buf.setdefault(h, [0, 0, 0, 0])  # [pos_hits,pos_n,neg_hits,neg_n]
                    b[0] += int(correct[pos].sum()); b[1] += int(pos.sum())
                    b[2] += int(correct[~pos].sum()); b[3] += int((~pos).sum())
        # accumulate head accs for the log
        if steps % 200 == 0:
            dt = time.time() - t0
            parts = ["[transfer] step %d/%d epoch %d loss=%.4f lr=%.2e elapsed=%.0fs"
                     % (steps, total, steps // steps_per_epoch + 1,
                        run_loss / max(run_n, 1), lr, dt)]
            for h in ["discard"] + list(NEW_HEADS):
                b = acc_buf.get(h)
                if b and (b[1] + b[3]) > 0:
                    acc = (b[0] + b[2]) / max(b[1] + b[3], 1)
                    if h == "discard":
                        parts.append("discard=%.3f" % acc)
                    else:
                        pos_acc = b[0] / max(b[1], 1)
                        neg_acc = b[2] / max(b[3], 1)
                        pos_rate = b[1] / max(b[1] + b[3], 1)
                        parts.append("%s=%.3f(p=%.3f,neg=%.3f,pos=%.4f)"
                                     % (h, acc, pos_acc, neg_acc, pos_rate))
                else:
                    parts.append("%s=--" % h)
            print(" ".join(parts), flush=True)
            run_loss = 0.0
            run_n = 0
            acc_buf = {}
        if steps % args.eval_every == 0:
            _eval_new_heads(model, loader, device, args, best, steps)
        if steps % steps_per_epoch == 0:
            epoch = steps // steps_per_epoch
            torch.save({"model": model.state_dict(), "config": vars(args),
                        "epoch": epoch, "heads": BINARY_HEADS},
                       os.path.join(args.out, "transfer_epoch%d.pt" % epoch))
            print("saved transfer_epoch%d.pt" % epoch, flush=True)
    torch.save({"model": model.state_dict(), "config": vars(args),
                "epoch": args.epochs, "heads": BINARY_HEADS},
               os.path.join(args.out, "transfer_final.pt"))
    print("TRANSFER DONE steps=%d elapsed=%.0fs" % (steps, time.time() - t0))


def _eval_new_heads(model, loader, device, args, best, steps):
    model.eval()
    # [pos_hits, pos_n, neg_hits, neg_n]；discard 只用前两个
    hits = {h: [0, 0, 0, 0] for h in ["discard"] + list(NEW_HEADS)}
    count = 0
    with torch.no_grad():
        for batch in loader.batches():
            x, mask, yd, yd_valid, drawn, heads = batch
            cands = {h: heads[h][2] for h in model.binary_heads}
            logits = model(x, masks={"discard": mask}, candidates=cands)
            for h in NEW_HEADS:
                y, v, _c = heads[h]
                if v.any():
                    pr = logits[h][v].argmax(1)
                    correct = pr == y[v]
                    yv = y[v]
                    pos = yv == 1
                    b = hits[h]
                    b[0] += int(correct[pos].sum().item()); b[1] += int(pos.sum().item())
                    b[2] += int(correct[~pos].sum().item()); b[3] += int((~pos).sum().item())
            dm = yd_valid
            if bool(dm.any()):
                pd = logits["discard"].argmax(1)[dm]
                b = hits["discard"]
                b[0] += int((pd == yd[dm]).sum().item()); b[1] += int(dm.sum().item())
            count += int(x.shape[0])
            if count >= 20000:
                break
    line = []
    for h in ["discard"] + list(NEW_HEADS):
        b = hits[h]
        total_n = b[1] + b[3]
        acc = (b[0] + b[2]) / max(total_n, 1)
        best.setdefault(h, 0.0)
        best[h] = max(best[h], acc)
        if h == "discard":
            line.append("discard=%.3f(best %.3f, n=%d)" % (acc, best[h], total_n))
        else:
            pos_acc = b[0] / max(b[1], 1)
            neg_acc = b[2] / max(b[3], 1)
            line.append("%s=%.3f(p=%.3f,neg=%.3f,best %.3f,n=%d)"
                        % (h, acc, pos_acc, neg_acc, best[h], total_n))
    print("[eval @%d] %s" % (steps, "  ".join(line)), flush=True)
    model.train()


if __name__ == "__main__":
    main()
