# -*- coding: utf-8 -*-
"""Partition-leak audit: verify no eval_holdout / val games leaked into shards.

Each shard stores fnv1a64(game_id) per sample (preprocess.py). This script
hashes the split-list game ids the same way and reports any intersection.
Exit 0 = clean, 1 = leak found.

Usage:
  python src/model/verify_partition.py --shards data/processed/tenhou/shards
      --holdout data/processed/tenhou/splits/eval_holdout_games.txt
      [--val data/processed/tenhou/splits/val_games.txt]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.preprocess import fnv1a64, load_shard  # noqa: E402


def load_hashes(ids_file: str) -> set:
    with open(ids_file, "r", encoding="utf-8") as fh:
        return {fnv1a64(ln.strip()) for ln in fh if ln.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description="shard partition-leak audit")
    ap.add_argument("--shards", required=True, help="dir with shard-*.npz.zst")
    ap.add_argument("--holdout", required=True, help="eval_holdout_games.txt")
    ap.add_argument("--val", default=None, help="val_games.txt (optional)")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(Path(args.shards) / "shard-*.npz.zst")))
    if not paths:
        print("no shards found")
        return 1
    holdout = load_hashes(args.holdout)
    val = load_hashes(args.val) if args.val else set()
    print(f"shards={len(paths)} holdout_ids={len(holdout)} val_ids={len(val)}")

    n_samples = 0
    holdout_hits = 0
    val_hits = 0
    for p in paths:
        d = load_shard(p)
        g = d["gid"]
        n_samples += len(g)
        if len(holdout):
            holdout_hits += int(np.isin(g, list(holdout)).sum())
        if len(val):
            val_hits += int(np.isin(g, list(val)).sum())

    print(f"total_samples={n_samples} holdout_leak={holdout_hits} "
          f"val_leak={val_hits}")
    if holdout_hits or val_hits:
        print("LEAK DETECTED — do not train on these shards")
        return 1
    print("PARTITION AUDIT CLEAN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
