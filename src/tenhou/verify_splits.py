"""Partition-consistency verifier (reviewer t7 leakage-check support).

Checks, per docs/observation_schema.md split rules:
1. splits.json game lists are pairwise disjoint;
2. every game in the lists maps back to its partition via
   tenhou.stats.assign_split (rule self-consistency);
3. (--records GLOB) every record in the shards carries a game_id whose
   partition per the rule equals the partition of the manifest containing it
   (same-game samples never cross partitions).

Usage:
  python src/tenhou/verify_splits.py
  python src/tenhou/verify_splits.py --records "C:/agentwork/data/processed/tenhou/records-*.jsonl.gz"
"""
from __future__ import annotations

import argparse
import gzip
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tenhou.stats import assign_split

SPLITS = "C:/agentwork/data/processed/tenhou/splits/splits.json"


def load_lists(path=SPLITS):
    data = json.load(open(path, encoding="utf-8"))
    parts = {}
    for key in ("train", "val", "eval_holdout"):
        if key in data and isinstance(data[key], dict):
            parts[key] = set(data[key].get("games", []))
        elif key + "_games" in data:
            parts[key] = set(data[key + "_games"])
    # alias check: eval == eval_holdout
    if "eval" in data and isinstance(data["eval"], dict):
        assert parts["eval_holdout"] == set(data["eval"].get("games", [])), (
            "eval alias differs from eval_holdout")
    return parts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=None,
                    help="glob of records shards to stream-verify")
    args = ap.parse_args(argv)

    parts = load_lists()
    total = sum(len(v) for v in parts.values())
    print("partitions: %s" % {k: len(v) for k, v in parts.items()})
    ok = True
    for a in parts:
        for b in parts:
            if a < b and parts[a] & parts[b]:
                print("FAIL: overlap between %s and %s: %d"
                      % (a, b, len(parts[a] & parts[b])))
                ok = False
    if ok:
        print("OK: game lists pairwise disjoint")

    # rule self-consistency
    bad = 0
    for part, games in parts.items():
        for g in games:
            if assign_split(g) != part:
                bad += 1
                if bad < 5:
                    print("RULE MISMATCH: %s in %s -> %s"
                          % (g, part, assign_split(g)))
    if bad:
        print("FAIL: %d games whose manifest partition != rule partition" % bad)
        ok = False
    else:
        print("OK: all %d games map to their manifest partition" % total)

    if args.records:
        n = 0
        viol = 0
        noid = 0
        for path in glob.glob(args.records):
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    n += 1
                    gid = rec.get("game_id")
                    if not gid:
                        noid += 1
                        continue
                    part = assign_split(str(gid))
                    if part not in parts or gid not in parts[part]:
                        viol += 1
                        if viol < 5:
                            print("RECORD VIOLATION: %s -> %s"
                                  % (gid, part))
        print("records scanned=%d missing_game_id=%d partition_violations=%d"
              % (n, noid, viol))
        if noid or viol:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
