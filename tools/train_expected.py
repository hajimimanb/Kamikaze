# -*- coding: utf-8 -*-
"""Exact train-partition record counter -> real expected shard count (truthful)."""
import collections, glob, gzip, json, os, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
from utils.paths import processed_dir, logs_dir

BASE = str(processed_dir() / "tenhou")
OUT = str(logs_dir() / "train" / "train_expected.json")

def main():
    t0 = time.time()
    train = set()
    with open(BASE + "/splits/train_games.txt", encoding="utf-8") as f:
        for ln in f:
            g = ln.strip()
            if g:
                train.add(g)
    total = 0
    tr = 0
    files = sorted(glob.glob(BASE + "/records-*.jsonl.gz"))
    for fi, f in enumerate(files):
        try:
            with gzip.open(f, "rt", encoding="utf-8") as g:
                for ln in g:
                    if not ln.strip():
                        continue
                    total += 1
                    try:
                        if json.loads(ln).get("game_id") in train:
                            tr += 1
                    except Exception:
                        pass
        except Exception as e:
            print("skip", os.path.basename(f), str(e)[:60], flush=True)
        if (fi + 1) % 20 == 0:
            print("scanned %d/%d files, total=%d train=%d" % (fi + 1, len(files), total, tr), flush=True)
    import math
    exp = int(math.ceil(tr / 65536.0)) if tr else 0
    json.dump({"total_records": total, "train_records": tr, "expected_shards": exp,
               "elapsed_s": round(time.time() - t0, 1), "done_at": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(OUT, "w", encoding="utf-8"))
    print("DONE total=%d train=%d expected_shards=%d" % (total, tr, exp), flush=True)

if __name__ == "__main__":
    main()