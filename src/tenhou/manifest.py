"""Shards manifest for downstream consumers (ml-engineer preprocess).

Writes data/processed/tenhou/shards_manifest.json:
- glob convention for record shards
- per-shard date / path / game & record counts / label histogram
- split manifests (train / val / eval_holdout) with counts
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import processed_dir

OUT = str(processed_dir() / "tenhou")


def main(argv=None) -> int:
    shards = sorted(glob.glob(os.path.join(OUT, "records-*.jsonl.gz")))
    entries = []
    for s in shards:
        date = os.path.basename(s)[len("records-"):-len(".jsonl.gz")]
        daystat = os.path.join(OUT, "daystats", "stats_%s.json" % date)
        games = records = None
        labels = {}
        if os.path.exists(daystat):
            try:
                d = json.load(open(daystat, encoding="utf-8"))
                games = d.get("games_parsed")
                records = d.get("records")
                labels = d.get("labels", {})
            except ValueError:
                pass
        entries.append({"date": date, "path": s, "games": games,
                        "records": records, "labels": labels})
    splits_meta = {}
    splits_path = os.path.join(OUT, "splits", "splits.json")
    if os.path.exists(splits_path):
        try:
            splits_meta = json.load(open(splits_path, encoding="utf-8"))
        except ValueError:
            pass
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "glob": str(processed_dir() / "tenhou" / "records-*.jsonl.gz"),
        "shards": entries,
        "splits": splits_meta,
    }
    with open(os.path.join(OUT, "shards_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("wrote shards_manifest.json (%d shards)" % len(entries))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
