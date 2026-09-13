"""Re-export MJAI streams for all days that have done markers.

Used after notation/schema alignment changes to rewrite
data/processed/tenhou/mjai/{game_id}.jsonl from the raw XML without touching
records shards. Idempotent; safe to rerun (only mjai/ files are written).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.paths import raw_dir, processed_dir
from tenhou.mjai_export import export_game_file

RAW = str(raw_dir() / "mjlog")
DONE_DIR = str(processed_dir() / "tenhou/done")
MJAI = str(processed_dir() / "tenhou/mjai")


def main(argv=None) -> int:
    done_dates = sorted(
        f.split(".")[0] for f in os.listdir(DONE_DIR)
        if f.endswith(".json") and not f.endswith(".json.bak")
    )
    if argv and argv[0] == "--all":
        done_dates = None
    files = []
    for year_dir in ("2025", "2026"):
        files.extend(glob.glob(os.path.join(RAW, year_dir, "*.xml.gz")))
    if done_dates is not None:
        files = [f for f in files
                 if os.path.basename(f)[:8] in set(done_dates)]
    files.sort()
    t0 = time.time()
    n = 0
    for f in files:
        try:
            export_game_file(f, MJAI)
            n += 1
        except Exception as e:  # noqa: BLE001
            print("FAIL", f, e)
        if n % 500 == 0:
            print("[%s] re-exported %d/%d" % (time.strftime("%H:%M:%S"),
                                              n, len(files)), flush=True)
    print("DONE %d files in %.0fs" % (n, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
