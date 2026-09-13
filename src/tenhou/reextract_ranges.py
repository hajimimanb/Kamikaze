"""Re-extract records shards for given date ranges with the current code.

Deletes done markers + shards first (the raw XMLs stay), then delegates to
tenhou.batch.main. Used after extract.py label fixes.
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.paths import processed_dir

OUT = str(processed_dir() / "tenhou")
DONE = os.path.join(OUT, "done")


def reset_date(date: str) -> None:
    marker = os.path.join(DONE, date + ".json")
    if os.path.exists(marker):
        os.remove(marker)
    for shard in glob.glob(os.path.join(OUT, "records-%s.jsonl.gz" % date)):
        os.remove(shard)


def main(argv=None) -> int:
    ranges = argv or ["20260101-20260120", "20260819-20260824"]
    for rng in ranges:
        start, end = rng.split("-")
        import datetime
        d = datetime.datetime.strptime(start, "%Y%m%d")
        e = datetime.datetime.strptime(end, "%Y%m%d")
        while d <= e:
            reset_date(d.strftime("%Y%m%d"))
            d += datetime.timedelta(days=1)
    from tenhou import batch
    for rng in ranges:
        start, end = rng.split("-")
        batch.main(["--start", start, "--end", end,
                    "--rate", "3", "--workers", "4",
                    "--log", os.path.join(OUT, "logs", "reextract.log")])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
