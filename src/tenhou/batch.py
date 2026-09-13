"""Background batch pipeline for the tenhou phoenix corpus.

For each date in [start, end]:
  1. listing  : scc hourly files -> 00a9 log ids (cached)
  2. download : mjlog XML -> data/raw/mjlog/{yyyy}/{logid}.xml.gz (resumable)
  3. extract  : parse + decision records -> records_{date}.jsonl.gz (+day stats)
  4. mark done -> processed/tenhou/done/{date}.json (resume skips it)

Usage:
  python -m tenhou.batch --start 20260101 --end 20260825 --rate 3 --workers 4       --extract-workers 4 --log data/processed/tenhou/logs/batch.log
"""
from __future__ import annotations

import argparse
import collections
import gzip
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.paths import raw_dir, processed_dir
from tenhou.listing import phoenix_ids_for_day
from tenhou.download import download_logs
from tenhou.mjlog_parser import parse_game_file
from tenhou.extract import extract_game, GameStats
from tenhou.mjai_export import export_game

MJAI_DIR = str(processed_dir() / "tenhou/mjai")

RAW_DIR = str(raw_dir() / "mjlog")
OUT_DIR = str(processed_dir() / "tenhou")
LIST_CACHE = str(raw_dir() / "listings")


def logline(log, msg: str) -> None:
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    if log is not None:
        log.write(line + "\n")
        log.flush()


def _extract_file(path: str):
    """Worker: parse+extract one file -> records + mjai events + stats."""
    try:
        game = parse_game_file(path)
    except Exception:
        return None
    logid = os.path.basename(path).split(".")[0]
    game["log_id"] = logid
    records = []
    stats = GameStats()
    extract_game(game, records, stats)
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    # mjai protocol stream (schema section 7.5): one file per game
    try:
        export_game(game, MJAI_DIR)
    except Exception:
        pass
    return {"logid": logid, "lines": lines,
            "stats": stats.as_dict()}


def process_date(date: str, args, log, done_file) -> dict:
    t0 = time.time()
    logline(log, "== date %s: listing" % date)
    rows = phoenix_ids_for_day(date, LIST_CACHE)
    ids = [r["logid"] for r in rows]
    logline(log, "date %s: %d phoenix ids" % (date, len(ids)))
    if not ids:
        _mark_done(done_file, {"date": date, "games": 0, "empty": True})
        return {"date": date, "games": 0}

    logline(log, "date %s: downloading %d logs" % (date, len(ids)))
    dl = download_logs(ids, RAW_DIR, args.rate, args.retries, args.workers,
                       log, progress_every=100)
    logline(log, "date %s: download done ok=%d skip=%d miss=%d fail=%d (%.0fs)"
            % (date, dl["ok"], dl["skipped"], dl["missing"], dl["failed"],
               dl["elapsed"]))

    logline(log, "date %s: extracting" % date)
    files = sorted(glob.glob(os.path.join(RAW_DIR, date[:4], date + "*.xml.gz")))
    out_gz = os.path.join(OUT_DIR, "records-%s.jsonl.gz" % date)
    # atomic write: consumers (ml-engineer preprocess) must never observe a
    # half-written shard; write to .tmp then os.replace on completion
    tmp_gz = out_gz + ".tmp"
    day_stats = {
        "date": date, "games_parsed": 0, "records": 0,
        "rounds": 0, "agari": 0, "ryuukyoku": 0,
        "ryuukyoku_types": {}, "anomalies": 0, "parse_fail": 0,
    }
    label_counter = collections.Counter()
    import multiprocessing
    n_extract = max(1, int(getattr(args, "extract_workers", 4)))
    with gzip.open(tmp_gz, "wt", encoding="utf-8") as out:
        done_n = 0
        with multiprocessing.Pool(n_extract) as pool:
            for res in pool.imap_unordered(_extract_file, files):
                if res is None:
                    day_stats["parse_fail"] += 1
                    done_n += 1
                    continue
                day_stats["games_parsed"] += 1
                day_stats["records"] += len(res["lines"])
                for line in res["lines"]:
                    out.write(line + "\n")
                    try:
                        label_counter[json.loads(line)["label"]["type"]] += 1
                    except (ValueError, KeyError):
                        pass
                st = res["stats"]
                day_stats["rounds"] += st["rounds"]
                day_stats["agari"] += st["agari"]
                day_stats["ryuukyoku"] += st["ryuukyoku"]
                day_stats["anomalies"] += st["anomalies"]
                for k, v in st["ryuukyoku_types"].items():
                    day_stats["ryuukyoku_types"][k] = day_stats["ryuukyoku_types"].get(k, 0) + v
                done_n += 1
                if done_n % 100 == 0:
                    logline(log, "date %s: extracted %d/%d files"
                            % (date, done_n, len(files)))
    day_stats["labels"] = dict(label_counter)
    day_stats["elapsed"] = time.time() - t0
    os.replace(tmp_gz, out_gz)
    stats_path = os.path.join(OUT_DIR, "daystats", "stats_%s.json" % date)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(day_stats, f, ensure_ascii=False, indent=1)
    _mark_done(done_file, day_stats)
    logline(log, "date %s: DONE games=%d records=%d (%.0fs)"
            % (date, day_stats["games_parsed"], day_stats["records"],
               day_stats["elapsed"]))
    return day_stats


def _mark_done(done_file: str, data: dict) -> None:
    os.makedirs(os.path.dirname(done_file), exist_ok=True)
    with open(done_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _is_done(done_file: str) -> bool:
    if not os.path.exists(done_file):
        return False
    try:
        d = json.load(open(done_file, encoding="utf-8"))
        return not d.get("empty", False) and d.get("games_parsed", 0) > 0
    except ValueError:
        return False


def iter_dates(start: str, end: str):
    import datetime
    d = datetime.datetime.strptime(start, "%Y%m%d")
    e = datetime.datetime.strptime(end, "%Y%m%d")
    while d <= e:
        yield d.strftime("%Y%m%d")
        d += datetime.timedelta(days=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="YYYYMMDD")
    ap.add_argument("--end", required=True, help="YYYYMMDD")
    ap.add_argument("--rate", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--log", default=None)
    ap.add_argument("--limit-days", type=int, default=0,
                    help="stop after N days (0 = no limit)")
    ap.add_argument("--extract-workers", type=int, default=4,
                    help="parallel parse/extract workers per date")
    args = ap.parse_args(argv)

    log = open(args.log, "a", encoding="utf-8") if args.log else None
    try:
        done_dir = os.path.join(OUT_DIR, "done")
        n_days = 0
        for date in iter_dates(args.start, args.end):
            if args.limit_days and n_days >= args.limit_days:
                logline(log, "limit-days reached (%d), stopping" % n_days)
                break
            done_file = os.path.join(done_dir, date + ".json")
            if _is_done(done_file):
                logline(log, "date %s already done, skipping" % date)
                continue
            process_date(date, args, log, done_file)
            n_days += 1
        logline(log, "BATCH DONE (%d days)" % n_days)
    finally:
        if log:
            log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
