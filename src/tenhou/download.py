"""Rate-limited, resumable mjlog XML downloader.

Download:  https://tenhou.net/0/log/?{logid}  ->  data/raw/mjlog/{yyyy}/{logid}.xml.gz

Features:
- total rate cap (default 3 req/s) with worker threads to hide per-request latency
- exponential backoff with jitter; 404 = missing (no retry)
- resume: existing valid gzip files are skipped
- atomic writes (tmp + replace), parseable progress log for background runs
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import raw_dir

XML_URL = "https://tenhou.net/0/log/?{logid}"
DEFAULT_OUT = str(raw_dir() / "mjlog")


def logid_year(logid: str) -> str:
    # log ids look like 2026082400gm-00a9-0000-8f7029ce
    return logid[:4]


def is_valid_gz(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        if len(head) < 4 or head[:2] != b"\x1f\x8b":
            return False
        with open(path, "rb") as f:
            gzip.decompress(f.read())
        return True
    except (OSError, EOFError):
        return False


def load_ids(path: str) -> List[str]:
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):  # ids.jsonl from listing
                try:
                    obj = json.loads(line)
                    ids.append(obj["logid"])
                except (ValueError, KeyError):
                    continue
            else:
                ids.append(line.split()[0])
    return ids


def fetch_one(logid: str, timeout: int = 40) -> Optional[bytes]:
    url = XML_URL.format(logid=logid)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; riichi-ai/0.1; data pipeline)",
        "Referer": "https://tenhou.net/0/",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not data.lstrip()[:20].startswith(b"<mjloggm"):
        return None  # HTML error page served as 200
    return data


class _RateLimiter:
    """Global cap on request starts (requests per second)."""

    def __init__(self, rate: float):
        self.interval = 1.0 / max(rate, 0.01)
        self.lock = threading.Lock()
        self.next_slot = 0.0

    def acquire(self) -> None:
        with self.lock:
            now = time.time()
            wait = self.next_slot - now
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            self.next_slot = max(now, self.next_slot) + self.interval


def _dl_one(logid: str, out_dir: str, retries: int, limiter: _RateLimiter,
            stats: dict, lock: threading.Lock) -> str:
    limiter.acquire()
    year = logid_year(logid)
    path = os.path.join(out_dir, year, logid + ".xml.gz")
    if os.path.exists(path) and is_valid_gz(path):
        with lock:
            stats["skipped"] += 1
        return "skipped"
    data = None
    outcome = "failed"
    for attempt in range(retries):
        try:
            data = fetch_one(logid)
            outcome = "ok" if data is not None else "missing"
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                outcome = "missing"
                break
            time.sleep((2 ** attempt) + random.random())
        except Exception:
            time.sleep((2 ** attempt) + random.random())
    if data is None:
        with lock:
            stats[outcome] += 1
        return outcome
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(gzip.compress(data, 6))
    os.replace(tmp, path)
    with lock:
        stats["ok"] += 1
        stats["bytes"] += len(data)
    return "ok"

def download_logs(ids: List[str], out_dir: str = DEFAULT_OUT, rate: float = 3.0,
                  retries: int = 6, workers: int = 4,
                  log: Optional[object] = None,
                  progress_every: int = 50) -> Dict:
    """Download ids concurrently (total rate capped at rate req/s)."""
    stats = {"total": len(ids), "ok": 0, "skipped": 0, "missing": 0,
             "failed": 0, "bytes": 0, "start": time.time()}
    limiter = _RateLimiter(rate)
    lock = threading.Lock()

    def logline(msg: str) -> None:
        if log is not None:
            with lock:
                log.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))
                log.flush()

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_dl_one, logid, out_dir, retries, limiter, stats,
                            lock): logid for logid in ids}
        for fut in as_completed(futs):
            done += 1
            if done % progress_every == 0:
                elapsed = time.time() - stats["start"]
                logline("progress %d/%d ok=%d skip=%d miss=%d fail=%d %.2f req/s"
                        % (done, stats["total"], stats["ok"], stats["skipped"],
                           stats["missing"], stats["failed"],
                           done / elapsed if elapsed > 0 else 0))

    stats["elapsed"] = time.time() - stats["start"]
    logline("DONE total=%d ok=%d skipped=%d missing=%d failed=%d elapsed=%.1fs"
            % (stats["total"], stats["ok"], stats["skipped"], stats["missing"],
               stats["failed"], stats["elapsed"]))
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", help="file with log ids (jsonl or plain text)")
    ap.add_argument("--date", help="YYYYMMDD: build id list via listing.py")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--rate", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--log", default=None, help="progress log file")
    args = ap.parse_args(argv)

    if args.ids:
        ids = load_ids(args.ids)
    elif args.date:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from tenhou.listing import phoenix_ids_for_day
        rows = phoenix_ids_for_day(args.date)
        ids = [r["logid"] for r in rows]
    else:
        ap.error("one of --ids / --date is required")

    log = open(args.log, "a", encoding="utf-8") if args.log else None
    try:
        stats = download_logs(ids, args.out, args.rate, args.retries,
                              args.workers, log)
    finally:
        if log:
            log.close()
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
