"""Parse tenhou scc{yyyymmdd}{hh}.html.gz phoenix-lobby listings.

Row format (2026)::
    00:02 | 16 | 三鳳南喰赤－ | <a href="http://tenhou.net/0/?log=2026082400gm-00b9-0000-bb9b0a36">牌譜</a> | names...

We keep only 四鳳南 (four-player south, lobby code 00a9). The lobby code is
the segment right after "gm-" in the log id (e.g. "00a9" = 0xA9:
bit 0x08 hanchan, bit 0x10 sanma; 0xA9 has hanchan set and sanma clear).

Older years (2009-2012) may have different row layouts; we extract the log id
and time fields leniently.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import urllib.request
from typing import Dict, List, Optional, Tuple

ROW_RE = re.compile(
    r"(\d{1,2}:\d{2})?[^<]*?<a href=\"[^\"]*[?&]log=([0-9A-Za-z_-]+)\""
)
PHOENIX_LOBBY = "00a9"

DAT_URL = "https://tenhou.net/sc/raw/dat/{rel}"  # rel may carry a {yyyy}/ prefix


def is_gzip(data: bytes) -> bool:
    return data[:2] == b"\x1f\x8b"


def fetch_scc(rel_path: str, timeout: int = 30, retries: int = 3) -> Optional[bytes]:
    """Download one scc listing file; None if missing (404) or unreadable."""
    url = DAT_URL.format(rel=rel_path)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError("failed to fetch %s: %s" % (url, last))


def parse_scc_bytes(data: bytes) -> List[Dict]:
    """Parse an scc html.gz body into rows.

    Returns [{"time": "HH:MM"|None, "logid": str, "lobby": str|None}...]
    """
    if not is_gzip(data):
        return []
    try:
        html = gzip.decompress(data).decode("utf-8", "replace")
    except OSError:
        return []
    rows = []
    for line in html.splitlines():
        m = re.search(r"log=([0-9A-Za-z_-]+)", line)
        if not m:
            continue
        logid = m.group(1)
        tm = re.match(r"\s*(\d{1,2}:\d{2})", line)
        parts = logid.split("-")
        lobby = parts[1] if len(parts) >= 2 else None
        rows.append({"time": tm.group(1) if tm else None,
                     "logid": logid, "lobby": lobby})
    return rows


def phoenix_rows(rows: List[Dict]) -> List[Dict]:
    return [r for r in rows if r.get("lobby") == PHOENIX_LOBBY]


_SCHEME_CACHE: Dict[str, str] = {}


def _fetch_cached(rel: str, cache_dir: Optional[str], date: str) -> Optional[bytes]:
    if cache_dir:
        cache_path = os.path.join(cache_dir, "scc_%s" % date,
                                  rel.replace("/", "_"))
        if os.path.exists(cache_path):
            return open(cache_path, "rb").read()
    data = fetch_scc(rel)
    if data is not None and cache_dir:
        cache_path = os.path.join(cache_dir, "scc_%s" % date,
                                  rel.replace("/", "_"))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(data)
    return data


def _fetch_zip(year: str, cache_dir: Optional[str]) -> Optional[bytes]:
    """Download scraw{year}.zip (cached at data/raw/scraw/{year}.zip).

    The yearly zips (~450MB) are currently 404 on the server; the captain
    runs a 30-minute probe and will announce when they return. When they do,
    this fallback lets the batch resume 2009-2025 without code changes.
    """
    zip_dir = "C:/agentwork/data/raw/scraw"
    zip_path = os.path.join(zip_dir, "scraw%s.zip" % year)
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
        try:
            with open(zip_path, "rb") as f:
                return f.read()
        except OSError:
            return None
    url = "https://tenhou.net/sc/raw/dat/scraw%s.zip" % year
    data = fetch_scc(url)
    if data is None:
        return None
    os.makedirs(zip_dir, exist_ok=True)
    with open(zip_path + ".tmp", "wb") as f:
        f.write(data)
    os.replace(zip_path + ".tmp", zip_path)
    return data


def _extract_scc_from_zip(data: bytes, date: str,
                          cache_dir: Optional[str]) -> Optional[bytes]:
    """Extract the scc listing file(s) of YYYYMMDD from a scraw zip body.

    Returns the concatenation of matching members (usually exactly one
    scc{yyyymmdd}.html.gz). Members are also cached individually.
    """
    import io
    import zipfile
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return None
    names = [n for n in zf.namelist()
             if os.path.basename(n).startswith("scc%s" % date)
             and n.endswith(".html.gz")]
    if not names:
        return None
    out = b""
    for n in sorted(names):
        body = zf.read(n)
        out += body
        if cache_dir:
            cache_path = os.path.join(cache_dir, "scc_%s" % date,
                                      os.path.basename(n))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(body)
    return out


def phoenix_ids_for_day(date: str, cache_dir: Optional[str] = None) -> List[Dict]:
    """Download the scc listing of YYYYMMDD and return 00a9 rows.

    Recent dates use hourly files (scc{yyyymmdd}{hh}.html.gz, no prefix);
    older dates only have DAILY files ({yyyy}/scc{yyyymmdd}.html.gz).
    Schemes are tried in order and remembered per date.
    """
    year = date[:4]
    out: List[Dict] = []
    scheme = _SCHEME_CACHE.get(date)
    if scheme == "daily":
        data = _fetch_cached("%s/scc%s.html.gz" % (year, date), cache_dir, date)
        if data is None:
            data = _fetch_cached("scc%s.html.gz" % date, cache_dir, date)
        if data is None:
            return out
        rows = phoenix_rows(parse_scc_bytes(data))
        for r in rows:
            r["date"] = date
            r["hour"] = None
        return rows
    if scheme == "hourly":
        for hour in range(24):
            rel = "scc%s%02d.html.gz" % (date, hour)
            data = _fetch_cached(rel, cache_dir, date)
            if data is None:
                continue
            rows = phoenix_rows(parse_scc_bytes(data))
            for r in rows:
                r["date"] = date
                r["hour"] = hour
            out.extend(rows)
        return out
    # unknown: try hourly (no prefix) first
    got = 0
    for hour in range(24):
        rel = "scc%s%02d.html.gz" % (date, hour)
        data = _fetch_cached(rel, cache_dir, date)
        if data is None:
            continue
        rows = phoenix_rows(parse_scc_bytes(data))
        got += len(rows)
        for r in rows:
            r["date"] = date
            r["hour"] = hour
        out.extend(rows)
    if got:
        _SCHEME_CACHE[date] = "hourly"
        return out
    # fall back to the daily file (year-prefixed first)
    data = _fetch_cached("%s/scc%s.html.gz" % (year, date), cache_dir, date)
    if data is None:
        data = _fetch_cached("scc%s.html.gz" % date, cache_dir, date)
    if data is not None:
        rows = phoenix_rows(parse_scc_bytes(data))
        _SCHEME_CACHE[date] = "daily"
        for r in rows:
            r["date"] = date
            r["hour"] = None
        return rows
    # zip fallback: scraw{year}.zip (2009-2025; currently 404, see captain
    # note). When the archive returns this resumes old years transparently.
    zip_data = _fetch_zip(year, cache_dir)
    if zip_data is None:
        return out
    data = _extract_scc_from_zip(zip_data, date, cache_dir)
    if data is None:
        return out
    rows = phoenix_rows(parse_scc_bytes(data))
    _SCHEME_CACHE[date] = "daily"
    for r in rows:
        r["date"] = date
        r["hour"] = None
    return rows


def ids_file_for_day(date: str, cache_dir: str, out_dir: str) -> str:
    rows = phoenix_ids_for_day(date, cache_dir)
    path = os.path.join(out_dir, "listings", "scc_%s" % date, "ids.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--cache-dir", default="C:/agentwork/data/raw/listings")
    ap.add_argument("--out-dir", default="C:/agentwork/data/processed/tenhou")
    ap.add_argument("--print", action="store_true", dest="print_rows")
    args = ap.parse_args(argv)

    rows = phoenix_ids_for_day(args.date, args.cache_dir)
    if args.print_rows:
        for r in rows:
            print("%s %s" % (r["time"], r["logid"]))
    else:
        path = ids_file_for_day(args.date, args.cache_dir, args.out_dir)
        print("phoenix 00a9 rows: %d -> %s" % (len(rows), path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
