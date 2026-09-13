# -*- coding: utf-8 -*-
"""amae-koromo mirror API client: paginate 王座 (mode=16) game lists.

Endpoint (per team contract / docs/observation_schema.md §7):
  GET {mirror}api/v2/pl4/games/{end_ms}/{start_ms}?limit=100&descending=true&mode=16
  -> JSON array of game summaries (uuid, start_time, end_time, players, ...)

Auth: authorization: Bearer <cap token> (see cap_client.py). A 429 whose body
contains "x-cap-token-required" means the token must be refreshed; generic 429s
are rate limits and are retried with backoff. Mirrors are rotated on failure.
"""
import io
import json
import os
import random
import time
import urllib.error
import urllib.request

import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import raw_dir
import cap_client

MIRRORS = [
    "https://5-data.amae-koromo.com/",
    "https://1.data.amae-koromo.com/",
    "https://4.data.amae-koromo.com/",
    "https://2.data.amae-koromo.com/",
]

API_V2 = "api/v2/pl4"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0")

GAME_LIST_FILE = str(raw_dir() / "majsoul/game_list.jsonl")
UUID_LIST_FILE = str(raw_dir() / "majsoul/uuid_list.txt")
CURSOR_FILE = str(raw_dir() / "majsoul/cursor.json")


class ApiError(Exception):
    def __init__(self, status, body, mirror):
        super().__init__("mirror %s http %d: %s" % (mirror, status, body[:200]))
        self.status = status
        self.body = body


class ApiClient:
    def __init__(self, token=None, mirrors=None):
        self.mirrors = list(mirrors or MIRRORS)
        self.token = token or cap_client.get_token()
        self.mirror_idx = 0
        self.stats = {"requests": 0, "429": 0, "token_refreshes": 0, "mirror_failovers": 0}

    def _token(self):
        cached = cap_client.load_cached_token()
        if cached:
            return cached
        self.token = cap_client.get_token(force=True)
        return self.token

    def get_json(self, path, retries=5, allow_token_refresh=True, cache_bust=False):
        """GET path (relative to api/v2/pl4), returning parsed JSON."""
        last_err = None
        for attempt in range(retries):
            mirror = self.mirrors[self.mirror_idx % len(self.mirrors)]
            url = mirror + API_V2 + "/" + path.lstrip("/")
            if cache_bust:
                url += ("&" if "?" in url else "?") + "cap_token_refreshed=%d" % int(time.time() * 1000)
            headers = {
                "User-Agent": UA,
                "Accept": "application/json",
                "authorization": "Bearer " + self._token(),
                "cache-control": "no-cache",
                "Referer": "https://amae-koromo.sapk.ch/",
            }
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=25) as r:
                    self.stats["requests"] += 1
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = b""
                try:
                    body = e.read()
                except Exception:
                    pass
                text = body.decode("utf-8", "replace")
                if e.code == 429 and "x-cap-token-required" in text and allow_token_refresh:
                    age = cap_client.cache_age() or 1e9
                    if age < 120:
                        # token is fresh; the 429 is per-request rate limiting -
                        # retry with cache-bust only (no browser round-trip)
                        print("[api] 429 but token fresh (%.0fs), retrying with cache-bust" % age, flush=True)
                        time.sleep(2)
                        return self.get_json(path, retries=retries, allow_token_refresh=False, cache_bust=True)
                    print("[api] 429 x-cap-token-required (token age %.0fs), refreshing CAP token" % age, flush=True)
                    self.stats["token_refreshes"] += 1
                    self.token = cap_client.get_token(force=True)
                    return self.get_json(path, retries=retries, allow_token_refresh=False, cache_bust=True)
                if e.code in (429, 502, 503, 504):
                    self.stats["429"] += 1
                    wait = 3 + attempt * 3 + random.uniform(0, 2)
                    print("[api] http %d on %s, retry %d/%d in %.1fs"
                          % (e.code, mirror, attempt + 1, retries, wait), flush=True)
                    time.sleep(wait)
                    last_err = ApiError(e.code, text, mirror)
                    continue
                last_err = ApiError(e.code, text, mirror)
                self.mirror_idx += 1
                self.stats["mirror_failovers"] += 1
                time.sleep(1)
            except Exception as e:
                print("[api] network error on %s: %s" % (mirror, e), flush=True)
                self.mirror_idx += 1
                self.stats["mirror_failovers"] += 1
                last_err = e
                time.sleep(1.5)
        raise RuntimeError("api request failed after retries: %s" % (last_err,))

    # ------------------------------------------------------------------
    def fetch_games(self, end_ms, start_ms, mode=16, limit=100):
        """One page: games with start_time in (start_ms, end_ms], newest first."""
        path = "games/%d/%d?limit=%d&descending=true&mode=%d" % (end_ms, start_ms, limit, mode)
        data = self.get_json(path)
        if not isinstance(data, list):
            print("[api] unexpected response shape: %s" % str(data)[:200], flush=True)
            return []
        return data

    def iter_all_games(self, mode=16, limit=100, min_games=None, max_pages=100000,
                       end_ms=None, start_ms=0):
        """Yield game dicts newest-first until exhaustion or min_games reached."""
        now_ms = int(time.time() * 1000)
        cur_end = end_ms or now_ms
        seen = set()
        count = 0
        pages = 0
        while pages < max_pages:
            page = self.fetch_games(cur_end, start_ms, mode=mode, limit=limit)
            pages += 1
            if not page:
                print("[api] exhausted at page %d (empty page), cur_end=%d" % (pages, cur_end), flush=True)
                break
            new_items = 0
            min_start = None
            for g in page:
                uid = g.get("uuid")
                if uid in seen:
                    continue
                seen.add(uid)
                yield g
                count += 1
                new_items += 1
                st = g.get("startTime", g.get("start_time"))
                if st is not None:
                    min_start = st if min_start is None else min(min_start, st)
            print("[api] page %d: %d items (%d new), total=%d"
                  % (pages, len(page), new_items, count), flush=True)
            if min_games and count >= min_games:
                break
            time.sleep(0.7)  # polite inter-page delay to avoid 429s
            if min_start is None or new_items == 0:
                break
            cur_end = int(min_start * 1000) - 1 if min_start < 1e12 else int(min_start) - 1
        return count


def load_existing_uuids(uuid_file=UUID_LIST_FILE):
    out = set()
    if os.path.exists(uuid_file):
        for line in io.open(uuid_file, encoding="utf-8"):
            u = line.strip()
            if u:
                out.add(u)
    return out


def download_game_list(min_games=5000, mode=16, hours_back=None, out_file=GAME_LIST_FILE,
                       uuid_file=UUID_LIST_FILE, append=True, resume=True):
    """Fetch >=min_games game summaries and persist them.

    append/resume: existing uuids are loaded and skipped; the cursor (oldest
    startTime reached) is persisted to CURSOR_FILE so a re-run continues
    exactly where the previous run stopped (断点续抓).
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    client = ApiClient()
    end_ms = None
    start_ms = 0
    if hours_back:
        start_ms = int((time.time() - hours_back * 3600) * 1000)
    games = []
    mode_write = "a" if (append and os.path.exists(out_file)) else "w"
    new_uuids = load_existing_uuids(uuid_file)
    # also merge uuids already present in the output file (crash-safe dedupe)
    if os.path.exists(out_file):
        for line in io.open(out_file, encoding="utf-8"):
            try:
                u = json.loads(line).get("uuid")
                if u:
                    new_uuids.add(u)
            except ValueError:
                pass
    base_count = len(new_uuids)
    if resume and base_count > 0 and os.path.exists(CURSOR_FILE):
        try:
            cur = json.load(io.open(CURSOR_FILE, encoding="utf-8"))
            if cur.get("cursor_ms"):
                end_ms = int(cur["cursor_ms"])
                print("[api] resume from cursor %d (already have %d uuids)"
                      % (end_ms, base_count), flush=True)
        except ValueError:
            pass
    t0 = time.time()
    min_start_seen = None
    with io.open(out_file, mode_write, encoding="utf-8") as f:
        for g in client.iter_all_games(mode=mode, min_games=base_count + min_games,
                                       end_ms=end_ms, start_ms=start_ms):
            uid = g.get("uuid")
            if uid and uid not in new_uuids:
                games.append(g)
                new_uuids.add(uid)
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
                st = g.get("startTime", g.get("start_time"))
                if st is not None:
                    min_start_seen = st if min_start_seen is None else min(min_start_seen, st)
            if min_start_seen is not None:
                cursor_ms = int(min_start_seen * 1000) - 1 if min_start_seen < 1e12 else int(min_start_seen) - 1
                json.dump({"cursor_ms": cursor_ms, "count": len(new_uuids),
                           "updated": time.time()}, io.open(CURSOR_FILE, "w", encoding="utf-8"))
    with io.open(uuid_file, "w", encoding="utf-8") as f:
        for u in sorted(new_uuids):
            f.write(u + "\n")
    # dedupe the output file (crash-safe: rewrite unique lines)
    try:
        seen = set()
        lines = io.open(out_file, encoding="utf-8").readlines()
        with io.open(out_file, "w", encoding="utf-8") as f:
            for line in lines:
                try:
                    u = json.loads(line).get("uuid")
                    if u is None or u in seen:
                        continue
                    seen.add(u)
                except ValueError:
                    continue
                f.write(line)
        print("[api] deduped %s (%d unique)" % (out_file, len(seen)), flush=True)
    except Exception as e:
        print("[api] dedupe failed: %s" % e, flush=True)
    print("[api] run: fetched %d new games in %.0fs (total %d in list)"
          % (len(games), time.time() - t0, len(new_uuids)), flush=True)
    return games, client.stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-games", type=int, default=5000)
    ap.add_argument("--hours-back", type=float, default=None)
    ap.add_argument("--probe", action="store_true", help="fetch one page and print first item")
    args = ap.parse_args()
    if args.probe:
        c = ApiClient()
        page = c.fetch_games(int(time.time() * 1000), 0, mode=16, limit=3)
        print(json.dumps(page[:3], ensure_ascii=False, indent=1)[:3000])
        print("STATS", json.dumps(c.stats))
    else:
        games, stats = download_game_list(min_games=args.min_games, hours_back=args.hours_back)
        print(json.dumps({"games": len(games), "stats": stats}))
