# -*- coding: utf-8 -*-
"""Resolve amae short game ids to real Majsoul uuids via player_records.

The v2 games list endpoint masks real uuids (short _id + possibly masked
players), but player_records/{accountId}/{start_ms}/{end_ms}?mode=16 always
returns real uuids. We fetch, for each distinct player in the game list, their
records over the list's time range and build (accountId, startTime) -> uuid.

Outputs:
  data/raw/majsoul/resolved_games.jsonl  (one line per game with real uuid)
  data/raw/majsoul/uuid_list.txt         (real uuids, one per line)
State:
  data/raw/majsoul/resolve_state.json    (players already processed; resumable)
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api_client
import cap_client

GAME_LIST_FILE = "C:/agentwork/data/raw/majsoul/game_list.jsonl"
RESOLVED_FILE = "C:/agentwork/data/raw/majsoul/resolved_games.jsonl"
UUID_LIST_FILE = "C:/agentwork/data/raw/majsoul/uuid_list.txt"
STATE_FILE = "C:/agentwork/data/raw/majsoul/resolve_state.json"


def load_game_list(path=GAME_LIST_FILE):
    games = []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            g = json.loads(line)
        except ValueError:
            continue
        if g.get("uuid") and len(str(g["uuid"])) > 20:
            g["_resolved"] = True
        else:
            g["_resolved"] = False
        games.append(g)
    return games


def resolve(games, min_games=None, max_players=None, delay=0.25):
    """Resolve short ids to real uuids; returns updated game list."""
    os.makedirs(os.path.dirname(RESOLVED_FILE), exist_ok=True)
    client = api_client.ApiClient(mirrors=["https://5-data.amae-koromo.com/",
                                           "https://1.data.amae-koromo.com/",
                                           "https://4.data.amae-koromo.com/"])
    if not games:
        games = load_game_list()
    if min_games is None:
        min_games = len(games)

    # global window
    times = [g["startTime"] for g in games if g.get("startTime")]
    if not times:
        return games
    tmin, tmax = min(times), max(times)
    start_ms = int((tmin - 3600) * 1000)
    end_ms = int((tmax + 3600) * 1000)

    # state
    state = {"players_done": {}, "uuid_map": {}}
    if os.path.exists(STATE_FILE):
        try:
            state = json.load(io.open(STATE_FILE, encoding="utf-8"))
        except ValueError:
            pass
    uuid_map = state.get("uuid_map", {})
    players_done = state.get("players_done", {})

    # candidate players (only from unresolved games)
    need = [g for g in games if not g.get("_resolved") and g.get("startTime") is not None]
    unresolved_ids = set()
    player_priority = []
    seen_players = set()
    for g in need:
        pid = g.get("_id")
        if pid:
            unresolved_ids.add(pid)
        for p in g.get("players") or []:
            aid = str(p.get("accountId"))
            if aid and aid not in seen_players:
                seen_players.add(aid)
                player_priority.append(aid)

    processed = 0
    for aid in player_priority:
        if max_players is not None and processed >= max_players:
            break
        if players_done.get(aid):
            continue
        try:
            recs = []
            cur_end = end_ms
            for _ in range(10):
                path = "player_records/%s/%d/%d?limit=100&mode=16" % (aid, start_ms, cur_end)
                try:
                    data = client.get_json(path)
                except Exception:
                    break
                if not isinstance(data, list) or not data:
                    break
                recs.extend(data)
                if len(data) < 100:
                    break
                min_st = min((r.get("startTime") or 0) for r in data if r.get("startTime"))
                if not min_st:
                    break
                cur_end = int(min_st * 1000) - 1
            for r in recs:
                u = r.get("uuid")
                st = r.get("startTime")
                if u and st:
                    uuid_map["%s|%d" % (aid, st)] = u
            players_done[aid] = True
            processed += 1
        except Exception as e:
            print("[resolve] player %s failed: %s" % (aid, str(e)[:120]), flush=True)
        time.sleep(delay)
        if processed % 100 == 0:
            print("[resolve] players processed: %d / %d, map size %d"
                  % (processed, len(player_priority), len(uuid_map)), flush=True)
            json.dump({"players_done": players_done, "uuid_map": uuid_map},
                      io.open(STATE_FILE, "w", encoding="utf-8"))

    # apply
    resolved = 0
    for g in games:
        if g.get("_resolved"):
            continue
        st = g.get("startTime")
        if st is None:
            continue
        for p in g.get("players") or []:
            u = uuid_map.get("%s|%d" % (p.get("accountId"), st))
            if u:
                g["uuid"] = u
                g["_resolved"] = True
                resolved += 1
                break
    json.dump({"players_done": players_done, "uuid_map": uuid_map},
              io.open(STATE_FILE, "w", encoding="utf-8"))
    print("[resolve] resolved %d games (target %d)" % (resolved, min_games), flush=True)
    return games


def save_resolved(games, resolved_file=RESOLVED_FILE, uuid_file=UUID_LIST_FILE):
    with io.open(resolved_file, "w", encoding="utf-8") as f:
        for g in games:
            if g.get("_resolved"):
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with io.open(uuid_file, "w", encoding="utf-8") as f:
        for g in games:
            if g.get("_resolved"):
                f.write(g["uuid"] + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-players", type=int, default=None)
    ap.add_argument("--min-games", type=int, default=None)
    args = ap.parse_args()
    games = load_game_list()
    games = resolve(games, min_games=args.min_games, max_players=args.max_players)
    save_resolved(games)
    resolved = [g for g in games if g.get("_resolved")]
    print("total resolved:", len(resolved), "of", len(games))
