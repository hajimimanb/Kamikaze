"""Full-corpus verification for the C1/C2 fixes (t14).

For every raw mjlog XML on disk:
1. C2: rebuild the mjai stream with the fixed exporter and compare each
   dahai's tsumogiri against the records' river tsumogiri per
   (round, honba, seat) — the extract.py path is an independent
   implementation of the "discard == seat's last drawn tile" rule.
2. C1: every emitted event type must be one of the canonical spellings
   (ryukyoku, not ryuukyoku).
3. Scan the ON-DISK mjai files for the old C2 defect pattern
   (tsumogiri=true directly after a same-actor meld) to quantify the
   pre-fix incidence; after the final reexport this count must be 0.

Usage: python src/tenhou/verify_corpus_consistency.py [--limit N]
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.paths import raw_dir, processed_dir
from tenhou.mjlog_parser import parse_game_file
from tenhou.mjai_export import build_events
from tenhou.extract import extract_game, GameStats

RAW = str(raw_dir() / "mjlog")
MJAI = str(processed_dir() / "tenhou/mjai")

BAKAZE_IDX = {"E": 0, "S": 1, "W": 2, "N": 3}


def ts2tile(s):
    # notation -> tile136 (kind*4, copy 0); caller compares // 4 (kind)
    if s.endswith("r"):
        return {"m": 16, "p": 52, "s": 88}[s[1]]
    suit = s[-1]
    if suit in "ESWNPFC":
        return (27 + "ESWNPFC".index(suit)) * 4
    return ({"m": 0, "p": 9, "s": 18}[suit] + int(s[:-1]) - 1) * 4


def check_game(path, game_id):
    problems = []
    g = parse_game_file(path)
    g["log_id"] = game_id
    records = []
    st = GameStats()
    extract_game(g, records, st)
    events = build_events(g)
    # C1: event type spelling
    for e in events:
        if e["type"] == "ryuukyoku":
            problems.append("C1 old spelling")
            break
    # C2: per (round, honba, seat) dahai vs record river
    round_dahai = {}
    cur_round = cur_honba = None
    for e in events:
        if e["type"] == "start_kyoku":
            cur_round = BAKAZE_IDX[e["bakaze"]] * 4 + e["kyoku"] - 1
            cur_honba = e["honba"]
        elif e["type"] == "dahai" and cur_round is not None:
            round_dahai.setdefault((cur_round, cur_honba, e["actor"]),
                                   []).append(e)
    rivers = {}
    for r in records:
        key = (r["round"], r["honba"], r["seat"])
        river = r["discards"][r["seat"]]
        if len(river) > len(rivers.get(key, [])):
            rivers[key] = river
    for key, mj in round_dahai.items():
        if key[0] > 7:
            continue
        rv = rivers.get(key, [])
        for e, d in zip(mj, rv):
            if ts2tile(e["pai"]) // 4 != d["tile"] // 4:
                problems.append("river kind mismatch %s" % (key,))
                break
            if e["tsumogiri"] != d["tsumogiri"]:
                problems.append("C2 tsumogiri mismatch %s" % (key,))
                break
    return problems


def scan_old_mjai(game_id):
    """Count the pre-fix defect pattern in an on-disk mjai file."""
    p = os.path.join(MJAI, game_id + ".jsonl")
    if not os.path.exists(p):
        return None
    n = 0
    last_actor_event = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        t = e["type"]
        if t in ("chi", "pon", "daiminkan", "ankan", "kakan"):
            last_actor_event[e["actor"]] = "meld"
        elif t == "dahai":
            if e.get("tsumogiri") and last_actor_event.get(e["actor"]) == "meld":
                n += 1
            last_actor_event[e["actor"]] = "dahai"
    return n


def main(argv=None) -> int:
    limit = 0
    if argv and argv[0] == "--limit":
        limit = int(argv[1])
    files = sorted(glob.glob(os.path.join(RAW, "*", "*.xml.gz")))
    if limit:
        files = files[:limit]
    bad = 0
    scanned = 0
    old_pattern_total = 0
    old_pattern_files = 0
    n = 0
    for f in files:
        game_id = os.path.basename(f).split(".")[0]
        problems = check_game(f, game_id)
        scanned += 1
        if problems:
            bad += 1
            if bad <= 5:
                print("BAD", game_id, problems[:3])
        cnt = scan_old_mjai(game_id)
        if cnt:
            old_pattern_total += cnt
            old_pattern_files += 1
        n += 1
        if n % 500 == 0:
            print("[%s] scanned %d/%d bad=%d old_pattern_total=%d (files=%d)"
                  % (__import__("time").strftime("%H:%M:%S"), n, len(files),
                     bad, old_pattern_total, old_pattern_files), flush=True)
    print("DONE scanned=%d bad_games=%d old_defect_total=%d old_defect_files=%d"
          % (scanned, bad, old_pattern_total, old_pattern_files))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
