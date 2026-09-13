"""Determine whether the phoenix lobby plays with kuitan (open tanyao).

Evidence rule (captain): count wins where the winner had open melds AND the
yaku list contains tanyao (yaku id 8):
  (a) open-meld + tanyao wins          > 0  -> kuitan allowed (weak evidence)
  (b) open-meld + tanyao-only wins
      (no other non-dora yaku)         > 0  -> kuitan allowed (strong evidence)
If both are 0 the lobby rules are kuitan-nashi.

Sample: two full days (20260101, 20260824) + 10 deterministic random days.
Output: docs/kuitan_stat.md.
"""
from __future__ import annotations

import glob
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.paths import raw_dir, docs_dir
from tenhou.mjlog_parser import parse_game_file

RAW = str(raw_dir() / "mjlog")
OUT = str(docs_dir() / "kuitan_stat.md")
TANYAO = 8
DORA_IDS = (52, 53, 54)
FIXED_DAYS = ("20260101", "20260824")
N_RANDOM = 10
SEED = 42


def day_files(date: str):
    return sorted(glob.glob(os.path.join(RAW, date[:4], date + "*.xml.gz")))


def analyze_day(date: str, stats: dict) -> None:
    files = day_files(date)
    for f in files:
        try:
            g = parse_game_file(f)
        except Exception:
            stats["parse_fail"] += 1
            continue
        stats["games"] += 1
        for rnd in g["rounds"]:
            for ev in rnd["events"]:
                if ev[0] != "agari":
                    continue
                a = ev[1]
                stats["wins"] += 1
                melds = a["melds"] or []
                yaku = a.get("yaku") or []
                ids = yaku[::2] if yaku else []
                open_melds = [m for m in melds if m["type"] != "ankan"]
                non_dora = {i for i in ids if i < DORA_IDS[0]}
                if not open_melds:
                    continue
                stats["open_wins"] += 1
                if TANYAO in ids:
                    stats["a_open_tanyao"] += 1
                    if non_dora == {TANYAO}:
                        stats["b_pure_kuitan"] += 1


def main(argv=None) -> int:
    stats = Counter(games=0, wins=0, open_wins=0, a_open_tanyao=0,
                    b_pure_kuitan=0, parse_fail=0)
    days = list(FIXED_DAYS)
    all_days = sorted({os.path.basename(f)[:8]
                       for f in glob.glob(os.path.join(RAW, "*", "*.xml.gz"))})
    rng = random.Random(SEED)
    days += rng.sample(all_days, min(N_RANDOM, len(all_days)))
    days = sorted(set(days))
    per_day = {}
    for d in days:
        before = dict(stats)
        analyze_day(d, stats)
        per_day[d] = {k: stats[k] - before.get(k, 0) for k in
                      ("games", "wins", "open_wins", "a_open_tanyao",
                       "b_pure_kuitan")}
    kuitan = stats["a_open_tanyao"] > 0 and stats["b_pure_kuitan"] > 0
    lines = []
    lines.append("# 凤凰桌 喰断（open tanyao）统计结论")
    lines.append("")
    lines.append("生成: python src/tenhou/kuitan_stat.py")
    lines.append("")
    lines.append("## 样本")
    lines.append("- 固定全天: %s" % ", ".join(FIXED_DAYS))
    lines.append("- 随机 %d 天 (seed %d): %s" % (N_RANDOM, SEED,
          ", ".join(d for d in days if d not in FIXED_DAYS)))
    lines.append("")
    lines.append("## 结果")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append("| 样本局数 | %d |" % stats["games"])
    lines.append("| 和牌总数 (AGARI) | %d |" % stats["wins"])
    lines.append("| 其中副露和牌 | %d |" % stats["open_wins"])
    lines.append("| (a) 副露+断幺九(役含 8) 和牌 | %d |" % stats["a_open_tanyao"])
    lines.append("| (b) 副露+纯断幺九(无其他役) 和牌 | %d |" % stats["b_pure_kuitan"])
    lines.append("")
    lines.append("## 逐日明细")
    lines.append("")
    lines.append("| 日期 | 局数 | 和牌 | 副露和牌 | (a) | (b) |")
    lines.append("|---|---|---|---|---|---|")
    for d in sorted(per_day):
        p = per_day[d]
        lines.append("| %s | %d | %d | %d | %d | %d |"
                     % (d, p["games"], p["wins"], p["open_wins"],
                        p["a_open_tanyao"], p["b_pure_kuitan"]))
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    if kuitan:
        lines.append("**凤凰桌 = 喰断あり**：(a)=%d>0 且 (b)=%d>0，")
        lines.append("样本中存在副露+断幺九和牌（含纯喰断），")
        lines.append("佐证大厅名「四鳳南喰赤－」的「喰」字（喰断）。")
        lines = [lines[0], lines[1]] + [
            l % (stats["a_open_tanyao"], stats["b_pure_kuitan"])
            if "%d" in l else l for l in lines[2:]]
    else:
        lines.append("**凤凰桌 = 喰断なし**：样本中未发现副露+断幺九和牌。")
    lines.append("")
    text = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
