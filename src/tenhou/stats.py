"""Dataset statistics: records shards -> docs/data_report.md.

Stats per docs/observation_schema.md section 3 + task requirements:
- games & records per year
- decision-type distribution
- seat balance
- riichi rate / call rate
- ron/tsumo distribution, tsumogiri fraction
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
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import docs_dir, processed_dir

DEFAULT_GLOB = str(processed_dir() / "tenhou" / "records-*.jsonl*")
DEFAULT_OUT = str(docs_dir() / "data_report.md")


def iter_records(patterns: List[str], pass_no=None):
    """Yield parsed records; with pass_no prints real-time progress lines
    (every 5 files + pass-done): scanned files / cumulative records / rate / elapsed."""
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    t0 = time.time()
    n = 0
    for i, f in enumerate(files, 1):
        opener = gzip.open if f.endswith(".gz") else open
        try:
            fh = opener(f, "rt", encoding="utf-8")
        except OSError:
            continue
        with fh:
            try:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    n += 1
                    yield rec
            except (EOFError, OSError):
                # tolerate a shard still being written by a live batch
                continue
        if pass_no and i % 5 == 0:
            dt = time.time() - t0
            print("progress stats pass=%s file=%d/%d records=%d rate=%.0f/s elapsed=%ds"
                  % (pass_no, i, len(files), n, n / max(dt, 1e-6), dt), flush=True)
    if pass_no:
        dt = time.time() - t0
        print("progress stats pass=%s done files=%d records=%d rate=%.0f/s elapsed=%ds"
              % (pass_no, len(files), n, n / max(dt, 1e-6), dt), flush=True)


def compute(patterns: List[str]):
    """Single full-corpus pass: all stats + per-game record counts (fed to
    build_splits), so the corpus is read exactly once, not 3 times.
    Returns (stats, games, records_per_game)."""
    label_types = collections.Counter()
    seat_labels = collections.Counter()
    seat_discard = collections.Counter()
    seat_riichi = collections.Counter()
    seat_call = collections.Counter()
    games = set()
    records_per_game = collections.Counter()
    records_by_year = collections.Counter()
    games_with_call = set()
    games_with_riichi = set()
    round_dist = collections.Counter()
    tsumogiri_t = 0
    tsumogiri_tsumo = 0
    n = 0
    for r in iter_records(patterns, 1):
        n += 1
        gid = r["game_id"]
        games.add(gid)
        records_per_game[gid] += 1
        records_by_year[gid[:4]] += 1
        lt = r["label"]["type"]
        label_types[lt] += 1
        seat_labels[r["seat"]] += 1
        round_dist[r["round"]] += 1
        if lt in ("discard", "riichi"):
            seat_discard[r["seat"]] += 1
            if lt == "riichi":
                seat_riichi[r["seat"]] += 1
                games_with_riichi.add(gid)
        elif lt in ("pon", "chow", "kan"):
            seat_call[r["seat"]] += 1
            games_with_call.add(gid)
        for d in r["discards"][r["seat"]]:
            tsumogiri_t += 1
            if d["tsumogiri"]:
                tsumogiri_tsumo += 1
    if n == 0:
        return {"records": 0, "games": 0}, set(), collections.Counter()
    games_by_year = collections.Counter(g[:4] for g in games)
    return {
        "records": n,
        "games": len(games),
        "games_by_year": dict(sorted(games_by_year.items())),
        "records_by_year": dict(sorted(records_by_year.items())),
        "label_types": dict(label_types.most_common()),
        "seat_labels": dict(sorted(seat_labels.items())),
        "seat_discard": dict(sorted(seat_discard.items())),
        "seat_riichi": dict(sorted(seat_riichi.items())),
        "seat_call": dict(sorted(seat_call.items())),
        "games_with_riichi": len(games_with_riichi),
        "games_with_call": len(games_with_call),
        "round_dist": dict(sorted(round_dist.items())),
        "riichi_rate": label_types["riichi"] / max(1, label_types["riichi"] + label_types["discard"]),
        "tsumogiri_frac": tsumogiri_tsumo / max(1, tsumogiri_t),
    }, games, records_per_game


def render_report(s: Dict) -> str:
    if s.get("records", 0) == 0:
        return "# Tenhou dataset report\n\nNo records found.\n"
    L = []
    A = L.append
    A("# 天凤凤凰桌数据集统计报告 (data_report)")
    A("")
    A("自动生成: python -m tenhou.stats  (src/tenhou/stats.py)")
    A("")
    A("## 概览")
    A("")
    A("| 指标 | 值 |")
    A("|---|---|")
    A("| 决策记录数 | %d |" % s["records"])
    A("| 对局数 | %d |" % s["games"])
    A("| 平均每局决策数 | %.1f |" % (s["records"] / max(1, s["games"])))
    A("")
    A("## 按年份统计")
    A("")
    A("| 年份 | 对局数 | 记录数 |")
    A("|---|---|---|")
    for y in sorted(s["games_by_year"]):
        A("| %s | %d | %d |" % (y, s["games_by_year"][y],
                                s["records_by_year"].get(y, 0)))
    A("")
    A("## 决策类型分布")
    A("")
    A("| 类型 | 数量 | 占比 |")
    A("|---|---|---|")
    total = s["records"]
    for t, c in sorted(s["label_types"].items(), key=lambda kv: -kv[1]):
        A("| %s | %d | %.2f%% |" % (t, c, 100.0 * c / total))
    A("")
    A("## 各座位均衡性")
    A("")
    A("| 座位 | 全部标签 | 切牌/立直 | 立直 | 鸣牌 |")
    A("|---|---|---|---|---|")
    for seat in range(4):
        A("| %d | %d | %d | %d | %d |"
          % (seat, s["seat_labels"].get(seat, 0),
             s["seat_discard"].get(seat, 0),
             s["seat_riichi"].get(seat, 0),
             s["seat_call"].get(seat, 0)))
    A("")
    A("## 行为率")
    A("")
    A("| 指标 | 值 |")
    A("|---|---|")
    A("| 立直率 (立直 / (立直+切牌)) | %.3f |" % s["riichi_rate"])
    A("| 有立直对局占比 | %.3f |" % (s["games_with_riichi"] / max(1, s["games"])))
    A("| 有鸣牌对局占比 | %.3f |" % (s["games_with_call"] / max(1, s["games"])))
    A("| 平均每局鸣牌次数 | %.2f |" % (sum(s["seat_call"].values()) / max(1, s["games"])))
    A("| 平均每局立直次数 | %.2f |" % (sum(s["seat_riichi"].values()) / max(1, s["games"])))
    A("| 摸切占比 (牌河观测) | %.3f |" % s["tsumogiri_frac"])
    A("")
    A("## 场次分布 (round index)")
    A("")
    A("| round | 记录数 |")
    A("|---|---|")
    for k in sorted(s["round_dist"]):
        A("| %d | %d |" % (k, s["round_dist"][k]))
    A("")
    if "splits" in s:
        sp = s["splits"]
        A("## 数据划分 (train / val / eval-holdout)")
        A("")
        A("| 分区 | 对局数 | 记录数 |")
        A("|---|---|---|")
        for k in ("train", "val", "eval_holdout"):
            A("| %s | %d | %d |" % (k, sp["games"].get(k, 0),
                                    sp["records"].get(k, 0)))
        A("")
        A("- eval-holdout 截止日期: %s（该日期及之后的牌谱永不进训练，"
          % sp["holdout_cutoff"])
        A("  专供阶段5 通道B 相似度测评与专家查验；")
        A("- val: 稳定哈希 logid % 20 == 0；train: 其余。")
        A("- 划分是 game_id 的纯函数：同一局样本永不跨分区。")
        A("- 清单文件: data/processed/tenhou/splits/{train,val,eval_holdout}_games.txt + splits.json")
        A("")
    A("## 说明")
    A("")
    A("- 采样规则见 docs/observation_schema.md 第3节：只采样有真实选择权的决策点；")
    A("  立直后的强制摸切不采样；流局不产生标签。")
    A("- 立直率分母为立直+普通切牌标签（鸣牌后的切牌也算切牌决策）。")
    A("- 和牌样本(ron/tsumo)按自然比例保留(1:1 不降权)。")
    return "\n".join(L)


SPLIT_RULES = """
Data split (train / val / eval-holdout), reproducible pure functions of the
game_id (per captain directive and reviewer t7 auditability requirements):

- eval-holdout: games dated 2026-08-01 or later (most recent window;
  captain final confirmation 2026-08-25: ~15k games, sufficient for
  channel-B similarity eval and L1 metrics).
  NEVER used for SL training; reserved for stage-5 channel-B similarity eval
  and expert review.  When the corpus grows (zip archive restoration
  2009-2025), the cut moves to the most recent complete window.
- val: stable hash(game_id) % 20 == 0 (5% of the remaining games),
  used for training-side early stopping.
- train: everything else.

A game's samples never cross partitions (the rule is a function of game_id).
Manifests are written to data/processed/tenhou/splits/ :
  train_games.txt / val_games.txt / eval_holdout_games.txt + eval_games.txt
  alias + splits.json (rules, counts, embedded game lists for
  review/check_leakage.py).
"""

HOLDOUT_CUTOFF = "20260801"


def assign_split(game_id: str) -> str:
    """train | val | eval_holdout (see SPLIT_RULES)."""
    date = game_id[:8]
    if date >= HOLDOUT_CUTOFF:
        return "eval_holdout"
    h = 0
    for ch in game_id:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    if h % 20 == 0:
        return "val"
    return "train"


def build_splits(games, records_per_game) -> Dict:
    """Write split manifests from already-collected data (no corpus re-read);
    return counts per split."""
    split_of = {g: assign_split(g) for g in games}
    counts = collections.Counter(split_of.values())
    rec_counts = collections.Counter()
    for g, s in split_of.items():
        rec_counts[s] += records_per_game[g]
    out_dir = str(processed_dir() / "tenhou/splits")
    os.makedirs(out_dir, exist_ok=True)
    # E3 fix: atomic writes (tmp + os.replace) so concurrent readers
    # (ml-engineer preprocess, reviewer check_leakage) never see half-written
    # manifests
    for split in ("train", "val", "eval_holdout"):
        gs = sorted(g for g in games if split_of[g] == split)
        path = os.path.join(out_dir, split + "_games.txt")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for g in gs:
                f.write(g + "\n")
        os.replace(tmp, path)
    # alias for the reviewer's t7 checker convention (train/val/eval_games.txt)
    for alias, target in (("eval_games.txt", "eval_holdout_games.txt"),):
        src = os.path.join(out_dir, target)
        dst = os.path.join(out_dir, alias)
        tmp = dst + ".tmp"
        with open(src, encoding="utf-8") as f:
            with open(tmp, "w", encoding="utf-8") as g:
                g.write(f.read())
        os.replace(tmp, dst)
    # checker-compatible embedded lists (review/check_leakage.py reads
    # {"train": {"games": [...]}, "eval": {"games": [...]}} or the flat
    # train_games/eval_games keys)
    game_lists = {}
    for split in ("train", "val", "eval_holdout"):
        game_lists[split] = sorted(g for g in games if split_of[g] == split)
    meta = {
        "rule": SPLIT_RULES.strip(),
        "holdout_cutoff": HOLDOUT_CUTOFF,
        "games": {k: counts.get(k, 0) for k in ("train", "val", "eval_holdout")},
        "records": {k: rec_counts.get(k, 0)
                    for k in ("train", "val", "eval_holdout")},
        # reviewer checker format
        "train": {"games": game_lists["train"], "records_file": None},
        "val": {"games": game_lists["val"], "records_file": None},
        "eval": {"games": game_lists["eval_holdout"], "records_file": None},
        "eval_holdout": {"games": game_lists["eval_holdout"],
                         "records_file": None},
    }
    spath = os.path.join(out_dir, "splits.json")
    stmp = spath + ".tmp"
    with open(stmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    os.replace(stmp, spath)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default=DEFAULT_GLOB, help="records shard pattern")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    s, games, rpg = compute([args.glob])
    splits = build_splits(games, rpg)
    s["splits"] = splits
    md = render_report(s)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print("wrote %s (%d records, %d games)" % (args.out, s.get("records", 0),
                                               s.get("games", 0)))
    print("splits:", splits["games"], "records:", splits["records"])
    print("consistency: records=%d by_year=%d per_game=%d labels=%d"
          % (s.get("records", 0), sum(s.get("records_by_year", {}).values()),
             sum(rpg.values()), sum(s.get("label_types", {}).values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
