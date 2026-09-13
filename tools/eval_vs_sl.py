# -*- coding: utf-8 -*-
"""评估协议 v3：策略 vs 基线独立强度对比（feedback1 修订版）。

指标口径（feedback1 §4–§6）：
- 主指标：Mean Rank（平均顺位） = 1/(4N)·Σ_seat Σ_game rank（1..4）
- 辅助指标：Rank 1 Rate（胜率，Win Rate ≡ Rank 1 Rate）、
           Placement Score = P(rank=1) + 0.5·P(rank=2)（不再叫 Win Rate）、pt/100 局
- 行为指标：和牌率（agari_rate，每半庄平均和牌局数）、
           放铳率（deal_in_rate，每半庄平均放铳局数；double-ron 不重复计）

统计口径（feedback1 §14–§15）：
- Bootstrap 95% CI（估计量的不确定性，resample N 次）——主体统一
- SD across seeds（seed 间离散）——附加信息，两者不混

输出：结构化 results.json（feedback1 §13）+ 终端摘要。
座次轮换：4 座位轮换 × games_per_seat × seeds；逐 seat、逐 seed 均记录。

均势参考线（我方与 3 个同强度对手对局，1 打 3）：
  mean_rank=2.5、rank1_rate=0.25、placement_score=0.375、pt/100=0。

用法：
  python eval_vs_sl.py --a rl_v1.pt --b transfer_final.pt --games 250 --seeds 3
  python eval_vs_sl.py --a rl_v1.pt --b random --games 20 --seeds 1    # sanity
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent.policy import RiichiPolicy
from env.riichi_game import RiichiGame, RiichiConfig
from utils.paths import repo_root, logs_dir

CI_LEVEL = 0.95
BOOTSTRAP_SEED = 0


def play_game_full(policy_a, policy_b, a_seat, seed, sample=False, temp=1.0):
    """跑一整局半庄，返回我方名次/点数/精算 pt/和牌/放铳的完整 dict。

    和牌（agari）：事件流中 ``actor == a_seat`` 的 hora 局数；
    放铳（dealin）：事件流中 ``target == a_seat`` 的 hora 局数（按局去重，double-ron 不重复计）。
    """
    from model.rl_reward import settlement_pt as _spt

    game = RiichiGame(RiichiConfig(), seed=seed)
    for _ in range(10000):
        if game.phase == "game_end":
            break
        if game.turn == a_seat:
            if sample:
                action = policy_a.sample_with_logp(
                    game.state.get_observation(), temp,
                    __import__("model.attn_modules", fromlist=["x"]).build_events_from_game(
                        game.events, game.round_idx) if policy_a.model.use_event_attn else None)[0]
            else:
                action = policy_a(game)   # __call__ 传真实时间线事件
        else:
            action = policy_b(game)
        game.step(action)
    else:
        raise RuntimeError("game did not finish")

    scores = game.scores
    rank = sorted(range(4), key=lambda i: -scores[i]).index(a_seat)

    agari_rounds, dealin_rounds = set(), set()
    for e in game.events:
        if e.get("type") != "hora":
            continue
        r = e.get("round")
        if e.get("actor") == a_seat:
            agari_rounds.add(r)
        if e.get("target") == a_seat:
            dealin_rounds.add(r)

    return {
        "rank": rank,            # 0..3
        "score": scores[a_seat],
        "pt": _spt(list(scores), a_seat, "tenhou"),
        "agari": len(agari_rounds),
        "dealin": len(dealin_rounds),
    }


def play_game(policy_a, policy_b, a_seat, seed, sample=False, temp=1.0):
    """兼容包装：返回 ``(rank, score, pt)`` 三元组。

    train_rl.py / train_rl_vec.py / train_rl_mp.py 依赖此签名解包，勿改。
    """
    r = play_game_full(policy_a, policy_b, a_seat, seed, sample, temp)
    return r["rank"], r["score"], r["pt"]


# --------------------------------------------------------------------------- utils

def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(repo_root()), check=True, timeout=5)
        return out.stdout.strip()
    except Exception:
        return ""


def _config_hash(config_arg: str) -> str:
    p = Path(config_arg)
    if not p.is_absolute():
        p = repo_root() / p
    try:
        if p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except Exception:
        pass
    return ""


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _sd(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    m = _mean(vals)
    return (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5


def _bootstrap_mean_ci(vals, n_boot, level, rng):
    """对 ``vals`` 的均值做 Bootstrap percentile CI（resample 单位 = 单局样本）。"""
    n = len(vals)
    if n == 0:
        return [0.0, 0.0]
    ests = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(n_boot))
    alpha = 1.0 - level
    lo = ests[max(0, int(alpha / 2 * n_boot))]
    hi = ests[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return [round(lo, 4), round(hi, 4)]


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="被评估策略 checkpoint（我方）")
    ap.add_argument("--b", required=True, help="基线 checkpoint，或 'random'（sanity）")
    ap.add_argument("--games", type=int, default=250, help="每座位局数（games_per_seat）")
    ap.add_argument("--seeds", type=int, default=3, help="评估种子数")
    ap.add_argument("--sample", action="store_true", help="温度采样（默认确定性 act）")
    ap.add_argument("--event-attn", action="store_true", help="启用事件因果注意力旁支")
    ap.add_argument("--n-bootstrap", type=int, default=10000, help="Bootstrap 重采样次数")
    ap.add_argument("--experiment", default="rl_vs_sl_v1", help="实验名（写入 results.json）")
    ap.add_argument("--config", default="configs/rl_v1.json", help="配置文件（用于 config_hash）")
    ap.add_argument("--out", default=None, help="results.json 输出路径（默认 logs/results_<exp>.json）")
    ap.add_argument("--pt", action="store_true", help="额外打印每局 settlement_pt 明细")
    ap.add_argument("--window", type=int, default=20, help="--pt 时逐局明细只打印最近 N 局")
    args = ap.parse_args()

    t0 = time.time()
    pa = RiichiPolicy(args.a, seed=0, use_event_attn=args.event_attn)
    if args.b.lower() == "random":
        class _RandomPolicy:
            def decide(self, obs):
                la = obs.get("legal_actions") or {}
                if la.get("tsumo"):
                    return {"type": "tsumo"}
                if la.get("ron"):
                    return {"type": "ron"}
                for h in ("kan", "pon", "chow"):
                    if la.get(h):
                        return {"type": h, "tiles": la[h][0]["tiles"]}
                if la.get("riichi"):
                    return {"type": "riichi", "tile": la["riichi"][0]}
                if la.get("discard"):
                    return {"type": "discard", "tile": random.choice(la["discard"])}
                return {"type": "pass"}
            def __call__(self, game):
                return self.decide(game.state.get_observation())
        pb = _RandomPolicy()
        print("对手: RANDOM（vs-random sanity，均势参考线见末尾）")
    else:
        pb = RiichiPolicy(args.b, seed=1)

    # ---- 收集样本：每个 (seed, seat) 格 games_per_seat 局 ----
    samples = []   # 元素 = {seed, seat, rank(0..3), score, pt, agari, dealin}
    for s in range(args.seeds):
        for seat in range(4):
            for g in range(args.games):
                r = play_game_full(pa, pb, seat, s * 100000 + g, args.sample)
                samples.append({
                    "seed": s, "seat": seat,
                    "rank": r["rank"], "score": r["score"],
                    "pt": r["pt"], "agari": r["agari"], "dealin": r["dealin"],
                })

    def cell_stats(rows):
        n = len(rows)
        if n == 0:
            return {"games": 0, "mean_rank": 0.0, "rank1_rate": 0.0,
                    "placement_score": 0.0, "pt_per_100": 0.0,
                    "agari_rate": 0.0, "deal_in_rate": 0.0}
        return {
            "games": n,
            "mean_rank": _mean([x["rank"] + 1 for x in rows]),
            "rank1_rate": sum(1 for x in rows if x["rank"] == 0) / n,
            "placement_score": sum(
                1.0 if x["rank"] == 0 else (0.5 if x["rank"] == 1 else 0.0)
                for x in rows) / n,
            "pt_per_100": _mean([x["pt"] for x in rows]) * 100.0,
            "agari_rate": _mean([x["agari"] for x in rows]),
            "deal_in_rate": _mean([x["dealin"] for x in rows]),
        }

    # 逐 (seed, seat) 明细（feedback1 §5 逐 seat 报告）
    seats = []
    for s in range(args.seeds):
        for seat in range(4):
            rows = [x for x in samples if x["seed"] == s and x["seat"] == seat]
            seats.append({"seed": s, "seat": seat, **cell_stats(rows)})

    # 逐 seed 汇总（用于 SD across seeds）
    seed_rows = [cell_stats([x for x in samples if x["seed"] == s])
                 for s in range(args.seeds)]

    # aggregate（全局统计 + Bootstrap 95% CI）
    rng = random.Random(BOOTSTRAP_SEED)
    ranks = [x["rank"] + 1 for x in samples]
    rank1 = [1.0 if x["rank"] == 0 else 0.0 for x in samples]
    place = [1.0 if x["rank"] == 0 else (0.5 if x["rank"] == 1 else 0.0)
             for x in samples]
    pt100 = [x["pt"] * 100.0 for x in samples]

    def ci(vals):
        return _bootstrap_mean_ci(vals, args.n_bootstrap, CI_LEVEL, rng)

    all_stats = cell_stats(samples)
    aggregate = {
        "total_games": len(samples),
        "mean_rank": round(all_stats["mean_rank"], 4),
        "mean_rank_ci": ci(ranks),
        "rank1_rate": round(all_stats["rank1_rate"], 4),
        "rank1_rate_ci": ci(rank1),
        "placement_score": round(all_stats["placement_score"], 4),
        "placement_score_ci": ci(place),
        "pt_per_100": round(all_stats["pt_per_100"], 2),
        "pt_per_100_ci": [round(v, 2) for v in ci(pt100)],
        "agari_rate": round(all_stats["agari_rate"], 4),
        "deal_in_rate": round(all_stats["deal_in_rate"], 4),
        "std_across_seeds": {
            "mean_rank": round(_sd([r["mean_rank"] for r in seed_rows]), 4),
            "rank1_rate": round(_sd([r["rank1_rate"] for r in seed_rows]), 4),
            "placement_score": round(_sd([r["placement_score"] for r in seed_rows]), 4),
            "pt_per_100": round(_sd([r["pt_per_100"] for r in seed_rows]), 2),
        },
    }

    results = {
        "experiment": args.experiment,
        "git_commit": _git_commit(),
        "config_hash": _config_hash(args.config),
        "checkpoint_a": args.a,
        "checkpoint_b": args.b,
        "games_per_seat": args.games,
        "seeds_count": args.seeds,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ci_level": CI_LEVEL,
        "sample": args.sample,
        "event_attn": args.event_attn,
        "seat_rotation": True,
        "seats": seats,
        "aggregate": aggregate,
    }

    # ---- 终端摘要 ----
    a = aggregate
    print("=== RESULT（feedback1 v3：Mean Rank 主指标） ===")
    print("games=%d（%d seeds × 4 seats × %d）  elapsed=%.0fs"
          % (a["total_games"], args.seeds, args.games, time.time() - t0))
    print("Mean Rank     = %.3f  (Bootstrap 95%% CI [%.3f, %.3f], SD-across-seeds %.3f)"
          % (a["mean_rank"], a["mean_rank_ci"][0], a["mean_rank_ci"][1],
             a["std_across_seeds"]["mean_rank"]))
    print("Rank 1 Rate   = %.3f  (CI [%.3f, %.3f])"
          % (a["rank1_rate"], a["rank1_rate_ci"][0], a["rank1_rate_ci"][1]))
    print("Placement     = %.3f  (CI [%.3f, %.3f])"
          % (a["placement_score"], a["placement_score_ci"][0],
             a["placement_score_ci"][1]))
    print("pt/100 局     = %+.1f  (CI [%+.1f, %+.1f])"
          % (a["pt_per_100"], a["pt_per_100_ci"][0], a["pt_per_100_ci"][1]))
    print("和牌率         = %.3f   放铳率 = %.3f"
          % (a["agari_rate"], a["deal_in_rate"]))
    print("参考（均势，1打3同强度对手）: mean_rank=2.5 rank1=0.25 placement=0.375 pt/100=0")

    if args.pt:
        pts = [x["pt"] for x in samples][-args.window:]
        print("逐局 settlement_pt（最近 %d 局）: %s" % (len(pts), [int(x) for x in pts]))

    # ---- 结构化输出 ----
    out_path = Path(args.out) if args.out else logs_dir() / ("results_%s.json" % args.experiment)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("results.json -> %s" % out_path)


if __name__ == "__main__":
    main()