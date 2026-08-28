# -*- coding: utf-8 -*-
"""评估协议 v2：策略 vs 基线自对弈胜率（评审 2.5 修订版）。

- 4 座位轮换 × games_per_seat 局 × --seeds 个种子（默认 3）
- 指标：win_rate = P(rank==1) + 0.5*P(rank==2)；avg_rank；95% 正态 CI
- 判定：mean ≥ 0.66 且 CI 下界 ≥ 0.60（评审 2.5：单点 66% 是掷硬币）
- 1 打 3 门槛校准：均势期望 win_rate=0.375（评审 2.5）
- 事件接线：策略经 __call__(game) 决策 → 真实时间线（旁支启用时）
- 决策：确定性 act()（阈值+argmax），--sample 可切换温度采样

用法：
  python eval_vs_sl.py --a rl_v1.pt --b transfer_final.pt --games 100 --seeds 3
"""
import argparse, math, os, sys, time
sys.path.insert(0, "C:/agentwork/src")
from agent.policy import RiichiPolicy
from env.riichi_game import RiichiGame, RiichiConfig


def play_game(policy_a, policy_b, a_seat, seed, sample=False, temp=1.0):
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
    # pt 公式：我方终局精算点数（tenhou：uma 20/10/-10/-20 + 千点差 + 供托）
    return rank, scores[a_seat], _spt(list(scores), a_seat, "tenhou")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=100, help="每座位局数（总 4×；评审 2.5 ≥100）")
    ap.add_argument("--seeds", type=int, default=3, help="种子数（多种子取均值报 CI）")
    ap.add_argument("--sample", action="store_true", help="温度采样（默认确定性 act）")
    ap.add_argument("--event-attn", action="store_true", help="启用事件因果注意力旁支（judge A1）")
    ap.add_argument("--pt", action="store_true",
                    help="胜率按 pt 定义：win_rate_pt = P(我方 settlement_pt > 0)（零和均势基准 0）")
    ap.add_argument("--window", type=int, default=20,
                    help="--pt 时只统计最近 N 局（默认 20）")
    args = ap.parse_args()

    t0 = time.time()
    pa = RiichiPolicy(args.a, seed=0, use_event_attn=args.event_attn)
    if args.b.lower() == "random":
        class _R:
            def decide(self, obs):
                la = obs.get("legal_actions") or {}
                import random
                if la.get("tsumo"): return {"type": "tsumo"}
                if la.get("ron"): return {"type": "ron"}
                for h in ("kan", "pon", "chow"):
                    if la.get(h): return {"type": h, "tiles": la[h][0]["tiles"]}
                if la.get("riichi"): return {"type": "riichi", "tile": la["riichi"][0]}
                if la.get("discard"): return {"type": "discard", "tile": random.choice(la["discard"])}
                return {"type": "pass"}
            def __call__(self, game): return self.decide(game.state.get_observation())
        pb = _R()
        print("对手: RANDOM（vs-random sanity，均势参考线 0.375）")
    else:
        pb = RiichiPolicy(args.b, seed=1)
    rates = []
    all_pts = []
    for s in range(args.seeds):
        dist = [0, 0, 0, 0]
        total = 0
        for seat in range(4):
            for g in range(args.games):
                rank, sc, spt = play_game(pa, pb, seat, s * 100000 + g, args.sample)
                dist[rank] += 1
                total += 1
                all_pts.append(spt)
        wr = (dist[0] + 0.5 * dist[1]) / max(total, 1)
        rates.append(wr)
        print("seed %d: games=%d rank_dist=%s win_rate=%.3f" % (s, total, dist, wr), flush=True)
    if args.pt:
        # pt 公式（用户定义）：最近 window 局内，我方 settlement_pt（4 人零和，均势 0）
        win_pts = all_pts[-args.window:]
        n = len(win_pts)
        avg_pt = float(sum(win_pts)) / max(n, 1)
        # ① 平均每百局 pt 增量：avg_pt × 100（>0 = 每百局净赚 pt）
        pt_per_100 = avg_pt * 100.0
        # ② pt 加权胜率（按 pt 加权而非普通胜率）：
        #    胜面占比 = Σmax(spt,0) / (Σmax(spt,0)+Σ|min(spt,0)|)，均势 0.5
        pos = sum(x for x in win_pts if x > 0)
        neg = sum(-x for x in win_pts if x < 0)
        wr_pt_w = pos / max(pos + neg, 1e-9)
        # ③ 普通 pt 胜率（保留对照）：P(spt>0)
        wr_pt = float(sum(1 for x in win_pts if x > 0)) / max(n, 1)
        import statistics
        sd_pt = statistics.pstdev(win_pts) if n > 1 else 0.0
        print("=== PT RESULT (最近 %d 局) ===" % n)
        print("平均每百局pt=%.1f   pt加权胜率=%.3f   普通pt胜率P(spt>0)=%.3f   平均pt=%.2f(pt_std=%.1f)" % (
            pt_per_100, wr_pt_w, wr_pt, avg_pt, sd_pt))
        print("每局 spt: %s" % [int(x) for x in win_pts])
        print("参考: 均势期望 avg_pt=0 / 每百局pt=0 / pt加权胜率=0.5")
        return
    mean = sum(rates) / len(rates)
    sd = (sum((r - mean) ** 2 for r in rates) / max(len(rates) - 1, 1)) ** 0.5
    se = sd / (len(rates) ** 0.5)
    # seeds<10 用 t 分布临界值（auditor 建议；df=seeds-1, 95% 双侧查表）
    _T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    z = _T95.get(len(rates) - 1, 1.96) if len(rates) > 1 else 1.96
    lo, hi = mean - z * se, mean + z * se
    print("=== RESULT ===")
    print("mean win_rate=%.3f (CI %.3f~%.3f, seeds=%d)" % (mean, lo, hi, args.seeds))
    print("判定: %s" % ("PASS (mean>=0.66 且下界>=0.60)" if mean >= 0.66 and lo >= 0.60
                        else "FAIL (需 mean>=0.66 且下界>=0.60)"))
    print("参考: 均势(1打3)期望=0.375；elapsed=%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()