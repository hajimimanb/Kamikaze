# -*- coding: utf-8 -*-
"""并行模型评估 v2（CPU 推理，永远在跑，永远最新 epoch）。"""
import json, os, sys, time

sys.path.insert(0, "C:/agentwork/src")
sys.path.insert(0, "C:/agentwork/tools")

import torch
from agent.policy import RiichiPolicy
from env.riichi_game import RiichiGame, RiichiConfig
from model.rl_reward import settlement_pt as _spt

CKPT = "C:/agentwork/checkpoints/sl/rl/rl_v1.pt"
SL_CKPT = "C:/agentwork/checkpoints/sl/transfer/transfer_final.pt"
OUT = "C:/agentwork/logs/vs_sl_sliding.jsonl"
PROGRESS = "C:/agentwork/logs/vs_sl_eval_progress.json"
GAMES_LOG = "C:/agentwork/logs/vs_sl_sliding_games.jsonl"
GAMES = 100
DEVICE = "cpu"


def _ckpt_epoch():
    try:
        ck = torch.load(CKPT, map_location="cpu")
        return int(ck.get("epoch", 0)), os.path.getmtime(CKPT)
    except Exception:
        return None, None


def _write_progress(d):
    # 原子写：先写临时文件再 os.replace —— 避免面板读到半写 JSON（进度块忽隐忽现的根因）
    try:
        tmp = PROGRESS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        os.replace(tmp, PROGRESS)
    except Exception:
        pass


def _round_label(idx):
    w = "东" if idx < 4 else "南"
    return "%s%d" % (w, idx % 4 + 1)


def play_one(policy, pb, seat, seed, epoch, gidx):
    game = RiichiGame(RiichiConfig(), seed=seed)
    results = []
    step = 0
    while game.phase != "game_end" and step < 10000:
        obs = game.state.get_observation()
        act = policy(game) if game.turn == seat else pb(game)
        res = game.step(act)
        for e in res.get("events", []):
            if e["type"] == "hora":
                actor = e.get("actor")
                target = e.get("target")
                results.append({"t": _round_label(game.round_idx),
                                "和": actor, "铳": target, "han": e.get("han") or 0,
                                "tsumo": target is None})
            elif e["type"] == "ryuukyoku":
                results.append({"t": _round_label(game.round_idx), "流局": True})
        _write_progress({"t": time.strftime("%H:%M:%S"), "epoch": epoch,
                         "game": gidx, "total": GAMES, "seat": seat,
                         "round": _round_label(game.round_idx), "round_idx": game.round_idx,
                         "scores": list(game.scores), "results": results[-10:],
                         "phase": "playing"})
        step += 1
    scores = game.scores
    rank = sorted(range(4), key=lambda i: -scores[i]).index(seat)
    spt = _spt(list(scores), seat, "tenhou")
    _write_progress({"t": time.strftime("%H:%M:%S"), "epoch": epoch,
                     "game": gidx, "total": GAMES, "seat": seat,
                     "round": "终局", "scores": list(scores),
                     "rank": rank + 1, "spt": int(round(spt)),
                     "results": results[-10:], "phase": "done"})
    return rank, spt, results


def _record_round(epoch, games_done, pts, ranks):
    n = len(pts)
    if n <= 0:
        return
    pos = sum(x for x in pts if x > 0)
    neg = sum(-x for x in pts if x < 0)
    wr_pt = pos / max(pos + neg, 1e-9)
    avg_pt = sum(pts) / n
    d = [0, 0, 0, 0]
    for r in ranks:
        if 0 <= r <= 3:   # play_one 返回 0-3 索引：0=1位 ... 3=4位
            d[r] += 1
    rank_wr = (d[0] + 0.5 * d[1]) / n
    avg_rank = sum(ranks) / n + 1.0   # 0-3 → 1-4 制
    import statistics
    pt_std = statistics.pstdev(pts) if n > 1 else 0.0
    row = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "epoch": epoch,
           "games_done": games_done, "wr_pt": round(wr_pt, 4),
           "avg_pt": round(avg_pt, 2), "avg_pt_per100": round(avg_pt * 100, 1),
           "wr_spt_pos": round(float(sum(1 for x in pts if x > 0)) / n, 4), "n": n,
           "rank_dist": d, "rank_wr": round(rank_wr, 4), "avg_rank": round(avg_rank, 2),
           "pt_std": round(pt_std, 1), "wins": sum(1 for x in pts if x > 0),
           "losses": n - sum(1 for x in pts if x > 0),
           "max_win": int(round(max((x for x in pts if x > 0), default=0))),
           "max_loss": int(round(min((x for x in pts if x < 0), default=0))),
           "elapsed_s": 0}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("[sliding] epoch %d 记录 %d 局: wr_pt=%.3f avg_pt=%+.1f"
          % (epoch, n, wr_pt, avg_pt), flush=True)


if __name__ == "__main__":
    print("[sliding] v2 started (cpu, games/round=%d, epoch-switch)" % GAMES, flush=True)
    last_epoch, last_mt = _ckpt_epoch()
    if last_epoch is None:
        print("[sliding] no ckpt, waiting", flush=True)
    pa = pb = None
    cur_pts, cur_ranks = [], []
    gidx = 0
    games_done = 0
    while True:
        try:
            ep, mt = _ckpt_epoch()
            if ep is None:
                time.sleep(30)
                continue
            if last_epoch is None or ep != last_epoch:
                if last_epoch is not None and cur_pts:
                    _record_round(last_epoch, games_done, cur_pts, cur_ranks)
                last_epoch, last_mt = ep, mt
                pa = RiichiPolicy(CKPT, seed=0, device=DEVICE, use_event_attn=True)
                pb = RiichiPolicy(SL_CKPT, seed=1, device=DEVICE)
                cur_pts, cur_ranks = [], []
                gidx = 0
                print("[sliding] 新 epoch %d 已加载，开始评估" % ep, flush=True)
            rank, spt, _rr = play_one(pa, pb, gidx % 4, last_epoch * 10000 + gidx,
                                      last_epoch, gidx + 1)
            cur_pts.append(spt)
            cur_ranks.append(rank)
            # 每局立即写对弈记录（最近 N 局实时更新）
            try:
                with open(GAMES_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "epoch": last_epoch, "seat": (gidx % 4) + 1,
                                        "rank": rank + 1, "spt": int(round(spt)),
                                        "win": spt > 0, "score": 0},
                                       ensure_ascii=False) + "\n")
            except Exception:
                pass
            gidx += 1
            games_done += 1
            if gidx >= GAMES:
                _record_round(last_epoch, games_done, cur_pts, cur_ranks)
                cur_pts, cur_ranks = [], []
                gidx = 0
        except Exception as e:
            import traceback
            print("[sliding] err: %s" % e, flush=True)
            traceback.print_exc()
            time.sleep(10)
