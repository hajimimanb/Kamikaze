# -*- coding: utf-8 -*-
"""GRU 全局奖励预测器训练（Suphx §3.2，rl_reward.RewardPredictor）。

数据：从 transfer_records 聚合每局每轮特征（16 维：记分板 12 + 手牌 4）+ 终局精算奖励标签。
- 按 game_id 分组，每轮最后一条记录 = 该轮结束状态（分数/庄/本场/宝牌指示）
- 手牌特征取该轮 seat0 最后一次决策记录（手牌快照；无则中性占位）
- round_scores[k] = scores[k] - scores[k-1]（seat 0 视角）
- 标签 R = settlement_pt(final scores, seat 0, "tenhou")
- 训练：MSE(Φ(x_1..k), R)，K 位置全部监督（Suphx Eq.4）

用法：python tools/train_reward_pred.py [--games-limit N 调试用]
产出：checkpoints/sl/rl/reward_predictor.pt + logs/reward_pred.txt
"""
import argparse, gzip, glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "C:/agentwork/src")
from model.rl_reward import RewardPredictor, _round_features, settlement_pt

RECORDS = "C:/agentwork/data/processed/transfer_records/records-*.jsonl.gz"
OUT = "C:/agentwork/checkpoints/sl/rl/reward_predictor.pt"
LOG = "C:/agentwork/logs/reward_pred.txt"


def collect_games(files, limit=0):
    """按 game_id 聚合每轮结束状态；返回 {gid: {"rounds": {r: rec}, "hands0": {r: rec0},
    "final": scores}}。rounds=轮末记录（分数/庄/本场/宝牌），hands0=该轮 seat0 最后
    决策记录（Φ 手牌特征快照）。"""
    games = {}
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gid = rec["game_id"]
                g = games.get(gid)
                if g is None:
                    g = games[gid] = {"rounds": {}, "hands0": {}}
                # 每轮只保留最后一条（该轮结束状态）
                r = rec.get("round", 0)
                key = r
                if key not in g["rounds"] or (rec.get("honba", 0) or 0) >= g["rounds"][key].get("honba", 0):
                    g["rounds"][key] = rec
                # 同轮同本场：保留 seat0 最后一次决策（其手牌=该轮内 seat0 最终手牌）
                if rec.get("seat") == 0:
                    prev0 = g["hands0"].get(key)
                    if prev0 is None or (rec.get("honba", 0) or 0) >= (prev0.get("honba", 0) or 0):
                        g["hands0"][key] = rec
                g["final"] = rec["scores"]  # 全量扫，最后覆盖=终局
        if limit and len(games) >= limit:
            break
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    t0 = time.time()
    files = sorted(glob.glob(RECORDS))
    games = collect_games(files, args.games_limit)
    print("games:", len(games), "%.0fs" % (time.time() - t0), flush=True)

    seqs, labels = [], []
    for gid, g in games.items():
        rounds = g["rounds"]
        if not rounds:
            continue
        order = sorted(rounds.keys())
        if len(order) < 2:
            continue
        cur = []
        for r in order:
            rec = rounds[r]
            cur.append(rec)
        # 每轮：scores（seat0）、dealer、honba、sticks、round + seat0 手牌特征
        round_scores, current_scores = [], []
        dealers, honbas, sticks, ridx = [], [], [], []
        hands, meld_counts, dora_ind = [], [], []
        prev = 25000
        for rec in cur:
            s0 = rec["scores"][0]
            round_scores.append(s0 - prev)
            current_scores.append(s0)
            prev = s0
            dealers.append(rec.get("oya", 0))
            honbas.append(rec.get("honba", 0))
            sticks.append(rec.get("riichi_sticks", 0))
            ridx.append(rec.get("round", 0))
            rec0 = g.get("hands0", {}).get(rec.get("round", 0))
            hands.append(rec0.get("hand") if rec0 else None)
            meld_counts.append(
                len((rec0.get("melds") or [[], [], [], []])[0]) if rec0 else 0)
            dora_ind.append(rec.get("dora_indicators") or [])
        seq = _round_features(round_scores, current_scores, dealers, honbas,
                              sticks, ridx, hands=hands,
                              meld_counts=meld_counts, dora_ind=dora_ind)  # (1,K,16)
        R = settlement_pt(g["final"], 0, "tenhou")
        seqs.append(seq.squeeze(0))
        labels.append(R)
    if not seqs:
        print("no data"); return
    print("seqs=%d maxK=%d" % (len(seqs), max(s.shape[0] for s in seqs)), flush=True)

    # padding 到最大 K
    Kmax = max(s.shape[0] for s in seqs)
    X = torch.zeros(len(seqs), Kmax, 16)
    for i, s in enumerate(seqs):
        X[i, :s.shape[0]] = s
    y = torch.tensor(labels, dtype=torch.float32)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardPredictor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n = len(X)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        nb = 0
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            xb, yb = X[idx].to(device), y[idx].to(device)
            # 每位置监督（Suphx Eq.4）：Φ(x_1..k) ~ R，逐时间步预测
            out, _ = model.gru(xb)            # (B,K,H)
            pred_k = model.fc(out).squeeze(-1)  # (B,K)
            mask = (xb.abs().sum(-1) > 0).float()  # 有效轮次
            loss = ((pred_k - yb.unsqueeze(1)) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        print("epoch %d: loss=%.4f" % (ep, tot / max(nb, 1)), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), "games": len(games),
                "kmax": Kmax, "config": "tenhou"}, args.out)
    print("REWARD_PRED DONE -> %s (%.0fs)" % (args.out, time.time() - t0))


if __name__ == "__main__":
    main()
