# -*- coding: utf-8 -*-
"""Φ 全局奖励预测器漂移监控 + 重训（P0-B，compliance 实现）。

依据（docs/rl_stage1_training_plan.md §4 + docs/rl_stage1_arch.md §7.6）:
1. 监控（检测）: 每 500 局用天凤日志评估 Φ 预测 MSE 与相关性 →
   logs/rl_phi_monitor.json；MSE 相对初始训练值恶化 >30% 触发重训
2. 重训（校正）: 用最新自对弈轨迹（终局精算 settlement_pt 为 ground truth）
   在线微调 Φ（lr 1e-4，~500 步，逐前缀监督）
3. 防遗忘混合: 微调数据 = 50% 最新自对弈 + 50% 人类日志

用法（由 train_rl.py 集成调用）:
    from tools.rl_phi_monitor import PhiMonitor
    mon = PhiMonitor(phi, device)
    mon.maybe_step(games_done, selfplay_buffer)   # 每 500 局自动评估/重训
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "C:/agentwork/src")
from model.rl_reward import RewardPredictor, _round_features, settlement_pt

RECORDS = "C:/agentwork/data/processed/transfer_records/records-*.jsonl.gz"
MONITOR_JSON = "C:/agentwork/logs/rl_phi_monitor.json"
PHI_OUT = "C:/agentwork/checkpoints/sl/rl/reward_predictor.pt"
LOG = "C:/agentwork/logs/rl_phi_monitor.txt"

DRIFT_RATIO = 1.3          # MSE 恶化 >30% 触发重训
EVAL_EVERY = 500           # 每 500 局评估一次（可被 train_rl 参数覆盖）
EVAL_GAMES = 1000          # 天凤日志评估局数（固定留出子集，可比性好）
RETRAIN_STEPS = 500        # 重训步数
RETRAIN_LR = 1e-4
RETRAIN_BATCH = 256
SELFPLAY_BUF = 2000        # 自对弈缓冲上限（局）


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s %s\n" % (time.strftime("%H:%M:%S"), msg))
    print("[phi-mon] %s" % msg, flush=True)


def collect_human_games(files, limit=EVAL_GAMES):
    """从天凤日志按 game_id 聚合每轮结束状态（同 train_reward_pred）。"""
    import gzip
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
                r = rec.get("round", 0)
                if r not in g["rounds"] or (rec.get("honba", 0) or 0) >= g["rounds"][r].get("honba", 0):
                    g["rounds"][r] = rec
                # 同轮同本场：保留 seat0 最后一次决策（Φ 手牌特征快照）
                if rec.get("seat") == 0:
                    prev0 = g["hands0"].get(r)
                    if prev0 is None or (rec.get("honba", 0) or 0) >= (prev0.get("honba", 0) or 0):
                        g["hands0"][r] = rec
                g["final"] = rec["scores"]
        if limit and len(games) >= limit:
            break
    return games


def games_to_seqs(games):
    """games -> (seqs list (K,16), labels list float, valid_mask list bool)。"""
    seqs, labels, valid = [], [], []
    for gid, g in games.items():
        rounds = g.get("rounds") or {}
        if not rounds or "final" not in g:
            continue
        order = sorted(rounds.keys())
        if len(order) < 2:
            continue
        round_scores, current_scores = [], []
        dealers, honbas, sticks, ridx = [], [], [], []
        hands, meld_counts, dora_ind = [], [], []
        prev = 25000
        for r in order:
            rec = rounds[r]
            s0 = rec["scores"][0]
            round_scores.append(s0 - prev)
            current_scores.append(s0)
            prev = s0
            dealers.append(rec.get("oya", 0))
            honbas.append(rec.get("honba", 0))
            sticks.append(rec.get("riichi_sticks", 0))
            ridx.append(rec.get("round", 0))
            rec0 = g.get("hands0", {}).get(r)
            hands.append(rec0.get("hand") if rec0 else None)
            meld_counts.append(
                len((rec0.get("melds") or [[], [], [], []])[0]) if rec0 else 0)
            dora_ind.append(rec.get("dora_indicators") or [])
        seq = _round_features(round_scores, current_scores, dealers, honbas,
                              sticks, ridx, hands=hands,
                              meld_counts=meld_counts,
                              dora_ind=dora_ind).squeeze(0)  # (K,16)
        R = settlement_pt(g["final"], 0, "tenhou")
        seqs.append(seq)
        labels.append(R)
        valid.append(True)
    return seqs, labels, valid


def eval_phi(phi, seqs, labels, device="cpu"):
    """Φ 对固定人类日志子集的预测 MSE + Pearson 相关性（逐前缀监督）。"""
    with torch.no_grad():
        preds, ys = [], []
        for seq, lab in zip(seqs, labels):
            if seq is None:
                continue
            out, _ = phi.gru(seq.unsqueeze(0).to(device))          # (1,K,H)
            pk = phi.fc(out).squeeze(-1).squeeze(0)                 # (K,)
            preds.append(pk.cpu().numpy())
            ys.append(np.full(pk.shape[0], lab, dtype=np.float32))
        if not preds:
            return float("nan"), float("nan"), 0
        P = np.concatenate(preds)
        Y = np.concatenate(ys)
        mse = float(np.mean((P - Y) ** 2))
        if P.std() < 1e-9 or Y.std() < 1e-9:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(P, Y)[0, 1])
        return mse, corr, int(len(P))


def _pad_and_batch(seqs, labels, device):
    Kmax = max(s.shape[0] for s in seqs)
    X = torch.zeros(len(seqs), Kmax, 16, dtype=torch.float32)
    for i, s in enumerate(seqs):
        X[i, :s.shape[0]] = s
    y = torch.tensor(labels, dtype=torch.float32)
    return X, y, Kmax


def retrain_phi(phi, selfplay, human_games, device="cpu",
                steps=RETRAIN_STEPS, lr=RETRAIN_LR, batch=RETRAIN_BATCH,
                out=PHI_OUT):
    """50% 自对弈 + 50% 人类日志微调 Φ（防遗忘混合，arch §7.6-3）。"""
    sp_seqs = [s for s, _ in selfplay if s is not None]
    sp_labels = [float(l) for s, l in selfplay if s is not None]
    if not sp_seqs:
        log("retrain skipped: no selfplay data")
        return None
    hu_seqs, hu_labels, hu_valid = games_to_seqs(human_games)
    hu_seqs = [s for s, v in zip(hu_seqs, hu_valid) if v]
    hu_labels = [l for l, v in zip(hu_labels, hu_valid) if v]
    # 混合：自对弈与人类各取 min(n_sp, n_hu) 局（50/50）
    n_mix = min(len(sp_seqs), len(hu_seqs))
    if n_mix < 8:
        log("retrain skipped: too few mixed games (%d)" % n_mix)
        return None
    sp_seqs, sp_labels = sp_seqs[:n_mix], sp_labels[:n_mix]
    hu_seqs, hu_labels = hu_seqs[:n_mix], hu_labels[:n_mix]
    Xs = sp_seqs + hu_seqs
    ys = sp_labels + hu_labels
    X, y, Kmax = _pad_and_batch(Xs, ys, device)
    model = phi.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.RandomState(0)
    n = len(X)
    t0 = time.time()
    for step in range(steps):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        xb, yb = X[idx].to(device), y[idx].to(device)
        h_out, _ = model.gru(xb)
        pred_k = model.fc(h_out).squeeze(-1)
        mask = (xb.abs().sum(-1) > 0).float()
        loss = ((pred_k - yb.unsqueeze(1)) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": model.state_dict(),
                "games": n_mix * 2, "kmax": Kmax,
                "config": "tenhou", "retrain_steps": steps,
                "retrain_time": time.strftime("%Y-%m-%d %H:%M:%S")}, out)
    log("Φ RETRAIN done: %d mixed games (%d sp + %d hu), %d steps, %.0fs -> %s"
        % (n_mix * 2, n_mix, n_mix, steps, time.time() - t0, out))
    return model


class PhiMonitor:
    """Φ 漂移监控器：每 EVAL_EVERY 局评估一次；MSE 恶化 >30% 自动重训。"""

    def __init__(self, phi, device="cpu", eval_every=EVAL_EVERY,
                 drift_ratio=DRIFT_RATIO, retrain_steps=RETRAIN_STEPS,
                 out_json=MONITOR_JSON):
        self.phi = phi
        self.device = device
        self.eval_every = eval_every
        self.drift_ratio = drift_ratio
        self.retrain_steps = retrain_steps
        self.out_json = out_json
        self.baseline_mse = None
        self.human_games = None
        self._load_existing()

    def _load_existing(self):
        if os.path.exists(self.out_json):
            try:
                d = json.load(open(self.out_json, encoding="utf-8"))
                self.baseline_mse = d.get("baseline_mse")
            except Exception:
                pass

    def _get_human_games(self):
        import glob
        if self.human_games is None:
            files = sorted(glob.glob(RECORDS))
            self.human_games = collect_human_games(files, EVAL_GAMES)
            log("human eval set: %d games" % len(self.human_games))
        return self.human_games

    def step(self, games_done, selfplay, force=False):
        """训练循环每局后调用；到达监控点执行评估，必要时重训。返回事件 dict 或 None。"""
        if not force and games_done % self.eval_every != 0:
            return None
        human = self._get_human_games()
        hu_seqs, hu_labels, hu_valid = games_to_seqs(human)
        hu_seqs = [s for s, v in zip(hu_seqs, hu_valid) if v]
        hu_labels = [l for l, v in zip(hu_labels, hu_valid) if v]
        mse, corr, npts = eval_phi(self.phi, hu_seqs, hu_labels, self.device)
        if self.baseline_mse is None or self.baseline_mse != self.baseline_mse:
            self.baseline_mse = mse
        drift = bool(mse > self.baseline_mse * self.drift_ratio) if self.baseline_mse else False
        ev = {"t": time.strftime("%Y-%m-%d %H:%M:%S"),
              "games_done": int(games_done),
              "baseline_mse": round(float(self.baseline_mse), 4),
              "mse": round(float(mse), 4),
              "corr": round(float(corr), 4),
              "n_points": int(npts),
              "drift_ratio": round(float(self.drift_ratio), 2),
              "drift_triggered": bool(drift),
              "retrain": False}
        log("eval games_done=%d mse=%.4f corr=%.4f (baseline=%.4f, drift=%s)"
            % (games_done, mse, corr, self.baseline_mse, drift))
        if drift:
            self.phi = retrain_phi(self.phi, selfplay[-SELFPLAY_BUF:], human,
                                   self.device, steps=self.retrain_steps)
            ev["retrain"] = True
            self.baseline_mse = None  # 重训后基线在下次评估重建
        self._write(ev)
        return ev

    def _write(self, ev):
        os.makedirs(os.path.dirname(self.out_json), exist_ok=True)
        hist = []
        if os.path.exists(self.out_json):
            try:
                hist = json.load(open(self.out_json, encoding="utf-8")).get("history", [])
            except Exception:
                pass
        hist.append(ev)
        with open(self.out_json, "w", encoding="utf-8") as f:
            json.dump({"history": hist, "baseline_mse": self.baseline_mse},
                      f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 独立 CLI：仅评估当前 Φ（不重训），验证监控链路
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phi = RewardPredictor().to(device)
    ck = torch.load(PHI_OUT, map_location="cpu")
    phi.load_state_dict(ck["model"])
    phi.eval()
    import glob
    games = collect_human_games(sorted(glob.glob(RECORDS)), EVAL_GAMES)
    seqs, labels, valid = games_to_seqs(games)
    seqs = [s for s, v in zip(seqs, valid) if v]
    labels = [l for l, v in zip(labels, valid) if v]
    mse, corr, n = eval_phi(phi, seqs, labels, device)
    print("PHI EVAL: mse=%.4f corr=%.4f points=%d games=%d" % (mse, corr, n, len(seqs)))