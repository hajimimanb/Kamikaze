# -*- coding: utf-8 -*-
"""决策头阈值/温度校准到人类数据集（RL 初始值）。

目标: 模型在"人类决策点"上的各头执行率 ≈ 人类实际执行率。
- tsumo/ron/kyushu/riichi/chow/pon/kan 七个决策头
- 对每个头: 温度 T（采样/评估概率缩放）+ 阈值 tau（act 确定性决策）
- 输出: checkpoints/sl/rl/calibration.json + checkpoints/sl/transfer/calibration.json
  （policy.load_calibration 读取；act() 用 tau，sample_with_logp 用 T）

用法:
  .venv/Scripts/python.exe tools/calibrate_human_prior.py [--ckpt ...] [--max-samples 80000] [--event-attn]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "C:/agentwork/src")
sys.path.insert(0, "C:/agentwork")

from agent.policy import RiichiPolicy

HEADS = ("tsumo", "ron", "chow", "pon", "kan", "riichi", "kyushu")
RECORDS = "C:/agentwork/data/processed/transfer_records/records-*.jsonl.gz"
OUT_RL = "C:/agentwork/checkpoints/sl/rl/calibration.json"
OUT_TF = "C:/agentwork/checkpoints/sl/transfer/calibration.json"


def collect(policy, files, max_samples):
    """在人类决策点上收集各头 (p_h, exec) 样本（批量前向加速）。"""
    from model.train_rl_vec import _batch_logits
    data = {h: {"p": [], "exec": []} for h in HEADS}
    n = 0
    buf = []   # (rec, legal_heads, at)
    BATCH = 256
    def flush():
        nonlocal buf
        if not buf:
            return
        obss = [r for r, _, _ in buf]
        lg = _batch_logits(policy.model, obss, [None] * len(obss),
                           policy.device, amp=False)
        for k, (rec, legal_heads, at) in enumerate(buf):
            for h in legal_heads:
                z = lg[h][k]
                p = float(torch.sigmoid((z[1] - z[0])
                                        / policy.temp.get(h, 1.0)).item())
                data[h]["p"].append(p)
                data[h]["exec"].append(1 if at == h else 0)
        buf = []
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                la = rec.get("legal_actions") or {}
                lab = rec.get("label") or {}
                at = lab.get("type") if isinstance(lab, dict) else None
                legal_heads = [h for h in HEADS if la.get(h)]
                if not legal_heads:
                    continue
                buf.append((rec, legal_heads, at))
                n += 1
                if len(buf) >= BATCH:
                    flush()
                if n >= max_samples:
                    flush()
                    return data, n
    flush()
    return data, n


def collect_bias(policy, n_games=30, seed0=8000):
    """自对弈局面各头 logits z 中位数 -> bias = -median（防饱和，梯度健康）。
    简单动作推进（仅收集合法时的 z，与动作无关）。"""
    from env.riichi_game import RiichiGame, RiichiConfig
    zs = {h: [] for h in HEADS}
    for g in range(n_games):
        env = RiichiGame(config=RiichiConfig(), seed=seed0 + g)
        for _ in range(10000):
            if env.phase == "game_end":
                break
            obs = env.state.get_observation()
            la = obs.get("legal_actions") or {}
            lg = policy._logits(obs, None, amp=False)
            for h in HEADS:
                if la.get(h):
                    zs[h].append(float((lg[h][0][1] - lg[h][0][0]).item()))
            try:
                env.step({"type": "discard", "tile": la["discard"][0]}
                         if la.get("discard") else {"type": "pass"})
            except Exception:
                break
    bias = {}
    for h in HEADS:
        if len(zs[h]) >= 5:
            bias[h] = -float(np.median(zs[h]))
    return bias, {h: len(v) for h, v in zs.items()}


def calibrate(data):
    """每头: 温度 T 使 mean(p**(1/T))≈人类率；阈值 tau 在展开后概率上匹配。

    梯度健康约束: 二分类头在自对弈分布可能 logits 饱和（碰 z≈-14、杠 z≈-99、
    吃 z≈+52）→ sigmoid 梯度≈0 → RL 无法学习。强制 T >= T_MIN 使 z/T 展开，
    p 不饱和（梯度可流动），tau 在展开后的 p' 上重算（保持人类频率匹配）。
    """
    temps = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    T_MIN = {"chow": 2.0, "pon": 2.5, "kan": 2.5, "riichi": 1.5,
             "tsumo": 2.0, "ron": 2.0, "kyushu": 1.2}
    taus = [round(0.02 * i, 2) for i in range(1, 50)]
    out = {}
    for h in HEADS:
        ps = np.array(data[h]["p"])
        ex = np.array(data[h]["exec"])
        n = len(ps)
        if n == 0:
            out[h] = {"T": 1.0, "tau": 0.5, "n": 0, "human_rate": None}
            continue
        human_rate = float(ex.mean())
        # 原温度（收集时 policy.temp，默认 1.0）→ logits z = T_old * logit(p)
        T_old = 1.0
        z = T_old * np.log(np.clip(ps, 1e-7, 1 - 1e-7) / (1 - np.clip(ps, 1e-7, 1 - 1e-7)))
        # 温度: mean(p**(1/T)) 拟合人类率，但受 T_MIN 约束（梯度健康）
        best_T, best_e = 1.0, 1e9
        for T in temps:
            if T < T_MIN.get(h, 0.5):
                continue
            pT = np.clip(ps, 1e-6, 1.0) ** (1.0 / max(T, 0.1))
            e = abs(float(pT.mean()) - human_rate)
            if e < best_e:
                best_e, best_T = e, T
        # 展开后的概率 p' = sigmoid(z / best_T)
        ps_exp = 1.0 / (1.0 + np.exp(-z / max(best_T, 0.1)))
        # 阈值: 展开概率上 (p' >= tau) 匹配人类率
        best_tau, best_te = 0.5, 1e9
        for tau in taus:
            rate = float((ps_exp >= tau).mean())
            e = abs(rate - human_rate)
            if e < best_te:
                best_te, best_tau = e, tau
        out[h] = {"T": best_T, "tau": best_tau, "n": n,
                  "human_rate": human_rate,
                  "model_T_rate": float((np.clip(ps, 1e-6, 1.0) ** (1.0 / max(best_T, 0.1))).mean()),
                  "model_tau_rate": float((ps_exp >= best_tau).mean()),
                  "z_median": float(np.median(z)),
                  "z_exp_median": float(np.median(z / max(best_T, 0.1)))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sl/rl/value_pretrain.pt")
    ap.add_argument("--max-samples", type=int, default=80000)
    ap.add_argument("--event-attn", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = RiichiPolicy(args.ckpt, seed=0, device=device,
                          use_event_attn=args.event_attn)
    files = sorted(glob.glob(RECORDS))
    print("records files:", len(files), flush=True)
    data, n = collect(policy, files, args.max_samples)
    print("collect %d decision points" % n, flush=True)
    for h in HEADS:
        print("  %-6s n=%d" % (h, len(data[h]["p"])), flush=True)
    res = calibrate(data)
    print("=== 校准结果 (人类率 -> 模型率) ===", flush=True)
    for h in HEADS:
        r = res[h]
        if r["n"] == 0:
            print("  %-6s n=0 (无样本)" % h, flush=True)
            continue
        print("  %-6s 人类 %.1f%% | T=%.2f → 模型T率 %.1f%% | tau=%.2f → 模型tau率 %.1f%% (n=%d)"
              % (h, 100 * r["human_rate"], r["T"], 100 * r["model_T_rate"],
                 r["tau"], 100 * r["model_tau_rate"], r["n"]), flush=True)
    # 自对弈 bias（logit 平移防饱和）
    bias, zcnt = collect_bias(policy, n_games=30)
    print("=== 自对弈 bias（logit 平移防饱和）===", flush=True)
    for h in HEADS:
        print("  %-6s z样本=%d bias=%+.1f" % (h, zcnt.get(h, 0), bias.get(h, 0.0)), flush=True)
    # 写 calibration.json（policy 加载格式）
    cal = {}
    for h in HEADS:
        r = res[h]
        cal[h] = {"thr": float(r["tau"]), "T": float(r["T"]),
                  "bias": float(bias.get(h, 0.0))}
    for out_p in (OUT_RL, OUT_TF):
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(cal, f, ensure_ascii=False, indent=1)
        print("written:", out_p, flush=True)


if __name__ == "__main__":
    main()
