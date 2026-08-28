# -*- coding: utf-8 -*-
"""D: 事件注意力门控梯度探针（evaluator 方案 D）。
判定 ctx_gate 是否真的在学习（② 可救性）：
- 用当前模型在真实局面 batch 上做带梯度前向（含事件旁支）
- loss = 所有 head logits 之和（对 gate 的间接梯度路径）
- 输出 ctx_gate 梯度量级 vs trunk 某层梯度量级（对比基线）
用法: .venv/Scripts/python.exe tools/gate_probe.py [--ckpt ...]
"""
import sys
sys.path.insert(0, "C:/agentwork/src"); sys.path.insert(0, "C:/agentwork")
import argparse
import numpy as np
import torch

from agent.policy import RiichiPolicy
from env.riichi_game import RiichiGame, RiichiConfig
from model.attn_modules import build_events
from model.features import BINARY_HEADS, build_features, discard_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sl/rl/rl_v1.pt")
    ap.add_argument("--n-obs", type=int, default=48)
    args = ap.parse_args()
    device = "cuda"
    pol = RiichiPolicy(args.ckpt, seed=1, device=device, use_event_attn=True)
    if pol.model.event_attn is None:
        print("模型无事件注意力旁支（use_event_attn=False），探针无效")
        return

    # 收集真实 obs（含 events）
    obss = []
    env = RiichiGame(config=RiichiConfig(), seed=8300)
    n = 0
    while len(obss) < args.n_obs:
        if env.phase == "game_end":
            env = RiichiGame(config=RiichiConfig(), seed=8300 + n); n += 1
        obs = env.state.get_observation()
        la = obs.get("legal_actions") or {}
        if la.get("discard"):
            obss.append(obs)
        try:
            env.step({"type": "discard", "tile": la["discard"][0]}
                     if la.get("discard") else {"type": "pass"})
        except Exception:
            break

    xs = np.stack([build_features(o, "full")[:, :, 0] for o in obss])
    dms = np.stack([discard_mask(o) for o in obss])
    cands = {}
    for h in BINARY_HEADS:
        rows = []
        for o in obss:
            la = o.get("legal_actions") or {}
            legal = la.get(h)
            k = -1
            if h == "riichi" and legal:
                k = legal[0] // 4
            elif h in ("chow", "pon", "kan") and legal:
                k = legal[0]["tiles"][0] // 4
            v = np.zeros(34, dtype=np.float32)
            if k >= 0:
                v[k] = 1.0
            rows.append(v)
        cands[h] = torch.from_numpy(np.stack(rows)).to(device)
    x = torch.from_numpy(xs).to(device)
    dm = torch.from_numpy(dms).to(device)
    ev = torch.stack([build_events(o) for o in obss]).float().to(device)

    pol.model.train()
    pol.model.zero_grad()
    lg = pol.model(x, masks={"discard": dm}, candidates=cands, events=ev)
    lg = {h: v.float() for h, v in lg.items()}
    loss = sum(lg[h].sum() for h in lg if h != "value") + lg["value"].sum()
    loss.backward()

    g = pol.model.ctx_gate.grad
    print("=== gate 梯度探针（%d obs, ckpt=%s）===" % (len(obss), args.ckpt))
    if g is None:
        print("ctx_gate.grad = None（梯度未流动 → ② 死）")
    else:
        print("ctx_gate.grad: mean=%.6f abs_mean=%.6f max=%.6f | nonzero=%.1f%%"
              % (g.mean().item(), g.abs().mean().item(), g.abs().max().item(),
                 100.0 * (g != 0).float().mean().item()))
    # 对照：trunk/head 梯度
    for tag, key in (("trunk blocks.0", "blocks.0."),
                     ("head riichi", "binary.riichi."),
                     ("stem", "stem.")):
        gs = []
        for n2, p in pol.model.named_parameters():
            if key in n2 and p.grad is not None:
                gs.append(p.grad.abs().mean().item())
        if gs:
            print("%s 梯度 abs_mean: %.8f" % (tag, sum(gs) / len(gs)))
    if g is not None:
        ratio = g.abs().mean().item() / max(1e-9, (sum(
            p.grad.abs().mean().item() for n2, p in pol.model.named_parameters()
            if "blocks.0." in n2 and p.grad is not None) or 1e-9))
        print("gate/trunk 梯度比: %.4f（>0.1 则旁支在学；<0.01 则近似死）" % ratio)


if __name__ == "__main__":
    main()
