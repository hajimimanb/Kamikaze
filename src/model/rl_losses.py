# -*- coding: utf-8 -*-
"""RL 损失函数定义（Suphx 式 PPO + 熵正则 + 全局奖励预测 + 神谕引导）。

全部为纯张量函数，便于单测；公式出处见 docs/rl_spec.md。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ppo_policy_loss(logp_old: torch.Tensor, logp_new: torch.Tensor,
                    advantages: torch.Tensor, clip_eps: float = 0.2) -> torch.Tensor:
    """PPO 裁剪代理目标（Suphx 用策略梯度，我们提供 PPO 稳定版）：
        L = -min( r·A, clip(r, 1±ε)·A ),  r = exp(logp_new - logp_old)
    返回逐样本损失（对外 mean() 后反向）。"""
    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(surr1, surr2)


def entropy_loss(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """策略熵 H = -Σ p·log p（逐样本，非负）；最大熵 => 最小化 -H。"""
    return -torch.sum(probs * torch.log(probs + eps), dim=-1)


def adaptive_entropy_coef(alpha: float, beta: float,
                          h_target: float, h_bar: float) -> float:
    """Suphx Eq.(3)：α ← α + β·(H_target − H̄(π))，熵高于目标则减小 α。"""
    return alpha + beta * (h_target - h_bar)


def value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """价值头（critic）MSE： (V(s) − R_t)²。"""
    return F.mse_loss(values, returns)


def gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
        gamma: float = 0.99, lam: float = 0.95):
    """GAE：A_t = Σ (γλ)^l·δ_{t+l}，δ_t = r_t + γ·V(s_{t+1}) − V(s_t)。
    返回 (advantages, returns=adv+values)。"""
    n = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val = 0.0 if (dones[t].item() or t + 1 >= n) else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * lam * (0.0 if dones[t].item() else last_gae)
        adv[t] = last_gae
    return adv, adv + values


def reward_predictor_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """全局奖励预测器（Suphx Eq.4）：MSE(Φ(x_1..k) − R)。"""
    return F.mse_loss(pred, target)


def oracle_dropout_mask(perfect_shape, gamma_t: float, device="cpu") -> torch.Tensor:
    """神谕引导（Suphx Eq.5）：完美特征按 P(δ=1)=γ_t 的伯努利掩码保留；
    γ_t 从 1 线性衰减到 0，最终转为纯普通代理。返回 0/1 掩码（1=保留）。"""
    return (torch.rand(perfect_shape, device=device) < gamma_t).float()


def per_round_reward(pred: torch.Tensor, pred_prev: torch.Tensor) -> torch.Tensor:
    """全局奖励预测的每局奖励：Φ(x_1..k) − Φ(x_1..k−1)（Suphx §3.2）。"""
    return pred - pred_prev
