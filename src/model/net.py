# -*- coding: utf-8 -*-
"""Multi-head CNN (Suphx-style shared trunk) — task t4 deliverable net.py.

Architecture (docs/phase0_plan.md §2.2):
  Input : (B, C, 34, 1) feature tensor from model.features (34 tile-kind columns)
  Trunk : 3x1 conv stem (256 ch) + 50 residual blocks (3x1 conv 256 -> ReLU
          -> 3x1 conv 256 + skip). Implemented as Conv1d(kernel=3) over the
          34-kind axis — mathematically identical to the 3x1 2D conv on
          (34,1) but with far lower kernel-launch overhead.
  Heads : discard head (3x1 conv 32 -> FC 1024 -> FC 256 -> softmax 34,
          masked by legal discard kinds)
          binary heads (2 logits each): riichi / chow / pon / kan with
          candidate-tile one-hot input and availability masking
          value head (RL-stage placeholder): shared features -> scalar

Stage-1 curriculum ("simple" tsumogiri baseline) instantiates this class with
binary_heads=() and fewer trunk blocks; stage-2 "full" uses all heads.

model_profile() prints: parameter count (~21M), FLOPs estimate, CPU forward
timing (batch 1).
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_modules import EventCausalAttn


class ResidualBlock(nn.Module):
    """3x1 conv residual block (Suphx: preserve 34-column tile-kind semantics)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(x + h)


class DiscardHead(nn.Module):
    """(B,256,34) -> (B,34) logits (3x1 conv 32 -> FC 1024 -> FC 256 -> 34)."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv = nn.Conv1d(channels, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 34, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 34)

    def forward(self, x):
        h = F.relu(self.conv(x))
        h = h.flatten(1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class BinaryHead(nn.Module):
    """(B,256,34) + candidate tile one-hot (B,34) -> (B,2) logits."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv = nn.Conv1d(channels, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 34 + 34, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x, candidate):
        h = F.relu(self.conv(x)).flatten(1)
        h = torch.cat([h, candidate], dim=1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class ValueHead(nn.Module):
    """Shared features -> scalar value (RL-stage interface placeholder)."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv = nn.Conv1d(channels, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 34, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        h = F.relu(self.conv(x)).flatten(1)
        h = F.relu(self.fc1(h))
        return self.fc2(h).squeeze(-1)


class MultiHeadRiichiNet(nn.Module):
    """Shared trunk + discard head + binary action heads + value head.

    forward(x, masks=None, candidates=None) -> dict of logits:
      masks      : "discard" (B,34) bool; "riichi"/"chow"/"pon"/"kan" (B,) bool
                   illegal discard kinds are masked to -inf; unavailable
                   binary heads have BOTH logits masked to -inf (env gates
                   on legal_actions; trainer computes CE on available only)
      candidates : {"riichi": (B,34) one-hot, ...} for binary heads
    Output keys : "discard" (B,34), binary heads (B,2) each,
                  "value" (B,) when include_value=True
    """

    def __init__(self, in_channels: int, channels: int = 256, n_blocks: int = 50,
                 binary_heads: tuple = ("riichi", "chow", "pon", "kan"),
                 include_value: bool = True, use_event_attn: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_blocks = n_blocks
        self.binary_heads = tuple(binary_heads)
        self.include_value = include_value
        self.use_event_attn = use_event_attn
        self.event_attn = EventCausalAttn(d_model=channels) if use_event_attn else None
        # 逐通道门控注入：σ(gate)∈(0,1)，初始 -1（σ≈0.27，评审：-2 时 σ≈0.12
        # 梯度仍偏小（gate/trunk 梯度比 0.18、门长期冻结 0.119）；-1 平衡保护旧权重
        # 与旁支可学习性，让 ② 事件因果注意力在自对弈中真正可学）
        self.ctx_gate = nn.Parameter(torch.full((channels,), -1.0)) if use_event_attn else None
        # 注入尺度（2026-08-27 根因修复）：BinaryHead 在 34列×256通道 上线性求和
        # （权重 L1 和 ~870），σ(gate)·tanh(ctx)≈0.24 的小注入会被放大成 logit
        # ±200~1000 的偏移 → 立直死（z=-1001 clamp 后 p≈0）与必碰副露（z=+250）。
        # 0.02 缩放后注入≈0.005 → logit 偏移 ≈4（可接受）；可经 logs/rl_hyper.json
        # 的 inj_scale 热调。
        self.inj_scale = 0.02
        self.stem = nn.Conv1d(in_channels, channels, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])
        self.discard = DiscardHead(channels)
        self.binary = nn.ModuleDict({h: BinaryHead(channels) for h in self.binary_heads})
        self.value = ValueHead(channels) if include_value else None

    def forward(self, x, masks=None, candidates=None, events=None):
        masks = masks or {}
        candidates = candidates or {}
        if x.dim() == 4:  # accept (B,C,34,1) for compat, squeeze to (B,C,34)
            x = x.squeeze(-1)
        h = F.relu(self.stem(x))
        h = self.blocks(h)
        if self.event_attn is not None and events is not None:
            # ② 事件因果注意力：事件上下文逐通道门控注入 trunk 特征列
            # h = h + σ(gate_c)·ctx_c（初始 gate=-1 → 注入≈0.27，训练中逐通道开启）
            pad = events.abs().sum(-1) == 0                # (B,T) 补零位
            ctx = torch.tanh(self.event_attn(events, pad))  # (B,C)（M0-A1 padding mask）
            gate = torch.sigmoid(self.ctx_gate)            # (C,)
            # 初始 gate=-1 → σ≈0.27；×inj_scale(0.02) 后注入≈0.005（防头放大爆炸）
            h = h + self.inj_scale * (gate.unsqueeze(0).unsqueeze(-1)
                                      * ctx.unsqueeze(-1))
        logits = {"discard": self.discard(h)}
        for head in self.binary_heads:
            cand = candidates.get(head)
            if cand is None:
                cand = torch.zeros(x.shape[0], 34, device=x.device)
            logits[head] = self.binary[head](h, cand)
        if self.value is not None:
            logits["value"] = self.value(h)
        # legal-action / availability masking
        m = masks.get("discard")
        if m is not None:
            logits["discard"] = logits["discard"].masked_fill(~m, float("-inf"))
        for head in self.binary_heads:
            avail = masks.get(head)
            if avail is not None:
                logits[head] = logits[head].masked_fill(~avail.unsqueeze(1),
                                                        float("-inf"))
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def discard_probs(self, x, masks=None):
        """Softmax over legal discards: (B,34) probs."""
        logits = self.forward(x, masks=masks)["discard"]
        return F.softmax(logits, dim=1)


def estimate_macs(model: MultiHeadRiichiNet, batch: int = 1) -> float:
    """Analytic MAC (multiply-add) count for one forward pass of batch samples.

    Conv1d(k=3, padding=1): output length 34, MACs = out*in*3*34 per sample.
    Linear: MACs = out*in per sample.
    """
    C = model.in_channels
    ch = model.channels
    macs = 0.0
    macs += ch * C * 3 * 34            # stem
    macs += model.n_blocks * 2 * ch * ch * 3 * 34   # residual blocks
    macs += 32 * ch * 3 * 34           # discard conv
    macs += 32 * 34 * 1024 + 1024 * 256 + 256 * 34    # discard FCs
    for _ in model.binary_heads:
        macs += 64 * ch * 3 * 34
        macs += (64 * 34 + 34) * 512 + 512 * 2
    if model.value is not None:
        macs += 32 * ch * 3 * 34 + 32 * 34 * 512 + 512
    return macs * batch



def model_profile(model: MultiHeadRiichiNet, device: str = "cpu",
                  batch: int = 1, warmup: int = 2, iters: int = 10) -> dict:
    """Params / FLOPs estimate / forward timing (CPU batch=1 by default)."""
    n_params = model.num_parameters()
    macs = estimate_macs(model, batch)
    flops_fwd = 2 * macs  # 1 MAC = 2 FLOPs
    model.to(device)
    x = torch.randn(batch, model.in_channels, 34, 1, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        dt = (time.perf_counter() - t0) / iters
    return {
        "params": n_params,
        "params_M": n_params / 1e6,
        "macs_fwd": macs,
        "flops_fwd_G": flops_fwd / 1e9,
        "flops_fwd_bwd_G": 3 * flops_fwd / 1e9,  # backward ~2x forward
        "fwd_ms": dt * 1e3,
        "batch": batch,
        "device": device,
        "in_channels": model.in_channels,
        "n_blocks": model.n_blocks,
    }


def get_device(prefer: str = "cuda") -> torch.device:
    """CUDA if available, else CPU (Blackwell sm_120 needs cu128+ torch build)."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _main():
    """CLI: python -m model.net [C] — print params / FLOPs / CPU timing."""
    import sys
    from model.features import feature_channels
    c = int(sys.argv[1]) if len(sys.argv) > 1 else feature_channels("full")
    torch.set_num_threads(1)
    net = MultiHeadRiichiNet(c, channels=256, n_blocks=50)
    prof = model_profile(net, device="cpu", batch=1)
    print("in_channels=" + str(c) + " params=" + format(prof["params_M"], ".2f")
          + "M fwd_FLOPs=" + format(prof["flops_fwd_G"], ".2f") + "G fwd+bwd="
          + format(prof["flops_fwd_bwd_G"], ".2f") + "G CPU_fwd="
          + format(prof["fwd_ms"], ".1f") + "ms (batch 1)")


if __name__ == "__main__":
    _main()
