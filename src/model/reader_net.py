# -*- coding: utf-8 -*-
"""ReaderNet：对手读牌模型（独立于主网络，阶段 1 预训练，阶段 2 门控注入主网络）。

目标：从"可见信息"推断单家对手的隐藏状态——
  waits  : 34 维 sigmoid —— 该家听牌时能和的牌（听牌标签）
  danger : 34 维 sigmoid —— 该家能和的牌（与 waits 同源，但按向听数加权训练，
           重点是"接近听牌"的对手；开局手牌危险度≈0）
  pt     : 1 维回归 —— 该家当前手牌的最大潜在和牌打点（pt 量纲，归一化）

输入：现有 build_features(obs, "full") 的 284 维特征（全局可见信息：
      所有家牌河/副露/场况/dora + seat0 手牌——读牌模型用"可见信息"推断隐藏手牌）。

训练要点（详见 tools/gen_reader_data.py 与 tools/train_reader.py）：
  1. 样本 = 一家视角：每步局面为 3 个对手各生成一个样本
  2. 向听数加权：w(shanten) = {0:1.0, 1:0.7, 2:0.5, 3:0.3, 4:0.15, 5+:0.05}
     ——越靠近听牌权重越大，开局低危险样本不干扰
  3. 损失 = Σ w×(BCE(waits)+BCE(danger)) + w×MSE(pt)
  4. 指标：AUC(waits/danger)、MAE(pt)、按向听数分层 AUC
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 3, padding=1, bias=False)
        self.c2 = nn.Conv1d(c, c, 3, padding=1, bias=False)
        self.n1 = nn.BatchNorm1d(c)
        self.n2 = nn.BatchNorm1d(c)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.relu(x + h)


class ReaderNet(nn.Module):
    """轻量读牌网络：284 特征 → stem → n_blocks ResBlock → 3 头。"""

    def __init__(self, in_channels: int = 284, channels: int = 128,
                 n_blocks: int = 16):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, channels, 3, padding=1, bias=False)
        self.n0 = nn.BatchNorm1d(channels)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        # 3 头（34 列卷积 → 池化 → FC）
        self.waits = nn.Sequential(
            nn.Conv1d(channels, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.danger = nn.Sequential(
            nn.Conv1d(channels, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.pt = nn.Sequential(
            nn.Conv1d(channels, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc_w = nn.Linear(64, 34)
        self.fc_d = nn.Linear(64, 34)
        self.fc_p = nn.Linear(32, 1)

    def forward(self, x):
        """x: (B, 284, 34) -> (waits(B,34), danger(B,34), pt(B,1))"""
        h = F.relu(self.n0(self.stem(x)))
        h = self.blocks(h)
        w = self.fc_w(self.waits(h).squeeze(-1))
        d = self.fc_d(self.danger(h).squeeze(-1))
        p = self.fc_p(self.pt(h).squeeze(-1))
        return w, d, p


def load_reader(path, device="cpu"):
    """加载训练好的 ReaderNet 权重。"""
    ck = torch.load(path, map_location="cpu")
    m = ReaderNet()
    m.load_state_dict(ck["model"])
    m.eval().to(device)
    return m, ck
