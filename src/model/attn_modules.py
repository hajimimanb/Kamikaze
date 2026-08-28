# -*- coding: utf-8 -*-
"""RL 架构创新模块（训练前准备，仅实现+单测，不训练）：

② EventCausalAttn —— 公开事件序列 → 因果注意力编码（阶段1 主干旁路）
   事件 token = (seat, tile kind, tsumogiri, riichi, meld kind, round) 混合 one-hot；
   因果掩码只看过去；输出 context 向量注入 trunk 特征列。

④ OpponentEquivariant —— 他家 3 家置换等变聚合（阶段1/2 通用）
   共享参数 per-opponent 编码 + 3 家间自注意力 + 平均池化；
   对"打乱 3 家座位"输出按同置换变化（等变）/ 全局池化不变。

③ BeliefDangerHead —— 信念注意力读牌（阶段2 danger head）
   交叉注意力：全局公开特征(query) × 他家事件序列(key/value) -> 他家听牌信念；
   输出每家 tenpai logit + 34 维危险度 logits；3 家共享参数（等变）。

设计要点：
- 输入全部来自 obs（公开信息），不依赖引擎改动
- 等变性单测：permute 3 家事件 -> 输出同置换
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_events(obs: dict, per_opp: int = 8, max_events: int = 64) -> torch.Tensor:
    """从 obs 构建公开事件 token 序列 (T, D)。

    事件源：4 家 discards（含 tsumogiri/riichi 标记）+ 3 家对手副露。
    每 token 维度：seat(4) + tile(34) + tsumogiri(1) + riichi(1) + meld_kind(5) + round(1) = 46
    """
    D = 46
    la = obs.get("legal_actions") or {}
    discards = obs.get("discards") or [[] for _ in range(4)]
    melds = obs.get("melds") or [[] for _ in range(4)]
    seat_now = obs.get("seat", 0)
    round_idx = obs.get("round", 0) % 8
    rows = []
    MELD = {"chi": 0, "pon": 1, "kan": 2, "chow": 0, "minkan": 3, "ankan": 4, "kakan": 4}

    def tok(seat, kind, tsug, riichi, meld_k):
        v = torch.zeros(D)
        v[seat] = 1.0
        if kind >= 0:
            v[4 + kind] = 1.0
        v[38] = float(tsug)
        v[39] = float(riichi)
        if meld_k >= 0:
            v[40 + meld_k] = 1.0
        v[45] = float(round_idx) / 8.0
        return v

    # 各家的最近打牌（最后 per_opp 张），保留时间序（discards 列表有序）
    for s in range(4):
        ds = discards[s][-per_opp:]
        for d in ds:
            rows.append(tok(s, (d.get("tile", 0) or 0) // 4,
                            d.get("tsumogiri", False), d.get("riichi", False), -1))
    # 对手副露 token（公开信息）
    for s in range(4):
        if s == seat_now:
            continue
        for m in melds[s][-2:]:
            mt = m.get("type", "")
            rows.append(tok(s, (m["tiles"][0] if m.get("tiles") else 0) // 4,
                            False, False, MELD.get(mt, 0)))
    # 截断/补零
    if len(rows) > max_events:
        rows = rows[-max_events:]
    while len(rows) < max_events:
        rows.insert(0, torch.zeros(D))
    return torch.stack(rows)  # (T, D)


class EventCausalAttn(nn.Module):
    """② 公开事件序列因果注意力 -> context 向量 (B, C)。

    因果掩码：事件按时间序，只看过去。输出最后 token 的隐状态作为上下文。
    """

    def __init__(self, d_model: int = 256, n_head: int = 8,
                 max_events: int = 64, event_dim: int = 46):
        super().__init__()
        self.d_model = d_model
        self.max_events = max_events
        self.proj = nn.Linear(event_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(max_events, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)
        # 冷启动修复（立直头被旁支压到 P≈0.015 的根因）：旁支输出小 scale，
        # 初期近乎零注入主干（tanh(0.01)≈0.01 × gate 0.12 ≈ 0.0012/通道），
        # 权重小但梯度路径完整（不饥饿），随训练自然开启
        nn.init.uniform_(self.out.weight, -0.0001, 0.0001)
        nn.init.zeros_(self.out.bias)

    def forward(self, events: torch.Tensor, padding_mask=None) -> torch.Tensor:
        """events: (B, T, D) -> context (B, C)。
        padding_mask: (B, T) bool，True=补零位（M0-A1：补零不参与注意力）。
        输出 = last-token + 掩码平均池化（A1 提交2：增强；提交1 仅 mask）。"""
        B, T, _ = events.shape
        if padding_mask is None:
            padding_mask = events.abs().sum(-1) == 0   # 全零行=补零（默认检测）
        x = self.proj(events) + self.pos[:T].unsqueeze(0)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=events.device), 1)
        x = self.attn(x, x, x, attn_mask=causal,
                      key_padding_mask=padding_mask, need_weights=False)[0]
        x = self.norm(x + self.proj(events) + self.pos[:T].unsqueeze(0))
        ctx = x[:, -1]  # 最后事件上下文
        return self.out(ctx)


class OpponentEquivariant(nn.Module):
    """④ 他家 3 家置换等变聚合。

    per-opponent 共享编码 + 3 家间自注意力（set attention，天然等变）
    + 平均池化（置换不变全局）。
    """

    def __init__(self, d_model: int = 256, n_head: int = 4):
        super().__init__()
        self.per = nn.Sequential(nn.Linear(d_model, d_model),
                                 nn.ReLU(), nn.Linear(d_model, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, opp_feats: torch.Tensor):
        """opp_feats: (B, 3, C) -> ((B,3,C) 每家等变输出, (B,C) 全局不变池化)。"""
        x = self.per(opp_feats)
        x = self.attn(x, x, x, need_weights=False)[0]
        x = self.norm(x + self.per(opp_feats))
        pooled = x.mean(dim=1)
        return x, self.out(pooled)


class BeliefDangerHead(nn.Module):
    """③ 信念注意力读牌（阶段2 danger head）。

    交叉注意力：全局公开特征(query) × 各家事件序列(key/value)
    -> 他家听牌信念。3 家共享参数（等变）：打乱 3 家事件，输出同置换。
    输出：tenpai_logits (B,3)、danger_logits (B,3,34)。
    """

    def __init__(self, d_model: int = 256, n_head: int = 4,
                 max_events: int = 64, event_dim: int = 46):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(event_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(max_events, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.cross = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.tenpai = nn.Linear(d_model, 1)
        self.danger = nn.Linear(d_model, 34)

    def forward(self, pub_feat: torch.Tensor, opp_events: torch.Tensor):
        """pub_feat (B,C)；opp_events (B,3,T,D) -> (tenpai (B,3), danger (B,3,34))。"""
        B, O, T, _ = opp_events.shape
        q = self.q(pub_feat).unsqueeze(1)  # (B,1,C) 每对手同一 query 源（共享）
        kv = self.kv(opp_events) + self.pos[:T].unsqueeze(0).unsqueeze(0)  # (B,O,T,C)
        kv = kv.reshape(B * O, T, -1)
        q = q.repeat(1, O, 1).reshape(B * O, 1, -1)
        x = self.cross(q, kv, kv, need_weights=False)[0]  # (B*O,1,C)
        x = self.norm(x + q)
        tenpai = self.tenpai(x).squeeze(1).reshape(B, O)          # (B,3)
        danger = self.danger(x).squeeze(1).reshape(B, O, 34)      # (B,3,34)
        return tenpai, danger


# ---------------------------------------------------------------------------
# 真实全局时间线（引擎 events，严格时间序）——RL 决策时使用
# ---------------------------------------------------------------------------

MELD_KIND = {"chi": 0, "chow": 0, "pon": 1, "minkan": 2, "daiminkan": 2,
             "ankan": 3, "kakan": 4}


def build_events_from_game(events, round_idx=0, max_events: int = 64,
                           event_dim: int = 46) -> torch.Tensor:
    """引擎 game.events（严格时间序）→ (max_events, event_dim)。

    只保留决策信号事件：dahai / reach / chi / pon / kan / daiminkan；
    tsumo/start/end/dora/hora/ryuukyoku 不产生 token（摸牌由 dahai 的
    tsumogiri 标记承载）。截断保留最近 max_events 个，补零在前。
    """
    D = event_dim
    rows = []
    r_norm = (round_idx % 8) / 8.0

    def tok(seat, kind, tsug, riichi, meld_k):
        v = torch.zeros(D)
        if 0 <= seat < 4:
            v[seat] = 1.0
        if kind >= 0:
            v[4 + kind] = 1.0
        v[38] = float(tsug)
        v[39] = float(riichi)
        if meld_k >= 0:
            v[40 + meld_k] = 1.0
        v[45] = r_norm
        return v

    for e in events:
        t = e.get("type")
        seat = e.get("seat", e.get("actor", -1))
        if t == "dahai":
            rows.append(tok(seat, (e.get("tile", -1) or -1) // 4,
                            e.get("tsumogiri", False), e.get("riichi", False), -1))
        elif t == "reach":
            rows.append(tok(seat, -1, False, True, -1))
        elif t in MELD_KIND:
            tiles = e.get("tiles") or []
            rows.append(tok(seat, (tiles[0] if tiles else -1) // 4,
                            False, False, MELD_KIND[t]))
    rows = rows[-max_events:]
    while len(rows) < max_events:
        rows.insert(0, torch.zeros(D))
    return torch.stack(rows)  # (T, D)
