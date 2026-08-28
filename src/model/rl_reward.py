# -*- coding: utf-8 -*-
"""全局奖励预测器（Suphx §3.2 / training_plan：2 层 GRU(512) + 2 FC）。

Φ 以"本局到第 k 轮为止的信息"预测终局名次奖励 R；每轮奖励 = Φ(x_1..k) − Φ(x_1..k−1)。
训练数据：人类对局日志（终局奖励 MSE，Suphx Eq.4）；自对弈时直接用 Φ 提供密集奖励。
"""
from __future__ import annotations

import torch
import torch.nn as nn

ROUND_FEAT_DIM = 16  # 记分板 12 + 手牌特征 4（向听/听牌/宝牌/副露）——Φ 升级：让奖励能评估手牌价值

# 终局奖励：按最终顺位（1..4）查表（名次点数，非打点）。
# 注意 3位→4位 的不对称惩罚 —— 模型应学会"避4"（Suphx 低 4 位率即来源于此）。
# 终局奖励表：统一与 SETTLEMENT_CONFIG 同源（评审 2.4——Φ 训练目标与 RL 终局奖励
# 必须同源，均用 settlement_pt；旧 rank-only 表已修正对齐）。
TERMINAL_REWARD = {
    "tenhou_hanchan": (20, 10, -10, -20), # 天凤四麻半庄（= SETTLEMENT_CONFIG["tenhou"].uma）
    "tenhou_east": (15, 5, -5, -15),
    "majsoul_4p": (15, 5, -5, -15),       # 雀魂（= SETTLEMENT_CONFIG["majsoul"]）
    "mleague": (50, 10, -10, -30),        # M.League（= SETTLEMENT_CONFIG["mleague"]）
}


def rank_from_scores(scores, agent_seat: int = 0) -> int:
    """按终局分数定名次（1-4）：分数降序，同分按座位号升序（日麻惯例）。"""
    order = sorted(range(len(scores)), key=lambda s: (-scores[s], s))
    return order.index(agent_seat) + 1


def score_settlement(scores, agent_seat: int = 0, riichi_sticks: int = 0,
                     return_base: int = 30000, oka: bool = True,
                     uma=(0.0, 0.0, 0.0, 0.0), kilo_scale: float = 1.0) -> float:
    """实际点数结算（25000点持ち・30000点返し）：
       kilo_scale × [ (S − 30000)/1000 + オカ(1位+20) + 供托(1位+棒数) ] + ウマ。

    实际点数对终局精算 pt 的作用（修正认识）：
    - 天凤/雀魂在线段位点 = 仅名次；但名次本身由终局实际点数决定（同点才按起家座次）
      —— 实际点数通过"决定顺位"间接决定 pt；
    - M.League/竞技精算 = 顺位点 + 点差加算（30000返し）：kilo_scale 即"千点→pt"折算
      （常见 1 或 3，以目标赛事官方规则为准）。
    """
    rank = rank_from_scores(scores, agent_seat)
    k = (scores[agent_seat] - return_base) / 1000.0
    if rank == 1:
        if oka:
            k += 20.0  # オカ：4×(30000−25000)/1000
        k += float(riichi_sticks)  # 供托：每根立直棒 1000 点 = 1 千点
    return kilo_scale * k + float(uma[rank - 1])


# 真实精算规则（2026-08-25 网页抓取核实：萌娘百科精算点数 / 天凤规则书 / 游民星空雀魂段位点）
# 精算点数 = (终局点数 − 基点)/1000 + 马点；一位 = −(其他三位之和)（自动含オカ与供托）
SETTLEMENT_CONFIG = {
    "tenhou":  {"base": 30000, "uma": (20.0, 10.0, -10.0, -20.0), "round": True},   # 30000返し・ウマ10-20・千位四舍五入
    "majsoul": {"base": 25000, "uma": (15.0, 5.0, -5.0, -15.0), "round": False},   # 25000基点・ウマ5-15（段位点公式）
    "mleague": {"base": 30000, "uma": (50.0, 10.0, -10.0, -30.0), "round": False}, # 順位点50/10/-10/-30 + 持ち点差
}


def settlement_pt(scores, agent_seat: int = 0, config: str = "tenhou",
                  riichi_sticks: int = 0) -> float:
    """终局精算点数（真实规则，单位 pt）：
       二~四位: (S − base)/1000 + uma[rank]
       一位  : −(其他三位精算之和) + 供托（千点）
    天然同时包含"名次收益(马点)"与"实际点数收益(千点差)"；避4由 uma 的不对称性体现。"""
    rank = rank_from_scores(scores, agent_seat)
    cfg = SETTLEMENT_CONFIG[config]
    if rank == 1:
        s = -sum((scores[o] - cfg["base"]) / 1000.0
                 + cfg["uma"][rank_from_scores(scores, o) - 1]
                 for o in range(len(scores)) if o != agent_seat)
        s += float(riichi_sticks)  # 供托：每根 1000 点归一位
    else:
        s = (scores[agent_seat] - cfg["base"]) / 1000.0 + cfg["uma"][rank - 1]
    return float(round(s)) if cfg["round"] else float(s)


def final_reward(scores, agent_seat: int = 0, config: str = "tenhou",
                 riichi_sticks: int = 0) -> float:
    """RL 终局奖励 = 精算点数（settlement_pt）。"""
    return settlement_pt(scores, agent_seat, config, riichi_sticks)


def _round_features(round_scores, current_scores, dealer, honba,
                    riichi_sticks, round_idx, hands=None, meld_counts=None,
                    dora_ind=None):
    """从一局的对局轨迹构造每轮特征向量 (K, ROUND_FEAT_DIM=16)。

    记分板 12 维（原） + 手牌特征 4 维（向听/听牌/宝牌/副露数）——
    Φ 升级：让奖励能评估"手牌价值"（见逃/杠决策的判断依据）。

    round_scores : (K,)  每轮结束时的本方得分（可负）
    current_scores : (K,)  每轮结束后本方累计分（0~100k 归一）
    dealer / honba / riichi_sticks / round_idx : (K,)  见原
    hands : (K,) 每轮结束本方 hand（tile136 list）；None=用中性占位
    meld_counts : (K,) 每轮本方副露数
    dora_ind : (K,) 每轮宝牌指示牌（tile136 list）
    """
    from riichi.tiles import next_dora_kind
    feats = []
    for i in range(len(round_scores)):
        f = [
            round_scores[i] / 40000.0,                     # 本轮得分（归一）
            current_scores[i] / 100000.0,                  # 累计分（归一）
            dealer[i] / 3.0,                               # 庄位
            honba[i] / 4.0,                                # 本场
            riichi_sticks[i] / 4.0,                        # 立直棒
            (round_idx[i] % 4) / 3.0,                      # 场风位
            (round_idx[i] // 4) / 3.0,                     # 半庄段
            current_scores[i] / 100000.0 - 0.25,           # 相对均分 25000
            min(max(round_scores[i], -16000), 48000) / 48000.0,  # 得分夹取
            1.0 if round_scores[i] > 0 else 0.0,           # 是否和牌
            riichi_sticks[i] > 0 and 1.0 or 0.0,           # 是否有立直
            round_scores[i] < 0 and 1.0 or 0.0,            # 是否被罚/放铳
        ]
        # 手牌特征 4 维
        if hands is not None and i < len(hands) and hands[i] is not None:
            from model.shanten_feat import shanten_of, counts34
            h = list(hands[i])
            sh = shanten_of(counts34(h))
            f.append(sh / 3.0)                             # 向听数（归一）
            f.append(1.0 if sh <= 0 else 0.0)              # 是否听牌
            dora_n = 0.0
            if dora_ind is not None and i < len(dora_ind) and dora_ind[i]:
                for d in dora_ind[i]:
                    dk = next_dora_kind(d // 4)
                    dora_n += h.count(dk * 4) + h.count(dk * 4 + 1)                               + h.count(dk * 4 + 2) + h.count(dk * 4 + 3)
            f.append(min(dora_n, 8.0) / 8.0)              # 宝牌数（归一）
            mc = (meld_counts[i] if meld_counts and i < len(meld_counts) else 0)
            f.append(min(mc, 4.0) / 4.0)                   # 副露数（归一）
        else:
            f += [0.5, 0.0, 0.0, 0.0]                      # 无手牌数据：中性占位
        feats.append(f)
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # (1, K, 16)


class RewardPredictor(nn.Module):
    """Φ：GRU(2层, hidden) -> 2 FC -> 终局奖励标量（对序列末位输出）。"""

    def __init__(self, feat_dim: int = ROUND_FEAT_DIM, hidden: int = 512):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq):  # seq: (B, K, F)
        out, _ = self.gru(seq)          # (B, K, H)
        last = out[:, -1, :]            # 取序列末位（第 k 轮）
        return self.fc(last).squeeze(-1)  # (B,)


def predict_trajectory(model, round_feats, gamma: float = 0.99):
    """对 K 轮特征预测每轮"已见前缀"的终局奖励 Φ(x_1..k)，返回 (K,)。
    自对弈中每轮奖励 = Φ(k) − Φ(k−1)（见 rl_losses.per_round_reward）。"""
    K = round_feats.shape[1]
    preds = []
    for k in range(1, K + 1):
        preds.append(model(round_feats[:, :k, :]).detach())
    return torch.stack(preds).squeeze(-1)  # (K,)
