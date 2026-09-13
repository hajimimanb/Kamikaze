# -*- coding: utf-8 -*-
"""自对弈 PPO 训练器 v1.4（10局/epoch × 1000 epoch，评估每10 epoch×2局vs SL）。

v1.4（用户决策 2026-09）:
- 局数/epoch: 10 x 20000 -> 10 x 1000（共 10,000 局，~33h @300局/时）
- 评估: --eval-every 10 epoch（=每 100 局）一次 vs-SL 评估，--eval-games 2 局
  （座位轮换 seat=g%4），替代原 16 局/100epoch；value 统计同节奏
- 存档: --save-every 100 epoch（=每 1000 局）历史 checkpoint，环形保留 5 个入对手池

v1.3（用户决策 2026-09）:
- 局数/epoch: 20x500 -> 10x20000（共 200,000 局；epoch=10 局训练+1 次指标行）
- 解耦: 细粒度 epoch 下 vs-SL 评估(16局)/value 统计/T5 校准按 --eval-every(默认 100
  epoch=1000 局) 节奏执行，否则评估开销(16局)超过训练开销(10局)
- 存档: 最新 rl_v1.pt 每 epoch 覆盖；历史 rl_v1_ep*.pt 按 --save-every 存档，
  环形保留 --pool-k 个（防 20,000 个文件爆盘），并入历史对手池

v1.2（用户决策 2026-09）:
- 去掉训练期随机对手课程（随机仅 smoke/eval sanity；文献: AlphaGo-Lee 打历史版本、
  AlphaGo Zero 纯自对弈、Suphx/OpenAI Five 历史版本池）
- 对手池: 当前策略(主) / 历史 epoch checkpoint / SL 基线(保留) —— 自对弈主体
- epoch 拆分: 5×2000 -> 20×500（每局即更新，epoch=检查点/评估粒度，细化 4 倍）

v1.1 修复（docs/rl_stage1_review.md §2.1-2.3）:
- critic 梯度：重放时重算 value（带梯度）；GAE 用 detach 值，critic loss 用带梯度值
- 熵张量化：重放算归一化混合熵（discard/ln34 + binary/ln2），作为张量进 loss
- logp 单一数据源：采样与重放共用 policy._logp_for_action（初始 ratio≈1）
- 事件接线：rollout 用 build_events_from_game（引擎真实时间线 20/20），轨迹存 events 重放
- 阈值头 RL 期 Bernoulli 采样（policy.sample_with_logp 已改）
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # 使 import tools.* 可用

from agent.policy import RiichiPolicy
from model.attn_modules import build_events_from_game
from model.features import BINARY_HEADS, build_features, discard_mask
from model.rl_losses import gae, ppo_policy_loss, value_loss
from model.rl_reward import RewardPredictor, _round_features, settlement_pt
from env.riichi_game import RiichiGame, RiichiConfig


_sl_ref = None  # vs-SL 自对战基线（main 里初始化）


class _SLOpponent:
    """SL 基线对手（对手池低权重成分）——与验收目标同分布。"""

    def __init__(self, sl):
        self.sl = sl

    def decide(self, obs, events=None):
        return self.sl.act(obs, events=events)


class SelfPlayOpponent:
    """自对弈对手：同一网络权重的化身（act=阈值+argmax，与部署一致）。
    全 4 座同一 θ：seat0=PPO 采样训练，他家=同权重贪心——『自己和自己打』。"""

    def __init__(self, policy):
        self.policy = policy

    def decide(self, obs, events=None):
        return self.policy.act(obs, events=events)


class OpponentPool:
    """自对弈对手池（v1.2，替代 T3 随机课程；文献: AlphaGo-Lee 随机历史版本 /
    AlphaGo Zero 当前版本自对弈 / OpenAI Five past-version / Suphx 历史版本）:
    - self : 当前策略本身（主成分，纯自对弈，主学习信号）
    - past : 最近 K 个 epoch 检查点（虚构自对弈，防策略坍缩/循环）
    - sl   : SL 基线 transfer_final（低权重保留，防遗忘 + 与验收目标同分布）
    随机对手不再用于训练（仅 smoke / eval sanity 参考线 0.375）。"""

    # 全局共享的历史模型缓存（vec=8 个 pool 共用同一份，防 GPU OOM：
    # 每 pool 独立缓存 8×N 个 122MB 模型会爆显存）。LRU 上限 _PAST_CACHE_MAX。
    _PAST_CACHE = {}
    _PAST_CACHE_ORDER = []
    _PAST_CACHE_MAX = 3

    def __init__(self, policy, sl, k=5, p_self=0.60, p_past=0.25, p_sl=0.15,
                 device="cpu", seed=0):
        self.policy = policy
        self.sl = sl
        self.k = max(k, 1)
        self.p_self = p_self
        self.p_past = p_past
        self.p_sl = p_sl
        self.device = device
        self.rng = np.random.default_rng(seed)
        self._past = []      # [(epoch, path)]

    def add_checkpoint(self, path, epoch):
        """每 epoch 末登记历史检查点（供 'past' 对手抽样）。"""
        self._past.append((epoch, path))
        self._past = self._past[-self.k:]

    def _load(self, path):
        pol = OpponentPool._PAST_CACHE.get(path)
        if pol is None:
            if len(OpponentPool._PAST_CACHE) >= OpponentPool._PAST_CACHE_MAX:
                old = OpponentPool._PAST_CACHE_ORDER.pop(0)
                OpponentPool._PAST_CACHE.pop(old, None)
            pol = RiichiPolicy(
                path, device=self.device, seed=0,
                use_event_attn=self.policy.model.use_event_attn)
            OpponentPool._PAST_CACHE[path] = pol
            OpponentPool._PAST_CACHE_ORDER.append(path)
        else:
            # LRU 触碰：移到队尾
            try:
                OpponentPool._PAST_CACHE_ORDER.remove(path)
                OpponentPool._PAST_CACHE_ORDER.append(path)
            except ValueError:
                pass
        return pol

    def pick(self, rng):
        """返回 (opponent, kind)；kind ∈ {self, past, sl}。"""
        r = rng.random()
        if r < self.p_self:
            return SelfPlayOpponent(self.policy), "self"
        if r < self.p_self + self.p_past and self._past:
            _ep, path = self._past[self.rng.integers(len(self._past))]
            return SelfPlayOpponent(self._load(path)), "past"
        return _SLOpponent(self.sl), "sl"


def _prune_ckpts(ckpt_dir, keep_paths):
    """删除 keep_paths 之外的 rl_v1_ep*.pt（环形保留，防细粒度 epoch 爆盘）。"""
    import glob as _glob
    for p in _glob.glob(os.path.join(ckpt_dir, "rl_v1_ep*.pt")):
        if os.path.abspath(p) not in {os.path.abspath(x) for x in keep_paths}:
            try:
                os.remove(p)
            except OSError:
                pass


class RandomOpponent:
    """随机合法动作（v1 对手，仅 smoke / eval sanity）。"""

    def decide(self, obs, events=None):
        la = obs.get("legal_actions") or {}
        if la.get("tsumo"):
            return {"type": "tsumo"}
        if la.get("ron"):
            return {"type": "ron"}
        for h in ("kan", "pon", "chow"):
            if la.get(h):
                return {"type": h, "tiles": la[h][0]["tiles"]}
        if la.get("riichi"):
            return {"type": "riichi", "tile": la["riichi"][0]}
        if la.get("discard"):
            return {"type": "discard", "tile": random.choice(la["discard"])}
        return {"type": "pass"}


def rollout_game(env, policy, opponent, temperature, phi=None, device="cpu",
               selfplay_buf=None):
    """一局自对弈：本方（seat0）策略采样（真实时间线事件），他家用 opponent。
    奖励: Φ 差分（judge No-Go 阻塞 1 修复）——轮末 r = Phi(x_1..k) - Phi(x_1..k-1)；
    无 phi 时回退轮差/100000（smoke 用）。轨迹存 obs/action/logp/value/r/ent + events。
    selfplay_buf（可选 list）：局末追加 (round_features (K,16), settlement_pt label)
    供 Φ 漂移重训（arch §7.6-2，自对弈 ground truth）。"""
    traj = []
    prev_score = None
    use_ea = policy.model.use_event_attn
    # Φ 轮特征缓冲（judge: 尺度比值 1.95 已验证方案可行）
    rd = []; cs = []; dl = []; hb = []; st = []; ri = []
    # Φ 16 维手牌特征：每轮结束的 step-before 快照（round_end 后引擎已推进新局）
    hands_seq = []; meld_seq = []; dora_seq = []
    _t0 = time.time()
    _stats = {"settlement": 0.0, "rank": 0, "final_score": 0, "wins": 0,
              "deals": 0, "riichi": 0, "melds": 0, "rounds": 0, "duration_ms": 0}
    prev_phi = None
    if phi is not None:
        prev_score = None
    while True:
        obs = env.state.get_observation()
        events = None
        if use_ea:
            events = build_events_from_game(env.events, env.round_idx)  # (64,46)
        if env.turn == 0:
            action, logp, value, ent = policy.sample_with_logp(obs, temperature, events)
            t = {"obs": obs, "action": action, "logp": logp.detach(),
                 "value": value.detach(), "r": 0.0, "ent": ent}
            if use_ea:
                t["events"] = events.numpy().astype(np.float16)
            traj.append(t)
            if action["type"] == "riichi":
                _stats["riichi"] += 1
            if action["type"] in ("pon", "chi", "kan"):
                _stats["melds"] += 1
        else:
            action = opponent.decide(obs, events)
        hand_before = list(env.hands[0])          # step-before 快照（防新局手牌污染）
        meld_before = len(env.melds[0])
        dora_before = list(env.dora_indicators)
        res = env.step(action)
        for _e in res.get("events", []):
            if _e["type"] == "hora":
                if _e.get("actor") == 0:
                    _stats["wins"] += 1
                if _e.get("target") == 0:
                    _stats["deals"] += 1
        if res.get("round_end") or res.get("game_end"):
            _stats["rounds"] += 1
            obs0 = env.state.get_observation(seat=0)
            score = obs0["scores"][0]
            if phi is not None and traj:
                # 构造本轮 16 维特征 → Φ 差分奖励
                delta = (score - prev_score) if prev_score is not None else 0.0
                rd.append(delta); cs.append(score)
                dl.append(obs0.get("oya", 0)); hb.append(obs0.get("honba", 0))
                st.append(obs0.get("riichi_sticks", 0)); ri.append(obs0.get("round", 0))
                hands_seq.append(hand_before)
                meld_seq.append(meld_before)
                dora_seq.append(dora_before)
                R = _round_features(rd, cs, dl, hb, st, ri,
                                    hands=hands_seq, meld_counts=meld_seq,
                                    dora_ind=dora_seq)  # (1,K,16)
                with torch.no_grad():
                    phi_k = float(phi(R.to(device)).item())
                r = (phi_k - prev_phi) if prev_phi is not None else 0.0
                prev_phi = phi_k
                traj[-1]["r"] += r
            elif prev_score is not None and traj:
                traj[-1]["r"] += (score - prev_score) / 100000.0
            prev_score = score
        if res.get("game_end"):
            if selfplay_buf is not None and phi is not None and len(rd) >= 2:
                R = _round_features(rd, cs, dl, hb, st, ri,
                                    hands=hands_seq, meld_counts=meld_seq,
                                    dora_ind=dora_seq)  # (1,K,16)
                label = settlement_pt(list(env.scores), 0, "tenhou")
                selfplay_buf.append((R.squeeze(0).detach().cpu(), float(label)))
            break
    from model.rl_reward import settlement_pt as _spt
    _stats["final_score"] = env.scores[0]
    _stats["rank"] = sorted(range(4), key=lambda i: -env.scores[i]).index(0) + 1
    _stats["settlement"] = _spt(list(env.scores), 0, "tenhou")
    _stats["duration_ms"] = int((time.time() - _t0) * 1000)
    return traj, _stats


def _batch_features(traj, device):
    """轨迹 obs -> (x, dmask, cands, events tensor)。events 可能为 None（无旁支）。"""
    xs, dms = [], []
    evs = [] if any("events" in t for t in traj) else None
    for t in traj:
        obs = t["obs"]
        xs.append(build_features(obs, "full")[:, :, 0])
        dms.append(discard_mask(obs))
        if evs is not None:
            evs.append(t["events"])
    x = torch.from_numpy(np.stack(xs)).to(device)
    dm = torch.from_numpy(np.stack(dms)).to(device)
    cands = {}
    for h in BINARY_HEADS:
        rows = []
        for t in traj:
            la = t["obs"].get("legal_actions") or {}
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
    events_t = None
    if evs is not None:
        events_t = torch.from_numpy(np.stack(evs).astype(np.float32)).to(device)
    return x, dm, cands, events_t


def _normalized_entropy(logits, dm, bin_avail, T, head_w=None):
    """归一化混合熵 H_bar（M0-T2 修正：合法行掩码 + 分母=合法行数）：
    H_bar = (H_disc/ln34 + sum_b w_b·H_b/ln2) / (1+sum_b w_b), b in {riichi,chow,pon,kan}
    - head_w（热干预）：{头: 权重}，对易两极分化的头加大熵权重（防单头饱和塌缩）
    - 每头只在合法行计算熵；分母=合法行数（防补零/非法行稀释，auditor Q6）
    - 非法行全部时不贡献（按实际合法头数加权平均）"""
    eps = 1e-8
    dev = logits["discard"].device
    H = torch.zeros(1, device=dev)
    n_heads = 0
    hw = head_w or {}
    # discard 熵（合法牌种）
    z = logits["discard"].masked_fill(~dm.bool(), 0.0)
    p = torch.softmax(z / max(T, 1e-3), dim=-1) * dm.float()
    denom = p.sum(-1, keepdim=True).clamp(eps)
    p = p / denom
    valid = (denom.squeeze(-1) > eps).float()
    if valid.sum() > 0:
        H = H + (-(p * torch.log(p + eps)).sum(-1) * valid).sum() / valid.sum() / math.log(34.0)
        n_heads += 1
    # binary 头熵（合法行掩码）
    for h in ("riichi", "chow", "pon", "kan"):
        av = bin_avail[h].float()
        if av.sum() <= 0:
            continue
        pb = torch.softmax(logits[h] / max(T, 1e-3), dim=-1)
        hb = -(pb * torch.log(pb + eps)).sum(-1) * av
        w_h = float(hw.get(h, 1.0)) if hw else 1.0
        H = H + w_h * hb.sum() / av.sum() / math.log(2.0)
        n_heads += w_h
    return H / max(n_heads, 1)


def ppo_update(policy, opt, traj, args, device):
    """一局（或池）PPO 更新。重放前向 → 新 logp（共享函数）/新 value（带梯度）/熵（张量）。"""
    if not traj:
        return None
    x, dm, cands, events_t = _batch_features(traj, device)
    for t in traj:
        t["action"]["_la"] = t["obs"].get("legal_actions") or {}
        t["action"]["_disc"] = (t["obs"].get("legal_actions") or {}).get("discard") or []
    policy.model.train()
    opt.zero_grad()
    # fp32 前向（logp 单一数据源：采样 batch=1 与重放 batch=N 在 fp32 下位级一致）
    lg = policy.model(x, masks={"discard": dm}, candidates=cands,
                      events=events_t)
    # 新 logp（与采样共用单一数据源 → 初始 ratio≈1）
    new_logps = []
    for i, t in enumerate(traj):
        step_lg = {h: lg[h][i:i + 1] for h in lg}
        new_logps.append(policy._logp_for_action(step_lg, t["obs"], t["action"], args.temperature))
    new_logp = torch.stack(new_logps)
    old_logp = torch.stack([t["logp"].reshape(1) for t in traj]).detach()
    # critic：重放 value 带梯度（评审 2.1 修复）
    new_value = lg["value"]                       # (N,) 有梯度
    with torch.no_grad():
        v_detach = new_value.detach()
    rewards = torch.tensor([t["r"] for t in traj], dtype=torch.float32, device=device)
    dones = torch.zeros(len(traj), dtype=torch.bool, device=device)
    adv, returns = gae(rewards, v_detach, dones, args.gamma, args.lam)
    # advantage 归一化（评审 2.4）
    adv = (adv - adv.mean()) / (adv.std().clamp(min=1e-6))
    # M0-T2: binary 头可用性（熵掩码分母）与稀有动作 loss 加权
    from collections import Counter
    bin_avail = {h: torch.zeros(len(traj), dtype=torch.bool, device=device)
                 for h in ("riichi", "chow", "pon", "kan")}
    types = [t["action"]["type"] for t in traj]
    cnt = Counter(types)
    w = torch.ones(len(traj), device=device)
    for h in ("riichi", "chow", "pon", "kan"):
        for i, t in enumerate(traj):
            la = t["obs"].get("legal_actions") or {}
            bin_avail[h][i] = bool(la.get(h))
        n = cnt.get(h, 0)
        if n > 0:
            w_h = min(max(1.0 / max(n / max(len(traj), 1), 0.02), 1.0), 8.0)
            for i, tt in enumerate(types):
                if tt == h:
                    w[i] = w_h
    lp_per = ppo_policy_loss(old_logp, new_logp, adv.unsqueeze(1), args.clip)
    loss_p = (lp_per * w.unsqueeze(1)).mean()
    # KL(pi_new||pi_old) ~ 0.5*mean((d logp)^2)（更新幅度诊断）
    dlogp = (new_logp - old_logp).detach()
    kl = float((0.5 * dlogp.square()).mean().item())
    # head 分项 loss（discard vs binary，T2 加权效果监控）
    with torch.no_grad():
        is_disc = torch.tensor([t["action"]["type"] == "discard" for t in traj],
                               dtype=torch.float32, device=device)
        lp_disc = float((lp_per * w.unsqueeze(1) * is_disc.unsqueeze(1)).sum()
                        / max(is_disc.sum(), 1))
        lp_bin = float((lp_per * w.unsqueeze(1) * (1 - is_disc).unsqueeze(1)).sum()
                       / max((1 - is_disc).sum(), 1))
    loss_v = value_loss(new_value, returns)       # critic 有梯度 ✓
    H = _normalized_entropy(lg, dm, bin_avail, args.temperature)
    loss = loss_p + args.vf_coef * loss_v - args.entropy_coef * H
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.model.parameters(), args.clip_grad)
    opt.step()
    policy.model.eval()
    with torch.no_grad():
        ratio = torch.exp(new_logp - old_logp)
        clip_rate = float(((ratio < 1 - args.clip) | (ratio > 1 + args.clip)).float().mean().item())
    return (float(loss_p.item()), float(loss_v.item()), float(H.item()),
            float(ratio.mean().item()), clip_rate, kl, lp_disc, lp_bin)


def main():
    ap = argparse.ArgumentParser(description="Self-play PPO v1.1 (7-head calibrated + event-attn)")
    ap.add_argument("--ckpt", default="checkpoints/sl/rl/value_pretrain.pt")
    ap.add_argument("--phi-ckpt", default="checkpoints/sl/rl/reward_predictor.pt",
                    help="GRU 全局奖励预测器（Φ 差分奖励；judge No-Go 阻塞 1 修复）")
    ap.add_argument("--out", default="checkpoints/sl/rl/rl_v1.pt")
    ap.add_argument("--games", type=int, default=10,
                    help="每 epoch 局数（用户指定: 10 局/epoch）")
    ap.add_argument("--epochs", type=int, default=1000,
                    help="epoch 总数（用户指定: 1000 → 共 10,000 局）")
    ap.add_argument("--eval-every", type=int, default=10,
                    help="vs-SL 评估+value 统计间隔（epochs；用户指定 10 = 每 100 局）")
    ap.add_argument("--eval-games", type=int, default=2,
                    help="每次 vs-SL 评估局数（用户指定 2 局；座位轮换 seat=g%%4）")
    ap.add_argument("--save-every", type=int, default=100,
                    help="历史检查点存档间隔（epochs；环形保留 --pool-k 个并入对手池）")
    ap.add_argument("--pool-k", type=int, default=5,
                    help="历史对手池容量（保留最近 K 个存档）")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--event-attn", action="store_true", help="启用事件因果注意力旁支")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--phi-monitor-every", type=int, default=500,
                    help="Phi drift monitor interval (games); 0=off (P0-B)")
    ap.add_argument("--t5-every", type=int, default=100,
                    help="T5 threshold recalibrate period (epochs; 10局/epoch 时 100=每1000局); 0=off (P0-C)")
    ap.add_argument("--t5-games", type=int, default=40,
                    help="T5 calibrate selfplay sample games")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    policy = RiichiPolicy(args.ckpt, seed=args.seed, device=device,
                          use_event_attn=args.event_attn)
    if args.event_attn:
        print("EVENT_ATTN 旁支启用（missing 12 参数随机初始化，门控逐通道开启）")
    global _sl_ref
    _sl_ref = RiichiPolicy("checkpoints/sl/transfer/transfer_final.pt", seed=999, device=device)

    phi = None
    if os.path.exists(args.phi_ckpt):
        phi = RewardPredictor().to(device)
        phi.load_state_dict(torch.load(args.phi_ckpt, map_location="cpu")["model"])
        phi.eval()
        print("PHI loaded:", args.phi_ckpt, "（Φ 差分奖励，judge No-Go 阻塞 1 修复）")
    # P1 分层 lr（compliance）：event_attn/ctx_gate 3e-4、trunk/heads 3e-5、value 3e-4
    opt = torch.optim.AdamW([
        {"params": [p for n, p in policy.model.named_parameters()
                    if "event_attn" in n or "ctx_gate" in n], "lr": args.lr},
        {"params": [p for n, p in policy.model.named_parameters()
                    if ("event_attn" not in n and "ctx_gate" not in n
                        and not n.startswith("value."))], "lr": args.lr / 10.0},
        {"params": [p for n, p in policy.model.named_parameters()
                    if n.startswith("value.")], "lr": args.lr},
    ], lr=args.lr)
    # P0-B Φ 漂移监控器 / P0-C T5 阈值再校准器（compliance）
    phi_mon = None
    t5 = None
    if phi is not None and args.phi_monitor_every > 0:
        from tools.rl_phi_monitor import PhiMonitor
        phi_mon = PhiMonitor(phi, device, eval_every=args.phi_monitor_every)
        print("PHI MONITOR armed: every %d games, drift>%.1fx -> retrain"
              % (args.phi_monitor_every, phi_mon.drift_ratio))
    if args.t5_every > 0:
        from tools.rl_t5_calibrate import T5Calibrator
        t5 = T5Calibrator(policy, min_pos=200, n_games=args.t5_games)
        print("T5 CALIBRATOR armed: every %d epochs, %d sample games"
              % (args.t5_every, args.t5_games))

    if args.smoke:
        env = RiichiGame(config=RiichiConfig(), seed=args.seed)
        traj, _gs = rollout_game(env, policy, RandomOpponent(), 1.0, phi, device)
        print("SMOKE: 1 game steps=%d" % len(traj))
        up = ppo_update(policy, opt, traj, args, device)
        print("ppo update (lp, lv, H, ratio):", up)
        assert up is not None and all(math.isfinite(x) for x in up if x is not None)
        # 评审/验收：初始 ratio 必须 ≈1（采样与重放 logp 单一数据源）
        assert abs(up[3] - 1.0) < 0.05, "initial ratio != 1: %.4f" % up[3]
        print("SMOKE OK (ratio=%.4f)" % up[3])
        return

    _rng = random.Random(args.seed + 7)
    t_start = time.time()
    metrics_path = "logs/rl_train_metrics.jsonl"
    os.makedirs("logs", exist_ok=True)
    import json as _json
    selfplay_buf = []   # Φ 漂移重训缓冲（最近局轮特征+终局精算）
    opp_pool = OpponentPool(policy, _sl_ref, k=args.pool_k, device=device, seed=args.seed + 7)
    for epoch in range(args.epochs):
        lp_sum = lv_sum = 0.0
        h_sum = r_sum = c_sum = kl_sum = ld_sum = lb_sum = 0.0
        rew_all = []
        opp_stats = {"self": 0, "past": 0, "sl": 0}
        for gi in range(args.games):
            opp, opp_kind = opp_pool.pick(_rng)
            opp_stats[opp_kind] += 1
            env = RiichiGame(config=RiichiConfig(), seed=args.seed + gi)
            traj, gstats = rollout_game(env, policy, opp, args.temperature, phi, device,
                                selfplay_buf=selfplay_buf)
            rew_all.extend([t["r"] for t in traj])
            gstats["game"] = gi
            gstats["epoch"] = epoch
            gstats["opp"] = opp_kind  # 对手类型（random/sl）供面板分组统计
            with open("logs/rl_train_games.jsonl", "a", encoding="utf-8") as _gf:
                _gf.write(_json.dumps(gstats, ensure_ascii=False) + "\n")
            if phi_mon is not None:
                ev = phi_mon.step((epoch * args.games + gi + 1), selfplay_buf)
                if ev is not None:
                    print("[rl] phi-monitor:", _json.dumps(ev, ensure_ascii=False), flush=True)
            up = ppo_update(policy, opt, traj, args, device)
            if up:
                lp_sum += up[0]; lv_sum += up[1]; h_sum += up[2]
                r_sum += up[3]; c_sum += up[4]; kl_sum += up[5]
                ld_sum += up[6]; lb_sum += up[7]
            if gi % 10 == 0:  # evaluator P0③: 进度粒度 5-10 局
                print("[rl] epoch %d game %d/%d avg_p=%.4f avg_v=%.4f H=%.3f ratio=%.4f clip=%.3f %.0fs"
                      % (epoch, gi, args.games, lp_sum / max(gi + 1, 1),
                         lv_sum / max(gi + 1, 1), h_sum / max(gi + 1, 1),
                         r_sum / max(gi + 1, 1), c_sum / max(gi + 1, 1),
                         time.time() - t_start), flush=True)
        n = max(args.games, 1)
        import numpy as _np
        _rw = _np.array(rew_all) if rew_all else _np.zeros(1)
        m = {"epoch": epoch, "games_done": (epoch + 1) * args.games,
             "opp_self": opp_stats["self"], "opp_past": opp_stats["past"],
             "opp_sl": opp_stats["sl"], "opp_random": 0,  # 兼容旧面板字段
             "lp": lp_sum / n, "lv": lv_sum / n, "H": h_sum / n,
             "ratio": r_sum / n, "clip_rate": c_sum / n,
             "kl": kl_sum / n, "lp_disc": ld_sum / n, "lp_bin": lb_sum / n,
             "reward_mean": float(_rw.mean()), "reward_std": float(_rw.std()),
             "throughput_gph": round(args.games / max(time.time() - t_start, 1) * 3600, 1),
             "elapsed_s": round(time.time() - t_start, 1)}
        # gate 统计（事件注意力启用时）
        if policy.model.ctx_gate is not None:
            g = torch.sigmoid(policy.model.ctx_gate).detach().cpu()
            m["gate_mean"] = float(g.mean().item()); m["gate_std"] = float(g.std().item())
        else:
            m["gate_mean"] = None; m["gate_std"] = None
        # —— 细粒度 epoch（10局）下，评估/校准/存档按 eval_every 节奏执行 ——
        #    （vs-SL 16 局评估开销 > 10 局训练开销；每 epoch 全做不现实）
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            # value 输出统计（尺度监控）
            with torch.no_grad():
                vs = []
                for _g in range(5):
                    envv = RiichiGame(config=RiichiConfig(), seed=args.seed + 9000 + _g)
                    for _ in range(400):
                        if envv.phase == "game_end": break
                        obs = envv.state.get_observation()
                        vs.append(policy.value(obs))
                        envv.step(policy.act(obs) if envv.turn == 0 else envv.random_action())
                v = np.array(vs)
                m["value_mean"] = float(v.mean()); m["value_std"] = float(v.std())
            # vs-SL 自对战（--eval-games 局，默认 2；座位轮换 seat=g%4，轻量趋势）
            from tools.eval_vs_sl import play_game as _pg
            d = [0, 0, 0, 0]
            for _g in range(args.eval_games):
                seat = _g % 4
                rank, _ = _pg(policy, _sl_ref, seat, args.seed + epoch * 1000 + _g)
                d[rank] += 1
            t = sum(d)
            m["wr_vs_sl"] = float((d[0] + 0.5 * d[1]) / max(t, 1)) if t else 0.0
            # P0-C T5 阈值再校准（细粒度 epoch 下按 t5_every 节奏，默认每 1000 局）
            if t5 is not None and (epoch + 1) % args.t5_every == 0:
                t5res = t5.run(epoch, seed=args.seed, device=device)
                m["t5_calibrated"] = bool(any(not r.get("kept", True)
                                              for r in t5res.values()))
                m["t5_thr"] = {h: (r.get("applied_tau") if not r.get("kept", True)
                                   else "kept") for h, r in t5res.items()}
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(m, ensure_ascii=False) + "\n")
        _wr_s = (" wr=%.3f" % m["wr_vs_sl"]) if "wr_vs_sl" in m else ""
        print("[rl] epoch %d done: %d games avg_p=%.4f avg_v=%.4f H=%.3f ratio=%.4f clip=%.3f%s %.0fs"
              % (epoch, args.games, lp_sum / n, lv_sum / n, h_sum / n, r_sum / n,
                 c_sum / n, _wr_s, time.time() - t_start), flush=True)
        # 检查点: 最新版每 epoch 覆盖保存；历史版按 save_every 存档 + 环形保留
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        ck = {"model": policy.model.state_dict(), "config": vars(args),
              "epoch": epoch, "heads": BINARY_HEADS,
              "thresholds": policy.thr, "temps": policy.temp}
        torch.save(ck, args.out)
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ep_path = os.path.join(os.path.dirname(args.out), "rl_v1_ep%05d.pt" % epoch)
            torch.save(ck, ep_path)
            opp_pool.add_checkpoint(ep_path, epoch)  # 历史版本入池（虚构自对弈）
            _prune_ckpts(os.path.dirname(args.out),
                         [p for _, p in opp_pool._past])
            print("[rl] ckpt saved: %s (past pool=%d)" % (ep_path, len(opp_pool._past)),
                  flush=True)
    print("RL_V1 DONE -> %s (%.0fs)" % (args.out, time.time() - t_start))


if __name__ == "__main__":
    main()