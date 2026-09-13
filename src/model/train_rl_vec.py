# -*- coding: utf-8 -*-
"""向量化自对弈 PPO v1.6（单进程多局并行 + batch 推理，充分利用 GPU/CPU）。

背景: v1.5 多进程 batch=1 推理在 GPU 上排队竞争，吞吐不升反降（~330 局/时 < 单进程 438）。
本版: 单进程内并行 --vec 局游戏，每步把多游戏观察合并为大 batch 一次前向
（GPU 算力利用，batch=N 前向远快于 N 次 batch=1），引擎逐步串行推进（CPU），
PPO 用池化更新（--pool-size 局池 + minibatch，复用 train_rl_mp.ppo_update_pool）。

指标/日志/存档格式与 v1.4/v1.5 一致（rl_train.txt / rl_train_metrics.jsonl /
rl_train_games.jsonl）-> 面板直接可用。AMP: 对手决策 fp16、seat0 采样 fp32
（logp 单一数据源不变量，初始 ratio=1）。
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import json
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.policy import RiichiPolicy
from env.riichi_game import RiichiGame, RiichiConfig
from model.attn_modules import build_events_from_game
from model.features import BINARY_HEADS, build_features, discard_mask
from model.rl_reward import RewardPredictor, _round_features
from model.train_rl import OpponentPool
from model.train_rl_mp import ppo_update_pool

SL_CKPT = "checkpoints/sl/transfer/transfer_final.pt"


def _batch_logits(model, obss, events_list, device, amp):
    """多 obs 一次前向（batch）。返回 {head: (N,...) fp32 tensor}。"""
    xs = [build_features(o, "full")[:, :, 0] for o in obss]
    dms = [discard_mask(o) for o in obss]
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
    ev = None
    if (model.use_event_attn and events_list and events_list[0] is not None):
        ev = torch.from_numpy(np.stack(events_list)).float().to(device)
    x = torch.from_numpy(np.stack(xs)).to(device)
    dm = torch.from_numpy(np.stack(dms)).to(device)
    model.eval()
    with torch.no_grad():
        if amp and str(device).startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(x, masks={"discard": dm}, candidates=cands, events=ev)
        else:
            lg = model(x, masks={"discard": dm}, candidates=cands, events=ev)
    return {h: v.float() for h, v in lg.items()}


def _reset_heads_from(policy, heads, src_ckpt):
    """把指定二值头（如 riichi）的权重从 src ckpt 选择性加载（评审裁决 2）。
    用于 reset 失控/死掉的头（|z_med| 极端、exec 塌缩）到干净起点，
    在固定 bias（无归中）的正确动力学下重新学习该决策。"""
    src = torch.load(src_ckpt, map_location="cpu")["model"]
    sd = policy.model.state_dict()
    n = 0
    with torch.no_grad():
        for k, v in src.items():
            if not any(k.startswith("binary." + h + ".") for h in heads):
                continue
            if k in sd and sd[k].shape == v.shape:
                sd[k].copy_(v)
                n += 1
    print("[rl] reset heads %s from %s (%d params)"
          % (heads, src_ckpt, n), flush=True)
    return n


def _gate_monitor(model, trajs, device):
    """② 事件旁支注入监控（F）：σ(gate) 均值 + 注入向量范数。
    每池更新后采样 1 条轨迹前向，记录到 logs/rl_gate_monitor.jsonl。"""
    import json as _json
    _row = None
    try:
        # pool_buf 是轨迹列表（每轨迹=步骤 dict 列表），取首条含 events 的步骤
        for _traj in trajs:
            for t in _traj:
                if t.get("events") is None:
                    continue
                obs = t["obs"]
                x = torch.from_numpy(
                    build_features(obs, "full")[:, :, 0]).unsqueeze(0).to(device)
                dm = torch.from_numpy(discard_mask(obs)).unsqueeze(0).to(device)
                ev_t = torch.from_numpy(
                    np.asarray(t["events"], dtype=np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    h = model.blocks(model.stem(x))
                    pad = ev_t.abs().sum(-1) == 0
                    ctx = torch.tanh(model.event_attn(ev_t, pad))
                    gate = torch.sigmoid(model.ctx_gate)
                    inj = gate.unsqueeze(0).unsqueeze(-1) * ctx.unsqueeze(-1)
                    _row = {
                        "sigma_gate_mean": float(gate.mean().item()),
                        "sigma_gate_std": float(gate.std().item()),
                        "ctx_norm": float(ctx.norm().item()),
                        "inj_norm": float(inj.norm().item()),
                        "inj_max_abs": float(inj.abs().max().item()),
                    }
                break
            if _row:
                break
    except Exception as _e:
        _row = {"err": str(_e)}
    if _row:
        try:
            with open("logs/rl_gate_monitor.jsonl", "a", encoding="utf-8") as _f:
                _f.write(_json.dumps(_row, ensure_ascii=False) + "\n")
        except Exception:
            pass
    return _row


# ---------------------------------------------------------------------------
# 热超参数实时调整（logs/rl_hyper.json）
# 训练每 ~10s / 每池 / 每 epoch 检查该文件，mtime 变化即应用：
#   - 普通标量（clip/entropy_coef/...）在下一池 ppo_update_pool 内直接生效
#   - lr_schedule="fixed" 时用 lr_gate/lr_head/lr_value 固定各分组 lr（跳过 cosine）
#   - reset_heads="riichi,kan" 触发一次从 transfer_final 重置指定头（一次性）
#   - action="pause"/"quit"/"run" 控制暂停/退出/继续（每池边界生效）
# 应用日志：stdout [hyper] 行 + logs/rl_hyper_history.jsonl
_HYPER_STATE = {"mtime": 0.0, "applied": 0}

_HYPER_KEYS = [
    "clip", "clip_grad", "entropy_coef", "vf_coef", "gamma", "lam",
    "kl_stop", "inner_epochs", "batch_size", "pool_size", "temperature",
    "bias_snap", "bias_ema", "win_bonus", "riichi_cost", "base_win_bonus",
    "win_bonus_cap", "base_deal_penalty", "deal_penalty", "deal_penalty_cap",
    "rare_w_cap", "bias_cap",
    "save_every", "eval_every", "eval_games", "t5_every", "t5_games",
    "phi_monitor_every", "t5_every", "zmon", "wr_window",
]
_HYPER_INT = {"inner_epochs", "batch_size", "pool_size", "save_every",
              "eval_every", "eval_games", "t5_every", "t5_games",
              "phi_monitor_every", "bias_snap", "zmon", "wr_window"}


def _write_default_hyper(args, path=None):
    """启动时若 logs/rl_hyper.json 不存在则写入当前默认值模板（用户可直接编辑）。"""
    path = path or getattr(args, "hyper_path", "logs/rl_hyper.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            return
        d = {k: getattr(args, k) for k in _HYPER_KEYS if hasattr(args, k)}
        d.update({
            "lr_schedule": getattr(args, "lr_schedule", "cosine"),
            "lr_gate": 1e-3,
            "lr_head": args.lr / 10.0,
            "lr_value": args.lr,
            "reset_heads": getattr(args, "reset_heads", ""),
            "snap_exclude": list(getattr(args, "snap_exclude", ["tsumo", "ron", "riichi"])),
            "head_entropy": dict(getattr(args, "head_entropy", None) or {}),
            "bias": {},   # 热干预：{头: 值} 直接设置决策头 bias（移出归中后需手动对齐 z）
            "head_z_cap": {},
            "inj_scale": 0.02,
            "always_win": int(getattr(args, "always_win", 1)),
            "wr_mode": getattr(args, "wr_mode", "rank"),
            "wr_window": int(getattr(args, "wr_window", 20)),
            "zmon": int(getattr(args, "zmon", 1)),
            "action": "run",
            "_comment": "热更新：改本文件后训练在下一池/每~10s 应用（logs/rl_hyper_history.jsonl 留痕）。lr_schedule=fixed 时用 lr_gate/lr_head/lr_value 固定 lr（跳过 cosine）；reset_heads=头名逗号列表触发一次从 transfer_final 重置；action=pause/quit/run。",
        })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        print("[hyper] wrote default %s" % path, flush=True)
    except Exception as e:
        print("[hyper] write default err: %s" % e, flush=True)


def _sync_hyper(args, opt, sched, vec, policy, path=None):
    """检查 logs/rl_hyper.json 并热应用。返回 True 若有变更。"""
    path = path or getattr(args, "hyper_path", "logs/rl_hyper.json")
    try:
        if not os.path.exists(path):
            return False
        mt = os.path.getmtime(path)
        if mt == _HYPER_STATE["mtime"]:
            return False
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        _HYPER_STATE["mtime"] = mt
    except Exception as e:
        print("[hyper] read error: %s" % e, flush=True)
        return False
    changed = []
    for k in _HYPER_KEYS:
        if k in d and hasattr(args, k):
            try:
                v = int(d[k]) if k in _HYPER_INT else float(d[k])
            except (TypeError, ValueError):
                continue
            if abs(v - float(getattr(args, k))) > 1e-12:
                setattr(args, k, v)
                changed.append("%s=%s" % (k, v))
    if "wr_mode" in d and d.get("wr_mode") in ("rank", "pt"):
        if d["wr_mode"] != getattr(args, "wr_mode", "rank"):
            args.wr_mode = d["wr_mode"]
            changed.append("wr_mode=%s" % d["wr_mode"])
    # 结构化热干预（决策头防两极分化）
    if "snap_exclude" in d and isinstance(d.get("snap_exclude"), list):
        _se = [str(x) for x in d["snap_exclude"]]
        if _se != list(getattr(args, "snap_exclude", None) or []):
            args.snap_exclude = _se
            changed.append("snap_exclude=%s" % _se)
    if "head_entropy" in d and isinstance(d.get("head_entropy"), dict):
        _he = {str(k): float(v) for k, v in d["head_entropy"].items()
               if isinstance(v, (int, float))}
        if _he != dict(getattr(args, "head_entropy", None) or {}):
            args.head_entropy = _he
            changed.append("head_entropy=%s" % _he)
    if "bias" in d and isinstance(d.get("bias"), dict):
        _pb = getattr(policy, "bias", {})
        _nb = {str(_h): float(_v) for _h, _v in d["bias"].items()
               if _h in _pb and isinstance(_v, (int, float))}
        if _nb and any(abs(_nb[_h] - _pb[_h]) > 1e-9 for _h in _nb):
            for _h, _v in _nb.items():
                _pb[_h] = _v
            changed.append("bias=%s" % {k: round(v, 2) for k, v in _nb.items()})
    if "always_win" in d:
        try:
            _aw = int(d["always_win"])
        except (TypeError, ValueError):
            _aw = None
        if _aw is not None and _aw != int(getattr(policy, "always_win", 0)):
            policy.always_win = bool(_aw)
            args.always_win = _aw
            changed.append("always_win=%d" % _aw)
    if "inj_scale" in d:
        try:
            _is = float(d["inj_scale"])
        except (TypeError, ValueError):
            _is = None
        if _is is not None and _is >= 0 and abs(_is - float(getattr(policy.model, "inj_scale", 0.02))) > 1e-12:
            policy.model.inj_scale = _is
            changed.append("inj_scale=%g" % _is)
    if "head_z_cap" in d and isinstance(d.get("head_z_cap"), dict):
        _zc = {str(k): float(v) for k, v in d["head_z_cap"].items()
               if isinstance(v, (int, float))}
        if _zc != dict(getattr(policy, "z_cap", None) or {}):
            policy.z_cap = _zc
            changed.append("head_z_cap=%s" % _zc)
    sm = d.get("lr_schedule")
    if sm and sm != getattr(args, "lr_schedule", "cosine"):
        setattr(args, "lr_schedule", str(sm))
        changed.append("lr_schedule=%s" % sm)
    if getattr(args, "lr_schedule", "cosine") == "fixed":
        for gi, key in ((0, "lr_gate"), (1, "lr_head"), (2, "lr_value")):
            if key in d:
                try:
                    v = float(d[key])
                except (TypeError, ValueError):
                    continue
                if abs(v - opt.param_groups[gi]["lr"]) > 1e-12:
                    opt.param_groups[gi]["lr"] = v
                    changed.append("%s=%g" % (key, v))
    elif "lr" in d:
        try:
            v = float(d["lr"])
            if abs(v - args.lr) > 1e-12:
                args.lr = v
                opt.param_groups[2]["lr"] = v
                changed.append("lr=%g" % v)
        except (TypeError, ValueError):
            pass
    vec.temperature = float(args.temperature)
    vec.win_bonus = float(args.win_bonus)
    vec.base_win_bonus = float(args.base_win_bonus)
    vec.win_bonus_cap = float(args.win_bonus_cap)
    vec.base_deal_penalty = float(args.base_deal_penalty)
    vec.deal_penalty = float(args.deal_penalty)
    vec.deal_penalty_cap = float(args.deal_penalty_cap)
    vec.riichi_cost = float(args.riichi_cost)
    rh = d.get("reset_heads", "")
    if rh:
        heads = [h.strip() for h in str(rh).split(",") if h.strip()]
        if heads:
            n = _reset_heads_from(policy, heads, SL_CKPT)
            changed.append("reset_heads=%s(%d)" % (heads, n))
            try:
                d.pop("reset_heads", None)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False, indent=1)
            except Exception:
                pass
    act = d.get("action", "run")
    if act != getattr(args, "action", "run"):
        setattr(args, "action", str(act))
        changed.append("action=%s" % act)
    if changed:
        _HYPER_STATE["applied"] += 1
        line = "[hyper] #%d " % _HYPER_STATE["applied"] + " ".join(changed)
        print(line, flush=True)
        try:
            with open("logs/rl_hyper_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "changes": changed}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return True
    return False


class VecRollout:
    """并行 --vec 局游戏：每步 seat0 采样 batch(fp32) + 对手决策 batch(fp16) 一次前向。"""

    def __init__(self, policy, sl, phi, device, vec, seed, temperature,
                 win_bonus=0.5, riichi_cost=1.0, base_win_bonus=0.5,
                 win_bonus_cap=20.0, base_deal_penalty=0.5, deal_penalty=0.5,
                 deal_penalty_cap=20.0):
        self.policy = policy
        self.sl = sl
        self.phi = phi
        self.device = device
        self.vec = vec
        self.seed = seed
        self.temperature = temperature
        # win_bonus：按打点缩放的和牌奖励 alpha（evaluator 终审第3项）
        #   r_win = min(5.0, win_bonus * 打点/1000)——番值梯度 1:8 恢复
        self.win_bonus = win_bonus
        # 奖励结构（C 方案）：基础和牌奖 + 打点缩放(封顶) / 基础放铳惩罚 + 铳点缩放(封顶)
        self.base_win_bonus = base_win_bonus
        self.win_bonus_cap = win_bonus_cap
        self.base_deal_penalty = base_deal_penalty
        self.deal_penalty = deal_penalty
        self.deal_penalty_cap = deal_penalty_cap
        # 立直成本净零 shaping（evaluator 终审第1项·部分）：动作步 -cost、轮末 +cost 返还
        self.riichi_cost = riichi_cost
        self.pools = [OpponentPool(policy, sl, k=8, device=device, seed=seed + 10 + i)
                      for i in range(vec)]
        self.rngs = [random.Random(seed + 100 + i) for i in range(vec)]
        self.envs = [None] * vec
        self.trajs = [None] * vec
        self.prev_scores = [None] * vec
        self.prev_phis = [None] * vec
        self.rd = [None] * vec
        self.cs = [None] * vec
        self.dl = [None] * vec
        self.hb = [None] * vec
        self.st = [None] * vec
        self.ri = [None] * vec
        self.hands_seq = [None] * vec   # 每轮结束 hand 快照（Φ 16 维手牌特征）
        self.meld_seq = [None] * vec
        self.dora_seq = [None] * vec
        self.opps = [None] * vec
        self.opp_kinds = [None] * vec
        self.stats = [None] * vec
        self.games_done = [0] * vec
        self._win_round = [False] * vec
        self._deal_round = [False] * vec
        self._win_wall = [-1] * vec   # 和牌轮次的本轮墙数（hora 时记录，防新轮重置）
        self._wall_before = [0] * vec  # 每次 step 前的墙数（引擎 step 内会摸牌/开新局）
        # Φ 手牌特征：step-before 快照（round_end 后引擎已推进新局，手牌会重置）
        self._hand_before = [None] * vec
        self._meld_before = [0] * vec
        self._dora_before = [None] * vec
        self._riichi_before = [None] * vec   # 本轮立直状态快照（立直成本返还用）
        for g in range(vec):
            self._reset_game(g)

    def register_past(self, path, epoch):
        """历史检查点登记到所有 pool（虚构自对弈对手池，防策略坍缩）。"""
        for p in self.pools:
            p.add_checkpoint(path, epoch)

    def _reset_game(self, g):
        self.envs[g] = RiichiGame(config=RiichiConfig(),
                                  seed=self.seed + g * 100000 + self.games_done[g])
        self.trajs[g] = []
        self.prev_scores[g] = 25000   # 引擎 start_score=25000：首轮 delta 可算
        self.prev_phis[g] = None
        self.rd[g] = []
        self.cs[g] = []
        self.dl[g] = []
        self.hb[g] = []
        self.st[g] = []
        self.ri[g] = []
        self.hands_seq[g] = []   # 每轮结束时的 hand 快照（Φ 16 维手牌特征）
        self.meld_seq[g] = []
        self.dora_seq[g] = []
        self.opps[g], self.opp_kinds[g] = self.pools[g].pick(self.rngs[g])
        self._win_round[g] = False
        self._deal_round[g] = False
        self._win_wall[g] = -1
        self.stats[g] = {"settlement": 0.0, "rank": 0, "final_score": 0,
                         "wins": 0, "deals": 0, "riichi": 0, "melds": 0,
                         "rounds": 0, "duration_ms": 0, "_t0": time.time(),
                         "tsumo_wins": 0, "ron_wins": 0,
                         "tsumo_opp": 0, "tsumo_exec": 0,
                         "ron_opp": 0, "ron_exec": 0,
                         "chow_opp": 0, "chow_exec": 0,
                         "pon_opp": 0, "pon_exec": 0,
                         "kan_opp": 0, "kan_exec": 0,
                         "riichi_opp": 0, "riichi_exec": 0,
                         "kyushu_opp": 0, "kyushu_exec": 0,
                         "chow_opp_mf": 0, "chow_exec_mf": 0,
                         "chow_opp_md": 0, "chow_exec_md": 0,
                         "pon_opp_mf": 0, "pon_exec_mf": 0,
                         "pon_opp_md": 0, "pon_exec_md": 0,
                         "kan_opp_mf": 0, "kan_exec_mf": 0,
                         "kan_opp_md": 0, "kan_exec_md": 0,
                         "ankan_opp": 0, "ankan_exec": 0,
                         "ryuukyoku": 0, "ryuukyoku_tenpai": 0,
                         "riichi_wins": 0, "meld_wins": 0, "silent_wins": 0,
                         "win_pt_sum": 0.0, "win_pt_n": 0,
                         "deal_pt_sum": 0.0, "deal_pt_n": 0,
                         "win_turns_sum": 0, "win_turns_n": 0,
                         "busted": 0}

    def step(self, selfplay_buf=None):
        """推进所有活跃游戏一步。返回 [(g, traj, gstats), ...] 已结束游戏。"""
        from model.rl_reward import settlement_pt as _spt
        finished = []
        actives = [g for g in range(self.vec)
                   if self.envs[g].phase != "game_end"]
        if not actives:
            return finished
        use_ea = self.policy.model.use_event_attn
        obs_all = {}
        ev_all = {}
        for g in actives:
            obs_all[g] = self.envs[g].state.get_observation()
            if use_ea:
                ev_all[g] = build_events_from_game(self.envs[g].events,
                                                   self.envs[g].round_idx)
        actions = {}
        # 模型分组推理（修复：past 历史版本必须用自己的权重决策，不能并进当前策略）：
        #   cur（seat0 采样 + self 对手）= 当前策略 fp32（logp 单一数据源）
        #   sl = SL 基线 fp16（无事件旁支）
        #   past = 每个历史模型各自 batch fp32
        groups = {}   # (kind, model) -> [g...]
        for g in actives:
            if self.envs[g].turn == 0 or self.opp_kinds[g] == "self":
                groups.setdefault(("cur", self.policy.model), []).append(g)
            elif self.opp_kinds[g] == "sl":
                groups.setdefault(("sl", self.sl.model), []).append(g)
            else:   # past
                groups.setdefault(("past", self.opps[g].policy.model), []).append(g)
        for (kind, mdl), g_pol in groups.items():
            lg = _batch_logits(mdl, [obs_all[g] for g in g_pol],
                               [ev_all.get(g) for g in g_pol],
                               self.device, amp=(kind == "sl"))
            for k, g in enumerate(g_pol):
                step_lg = {h: lg[h][k:k + 1] for h in lg}
                if self.envs[g].turn == 0:
                    st0 = self.stats[g]
                    la0 = obs_all[g].get("legal_actions") or {}
                    hand0 = obs_all[g].get("hand") or []
                    mf0 = len((obs_all[g].get("melds") or [[], [], [], []])[0]) == 0
                    for _h in ("tsumo", "ron", "chow", "pon", "kan",
                               "riichi", "kyushu"):
                        if la0.get(_h):
                            st0[_h + "_opp"] += 1
                    # 吃碰杠细分：门清/副露 机会；暗杠（4 张全在手）单独
                    for _h in ("chow", "pon", "kan"):
                        if la0.get(_h):
                            _tiles = la0[_h][0]["tiles"]
                            if _h == "kan" and all(_t in hand0 for _t in _tiles):
                                st0["ankan_opp"] += 1
                            elif mf0:
                                st0[_h + "_opp_mf"] += 1
                            else:
                                st0[_h + "_opp_md"] += 1
                    action, logp, value, ent = self.policy._sample_from_logits(
                        step_lg, obs_all[g], self.temperature)
                    _at = action["type"]
                    if _at in ("chow", "chi"):
                        st0["chow_exec"] += 1
                        st0["chow_exec_" + ("mf" if mf0 else "md")] += 1
                    elif _at == "pon":
                        st0["pon_exec"] += 1
                        st0["pon_exec_" + ("mf" if mf0 else "md")] += 1
                    elif _at == "kan":
                        st0["kan_exec"] += 1
                        if all(_t in hand0 for _t in (action.get("tiles") or [])):
                            st0["ankan_exec"] += 1
                        elif mf0:
                            st0["kan_exec_mf"] += 1
                        else:
                            st0["kan_exec_md"] += 1
                    elif _at in ("tsumo", "ron", "riichi", "kyushu"):
                        st0[_at + "_exec"] += 1
                    t = {"obs": obs_all[g], "action": action, "logp": logp.detach(),
                         "value": value.detach(), "r": 0.0, "ent": ent}
                    if use_ea:
                        t["events"] = ev_all[g].numpy().astype(np.float16)
                    if action["type"] == "riichi":
                        # 立直成本净零 shaping（evaluator 终审）：动作步扣成本，轮末返还
                        t["r"] -= self.riichi_cost
                    self.trajs[g].append(t)
                    if action["type"] == "riichi":
                        self.stats[g]["riichi"] += 1
                    if action["type"] in ("pon", "chi", "chow", "kan"):
                        self.stats[g]["melds"] += 1
                    actions[g] = action
                elif kind == "cur":
                    actions[g] = self.policy._act_from_logits(step_lg, obs_all[g])
                elif kind == "sl":
                    actions[g] = self.sl._act_from_logits(step_lg, obs_all[g])
                elif kind == "past":
                    actions[g] = self.opps[g].policy._act_from_logits(step_lg, obs_all[g])
        # 引擎逐步 + 轮末/终局处理
        for g in actives:
            self._wall_before[g] = self.envs[g].wall_left   # step 前快照（引擎可能开新局）
            # Φ 手牌特征：step-before 快照（round_end 后引擎已推进新局，手牌重置）
            self._hand_before[g] = list(self.envs[g].hands[0])
            self._meld_before[g] = len(self.envs[g].melds[0])
            self._dora_before[g] = list(self.envs[g].dora_indicators)
            self._riichi_before[g] = list(self.envs[g].riichi_declared)
            res = self.envs[g].step(actions[g])
            _drew = False   # 本步是否有摸牌（墙 -1）——修正墙数快照
            for _e in res.get("events", []):
                if _e["type"] == "tsumo":
                    _drew = True
                elif _e["type"] == "hora":
                    if _e.get("actor") == 0:
                        self.stats[g]["wins"] += 1
                        self._win_round[g] = True
                        self._win_wall[g] = self._wall_before[g] - (1 if _drew else 0)
                        if _e.get("target") is None:   # 无 target = 自摸
                            self.stats[g]["tsumo_wins"] += 1
                        else:
                            self.stats[g]["ron_wins"] += 1
                        # 和牌方式：用本步 obs（快照未重置），非 round_end 后
                        _rd0 = bool((obs_all[g].get("riichi_declared") or [0] * 4)[0])
                        _md0 = len((obs_all[g].get("melds") or [[], [], [], []])[0]) > 0
                        if _rd0:
                            self.stats[g]["riichi_wins"] += 1
                        elif _md0:
                            self.stats[g]["meld_wins"] += 1
                        else:
                            self.stats[g]["silent_wins"] += 1
                    if _e.get("target") == 0:
                        self.stats[g]["deals"] += 1
                        self._deal_round[g] = True
            if res.get("round_end"):
                self.stats[g]["rounds"] += 1
                obs0 = self.envs[g].state.get_observation(seat=0)
                score = obs0["scores"][0]
                rr = getattr(self.envs[g], "round_result", None)
                # 流局/流听（罚符）
                if rr is not None and str(rr.get("type", "")).startswith("ryuukyoku"):
                    self.stats[g]["ryuukyoku"] += 1
                    if 0 in (rr.get("tenpai") or []):
                        self.stats[g]["ryuukyoku_tenpai"] += 1
                delta = (score - self.prev_scores[g]) if self.prev_scores[g] is not None else 0.0
                # 和牌分析：打点（轮末分数差）、了巡（墙牌估算）、和牌方式
                if self._win_round[g]:
                    st = self.stats[g]
                    st["win_pt_sum"] += delta
                    st["win_pt_n"] += 1
                    wl = self._win_wall[g]
                    if wl >= 0:
                        st["win_turns_sum"] += max(1, (70 - int(wl)) // 4 + 1)
                        st["win_turns_n"] += 1
                    self._win_wall[g] = -1
                if self._deal_round[g]:
                    st = self.stats[g]
                    st["deal_pt_sum"] += max(-delta, 0)
                    st["deal_pt_n"] += 1
                _won_r = self._win_round[g]   # 本轮是否和牌（奖励分支用）
                _deal_r = self._deal_round[g]  # 本轮是否被铳（奖励分支用）
                self._win_round[g] = False
                self._deal_round[g] = False
                # Φ 16 维：每轮追加 step-before 手牌快照（含杠/宝牌/副露信息）
                self.hands_seq[g].append(self._hand_before[g])
                self.meld_seq[g].append(self._meld_before[g])
                self.dora_seq[g].append(self._dora_before[g])
                if self.phi is not None and self.trajs[g]:
                    self.rd[g].append(delta)
                    self.cs[g].append(score)
                    self.dl[g].append(obs0.get("oya", 0))
                    self.hb[g].append(obs0.get("honba", 0))
                    self.st[g].append(obs0.get("riichi_sticks", 0))
                    self.ri[g].append(obs0.get("round", 0))
                    R = _round_features(self.rd[g], self.cs[g], self.dl[g],
                                        self.hb[g], self.st[g], self.ri[g],
                                        hands=self.hands_seq[g],
                                        meld_counts=self.meld_seq[g],
                                        dora_ind=self.dora_seq[g])
                    with torch.no_grad():
                        phi_k = float(self.phi(R.to(self.device)).item())
                    r = (phi_k - self.prev_phis[g]) if self.prev_phis[g] is not None else 0.0
                    if _won_r:
                        # C 方案：基础和牌奖 + 打点缩放（封顶 20）——高打点充分区分、防失控
                        r += self.base_win_bonus + min(
                            self.win_bonus_cap, self.win_bonus * max(delta, 0.0) / 1000.0)
                    if _deal_r:
                        # C 方案：基础放铳惩罚 + 铳点缩放（封顶 20）——与和牌对称
                        r -= self.base_deal_penalty + min(
                            self.deal_penalty_cap, self.deal_penalty * max(-delta, 0.0) / 1000.0)
                    # 立直成本返还（净零 shaping：动作步已扣 riichi_cost）
                    if self._riichi_before[g] and self._riichi_before[g][0]:
                        r += self.riichi_cost
                    self.prev_phis[g] = phi_k
                    self.trajs[g][-1]["r"] += r
                elif self.prev_scores[g] is not None and self.trajs[g]:
                    self.trajs[g][-1]["r"] += (score - self.prev_scores[g]) / 100000.0
                self.prev_scores[g] = score
            if res.get("game_end"):
                st = self.stats[g]
                st["final_score"] = self.envs[g].scores[0]
                st["busted"] = 1 if st["final_score"] < 0 else 0
                st["rank"] = sorted(range(4), key=lambda i: -self.envs[g].scores[i]).index(0) + 1
                st["settlement"] = _spt(list(self.envs[g].scores), 0, "tenhou")
                st["duration_ms"] = int((time.time() - st["_t0"]) * 1000)
                gstats = dict(st)
                gstats.pop("_t0", None)
                if self.phi is not None and len(self.rd[g]) >= 2:
                    R = _round_features(self.rd[g], self.cs[g], self.dl[g],
                                        self.hb[g], self.st[g], self.ri[g],
                                        hands=self.hands_seq[g],
                                        meld_counts=self.meld_seq[g],
                                        dora_ind=self.dora_seq[g])
                    label = _spt(list(self.envs[g].scores), 0, "tenhou")
                    if selfplay_buf is not None:
                        selfplay_buf.append((R.squeeze(0).detach().cpu(), float(label)))
                gstats["opp"] = self.opp_kinds[g]
                finished.append((g, self.trajs[g], gstats))
                self.games_done[g] += 1
                self._reset_game(g)
        return finished


def _epoch_analysis_log(args, _json):
    """epoch 完成后写完整分析日志（本 epoch 全指标 + 与上 epoch 变化）。
    输出 logs/rl_train_epoch_analysis.jsonl（每 epoch 一行，数据分析对象）。"""
    try:
        from tools._games_block import epoch_analytics as _ea
        _all_g = []
        with open(args.games_path, encoding="utf-8") as _gf:
            for _ln in _gf:
                _ln = _ln.strip()
                if _ln:
                    _all_g.append(_json.loads(_ln))
        _el = _ea(_all_g)
        if not _el:
            return
        _ce, _pe = _el[-1], (_el[-2] if len(_el) >= 2 else None)

        def _d(v1, v0):
            return (v1 - v0) if (v1 is not None and v0 is not None) else None

        _summ = {"epoch": _ce["epoch"], "games": _ce["n"]}
        for _k in ("win_rate", "deal_rate", "ryuukyoku_rate", "bust_rate",
                   "avg_win", "avg_deal", "avg_turns", "avg_rank",
                   "silent_rate", "riichi_rate", "meld_rate",
                   "tsumo_rate", "ron_rate", "tenpai_rate"):
            _summ[_k] = _ce.get(_k)
            _summ["delta_" + _k] = _d(_ce.get(_k), _pe.get(_k) if _pe else None)
        for _h in ("tsumo", "ron", "riichi", "kyushu"):
            _summ[_h + "_exec"] = _ce.get(_h + "_exec", 0)
            _summ[_h + "_opp"] = _ce.get(_h + "_opp", 0)
            _summ["delta_" + _h + "_exec_rate"] = _d(
                _ce.get(_h + "_exec_rate"), _pe.get(_h + "_exec_rate") if _pe else None)
        _summ["meld_exec"] = _ce.get("meld_exec", 0)
        _summ["meld_opp"] = _ce.get("meld_opp", 0)
        _summ["delta_meld_exec_rate"] = _d(
            _ce.get("meld_exec_rate"), _pe.get("meld_exec_rate") if _pe else None)
        for _h, _st in (("chow", "mf"), ("chow", "md"), ("pon", "mf"),
                        ("pon", "md"), ("kan", "mf"), ("kan", "md")):
            _rk = _h + "_exec_" + _st + "_rate"
            _summ[_h + "_exec_" + _st] = _ce.get(_h + "_exec_" + _st, 0)
            _summ[_h + "_opp_" + _st] = _ce.get(_h + "_opp_" + _st, 0)
            _summ["delta_" + _rk] = _d(_ce.get(_rk), _pe.get(_rk) if _pe else None)
        _summ["ankan_exec"] = _ce.get("ankan_exec", 0)
        _summ["ankan_opp"] = _ce.get("ankan_opp", 0)
        _summ["delta_ankan_exec_rate"] = _d(
            _ce.get("ankan_exec_rate"), _pe.get("ankan_exec_rate") if _pe else None)
        os.makedirs("logs", exist_ok=True)
        with open("logs/rl_train_epoch_analysis.jsonl", "a",
                  encoding="utf-8") as _af:
            _af.write(_json.dumps(_summ, ensure_ascii=False) + "\n")
        # 详细日志（实时可读）：关键执行率 + 顺位 + 变化
        def _pct(ex, op):
            return ("%.0f%%" % (100.0 * ex / op)) if op else "N/A"
        print("[ana] ep%d 顺位=%.2f(%s) 和率=%.1f%% 放铳=%.1f%% | 立直=%s(%s) 吃门清=%s 吃副露=%s "
              "碰门清=%s 碰副露=%s 自摸=%s 荣和=%s" % (
                  _summ["epoch"], _summ.get("avg_rank") or 0,
                  ("%+.2f" % _summ.get("delta_avg_rank")) if _summ.get("delta_avg_rank") is not None else "-",
                  _summ.get("win_rate") or 0, _summ.get("deal_rate") or 0,
                  _pct(_summ.get("riichi_exec", 0), _summ.get("riichi_opp", 0)),
                  ("%+.1fpp" % _summ.get("delta_riichi_exec_rate")) if _summ.get("delta_riichi_exec_rate") is not None else "-",
                  _pct(_summ.get("chow_exec_mf", 0), _summ.get("chow_opp_mf", 0)),
                  _pct(_summ.get("chow_exec_md", 0), _summ.get("chow_opp_md", 0)),
                  _pct(_summ.get("pon_exec_mf", 0), _summ.get("pon_opp_mf", 0)),
                  _pct(_summ.get("pon_exec_md", 0), _summ.get("pon_opp_md", 0)),
                  _pct(_summ.get("tsumo_exec", 0), _summ.get("tsumo_opp", 0)),
                  _pct(_summ.get("ron_exec", 0), _summ.get("ron_opp", 0))), flush=True)
    except Exception as _exc:
        print("[rl] epoch-analysis warn:", _exc, flush=True)


def _eval_block(args, epoch, policy, sl_ref, device, t5):
    from tools.eval_vs_sl import play_game as _pg
    m = {}
    with torch.no_grad():
        vs = []
        for _g in range(5):
            envv = RiichiGame(config=RiichiConfig(), seed=args.seed + 9000 + _g)
            for _ in range(400):
                if envv.phase == "game_end":
                    break
                obs = envv.state.get_observation()
                vs.append(policy.value(obs))
                envv.step(policy.act(obs) if envv.turn == 0 else envv.random_action())
        v = np.array(vs)
        m["value_mean"] = float(v.mean())
        m["value_std"] = float(v.std())
    d = [0, 0, 0, 0]
    pts = []
    for _g in range(args.eval_games):
        seat = _g % 4
        rank, _, _spt = _pg(policy, sl_ref, seat, args.seed + epoch * 1000 + _g)
        d[rank] += 1
        pts.append(_spt)
    t = sum(d)
    m["wr_vs_sl"] = float((d[0] + 0.5 * d[1]) / max(t, 1)) if t else 0.0
    # pt 公式（热键 wr_mode=rank|pt，wr_window 窗口局数）：P(我方 settlement_pt > 0)
    if getattr(args, "wr_mode", "rank") == "pt":
        _p = pts[-int(getattr(args, "wr_window", 20) or 20):]
        _n = len(_p)
        # pt 加权胜率（按 pt 加权）：胜面占比 Σmax(spt,0)/(Σmax(spt,0)+Σ|min(spt,0)|)
        _pos = sum(x for x in _p if x > 0)
        _neg = sum(-x for x in _p if x < 0)
        m["wr_pt_vs_sl"] = float(_pos) / max(_pos + _neg, 1e-9)
        m["avg_pt_vs_sl"] = float(sum(_p)) / max(_n, 1)
        # 平均每百局 pt 增量
        m["avg_pt_per100_vs_sl"] = float(sum(_p)) / max(_n, 1) * 100.0
        m["wr_pt_window"] = _n
    if t5 is not None and (epoch + 1) % args.t5_every == 0:
        t5res = t5.run(epoch, seed=args.seed, device=device)
        m["t5_calibrated"] = bool(any(not r.get("kept", True) for r in t5res.values()))
        m["t5_thr"] = {h: (r.get("applied_tau") if not r.get("kept", True)
                           else "kept") for h, r in t5res.items()}
    return m


def _save_ckpt(args, policy, epoch):
    from model.train_rl_mp import _save_ckpt as _mp_save
    return _mp_save(args, policy, epoch)   # 返回 ep_path 或 None（供 past 池登记）


def main():
    ap = argparse.ArgumentParser(description="Vectorized self-play PPO v1.6")
    ap.add_argument("--ckpt", default="checkpoints/sl/rl/value_pretrain.pt")
    ap.add_argument("--phi-ckpt", default="checkpoints/sl/rl/reward_predictor.pt")
    ap.add_argument("--out", default="checkpoints/sl/rl/rl_v1.pt")
    ap.add_argument("--vec", type=int, default=4, help="并行游戏数（向量化 batch）")
    ap.add_argument("--games", type=int, default=10, help="每 epoch 局数")
    ap.add_argument("--epochs", type=int, default=1000, help="epoch 总数")
    ap.add_argument("--pool-size", type=int, default=32, help="轨迹池大小（局）")
    ap.add_argument("--kl-stop", type=float, default=0.5,
                    help="KL 早停阈值（每 inner epoch 近似 KL，超阈值停止；0=关闭）")
    ap.add_argument("--bias-ema", type=float, default=0.05,
                    help="动态 bias EMA 系数（bias-snap=0 时使用；0=关闭）")
    ap.add_argument("--bias-snap", type=int, default=1,
                    help="bias 快照归中模式：1=每池 bias=-median(z)（推荐，估计器无偏）；0=EMA 慢追")
    ap.add_argument("--reset-heads", default="",
                    help="逗号分隔的决策头名（如 riichi），从 SL_CKPT(transfer_final) 选择性重置该头权重")
    ap.add_argument("--win-bonus", type=float, default=0.5,
                    help="和牌奖励 alpha（按打点缩放：r=min(5, alpha·打点/1000)；热调）")
    ap.add_argument("--riichi-cost", type=float, default=1.0,
                    help="立直成本净零 shaping：动作步扣 cost、轮末返还（热调）")
    ap.add_argument("--base-win-bonus", type=float, default=0.5, help="基础和牌奖（任何和牌保底 pt；热调）")
    ap.add_argument("--win-bonus-cap", type=float, default=20.0, help="和牌打点奖励封顶 pt（热调）")
    ap.add_argument("--base-deal-penalty", type=float, default=0.5, help="基础放铳惩罚（任何被铳扣 pt；热调）")
    ap.add_argument("--deal-penalty", type=float, default=0.5, help="放铳铳点缩放系数 pt/千点（热调）")
    ap.add_argument("--deal-penalty-cap", type=float, default=20.0, help="放铳惩罚封顶 pt（热调）")
    ap.add_argument("--rare-w-cap", type=float, default=8.0,
                    help="稀有动作权重上限（ppo_update_pool；热超参数可调）")
    ap.add_argument("--bias-cap", type=float, default=400.0,
                    help="bias 快照钳位上限（热超参数可调）")
    ap.add_argument("--lr-schedule", default="cosine",
                    help="lr 调度：cosine（默认）/ fixed（用 logs/rl_hyper.json 的 lr_gate/lr_head/lr_value 固定）")
    ap.add_argument("--hyper-path", default="logs/rl_hyper.json",
                    help="热超参数文件（每 ~10s/每池检查，编辑即实时应用）")
    ap.add_argument("--snap-exclude", default="tsumo,ron,riichi",
                    help="bias-snap 排除头（逗号分隔；热干预，logs/rl_hyper.json 可改）")
    ap.add_argument("--head-entropy", default="",
                    help="per-head 熵权重 JSON，如 {\"riichi\": 3.0}（防单头两极分化；热干预）")
    ap.add_argument("--zmon", type=int, default=1,
                    help="每池输出各头 z 分布到 logs/rl_head_zmon.jsonl（热干预观察依据）")
    ap.add_argument("--always-win", type=int, default=1,
                    help="阶段训练：强制能和就和（tsumo/ron 头冻结，logp=0）；能力提升后热改 always_win=0 恢复见逃学习")
    ap.add_argument("--wr-mode", default="rank",
                    help="vs-SL 胜率公式：rank（1位+0.5×2位）| pt（P(settlement_pt>0) 最近 wr_window 局）")
    ap.add_argument("--wr-window", type=int, default=20,
                    help="pt 胜率窗口局数（默认 20）")
    ap.add_argument("--inner-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--event-attn", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=30, help="vs-SL 评估间隔（epochs；0=关）")
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--pool-k", type=int, default=8)
    ap.add_argument("--t5-every", type=int, default=100)
    ap.add_argument("--t5-games", type=int, default=40)
    ap.add_argument("--phi-monitor-every", type=int, default=500)
    ap.add_argument("--metrics-path", default="logs/rl_train_metrics.jsonl")
    ap.add_argument("--games-path", default="logs/rl_train_games.jsonl")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.action = "run"   # 运行时热控制（pause/quit/run，经 logs/rl_hyper.json）
    args.snap_exclude = [h.strip() for h in args.snap_exclude.split(",") if h.strip()]
    try:
        args.head_entropy = json.loads(args.head_entropy) if args.head_entropy else {}
    except Exception:
        args.head_entropy = {}

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.smoke:
        args.vec = 1
        args.games = 1
        args.epochs = 1
        args.pool_size = 1
        args.eval_every = 0
        args.t5_every = 0
        args.phi_monitor_every = 0
        args.save_every = 1

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    policy = RiichiPolicy(args.ckpt, seed=args.seed, device=device,
                          use_event_attn=args.event_attn)
    policy.always_win = bool(args.always_win)
    if args.reset_heads:
        _reset_heads_from(policy, [h.strip() for h in args.reset_heads.split(",") if h.strip()],
                          SL_CKPT)
    sl_ref = RiichiPolicy(SL_CKPT, seed=999, device=device)
    phi = None
    if os.path.exists(args.phi_ckpt):
        phi = RewardPredictor().to(device)
        phi.load_state_dict(torch.load(args.phi_ckpt, map_location="cpu")["model"])
        phi.eval()
    # 分层 lr（H/F：头/主干 lr/10 真学习但慢；门控 lr 1e-3 使 ② 旁支可学；value 正常）
    opt = torch.optim.AdamW([
        {"params": [p for n, p in policy.model.named_parameters()
                    if "event_attn" in n or "ctx_gate" in n], "lr": 1e-3},
        {"params": [p for n, p in policy.model.named_parameters()
                    if ("event_attn" not in n and "ctx_gate" not in n
                        and not n.startswith("value."))], "lr": args.lr / 10.0},
        {"params": [p for n, p in policy.model.named_parameters()
                    if n.startswith("value.")], "lr": args.lr},
    ], lr=args.lr)
    # 动态 lr：warmup（~40 池 ≈ 500 更新步）→ cosine 衰减（决策头稳定，防后期分化）
    from torch.optim.lr_scheduler import LambdaLR
    _total_pools = max(1, args.games * args.epochs // max(args.pool_size, 1))
    _warmup_pools = min(40, max(1, _total_pools // 4))

    def _lr_lambda(step):
        if step < _warmup_pools:
            return step / max(_warmup_pools, 1)
        p = (step - _warmup_pools) / max(_total_pools - _warmup_pools, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = LambdaLR(opt, lr_lambda=_lr_lambda)
    phi_mon = None
    t5 = None
    if phi is not None and args.phi_monitor_every > 0:
        from tools.rl_phi_monitor import PhiMonitor
        phi_mon = PhiMonitor(phi, device, eval_every=args.phi_monitor_every)
    if args.t5_every > 0:
        from tools.rl_t5_calibrate import T5Calibrator
        t5 = T5Calibrator(policy, min_pos=200, n_games=args.t5_games)

    import json as _json
    os.makedirs("logs", exist_ok=True)
    _write_default_hyper(args)
    vec = VecRollout(policy, sl_ref, phi, device, args.vec, args.seed,
                     args.temperature, win_bonus=args.win_bonus,
                     riichi_cost=args.riichi_cost,
                     base_win_bonus=args.base_win_bonus,
                     win_bonus_cap=args.win_bonus_cap,
                     base_deal_penalty=args.base_deal_penalty,
                     deal_penalty=args.deal_penalty,
                     deal_penalty_cap=args.deal_penalty_cap)
    _sync_hyper(args, opt, sched, vec, policy)   # 启动即应用 logs/rl_hyper.json 覆盖
    selfplay_buf = []
    pool_buf = []
    games_done = 0
    total = args.games * args.epochs
    t_start = time.time()
    lp_s = lv_s = h_s = r_s = c_s = kl_s = ld_s = lb_s = 0.0
    n_upd = 0
    prev = {"lp": 0.0, "lv": 0.0, "H": 0.0, "ratio": 0.0, "clip": 0.0,
            "kl": 0.0, "lp_disc": 0.0, "lp_bin": 0.0}
    opp_stats = {"self": 0, "past": 0, "sl": 0}
    rew_all = []
    last_epoch_done = -1
    _last_hyper_t = 0.0

    while games_done < total:
        # 热超参数：每 ~10s 轮询 logs/rl_hyper.json（下一池即生效）
        if time.time() - _last_hyper_t > 10.0:
            _last_hyper_t = time.time()
            _sync_hyper(args, opt, sched, vec, policy)
        if getattr(args, "action", "run") == "pause":
            print("[hyper] PAUSED（改 logs/rl_hyper.json 的 action=run 恢复）", flush=True)
            time.sleep(5)
            continue
        if getattr(args, "action", "run") == "quit":
            print("[hyper] QUIT 请求，本池后停止", flush=True)
            break
        finished = vec.step(selfplay_buf)
        for g, traj, gstats in finished:
            opp_stats[gstats.get("opp", "sl")] += 1
            gstats["game"] = games_done % args.games
            gstats["epoch"] = games_done // args.games
            with open(args.games_path, "a", encoding="utf-8") as gf:
                gf.write(_json.dumps(gstats, ensure_ascii=False) + "\n")
            rew_all.extend([t["r"] for t in traj])
            pool_buf.append(traj)
            games_done += 1
            if phi_mon is not None:
                ev = phi_mon.step(games_done, selfplay_buf)
                if ev is not None:
                    print("[rl] phi-monitor:", _json.dumps(ev, ensure_ascii=False),
                          flush=True)
        if len(pool_buf) >= args.pool_size:
            up = ppo_update_pool(policy, opt, pool_buf, args, device)
            if getattr(policy.model, "use_event_attn", False) and up:
                _gm = _gate_monitor(policy.model, pool_buf, device)
                if _gm:
                    print("[gate] %s" % _json.dumps(_gm, ensure_ascii=False),
                          flush=True)
            pool_buf.clear()
            if getattr(args, "lr_schedule", "cosine") != "fixed":
                sched.step()   # 动态 lr 步进（warmup+cosine；lr_schedule=fixed 时跳过）
            if up:
                lp_s += up[0]; lv_s += up[1]; h_s += up[2]; r_s += up[3]
                c_s += up[4]; kl_s += up[5]; ld_s += up[6]; lb_s += up[7]
                n_upd += 1
            if args.smoke:
                assert abs(up[3] - 1.0) < 0.05, "initial ratio != 1: %.4f" % up[3]
                print("SMOKE OK (ratio=%.4f)" % up[3], flush=True)
        # epoch 边界（修复 vec 批处理跳过 10 倍数）：处理所有已完成的 epoch
        while (games_done > 0
               and (last_epoch_done + 2) * args.games <= games_done
               and last_epoch_done + 1 < args.epochs):
            epoch = last_epoch_done + 1
            last_epoch_done = epoch
            n_p = max(n_upd - prev.get("_n", 0), 1)
            _rw = np.array(rew_all) if rew_all else np.zeros(1)
            m = {"epoch": epoch, "games_done": games_done,
                 "opp_self": opp_stats["self"], "opp_past": opp_stats["past"],
                 "opp_sl": opp_stats["sl"], "opp_random": 0,
                 "lp": (lp_s - prev["lp"]) / n_p,
                 "lv": (lv_s - prev["lv"]) / n_p,
                 "H": (h_s - prev["H"]) / n_p,
                 "ratio": (r_s - prev["ratio"]) / n_p,
                 "clip_rate": (c_s - prev["clip"]) / n_p,
                 "kl": (kl_s - prev["kl"]) / n_p,
                 "lp_disc": (ld_s - prev["lp_disc"]) / n_p,
                 "lp_bin": (lb_s - prev["lp_bin"]) / n_p,
                 "reward_mean": float(_rw.mean()),
                 "reward_std": float(_rw.std()),
                 "throughput_gph": round(games_done / max(time.time() - t_start, 1) * 3600, 1),
                 "elapsed_s": round(time.time() - t_start, 1)}
            rew_all = []
            if policy.model.ctx_gate is not None:
                g_ = torch.sigmoid(policy.model.ctx_gate).detach().cpu()
                m["gate_mean"] = float(g_.mean().item())
                m["gate_std"] = float(g_.std().item())
            else:
                m["gate_mean"] = None
                m["gate_std"] = None
            if (args.eval_every > 0
                    and ((epoch + 1) % args.eval_every == 0
                         or epoch == args.epochs - 1)):
                m.update(_eval_block(args, epoch, policy, sl_ref, device, t5))
            with open(args.metrics_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(m, ensure_ascii=False) + "\n")
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()   # 防 PyTorch 显存缓存膨胀/碎片拖慢前向
            _epoch_analysis_log(args, _json)   # 新增：epoch 完整分析日志（全指标+变化）
            _wr_s = (" wr=%.3f" % m["wr_vs_sl"]) if "wr_vs_sl" in m else ""
            print("[rl] epoch %d game %d/%d avg_p=%.4f avg_v=%.4f H=%.3f "
                  "ratio=%.4f clip=%.3f%s %.0fs"
                  % (epoch, args.games, args.games,
                     (lp_s - prev["lp"]) / n_p, (lv_s - prev["lv"]) / n_p,
                     (h_s - prev["H"]) / n_p, (r_s - prev["ratio"]) / n_p,
                     (c_s - prev["clip"]) / n_p, _wr_s,
                     time.time() - t_start), flush=True)
            prev.update({"lp": lp_s, "lv": lv_s, "H": h_s, "ratio": r_s,
                         "clip": c_s, "kl": kl_s, "lp_disc": ld_s,
                         "lp_bin": lb_s, "_n": n_upd})
            _ep = _save_ckpt(args, policy, epoch)
            if _ep:
                vec.register_past(_ep, epoch)   # 历史版本入池（虚构自对弈）
    if pool_buf:
        if len(pool_buf) >= min(args.pool_size, max(total // 4, 1)):
            up = ppo_update_pool(policy, opt, pool_buf, args, device)
            pool_buf.clear()
            if up:
                print("SMOKE OK (ratio=%.4f)" % up[3], flush=True) if args.smoke else None
        else:
            pool_buf.clear()
    print("RL_V1 DONE -> %s (%d games, %.0fs)"
          % (args.out, games_done, time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
