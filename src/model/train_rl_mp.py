# -*- coding: utf-8 -*-
"""多进程 actor-learner 自对弈 PPO v1.5（加速版）。

架构（Windows spawn；引擎不可改，多实例并行即可）:
- learner(main): 收 worker 轨迹 -> 池化 PPO（--pool-size 局，minibatch
  --inner-epochs 遍历，GAE 按局边界）-> 更新权重 -> 存 sync.pt（worker 权重同步）/
  rl_v1.pt / rl_v1_ep%05d.pt（历史对手池）+ past_ckpts.txt
  -> 周期 vs-SL 评估 / value 统计 / T5 校准 / Φ 漂移监控 / metrics
- workers(--workers 个): 各持 policy+Φ 在 GPU，CPU 并行跑整局（复用 rollout_game），
  轨迹 pickle 落盘 logs/traj_buf/，gstats(含 Φ 重训缓冲) 走队列
- 权重同步: learner 每池存 sync.pt（state_dict），worker 每局前查 mtime 重载
  （权重滞后 <=1 池，标准 actor-learner）
- 历史对手池同步: learner 存档后写 past_ckpts.txt，worker 增量登记（懒加载）

日志/指标格式与单进程 v1.4 完全一致（rl_train.txt / rl_train_metrics.jsonl /
rl_train_games.jsonl）-> 面板无需改动。AMP: 推理+训练 autocast fp16（CUDA）。
"""
from __future__ import annotations

import argparse
import glob
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, "C:/agentwork")

from agent.policy import RiichiPolicy
from env.riichi_game import RiichiGame, RiichiConfig
from model.features import BINARY_HEADS, build_features, discard_mask
from model.rl_losses import gae, ppo_policy_loss, value_loss
from model.rl_reward import RewardPredictor
from model.train_rl import (OpponentPool, RandomOpponent, _normalized_entropy,
                            _prune_ckpts, rollout_game)

TRAJ_DIR = "logs/traj_buf"
SYNC_PATH = "logs/traj_buf/sync.pt"
PAST_FILE = "logs/traj_buf/past_ckpts.txt"
SL_CKPT = "checkpoints/sl/transfer/transfer_final.pt"


# --------------------------------------------------------------------------
# 池化 PPO（多局拼接 + 局边界 GAE + minibatch）
# --------------------------------------------------------------------------

def ppo_update_pool(policy, opt, pool, args, device):
    """一池（多局）PPO 更新。返回 (lp, lv, H, ratio, clip_rate, kl, lp_disc, lp_bin)。"""
    if not pool:
        return None
    xs, dms, obss, acts, logps_old, rs, evs = [], [], [], [], [], [], []
    lens = []
    for traj in pool:
        lens.append(len(traj))
        for t in traj:
            obss.append(t["obs"]); acts.append(t["action"])
            logps_old.append(t["logp"]); rs.append(t["r"])
            xs.append(build_features(t["obs"], "full")[:, :, 0])
            dms.append(discard_mask(t["obs"]))
            if "events" in t:
                evs.append(t["events"])
    evs = evs or None
    x = torch.from_numpy(np.stack(xs)).to(device)
    dm = torch.from_numpy(np.stack(dms)).to(device)
    cands = {}
    for h in BINARY_HEADS:
        rows = []
        for obs in obss:
            la = obs.get("legal_actions") or {}
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

    for i, obs in enumerate(obss):
        acts[i]["_la"] = obs.get("legal_actions") or {}
        acts[i]["_disc"] = (obs.get("legal_actions") or {}).get("discard") or []

    # 阶段1: no_grad 全量前向（仅 value 用于 GAE；不保留激活 -> 内存安全）
    policy.model.eval()
    with torch.no_grad():
        lg0 = policy.model(x, masks={"discard": dm}, candidates=cands,
                           events=events_t)
        lg0 = {h: v.float() for h, v in lg0.items()}
    v0 = lg0["value"]
    old_logp = torch.stack([lp.reshape(1) for lp in logps_old]).detach().to(device)
    rewards = torch.tensor(rs, dtype=torch.float32, device=device)
    dones = torch.zeros(len(rs), dtype=torch.bool, device=device)
    pos = 0
    for L in lens:
        pos += L
        dones[pos - 1] = True
    adv, returns = gae(rewards, v0, dones, args.gamma, args.lam)
    adv = (adv - adv.mean()) / adv.std().clamp(min=1e-6)

    from collections import Counter
    bin_avail = {h: torch.zeros(len(rs), dtype=torch.bool, device=device)
                 for h in ("riichi", "chow", "pon", "kan")}
    types = [a["type"] for a in acts]
    cnt = Counter(types)
    w = torch.ones(len(rs), device=device)
    for h in ("riichi", "chow", "pon", "kan"):
        for i, obs in enumerate(obss):
            la = obs.get("legal_actions") or {}
            bin_avail[h][i] = bool(la.get(h))
        n = cnt.get(h, 0)
        _rcap = float(getattr(args, "rare_w_cap", 8.0) or 8.0)  # 热超参数
        if n > 0:
            w_h = min(max(1.0 / max(n / max(len(rs), 1), 0.02), 1.0), _rcap)
            for i, tt in enumerate(types):
                if tt == h:
                    w[i] = w_h

    # 阶段2: per-minibatch 独立带梯度前向（峰值内存 = 1 个 minibatch，防 CUDA OOM）
    N = len(rs)
    idx = torch.randperm(N, device=device)
    mb = max(1, min(args.batch_size, N))
    n_mb = max(1, math.ceil(N / mb))
    lp_s = lv_s = h_s = 0.0
    policy.model.train()
    _kl_stop = float(getattr(args, "kl_stop", 0) or 0)
    for _ep in range(max(1, args.inner_epochs)):
        for b in range(n_mb):
            sl = idx[b * mb:(b + 1) * mb]
            opt.zero_grad()
            lg_mb = policy.model(
                x[sl], masks={"discard": dm[sl]},
                candidates={h: cands[h][sl] for h in cands},
                events=events_t[sl] if events_t is not None else None)
            lg_mb = {h: v.float() for h, v in lg_mb.items()}
            sl_l = sl.tolist()
            new_logps_mb = []
            for j, i in enumerate(sl_l):
                step_lg = {h: lg_mb[h][j:j + 1] for h in lg_mb}
                new_logps_mb.append(policy._logp_for_action(
                    step_lg, obss[i], acts[i], args.temperature))
            new_logp_mb = torch.stack(new_logps_mb)
            loss_p = (ppo_policy_loss(old_logp[sl], new_logp_mb,
                                      adv[sl].unsqueeze(1), args.clip)
                      * w[sl]).mean()
            loss_v = value_loss(lg_mb["value"], returns[sl])
            H = _normalized_entropy(lg_mb, dm[sl],
                                    {h: bin_avail[h][sl] for h in bin_avail},
                                    args.temperature,
                                    head_w=getattr(args, "head_entropy", None) or None)
            loss = loss_p + args.vf_coef * loss_v - args.entropy_coef * H
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(),
                                           args.clip_grad)
            opt.step()
            lp_s += float(loss_p.item())
            lv_s += float(loss_v.item())
            h_s += float(H.item())
        # KL 早停（稳定性）：更新后近似 KL = 0.5*mean((new_logp-old_logp)^2)
        if _kl_stop > 0:
            with torch.no_grad():
                _kl_ep = float((0.5 * (new_logp_mb - old_logp[sl]).square()).mean().item())
            if _kl_ep > _kl_stop:
                print("[rl] KL early-stop: %.3f > %.3f (inner epoch %d)"
                      % (_kl_ep, _kl_stop, _ep), flush=True)
                break
    policy.model.eval()
    # 诊断: no_grad 全量重算 new_logp（更新后）-> ratio/clip/kl/lp_disc/lp_bin
    with torch.no_grad():
        lg_f = policy.model(x, masks={"discard": dm}, candidates=cands,
                            events=events_t)
        lg_f = {h: v.float() for h, v in lg_f.items()}
        new_logps_f = []
        for i, obs in enumerate(obss):
            step_lg = {h: lg_f[h][i:i + 1] for h in lg_f}
            new_logps_f.append(policy._logp_for_action(step_lg, obs, acts[i],
                                                       args.temperature))
        new_logp_f = torch.stack(new_logps_f)
        ratio = torch.exp(new_logp_f - old_logp)
        clip_rate = float(((ratio < 1 - args.clip) | (ratio > 1 + args.clip))
                          .float().mean().item())
        kl = float((0.5 * (new_logp_f - old_logp).square()).mean().item())
        lp_f = ppo_policy_loss(old_logp, new_logp_f, adv.unsqueeze(1), args.clip)
        is_disc = torch.tensor([a["type"] == "discard" for a in acts],
                               dtype=torch.float32, device=device)
        lp_disc = float((lp_f * w * is_disc).sum() / max(is_disc.sum(), 1))
        lp_bin = float((lp_f * w * (1 - is_disc)).sum()
                       / max((1 - is_disc).sum(), 1))
    # 动态 bias（EMA 跟踪 logits 中位数，防决策头饱和/漂移极端化）：
    # 训练中模型 logits 分布漂移（立直 -57/碰 +41），固定 bias 失效反而加剧饱和。
    # 每池重放完成后更新（不影响本次 logp 一致性；下次采样用新 bias）。
    # A: bias 快照归中（evaluator 方案 A）——每池 bias = -median(z)（合法行），
    # 池内冻结（采样/重放/评估三处共享 policy.bias 快照，估计器无偏），
    # 替代慢速 EMA（α=0.05 追不上漂移，chow 塌缩/pon 通胀根因）。
    _bias_snap = bool(getattr(args, "bias_snap", 1) or 0)
    # 热干预：bias-snap 排除名单（默认 tsumo/ron/riichi——"应饱和"的决策头
    # 归中会封顶执行率或追 logit 失控；可经 logs/rl_hyper.json 热调）
    _snap_excl = list(getattr(args, "snap_exclude", None) or ["tsumo", "ron", "riichi"])
    if hasattr(policy, "bias") and lg_f:
        with torch.no_grad():
            for _h in list(policy.bias.keys()):
                if _h in _snap_excl:
                    continue
                if _h not in lg_f or lg_f[_h].shape[0] == 0:
                    continue
                _mask = torch.tensor(
                    [bool((obss[i].get("legal_actions") or {}).get(_h))
                     for i in range(len(obss))],
                    dtype=torch.bool, device=lg_f[_h].device)
                if not bool(_mask.any().item()):
                    continue
                _z = (lg_f[_h][_mask, 1] - lg_f[_h][_mask, 0]).detach().cpu()
                if _z.numel() >= 5:
                    _b_old = policy.bias[_h]
                    _med = float(_z.median().item())
                    if _bias_snap:
                        _new = -_med
                    else:
                        _alpha = float(getattr(args, "bias_ema", 0.05) or 0)
                        _new = (1 - _alpha) * _b_old + _alpha * (-_med)
                    _bcap = float(getattr(args, "bias_cap", 400.0) or 400.0)  # 热超参数
                    _new = max(-_bcap, min(_bcap, _new))   # 上限 250→400（kan z_med −253 已顶格，放宽防死）
                    if abs(_new - _b_old) > 1e-3:
                        print("[rl] bias %s %s: %.2f -> %.2f (z_med=%.1f, n=%d)"
                              % (_h, "snap" if _bias_snap else "ema", _b_old,
                                 _new, _med, _z.numel()), flush=True)
                    policy.bias[_h] = _new
        # 自动防 runaway（riichi 事件重演保护）：归中头 |z_med|>300 连续 2 池
        # → 自动加入排除名单（bias 固定），并提示可重置该头。
        _ex_hist = getattr(args, "_exclude_hist", None)
        if _ex_hist is None:
            _ex_hist = args._exclude_hist = {}
        for _h in list(policy.bias.keys()):
            if _h in _snap_excl or _h not in lg_f:
                continue
            _m2 = torch.tensor(
                [bool((obss[i].get("legal_actions") or {}).get(_h))
                 for i in range(len(obss))],
                dtype=torch.bool, device=lg_f[_h].device)
            if not bool(_m2.any().item()):
                continue
            _z2 = (lg_f[_h][_m2, 1] - lg_f[_h][_m2, 0]).detach().cpu()
            if _z2.numel() < 5:
                continue
            _med2 = float(_z2.median().item())
            if abs(_med2) > 300.0:
                _ex_hist[_h] = _ex_hist.get(_h, 0) + 1
                if _ex_hist[_h] >= 2 and _h not in _snap_excl:
                    _snap_excl.append(_h)
                    args.snap_exclude = list(_snap_excl)
                    print("[auto] 头 %s |z_med|=%.0f>300 连续 2 池 → 已自动移出 bias-snap"
                          "（bias 固定）；如仍失控可 logs/rl_hyper.json reset_heads=%s"
                          % (_h, _med2, _h), flush=True)
            else:
                _ex_hist[_h] = 0
        # [zmon] 每池输出各头 z 分布（含排除头），写 logs/rl_head_zmon.jsonl
        _zrow = {}
        for _h in list(policy.bias.keys()):
            if _h not in lg_f or lg_f[_h].shape[0] == 0:
                continue
            _m3 = torch.tensor(
                [bool((obss[i].get("legal_actions") or {}).get(_h))
                 for i in range(len(obss))],
                dtype=torch.bool, device=lg_f[_h].device)
            if not bool(_m3.any().item()):
                continue
            _z3 = (lg_f[_h][_m3, 1] - lg_f[_h][_m3, 0]).detach().cpu()
            if _z3.numel() >= 5:
                _q = torch.quantile(_z3, torch.tensor([0.1, 0.5, 0.9])).tolist()
                _zrow[_h] = {"p10": round(_q[0], 1), "med": round(_q[1], 1),
                             "p90": round(_q[2], 1), "n": int(_z3.numel()),
                             "bias": round(policy.bias.get(_h, 0.0), 1),
                             "in_snap": _h not in _snap_excl}
        if _zrow and getattr(args, "zmon", 1):
            import json as _json
            print("[zmon] " + _json.dumps(_zrow, ensure_ascii=False), flush=True)
            try:
                with open("logs/rl_head_zmon.jsonl", "a", encoding="utf-8") as _f:
                    _f.write(_json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"),
                                          "z": _zrow}, ensure_ascii=False) + "\n")
            except Exception:
                pass
    n_upd = max(1, max(1, args.inner_epochs) * n_mb)
    return (lp_s / n_upd, lv_s / n_upd, h_s / n_upd,
            float(ratio.mean().item()), clip_rate, kl, lp_disc, lp_bin)


# --------------------------------------------------------------------------
# worker：并行 rollout
# --------------------------------------------------------------------------

def _worker_main(worker_id, args, queue, stop_evt, ready_evt):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke
                          else "cpu")
    policy = RiichiPolicy(args.ckpt, seed=args.seed + worker_id, device=device,
                          use_event_attn=args.event_attn)
    sl = RiichiPolicy(SL_CKPT, seed=999 + worker_id, device=device)
    phi = None
    if os.path.exists(args.phi_ckpt):
        phi = RewardPredictor().to(device)
        phi.load_state_dict(torch.load(args.phi_ckpt, map_location="cpu")["model"])
        phi.eval()
    pool = OpponentPool(policy, sl, k=args.pool_k, device=device,
                        seed=args.seed + worker_id)
    rng = random.Random(args.seed + 7 + worker_id)
    _sync_mtime = 0.0
    _past_mtime = 0.0
    games_done = 0
    ready_evt.set()
    while not stop_evt.is_set():
        # 权重同步（learner 每池更新一次）
        if os.path.exists(SYNC_PATH):
            m = os.path.getmtime(SYNC_PATH)
            if m != _sync_mtime:
                sd = torch.load(SYNC_PATH, map_location="cpu")
                policy.model.load_state_dict(sd, strict=False)
                policy.model.eval()
                _sync_mtime = m
        # 历史对手池同步（learner 存档后写 past_ckpts.txt）
        if os.path.exists(PAST_FILE):
            m = os.path.getmtime(PAST_FILE)
            if m != _past_mtime:
                try:
                    for line in open(PAST_FILE, encoding="utf-8"):
                        line = line.strip()
                        if line and line not in [p for _, p in pool._past]:
                            ep = int(os.path.basename(line)
                                     .split("_ep")[1].split(".")[0])
                            pool.add_checkpoint(line, ep)
                except Exception:
                    pass
                _past_mtime = m
        env = RiichiGame(config=RiichiConfig(),
                         seed=args.seed + worker_id * 100000 + games_done)
        opp, opp_kind = pool.pick(rng)
        sbuf = []
        traj, gstats = rollout_game(env, policy, opp, args.temperature, phi,
                                    device, selfplay_buf=sbuf)
        gstats["opp"] = opp_kind
        gstats["worker"] = worker_id
        gstats["sbuf"] = [(R.numpy(), float(label)) for R, label in sbuf]
        # 轨迹送 learner：tensor 转 cpu，pickle 落盘（避免队列大对象/跨设备）
        for t in traj:
            t["logp"] = t["logp"].cpu()
            t["value"] = t["value"].cpu()
        os.makedirs(TRAJ_DIR, exist_ok=True)
        pkl = os.path.join(TRAJ_DIR, "g_%02d_%06d.pkl" % (worker_id, games_done))
        with open(pkl, "wb") as f:
            pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)
        _item = ("game", worker_id, games_done, pkl, gstats)
        while True:  # put 带超时：队列满时也能响应 stop（防死锁）
            try:
                queue.put(_item, timeout=1.0)
                break
            except Exception:
                if stop_evt.is_set():
                    break
        games_done += 1
    # 收尾：清理本 worker 未消费轨迹
    for f in glob.glob(os.path.join(TRAJ_DIR, "g_%02d_*.pkl" % worker_id)):
        try:
            os.remove(f)
        except OSError:
            pass


# --------------------------------------------------------------------------
# learner：池化 PPO + 评估 + 存档 + 指标
# --------------------------------------------------------------------------

def _save_sync(policy):
    os.makedirs(TRAJ_DIR, exist_ok=True)
    torch.save(policy.model.state_dict(), SYNC_PATH)


def _save_ckpt(args, policy, epoch):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ck = {"model": policy.model.state_dict(), "config": vars(args),
          "epoch": epoch, "heads": BINARY_HEADS,
          "thresholds": policy.thr, "temps": policy.temp,
          "bias": dict(getattr(policy, "bias", {}) or {})}
    torch.save(ck, args.out)
    if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
        ep_path = os.path.join(os.path.dirname(args.out),
                               "rl_v1_ep%05d.pt" % epoch)
        torch.save(ck, ep_path)
        with open(PAST_FILE, "a", encoding="utf-8") as f:
            f.write(ep_path + "\n")
        # 环形：past_ckpts.txt 与磁盘 ckpt 都只留最近 pool_k 个
        lines = [ln.strip() for ln in open(PAST_FILE, encoding="utf-8")
                 if ln.strip()][-args.pool_k:]
        with open(PAST_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        _prune_ckpts(os.path.dirname(args.out), lines)
        print("[rl] ckpt saved: %s (past=%d)" % (ep_path, len(lines)),
              flush=True)
        return ep_path
    return None


def _eval_block(args, epoch, policy, sl_ref, device, t5, t_start):
    m = {}
    # value 输出统计（尺度监控）
    with torch.no_grad():
        vs = []
        for _g in range(5):
            envv = RiichiGame(config=RiichiConfig(), seed=args.seed + 9000 + _g)
            for _ in range(400):
                if envv.phase == "game_end":
                    break
                obs = envv.state.get_observation()
                vs.append(policy.value(obs))
                envv.step(policy.act(obs) if envv.turn == 0
                          else envv.random_action())
        v = np.array(vs)
        m["value_mean"] = float(v.mean())
        m["value_std"] = float(v.std())
    # vs-SL 自对战（--eval-games 局，座位轮换 seat=g%4）
    from tools.eval_vs_sl import play_game as _pg
    d = [0, 0, 0, 0]
    for _g in range(args.eval_games):
        seat = _g % 4
        rank, _ = _pg(policy, sl_ref, seat, args.seed + epoch * 1000 + _g)
        d[rank] += 1
    t = sum(d)
    m["wr_vs_sl"] = float((d[0] + 0.5 * d[1]) / max(t, 1)) if t else 0.0
    # T5 阈值再校准
    if t5 is not None and (epoch + 1) % args.t5_every == 0:
        t5res = t5.run(epoch, seed=args.seed, device=device)
        m["t5_calibrated"] = bool(any(not r.get("kept", True)
                                      for r in t5res.values()))
        m["t5_thr"] = {h: (r.get("applied_tau") if not r.get("kept", True)
                           else "kept") for h, r in t5res.items()}
    return m


def _learner_main(args, queue, stop_evt):
    import json as _json
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke
                          else "cpu")
    policy = RiichiPolicy(args.ckpt, seed=args.seed, device=device,
                          use_event_attn=args.event_attn)
    sl_ref = RiichiPolicy(SL_CKPT, seed=999, device=device)
    phi = None
    if os.path.exists(args.phi_ckpt):
        phi = RewardPredictor().to(device)
        phi.load_state_dict(torch.load(args.phi_ckpt, map_location="cpu")["model"])
        phi.eval()
    opt = torch.optim.AdamW([
        {"params": [p for n, p in policy.model.named_parameters()
                    if "event_attn" in n or "ctx_gate" in n], "lr": args.lr},
        {"params": [p for n, p in policy.model.named_parameters()
                    if ("event_attn" not in n and "ctx_gate" not in n
                        and not n.startswith("value."))], "lr": args.lr / 10.0},
        {"params": [p for n, p in policy.model.named_parameters()
                    if n.startswith("value.")], "lr": args.lr},
    ], lr=args.lr)
    phi_mon = None
    t5 = None
    if phi is not None and args.phi_monitor_every > 0:
        from tools.rl_phi_monitor import PhiMonitor
        phi_mon = PhiMonitor(phi, device, eval_every=args.phi_monitor_every)
        print("PHI MONITOR armed: every %d games" % args.phi_monitor_every,
              flush=True)
    if args.t5_every > 0:
        from tools.rl_t5_calibrate import T5Calibrator
        t5 = T5Calibrator(policy, min_pos=200, n_games=args.t5_games)
        print("T5 CALIBRATOR armed: every %d epochs" % args.t5_every,
              flush=True)

    os.makedirs(TRAJ_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(TRAJ_DIR, "g_*.pkl")):
        try:
            os.remove(f)
        except OSError:
            pass
    if os.path.exists(PAST_FILE):
        os.remove(PAST_FILE)   # 历史池从本次训练重新积累
    if os.path.exists(SYNC_PATH):
        os.remove(SYNC_PATH)   # 清理上次训练残留权重，防 worker 加载旧权重

    metrics_path = args.metrics_path
    games_path = args.games_path
    os.makedirs("logs", exist_ok=True)
    pool_buf = []
    selfplay_buf = []
    games_done = 0
    total = args.games * args.epochs
    t_start = time.time()
    lp_s = lv_s = h_s = r_s = c_s = kl_s = ld_s = lb_s = 0.0
    n_upd = 0
    prev = {"lp": 0.0, "lv": 0.0, "H": 0.0, "ratio": 0.0, "clip": 0.0,
            "kl": 0.0, "lp_disc": 0.0, "lp_bin": 0.0}
    opp_stats = {"self": 0, "past": 0, "sl": 0}
    rew_all = []

    while games_done < total:
        item = queue.get()
        if item is None:
            break
        _kind, _wid, _gid, pkl, gstats = item
        with open(pkl, "rb") as f:
            traj = pickle.load(f)
        try:
            os.remove(pkl)
        except OSError:
            pass
        for R, label in gstats.pop("sbuf", []):
            selfplay_buf.append((torch.from_numpy(R), label))
        opp_stats[gstats.get("opp", "sl")] += 1
        gstats["game"] = games_done % args.games
        gstats["epoch"] = games_done // args.games
        with open(games_path, "a", encoding="utf-8") as gf:
            gf.write(_json.dumps(gstats, ensure_ascii=False) + "\n")
        rew_all.extend([t["r"] for t in traj])
        pool_buf.append(traj)
        games_done += 1
        if phi_mon is not None:
            ev = phi_mon.step(games_done, selfplay_buf)
            if ev is not None:
                print("[rl] phi-monitor:",
                      _json.dumps(ev, ensure_ascii=False), flush=True)
        # 池满 -> PPO 更新 + 权重同步
        if len(pool_buf) >= args.pool_size:
            up = ppo_update_pool(policy, opt, pool_buf, args, device)
            pool_buf.clear()
            if up:
                lp_s += up[0]; lv_s += up[1]; h_s += up[2]; r_s += up[3]
                c_s += up[4]; kl_s += up[5]; ld_s += up[6]; lb_s += up[7]
                n_upd += 1
            _save_sync(policy)
            if args.smoke:
                assert abs(up[3] - 1.0) < 0.05, "initial ratio != 1: %.4f" % up[3]
                print("SMOKE OK (ratio=%.4f)" % up[3], flush=True)
        # epoch 边界：指标行（delta）+ 周期评估 + 存档
        if games_done % args.games == 0:
            epoch = games_done // args.games - 1
            n_p = max(n_upd - prev.get("_n", 0), 1)
            _rw = np.array(rew_all) if rew_all else np.zeros(1)
            m = {"epoch": epoch,
                 "games_done": games_done,
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
                 "throughput_gph": round(games_done
                                         / max(time.time() - t_start, 1)
                                         * 3600, 1),
                 "elapsed_s": round(time.time() - t_start, 1)}
            rew_all = []
            if policy.model.ctx_gate is not None:
                g = torch.sigmoid(policy.model.ctx_gate).detach().cpu()
                m["gate_mean"] = float(g.mean().item())
                m["gate_std"] = float(g.std().item())
            else:
                m["gate_mean"] = None
                m["gate_std"] = None
            # 周期评估（eval_every>0 时每 eval_every epoch 一次）
            if (args.eval_every > 0
                    and ((epoch + 1) % args.eval_every == 0
                         or epoch == args.epochs - 1)):
                m.update(_eval_block(args, epoch, policy, sl_ref, device, t5,
                                     t_start))
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(m, ensure_ascii=False) + "\n")
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
            _save_ckpt(args, policy, epoch)
    # 池尾清算（不足一池的轨迹；仅在轨迹足够多时更新，防短测收尾长卡）
    if pool_buf:
        if len(pool_buf) >= min(args.pool_size, max(total // 4, 1)):
            up = ppo_update_pool(policy, opt, pool_buf, args, device)
            if up:
                _save_sync(policy)
        else:
            print("[rl] tail pool skipped (%d trajs < %d)"
                  % (len(pool_buf), min(args.pool_size, max(total // 4, 1))),
                  flush=True)
        pool_buf.clear()
    stop_evt.set()
    print("LEARNER_DONE total=%d (%.0fs)" % (games_done,
                                             time.time() - t_start), flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Multi-process actor-learner self-play PPO v1.5")
    ap.add_argument("--ckpt", default="checkpoints/sl/rl/value_pretrain.pt")
    ap.add_argument("--phi-ckpt", default="checkpoints/sl/rl/reward_predictor.pt")
    ap.add_argument("--out", default="checkpoints/sl/rl/rl_v1.pt")
    ap.add_argument("--games", type=int, default=10, help="每 epoch 局数")
    ap.add_argument("--epochs", type=int, default=1000, help="epoch 总数")
    ap.add_argument("--workers", type=int, default=4, help="rollout worker 数")
    ap.add_argument("--pool-size", type=int, default=32,
                    help="learner 轨迹池大小（局）")
    ap.add_argument("--inner-epochs", type=int, default=1,
                    help="PPO 池内遍历次数")
    ap.add_argument("--batch-size", type=int, default=512, help="minibatch 大小")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--event-attn", action="store_true",
                    help="启用事件因果注意力旁支")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=30,
                    help="vs-SL 评估间隔（epochs；0=关闭）")
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=50,
                    help="历史检查点存档间隔（epochs）")
    ap.add_argument("--pool-k", type=int, default=8, help="历史对手池容量")
    ap.add_argument("--t5-every", type=int, default=100,
                    help="T5 校准间隔（epochs；0=关闭）")
    ap.add_argument("--t5-games", type=int, default=40)
    ap.add_argument("--phi-monitor-every", type=int, default=500,
                    help="Φ 漂移监控间隔（局；0=关闭）")
    ap.add_argument("--metrics-path", default="logs/rl_train_metrics.jsonl",
                    help="metrics 输出（面板读取）")
    ap.add_argument("--games-path", default="logs/rl_train_games.jsonl",
                    help="games 输出（面板读取）")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.smoke:  # 最短链路：1 worker × 1 局 × 1 epoch
        args.workers = 1
        args.games = 1
        args.epochs = 1
        args.pool_size = 1
        args.eval_every = 0
        args.t5_every = 0
        args.phi_monitor_every = 0
        args.save_every = 1

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=args.workers * 4)
    stop_evt = ctx.Event()
    ready = ctx.Event()
    procs = []
    for w in range(args.workers):
        p = ctx.Process(target=_worker_main,
                        args=(w, args, queue, stop_evt, ready))
        p.start()
        procs.append(p)
    print("WORKERS started: %d (games=%d epochs=%d total=%d)"
          % (args.workers, args.games, args.epochs,
             args.games * args.epochs), flush=True)
    try:
        _learner_main(args, queue, stop_evt)
    finally:
        stop_evt.set()
        for p in procs:
            p.join(timeout=10)
    print("RL_V1 DONE -> %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
