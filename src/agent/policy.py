# -*- coding: utf-8 -*-
"""RiichiPolicy — 神经网络决策模块（引擎观察 -> 引擎动作）。

对接 src/env/riichi_game.py：
- 输入：game.state.get_observation(seat)（schema §2 rec 格式）
- 输出：engine step action（{"type": ...}）
- 特征：model.features.build_features / discard_mask（与 SL 训练一致）
- 阈值：ron/tsumo/kyushu 使用校准后的代价敏感阈值
  （默认从 ckpt 同目录 calibration.json 读取；tsumo tau=0.86/T=3.74，
   kyushu tau=0.83/T=3.33，ron 无负样本 -> 0.5）
- RL 接口：probs(obs) 返回各动作头分布（discard (34,) / binary (2,)）；
  act() 为阈值+argmax 决策；sample() 支持温度采样。

用法：
    from agent.policy import RiichiPolicy
    policy = RiichiPolicy("checkpoints/sl/transfer/transfer_final.pt")
    while game.phase != "game_end":
        game.step(policy(game))          # 决策者 = game.turn
"""
import json, os
import numpy as np
import torch
import torch.nn.functional as F

from model.attn_modules import build_events
from model.features import BINARY_HEADS, build_features, discard_mask, feature_channels
from model.model import MultiHeadRiichiNet, get_device

DEFAULT_THR = {"ron": 0.5, "tsumo": 0.86, "kyushu": 0.83,
               "riichi": 0.5, "chow": 0.5, "pon": 0.5, "kan": 0.5}
DEFAULT_TEMP = {"ron": 1.0, "tsumo": 3.74, "kyushu": 3.33,
                "riichi": 1.0, "chow": 1.0, "pon": 1.0, "kan": 1.0}
BINARY_ACTION_THR = 0.5  # 兼容常量（实际用 self.thr[h]）


def _onehot(kind, n=34):
    v = np.zeros(n, dtype=np.float32)
    if 0 <= kind < n:
        v[kind] = 1.0
    return v


def load_calibration(ckpt_path: str):
    """从 ckpt 同目录 calibration.json 读全部决策头阈值/温度/偏置。
    新格式 {head: {"thr": float, "T": float, "bias": float}}（人类执行率校准
    + logit 偏置防饱和）；兼容旧格式 {h: {"thr": {"Cfp=5": {"tau": ...}}, "T": ...}}。"""
    d = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "calibration.json")
    thr, temp, bias = dict(DEFAULT_THR), dict(DEFAULT_TEMP), {}
    if os.path.exists(d):
        try:
            data = json.load(open(d, encoding="utf-8"))
            for h in list(thr.keys()):
                if h not in data:
                    continue
                v = data[h]
                if not isinstance(v, dict):
                    continue
                t2 = v.get("thr", v.get("tau"))
                if isinstance(t2, dict) and "Cfp=5" in t2:
                    thr[h] = float(t2["Cfp=5"]["tau"])
                elif isinstance(t2, (int, float)):
                    thr[h] = float(t2)
                temp[h] = float(v.get("T", 1.0))
                b = v.get("bias", 0.0)
                if isinstance(b, (int, float)) and b != 0:
                    bias[h] = float(b)
        except Exception:
            pass
    return thr, temp, bias


class RiichiPolicy:
    """基于 MultiHeadRiichiNet 的策略。线程安全只读推理（no_grad）。"""

    def __init__(self, ckpt: str, thresholds=None, temps=None,
                 device=None, seed: int = 0, use_event_attn: bool = False):
        self.device = get_device() if device is None else device
        ck = torch.load(ckpt, map_location="cpu")
        cfg = ck.get("config") or {}
        # 新模块（event_attn/ctx_gate 等 missing 参数）跨进程随机初始化一致：
        # MP actor-learner 下 worker/learner 各自构造模型，若随机不同则 logp 不一致
        torch.manual_seed(0)
        self.model = MultiHeadRiichiNet(
            feature_channels("full"),
            channels=int(cfg.get("channels", 256)),
            n_blocks=int(cfg.get("blocks", 50)),
            binary_heads=BINARY_HEADS,
            use_event_attn=use_event_attn)
        missing, unexpected = self.model.load_state_dict(ck["model"], strict=False)
        if missing:
            print("policy: 新模块初始化（missing %d 参数）: %s"
                  % (len(missing), "event_attn/ctx_gate" if any("event_attn" in k or "ctx_gate" in k for k in missing) else "?"))
        if unexpected:
            print("policy: unexpected keys:", len(unexpected))
        self.model.eval().to(self.device)
        self.thr, self.temp, self.bias = load_calibration(ckpt)
        # 动态 bias 持久化（评审必改 1）：RL ckpt 存训练期每池归中的 bias，
        # 加载时覆盖 calibration 静态值——否则验收 eval 用 SL 尺度 bias 作用在
        # 归中训练过的模型上，行为≠训练行为。
        _ck_bias = ck.get("bias")
        if isinstance(_ck_bias, dict) and _ck_bias:
            for _h, _b in _ck_bias.items():
                if _h in self.bias and isinstance(_b, (int, float)):
                    self.bias[_h] = float(_b)
        if thresholds:
            self.thr.update(thresholds)
        if temps:
            self.temp.update(temps)
        self.z_cap = {}   # 热干预：{head: C} 对该头 z 做软限幅 C·tanh(z/C)（防两极分化，|z|>C 梯度衰减；0/缺省=关闭）
        self.always_win = False  # 阶段训练：强制能和就和（tsumo/ron 采样+logp=0 → 头零梯度冻结）；
        #   能力提升后再热改 logs/rl_hyper.json always_win=0 恢复学习见逃
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ 候选/特征

    def _cands(self, obs: dict) -> dict:
        """每头候选 tile one-hot（与训练一致：meld 取第一个合法选项）。"""
        la = obs.get("legal_actions") or {}
        out = {}
        for h in BINARY_HEADS:
            legal = la.get(h)
            k = -1
            if h == "riichi":
                if legal:
                    k = legal[0] // 4
            elif h in ("chow", "pon", "kan"):
                if legal:
                    k = legal[0]["tiles"][0] // 4
            out[h] = _onehot(k)
        return out

    def _logits(self, obs: dict, events=None, amp: bool = True) -> dict:
        """单观察前向。amp=True 用 fp16（act/probs/value；无 logp 记账），
        amp=False 用 fp32（sample_with_logp：保证 worker/learner 两侧 logp
        batch 无关一致，初始 ratio≈1）。输出一律转回 fp32。"""
        x = torch.from_numpy(build_features(obs, "full")[None]).to(self.device)
        dmask = torch.from_numpy(discard_mask(obs)[None]).to(self.device)
        cands = {h: torch.from_numpy(v[None]).to(self.device)
                 for h, v in self._cands(obs).items()}
        ev = None
        if self.model.use_event_attn:
            if events is None:
                ev = build_events(obs).unsqueeze(0).to(self.device)   # 离线近似（座位拼接）
            elif events.dim() == 2:
                ev = events.unsqueeze(0).to(self.device)               # 引擎真实时间线
            else:
                ev = events.to(self.device)
        with torch.no_grad():
            if amp and str(self.device).startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = self.model(x, masks={"discard": dmask}, candidates=cands,
                                    events=ev)
            else:
                lg = self.model(x, masks={"discard": dmask}, candidates=cands,
                                events=ev)
        return {h: v.float() for h, v in lg.items()}

    def _p(self, logits: dict, head: str, temp: float) -> float:
        lg = logits[head][0]  # (2,)
        z = lg[1] - lg[0] + self.bias.get(head, 0.0)
        z = z.clamp(-12.0, 12.0)   # logits 裁剪：防二次极端分化（梯度在裁剪处为 0）
        return float(torch.sigmoid(z / max(temp, 1e-3)).item())

    # ------------------------------------------------------------ 决策

    def act(self, obs: dict, events=None) -> dict:
        """观察 -> 引擎动作（阈值决策 + discard argmax）。obs 必须含 legal_actions。"""
        logits = self._logits(obs, events)
        return self._act_from_logits(logits, obs)

    def _act_from_logits(self, logits: dict, obs: dict) -> dict:
        """从已算好的 logits 决策（向量化 batch 前向后逐 obs 调用；行为与 act() 完全一致）。"""
        la = obs.get("legal_actions") or {}
        # 自摸 / 九种九牌（draw 阶段决策）
        if la.get("tsumo") and self._p(logits, "tsumo", self.temp["tsumo"]) >= self.thr["tsumo"]:
            return {"type": "tsumo"}
        if la.get("kyushu") and self._p(logits, "kyushu", self.temp["kyushu"]) >= self.thr["kyushu"]:
            return {"type": "kyushu"}
        # 阶段判定只看 legal：有 discard = 打牌阶段；无 discard + 有声明选项
        # = 声明阶段（含 kan-only 明杠、ron-only、chankan 抢杠——都是合法场景）
        has_decl = bool(la.get("ron") or la.get("pon") or la.get("chow") or la.get("kan"))
        in_claim = has_decl and not la.get("discard")
        if in_claim:
            if la.get("ron") and self._p(logits, "ron", self.temp["ron"]) >= self.thr["ron"]:
                return {"type": "ron"}
            for h in ("kan", "pon", "chow"):
                if la.get(h) and self._p(logits, h, self.temp[h]) >= self.thr[h]:
                    return {"type": h, "tiles": list(la[h][0]["tiles"])}
            return {"type": "pass"}
        # 摸牌阶段：立直 / 杠 / 打牌
        if la.get("riichi") and self._p(logits, "riichi", self.temp["riichi"]) >= self.thr["riichi"]:
            return {"type": "riichi", "tile": la["riichi"][0]}
        if la.get("kan") and self._p(logits, "kan", self.temp["kan"]) >= self.thr["kan"]:
            return {"type": "kan", "tiles": list(la["kan"][0]["tiles"])}
        return self._pick_discard(la, logits)

    def _pick_discard(self, la: dict, logits: dict) -> dict:
        disc = la.get("discard") or []
        assert disc, "no legal discard"
        lg = logits["discard"][0].cpu().numpy()
        kind = int(np.argmax(lg))
        tiles = [t for t in disc if t // 4 == kind]
        if not tiles:  # argmax kind 不在合法列表（数值兜底）
            kind = max({t // 4 for t in disc}, key=lambda k: lg[k])
            tiles = [t for t in disc if t // 4 == kind]
        return {"type": "discard", "tile": tiles[0]}

    def __call__(self, game) -> dict:
        """引擎对接：当前决策者（game.turn）的观察 -> 动作。
        启用 event_attn 时使用引擎全局事件构建真实因果时间线。"""
        obs = game.state.get_observation()
        events = None
        if self.model.use_event_attn:
            from model.attn_modules import build_events_from_game
            events = build_events_from_game(game.events, game.round_idx)
        return self.act(obs, events)

    # ------------------------------------------------------------ RL 接口

    def logits(self, obs: dict, requires_grad: bool = False) -> dict:
        """原始 logits（含 "value"），RL 反向传播时 requires_grad=True。"""
        x = torch.from_numpy(build_features(obs, "full")[None]).to(self.device)
        if requires_grad:
            x.requires_grad_(True)
        dmask = torch.from_numpy(discard_mask(obs)[None]).to(self.device)
        cands = {h: torch.from_numpy(v[None]).to(self.device)
                 for h, v in self._cands(obs).items()}
        if requires_grad:
            self.model.train()
        with torch.set_grad_enabled(requires_grad):
            return self.model(x, masks={"discard": dmask}, candidates=cands)

    def probs(self, obs: dict) -> dict:
        """各动作头分布（RL 采样/熵用）：
        discard: (34,) softmax（非法位置已 -inf）；binary 头: (2,)。"""
        logits = self._logits(obs)
        out = {"discard": F.softmax(logits["discard"][0], dim=0).cpu().numpy()}
        for h in BINARY_HEADS:
            out[h] = F.softmax(logits[h][0], dim=0).cpu().numpy()
        return out

    def value(self, obs: dict) -> float:
        """value head 输出（RL critic；分数量纲，训练时归一化到 [-25,+25]）。"""
        lg = self._logits(obs)
        return float(lg["value"][0].item())

    def policy_value(self, obs: dict) -> tuple:
        """单次前向返回 (probs dict, value float)——RL 每步的标准取用接口。"""
        logits = self._logits(obs)
        out = {"discard": F.softmax(logits["discard"][0], dim=0).cpu().numpy()}
        for h in BINARY_HEADS:
            out[h] = F.softmax(logits[h][0], dim=0).cpu().numpy()
        v = float(logits.get("value", torch.zeros(1))[0].item())
        return out, v

    def sample(self, obs: dict, temperature: float = 1.0) -> dict:
        """从策略分布采样一个动作（RL 探索用）。"""
        la = obs.get("legal_actions") or {}
        probs = self.probs(obs)
        # 和牌类：按阈值概率伯努利（探索时用温度软化）
        if la.get("tsumo"):
            p = probs["tsumo"][1] ** (1.0 / max(temperature, 1e-3))
            if self.rng.random() < p:
                return {"type": "tsumo"}
        if la.get("kyushu"):
            p = probs["kyushu"][1] ** (1.0 / max(temperature, 1e-3))
            if self.rng.random() < p:
                return {"type": "kyushu"}
        if la.get("ron"):
            p = probs["ron"][1] ** (1.0 / max(temperature, 1e-3))
            if self.rng.random() < p:
                return {"type": "ron"}
        if la.get("pon") or la.get("chow") or la.get("kan"):
            for h in ("kan", "pon", "chow"):
                if la.get(h) and self.rng.random() < probs[h][1] ** (1.0 / max(temperature, 1e-3)):
                    return {"type": h, "tiles": list(la[h][0]["tiles"])}
            return {"type": "pass"}
        if la.get("riichi") and self.rng.random() < probs["riichi"][1] ** (1.0 / max(temperature, 1e-3)):
            return {"type": "riichi", "tile": la["riichi"][0]}
        if la.get("kan") and self.rng.random() < probs["kan"][1] ** (1.0 / max(temperature, 1e-3)):
            return {"type": "kan", "tiles": list(la["kan"][0]["tiles"])}
        # discard 温度采样
        dp = probs["discard"].copy()
        disc = la.get("discard") or []
        mask = np.zeros(34, dtype=bool)
        for t in disc:
            mask[t // 4] = True
        dp[~mask] = 0.0
        if dp.sum() <= 0:
            kind = disc[0] // 4
        else:
            dp = dp ** (1.0 / max(temperature, 1e-3))
            dp /= dp.sum()
            kind = int(self.rng.choice(34, p=dp))
        tiles = [t for t in disc if t // 4 == kind]
        return {"type": "discard", "tile": tiles[0] if tiles else disc[0]}


    def sample_with_logp(self, obs, temperature: float = 1.0, events=None):
        """RL 采样接口（单 obs）：(action dict, logp (1,), value (1,), ent float)。
        logp 单一数据源（fp32）。"""
        lg = self._logits(obs, events, amp=False)   # fp32: logp 单一数据源不变量
        return self._sample_from_logits(lg, obs, temperature)

    def _sample_from_logits(self, lg, obs, temperature):
        """从已算好的 logits 采样（向量化 batch 前向后逐 obs 调用，与 sample_with_logp
        行为完全一致）。7 头 + 校准阈值（ron/tsumo/kyushu 阈值决策；riichi/chow/pon/kan
        采样；discard 合法牌种温度 softmax）。logp 覆盖所有分支（pass 用 1-p）。"""
        la = obs.get("legal_actions") or {}
        value = lg["value"][0].reshape(1)
        T = max(temperature, 1e-3)
        eps = 1e-8

        def p_bin(head):
            z = lg[head][0]
            z = self._zc(head, z[1] - z[0] + self.bias.get(head, 0.0)).clamp(-12.0, 12.0)
            return torch.sigmoid(z / self.temp.get(head, 1.0))

        # RL 期：阈值头 Bernoulli 采样（行为=计 logp 策略）；评估用 act() 阈值判定
        for h in ("tsumo", "kyushu", "ron"):
            if la.get(h):
                # 阶段训练（always_win）：能和就和——强制执行，logp=0（概率 1），
                # 与重放 _logp_for_action 一致 → 和牌头 PPO 零梯度冻结，只学其他决策
                if getattr(self, "always_win", False) and h in ("tsumo", "ron"):
                    return {"type": h}, torch.zeros(1, device=value.device), value, 0.0
                p = p_bin(h)
                if self.rng.random() < p ** (1.0 / T):
                    return {"type": h}, torch.log(p.clamp(eps)).reshape(1), value, 0.0
        if la.get("pon") or la.get("chow"):
            names, ps = self._claim_probs(lg, la, T)
            idx = int(self.rng.choice(len(names), p=ps.detach().cpu().numpy()))
            name = names[idx]
            if name == "pass":
                return {"type": "pass"}, torch.log(ps[idx].clamp(eps)).reshape(1), value, 0.0
            return ({"type": name, "tiles": list(la[name][0]["tiles"])},
                    torch.log(ps[idx].clamp(eps)).reshape(1), value, 0.0)
        if la.get("riichi"):
            p = float(p_bin("riichi"))
            if self.rng.random() < p ** (1.0 / T):
                return ({"type": "riichi", "tile": la["riichi"][0]},
                        torch.log(torch.tensor(max(p, eps), device=value.device)).reshape(1),
                        value, 0.0)
        if la.get("kan"):
            p = float(p_bin("kan"))
            if self.rng.random() < p ** (1.0 / T):
                return ({"type": "kan", "tiles": list(la["kan"][0]["tiles"])},
                        torch.log(torch.tensor(max(p, eps), device=value.device)).reshape(1),
                        value, 0.0)
        disc = la.get("discard") or []
        if not disc:
            return {"type": "pass"}, torch.zeros(1, device=value.device), value, 0.0
        probs = torch.softmax(lg["discard"][0] / T, dim=0)
        kinds = sorted({t // 4 for t in disc})
        mask = torch.full_like(probs, 0.0)
        mask[kinds] = 1.0
        probs = probs * mask
        probs = probs / probs.sum().clamp(eps)
        idx = int(self.rng.choice(34, p=probs.cpu().numpy()))
        tile = next((t for t in disc if t // 4 == idx), disc[0])
        ent = float(-(probs * torch.log(probs + eps)).sum().item())
        return ({"type": "discard", "tile": tile},
                torch.log(probs[idx].clamp(eps)).reshape(1), value, ent)


    def _zc(self, head, z):
        """软限幅（热干预）：z → C·tanh(z/C)。保梯度、防 |z| 无限外推；C=0/缺省=原样。"""
        c = self.z_cap.get(head)
        if c and c > 0:
            return c * torch.tanh(z / c)
        return z

    def _claim_probs(self, logits, la, T):
        """声明阶段分布（温度缩放+归一化）——采样与 PPO 重放共用（单一数据源）。
        评审必改：保持张量梯度链（float()+torch.tensor 重建会断链，导致吃/碰/
        声明杠/过 的 logp 无梯度，PPO 对声明决策零梯度）。"""
        eps = 1e-8
        opts = []
        for h in ("kan", "pon", "chow"):
            if la.get(h):
                z = logits[h][0]
                zz = self._zc(h, z[1] - z[0] + self.bias.get(h, 0.0))
                opts.append((h, torch.sigmoid(zz.clamp(-12.0, 12.0))))
        dev = next(iter(logits.values())).device if isinstance(logits, dict) and logits else "cpu"
        p_pass = torch.ones((), device=dev)
        for _h, p in opts:
            p_pass = p_pass * (1.0 - p)          # 保持张量链：P(不声明)
        if opts:
            ps = torch.stack([p_pass.clamp(min=eps)]
                             + [p.clamp(min=eps) for _h, p in opts])
        else:
            ps = p_pass.clamp(min=eps).reshape(1)
        ps = ps ** (1.0 / max(T, 1e-3))
        ps = ps / ps.sum()
        return ["pass"] + [h for h, _ in opts], ps

    def _logp_for_action(self, logits, obs, action, T=1.0):
        """动作在新策略下的 logp——采样与重放共用，保证 PPO ratio 一致。"""
        la = obs.get("legal_actions") or {}
        eps = 1e-8
        at = action["type"]

        def p_bin(h):
            z = logits[h][0]
            z = self._zc(h, z[1] - z[0] + self.bias.get(h, 0.0)).clamp(-12.0, 12.0)
            return torch.sigmoid(z / self.temp.get(h, 1.0))

        # 声明阶段（la 含 pon/chow）：kan/pon/chow/pass 都用联合分布 _claim_probs
        # （与采样一致，auditor 验收：原始 sigmoid 未归一化会导致双选项初始 ratio=1.26）
        # 阶段训练（always_win）：和牌 logp=0 常数（与采样一致，头零梯度）
        if getattr(self, "always_win", False) and at in ("tsumo", "ron"):
            _dev = next(iter(logits.values())).device
            return torch.zeros(1, device=_dev)
        in_claim = bool(la.get("pon") or la.get("chow"))
        if in_claim and at in ("kan", "pon", "chow", "pass"):
            names, ps = self._claim_probs(logits, la, T)
            return torch.log(ps[names.index(at)].clamp(eps)).reshape(1)
        # draw 阶段自杠 / riichi / 阈值头 / 非声明 kan：原始 p_bin
        if at in ("tsumo", "ron", "kyushu", "riichi", "kan", "pon", "chow"):
            return torch.log(p_bin(at).clamp(eps)).reshape(1)
        if at == "pass":
            names, ps = self._claim_probs(logits, la, T)
            return torch.log(ps[names.index("pass")].clamp(eps)).reshape(1)
        disc = la.get("discard") or []
        probs = torch.softmax(logits["discard"][0] / max(T, 1e-3), dim=0)
        kinds = sorted({t // 4 for t in disc})
        mask = torch.full_like(probs, 0.0)
        mask[kinds] = 1.0
        probs = probs * mask
        probs = probs / probs.sum().clamp(eps)
        return torch.log(probs[action["tile"] // 4].clamp(eps)).reshape(1)


def play_game(policy, seed=1, max_steps=10000):
    """用策略打一局（4 家同一策略），返回 game_result。"""
    from env.riichi_game import RiichiGame, RiichiConfig
    game = RiichiGame(RiichiConfig(), seed=seed)
    for _ in range(max_steps):
        if game.phase == "game_end":
            break
        game.step(policy(game))
    else:
        raise RuntimeError("game did not finish in %d steps" % max_steps)
    return game.game_result
