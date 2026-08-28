# RL 阶段 1 模型架构与训练规格（v1，2026-08-26）

> 本文档为机器可读实现规格：任何 AI/工程师按此可实现 RL 阶段 1 的完整链路。
> 代码位置索引见 §9；训练验收见 §8。

## 1. 总览

RL 阶段 1 = 自对弈 PPO，策略为「7 头校准决策 + 事件因果注意力（逐通道门控注入）」的多头网络。

```
obs ──build_features──▶ x (B,284,34) ──▶ stem+50blocks ──▶ h (B,256,34)
  └─build_events──▶ ev (B,64,46) ──▶ EventCausalAttn ──▶ ctx (B,256)
                                                        │
                     h' = h + σ(ctx_gate)⊙tanh(ctx)    ◀┘ (逐通道门控)
                     h' ──▶ discard/binary×7/value heads
```

## 2. 输入规格

### 2.1 特征张量 x（B, 284, 34, float32）
- 来源: `model.features.build_features(obs, mode="full")[:, :, 0]`
- 284 通道分组（`channel_layout("full")`）:
  | 组 | 通道数 | 含义 |
  |---|---:|---|
  | river | 192 | 4家×48位置的河牌+手切标记（静态内容） |
  | state | 25 | 局况（杠数/立直/分数等） |
  | melds | 16 | 副露 |
  | dora | 12 | 宝牌 |
  | discards | 12 | 打牌统计 |
  | basic | 10 | 轮次/庄家/本场 |
  | shanten | 9 | 自家向听 |
  | hand | 6 | 自家手牌 |
  | visible | 2 | 可见牌 |

### 2.2 事件序列 ev（B, 64, 46, float32）
- 来源（RL 决策）: `model.attn_modules.build_events_from_game(engine.events)` — 引擎全局事件严格时序（20/20 验证）；离线 records 回退 `build_events(obs)`（座位拼接近似，仅评估/数据生成用）
- 46 维 token 编码（每个事件一行）:
  | 段 | 维度 | 编码 |
  |---|---:|---|
  | seat | 0-3 | one-hot(4) |
  | tile kind | 4-37 | one-hot(34) |
  | tsumogiri | 38 | 0/1 手切标记 |
  | riichi | 39 | 0/1 立直打出 |
  | meld_kind | 40-44 | one-hot(5): chi/pon/minkan/ankan/kakan |
  | round | 45 | round_idx/8 归一化 |
- 64 token 构成（build_events_from_game，引擎全局事件严格时序）: dahai/reach/chi/pon/kan 事件按时间序取最近 64 个；补零在前
- 离线回退 build_events（座位拼接近似）仅用于无引擎事件的 records 数据生成（RL 决策一律用真实时间线）

## 3. 主干（继承自 SL，仅微调）

### 3.1 stem + 50 残差块（Suphx 风格）
- stem: `nn.Conv1d(284, 256, 3, padding=1, bias=False)`
- 50 × ResidualBlock: `Conv1d(256,256,3,padding=1,bias=False) → ReLU → Conv1d(256,256,3) → ReLU(x+残差)`
- 旧权重来源: `checkpoints/sl/transfer/transfer_final.pt`（strict=False 加载）

## 4. 事件注意力旁支 EventCausalAttn（新增）

### 4.1 模块定义（`src/model/attn_modules.py`）
```
EventCausalAttn(d_model=256, n_head=8, max_events=64, event_dim=46)
├─ proj:      Linear(46 → 256)            # 事件嵌入（kaiming init）
├─ pos:       nn.Parameter(64, 256)        # 可学习位置编码（N(0,0.02)）
├─ attn:      MultiheadAttention(256, 8头, batch_first=True)
│             # in_proj_weight (768,256), in_proj_bias (768)
│             # out_proj.weight (256,256), out_proj.bias (256)
├─ norm:      LayerNorm(256)               # 残差后归一化
└─ out:       Linear(256 → 256)            # 末 token → 上下文（kaiming）
```

### 4.2 前向
```
x = proj(ev) + pos[:T]                 # (B,T,256) 嵌入+位置
causal = triu(-inf, diagonal=1)        # 因果掩码（只看过去）
x = attn(x,x,x, attn_mask=causal)[0]   # 自注意力
x = norm(x + proj(ev) + pos[:T])       # 残差
ctx = out(x[:, -1])                    # 取最后事件 → (B,256)
```

### 4.3 注入（`src/model/net.py` forward，逐通道门控）
```
if model.event_attn is not None and events is not None:
    ctx = torch.tanh(model.event_attn(events))          # (B,256) 压缩到 [-1,1]
    gate = torch.sigmoid(model.ctx_gate)                # (256,) 逐通道
    h = h + gate.unsqueeze(0).unsqueeze(-1) * ctx.unsqueeze(-1)
```
- `ctx_gate`: `nn.Parameter(full((256,), -2.0))` — 初始 σ(-2)≈0.12（评审 1.1：-5 梯度饥饿，旁支有效学习仅 0.067×主干；-2 平衡保护与可学习性）
- **训练期望**: gate 逐通道开启（σ(-2) 起步随训练增大）——旁支合理性证据改为「注入前后 head 输出差异」与 gate 分布统计（评审 1.2：gate 对应 trunk 潜在通道，非输入通道组）

## 5. 决策头（7 头 + value）

| 头 | 输出 | 结构 | 决策方式 |
|---|---|---|---|
| discard | (B,34) | DiscardHead: Conv1d(256→32,3) → FC1024 → FC256 → FC34 | 合法牌种 softmax（温度采样） |
| riichi | (B,2) | BinaryHead: Conv1d(256→64,3) → concat(候选one-hot(34)) → FC512 → FC2 | 采样 P(1)=sigmoid(z1−z0) |
| chow | (B,2) | 同上（candidate=吃牌第一选项 tile） | 采样 |
| pon | (B,2) | 同上 | 采样 |
| kan | (B,2) | 同上（claim=明杠 / draw=自杠，阶段判定见 §6.2） | 采样 |
| ron | (B,2) | 同上（bool 头，candidate=全零） | RL 期 Bernoulli 采样；评估 act() 阈值 τ=0.5 |
| tsumo | (B,2) | 同上 | RL 期 Bernoulli 采样；评估阈值 τ=0.86, T=3.74 |
| kyushu | (B,2) | 同上 | RL 期 Bernoulli 采样；评估阈值 τ=0.83, T=3.33 |
| value | (B,) | ValueHead: Conv1d(256→32,3) → FC512 → FC1 | critic（value_pretrain.pt 初始化） |

- 校准概率: `p = sigmoid((z[1]−z[0]) / T)`（T 来自 calibration.json）
- 阈值/温度来源: `checkpoints/sl/transfer/calibration.json`

## 6. 决策策略（`src/agent/policy.py`）

### 6.1 采样接口 `sample_with_logp(obs, temperature) -> (action, logp, value, ent)`
分支优先级（与引擎 legal 一致）:
1. tsumo: Bernoulli(p^T) → 自摸（评估用 act() 阈值）
2. kyushu: Bernoulli(p^T) → 流局
3. ron: Bernoulli(p^T) → 荣和
4. 声明阶段（la 含 pon/chow，或 kan 且无 discard）：按 {pass, kan, pon, chow} 概率温度采样
5. riichi: 伯努利 p^T → 立直
6. kan（draw 自杠）: 伯努利 → 杠
7. discard: 合法牌种温度 softmax → 打牌

### 6.2 阶段判定（关键，避免非法 pass）
```
in_claim = has_decl and not la["discard"]   # has_decl = 存在 ron/pon/chow/kan
# draw 阶段自杠 kan 不算声明；claim 的 kan-only（暗刻明杠）算
```

## 7. 训练方式（自对弈 PPO）

### 7.1 初始化
- 新模型 `MultiHeadRiichiNet(284, 256, 50块, BINARY_HEADS, use_event_attn=True)`
- `load_state_dict(transfer_final.pt, strict=False)` → missing=12（event_attn 11 + ctx_gate 1）
- value head 用 `value_pretrain.pt` 覆盖（critic 预训练）
- GRU 奖励预测器 `RewardPredictor` ← `checkpoints/sl/rl/reward_predictor.pt`

### 7.2 自对弈采样
```
N 个引擎进程（CPU），seat0=本方（GPU 推理），其他=对手池
对手池（v1.2 自对弈主体，随机对手仅 smoke/eval sanity）:
  当前策略 θ（主, 60%）= 纯自对弈（全 4 座同一 θ，seat0 采样训练、他家阈值贪心）
  历史 epoch checkpoint（25%）= 虚构自对弈（AlphaGo-Lee/OpenAI Five/Suphx 做法，防坍缩）
  SL(transfer_final)（15%）= 保留强基线，防遗忘 + 与验收同分布
轨迹: {obs, action, logp_old, value, r, ent}（obs 全量保存用于重放）
轮末奖励: r = Φ(x_1..k) − Φ(x_1..k−1)（GRU 差分）
终局: + 精算点数 settlement_pt(final, seat0, "tenhou")
```

### 7.3 PPO 更新
```
L = E[ −min(r_t·A, clip(r_t, 0.8, 1.2)·A) ]   # r_t = exp(logp_new − logp_old)
  + 0.5 · (V(s)−R_t)²                          # value MSE
  − 0.01 · H̄(π)                                # 归一化混合熵（定义见 §7.4）
A = GAE(γ=0.99, λ=0.95)
重放: 轨迹 obs → batch 前向（x+events）→ 新 logp → ratio
```
- **更新循环（明确粒度）**:
  1. 采集 32 局轨迹 → 拼接为轨迹池（episode 边界标记保存）
  2. GAE 按 episode 边界计算（dones 在每局末尾置 True，γ/λ 不回跨局）
  3. 池内 shuffle → minibatch 512 → **3 个 inner epoch** 遍历池
  4. 每次 minibatch 一次 Adam 步（PPO 单 epoch 内重复利用 on-policy 数据，符合标准 PPO）
  5. 更新后丢弃旧池，重新采样（on-policy）
- 实现: `src/model/train_rl.py`（rollout_game / ppo_update / _batch_features）；logp 重放共用 `policy._logp_for_action`（单一数据源）

### 7.4 超参与分层学习率
```
优化器: AdamW · clip_grad 1.0 · warmup 500 步 → cosine
分层 lr（推荐，消融可选）:
  event_attn.* : 3e-4（新模块正常学习）
  ctx_gate     : 3e-4（门控快速开启）
  trunk/heads  : 3e-5（SL 权重 1/10 微调）
  value        : 3e-4
熵自适应: α ← α + β(H_target − H̄)
批量: 每 32 局一更新（on-policy 轨迹池）→ 3 inner epoch × minibatch 512
epochs × games: 10 × 1000（v1.4：总 10,000 局，用户指定）
  - 每局即 PPO 更新；epoch = 10 局训练 + 1 条 metrics
  - vs-SL 评估按 --eval-every(默认 10 epoch=100 局) 执行，--eval-games(默认 2 局)，
    座位轮换 seat=g%4；value 统计同节奏；T5 校准 --t5-every 100 epoch
  - 历史存档按 --save-every(默认 100 epoch) 存 rl_v1_ep%05d.pt，
    环形保留 --pool-k(默认 5) 个并入历史对手池
**归一化混合熵（评审修正量纲失衡）**:
  H_bar = (H_disc/ln34 + sum_b H_b/ln2) / 5,  b in {riichi,chow,pon,kan}
  - 各头除以 max 熵（ln34~3.53、ln2~0.69）→ 每头贡献 [0,1]，量纲一致
  - ron/tsumo/kyushu 为阈值决策（非采样）不参与熵
  - H_target = min(0.35, H_bar_sl + 0.1)（熵对标 SL 实测，评审 2.6）
保存: 每 epoch rl_v1_epoch*.pt（含 thresholds/temps/heads 元数据）
```

### 7.5 训练监控
- gate 均值/方差（判断旁支是否开启/分化）
- 策略熵、PPO clip 率（应 <10%）、value loss 趋势
- 每 500 局对固定评估种子自对战（趋势）

### 7.6 奖励模型（Φ）漂移控制（评审补充）

**问题**: Φ 在人类日志分布训练，自对弈策略演化 → 轨迹分布偏移 → Φ 分布外（OOD）→ 奖励误差自举。

**三管齐下**:
1. **监控（检测）**: 每 500 局，用天凤日志 1 万局评估 Φ 预测 MSE 与相关性（logs/rl_phi_monitor.json）；
   MSE 相对初始训练值恶化 >30% → 触发重训
2. **重训（校正，核心）**: 每 2000 局，用最新自对弈轨迹（终局精算 settlement_pt 为 ground truth）
   在线微调 Φ（lr 1e-4，~500 步，MSE 同 §7.3 逐前缀监督）
3. **防遗忘混合**: 微调数据 = 50% 最新自对弈 + 50% 人类日志（随机抽样），防止 Φ 丢失人类先验

**为什么自对弈可重训**: 引擎自对弈终局分数精确已知 → 终局精算 ground truth 零偏差（优于人类日志的间接监督）。

## 8. 验收标准（硬门槛）
```
tools/eval_vs_sl.py --a rl_v1.pt --b transfer_final.pt --games 100 --seeds 3
win_rate = 1位率 + 0.5×2位率；判定: mean≥0.66 且 95%CI 下界≥0.60（评审 2.5：单点 66% 是掷硬币）
1 打 3 门槛校准：均势期望 win_rate=0.375（评审 2.5）→ 66% 需约 45% 首位率
否则: 检查 gate 值/对手池/熵，调整重训
消融: use_event_attn on/off 各跑验收 → ② 的独立贡献
```

## 9. 代码文件索引

| 文件 | 内容 |
|---|---|
| `src/model/net.py` | MultiHeadRiichiNet（use_event_attn/ctx_gate 门控注入） |
| `src/model/attn_modules.py` | build_events / EventCausalAttn / OpponentEquivariant / BeliefDangerHead |
| `src/model/features.py` | build_features / discard_mask / channel_layout（284 通道） |
| `src/agent/policy.py` | RiichiPolicy（act/sample_with_logp/probs/value/policy_value） |
| `src/model/train_rl.py` | 自对弈 PPO（rollout/ppo_update/main） |
| `src/model/rl_losses.py` | ppo/entropy/gae/value 损失（16/16 单测） |
| `src/model/rl_reward.py` | RewardPredictor GRU / settlement_pt / _round_features |
| `tools/eval_vs_sl.py` | 66% 验收协议 |
| `tools/train_reward_pred.py` | GRU Φ 训练 |
| `checkpoints/sl/rl/` | value_pretrain.pt / reward_predictor.pt / rl_v1*.pt |

## 10. 关键约束与陷阱

1. 引擎不可修改（用户管辖）；obs 无 phase 字段 → 阶段判定靠 legal 结构
2. 旧 ckpt strict=False 加载；新模块（event_attn/ctx_gate）随机/固定初始
3. build_events 的近似时序（座位拼接）是已知缺陷，消融时评估
4. gate 初始 -2（σ≈0.12）；若训练 1000 步后 gate 均值仍 <0.02 → 旁支梯度流异常（检查分层 lr 与注入是否接线）
5. 非法 pass 防御: 声明阶段判定必须用 §6.2 规则（kan-only 明杠 / draw 自杠）
6. GRU Φ 奖励需 reward_predictor.pt 训练完成（train_reward_pred.py 全量 28K 局）
## 11. 评审修复记录（critic 评审 2026-08-26，docs/rl_stage1_review.md）

已修复的正确性缺陷（全部经 smoke 验证）:
1. critic 梯度：ppo_update 重放重算 value（带梯度）；GAE 用 detach 值，critic loss 用带梯度值
2. 熵张量化：归一化混合熵 H_bar 作为张量进 loss（全非法行掩码防 nan）
3. logp 单一数据源：采样与重放共用 policy._logp_for_action/_claim_probs → smoke 断言初始 ratio=1.0 OK
4. 事件接线：rollout 用 build_events_from_game（引擎真实时间线），轨迹存 events（float16）重放
5. 阈值头 RL 期 Bernoulli 采样（评估/部署仍用 act() 阈值）
6. 门控初始 -5 → -2（sigma≈0.12，消除梯度饥饿）
7. 双奖励表统一（TERMINAL_REWARD 对齐 SETTLEMENT_CONFIG）
8. 验收样本量默认 100 局/座（400 局）+ advantage 归一化
已采纳但后置: 输入相关门控、32局池+minibatch、分层lr、对手池、多进程（评审 §4 可砍/后置项）

验证: train_rl --smoke OK（ratio=1.0）| --smoke --event-attn OK（ratio=1.0）
### 7.7 验收补充落地（auditor 二轮验收回应）
- 评估协议 v2：eval_vs_sl.py --games 100 --seeds 3（多种子均值 + 95% CI + 下界判定）
- eval 侧事件接线：play_game 经 policy(game)（__call__ 传 build_events_from_game 真实时间线）
- 集成测试 tools/test_rl_grads.py：声明阶段双选项 logp 一致（auditor 1.26 场景修复验证）、
  critic/policy 梯度非零、初始 ratio=1.0000（全部 PASS）
- smoke 断言：--smoke 增加 abs(ratio-1)<0.05 断言（通过）
- 门控可解释性修正（评审 1.2）：ctx_gate 对应 trunk 256 潜在通道（非输入通道组），
  可解释证据改为「注入前后 head 输出差异」+ gate 分布统计，不再宣称 river/hand 组对应
- 熵对标 SL：训练前先测 SL 策略熵（H_bar_sl），H_target 取 min(0.35, H_bar_sl+0.1)（评审 2.6）