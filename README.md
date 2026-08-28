# Kamikaze — Riichi Mahjong AI（神风 · 日麻 AI）

副露流麻将 AI + 可独立部署的日麻规则引擎。
模型 trained on Tenhou logs with supervised learning (SL) and reinforcement learning (RL)。

## 这是什么

**Kamikaze 是一个日麻（立直麻将）项目，包含两大部分**：

1. **日麻规则引擎**（`src/env/riichi_game.py`）：完整实现立直麻将规则（立直/一发/宝牌/杠/流局/高点法），
   规则与天凤（Tenhou）逐条 oracle 验证。可**单独部署**，通过 **mjai JSON 协议**驱动对局，
   接入任何外部决策者（人类、AI、脚本）。
2. **日麻 AI 模型**：基于天凤牌谱训练（SL 迁移 → Φ 全局奖励预测 → PPO 自对弈强化学习），
   当前打法为**副露流**（激进吃碰杠），含完整训练/评估/监控工具链。

## 快速开始

### 方式 A：只使用规则引擎（mjai 协议）

引擎不依赖任何训练产物，单独导入即可驱动对局：

```python
from env.riichi_game import RiichiGame, RiichiConfig

game = RiichiGame(RiichiConfig(), seed=42)     # 新建一局（东四局制）
while game.phase != "game_end":
    obs = game.state.get_observation()          # 可观察状态（手牌/牌河/副露/宝牌…）
    la = game.legal_actions()                   # 合法动作（discard/riichi/pon/chi/kan/ron/tsumo…）
    # 由你的决策者给出动作（mjai 风格字典）：
    action = game.random_action()               # 或 game.step({"type": "discard", "tile": 0})
    res = game.step(action)                     # 推进引擎
    for ev in res["events"]:                    # mjai JSON 事件流
        print(ev)                               # start_kyoku/dahai/reach/pon/chi/kan/hora/ryuukyoku…
```

**动作协议**（mjai 风格 JSON）：
- 切牌/立直：`{"type": "discard", "tile": 0}` / `{"type": "riichi", "tile": 5}`
- 吃/碰/杠：`{"type": "pon", "tiles": [1, 2, 3]}` / `{"type": "chow", "tiles": [...]}` / `{"type": "kan", "tiles": [...]}`
- 和/过：`{"type": "ron"}` / `{"type": "tsumo"}` / `{"type": "pass"}`
- 牌编码：0–135 整数（0=1m…33=9m, 34=1p…67=9p, 68=1s…101=9s, 102=1z…135=7z），
  与 mjai 牌串互转：`mjai_pai(tile)` / `parse_mjai_pai("1m")`

**事件流**：mjai JSON（含 `start_kyoku`/`dahai`/`reach`/`pon`/`chi`/`kan`/`hora`/`ryuukyoku`/`end_game`），
与 [mjai.app](https://github.com/mjai/mjai) 协议对齐，可直接对接外部对局环境。

### 方式 B：训练 AI

```bash
pip install -r requirements.txt
# 1) Φ 奖励预测器（16 维特征，Tenhou 27.5K 局牌谱）
python tools/train_reward_pred.py
# 2) RL 自对弈（20 局/epoch × 500 epoch = 10K 局；单 GPU 约数天）
python tools/start_rl_train.py
```

### 方式 C：评估模型

```bash
# 验收（1200 局：rank 均值≥0.66 且 CI 下界≥0.60 通过）
python tools/eval_vs_sl.py --a checkpoints/sl/rl/rl_v1.pt \
    --b checkpoints/sl/transfer/transfer_final.pt --games 100 --seeds 3 --event-attn
# pt 口径（最近 20 局：pt 加权胜率 + 每百局 pt）
python tools/eval_vs_sl.py --a checkpoints/sl/rl/rl_v1.pt \
    --b checkpoints/sl/transfer/transfer_final.pt --games 5 --seeds 1 --pt --window 20 --event-attn
```

### 方式 D：实时监控面板

```bash
python tools/rl_panel_server.py   # render 循环 + http.server 8090
# 浏览器打开 http://127.0.0.1:8090/rl_train_panel.html
```

## 模型架构

### 主干：MultiHeadRiichiNet（src/model/net.py）
单模型共享主干 + 多决策头（约 2100 万参数，fp32 ~122MB）：

```
输入: 284 通道特征 × 34 牌位（手牌/牌河/副露/宝牌/风/自风/赤宝/危险度等）
  ↓
Stem: Conv1d(284 → 256, kernel=3, padding=1)
  ↓
Trunk: 50 × ResidualBlock(256)         # Conv1d(256,256,3)×2 + ReLU 残差
  ↓  ← ② 事件因果注意力注入（可选，默认开启）
Decoupled Heads:
  ├─ DiscardHead(34)     切牌头: Conv(256→32) + FC(32×34→1024→256→34)
  ├─ BinaryHead × 7      二值决策头: Conv(256→64) + FC(64×34+34→512→2)
  │    ├─ 立直/吃/碰/杠（学习执行时机）
  │    └─ 荣和/自摸/九种九牌
  └─ ValueHead(1)        价值头（PPO critic）
```

### ② 事件因果注意力（src/model/attn_modules.py）
- 8 头自注意力：输入公开事件序列（64 事件 × 46 维），因果掩码只看过去
- 门控注入 `h += inj_scale(0.02)×σ(gate)×tanh(ctx)`，防 logit 放大失控

### Φ 全局奖励预测器（src/model/rl_reward.py，Suphx 式）
- GRU 2 层(512)+2 FC；输入 16 维轮级特征（记分板 12：得分/累计/庄/本场/棒/风/夹取/标志 + 手牌 4：向听/听牌/宝牌/副露）
- 标签 = 终局精算点数 settlement_pt；奖励 = Φ(前缀k) − Φ(前缀k−1) 差分分配

## 训练方法

### 奖励结构（全部 pt 量纲，热可调）
```
r = Φ差分(局面→最终pt)
  + 和牌轮: +0.5(基础奖) + min(20, 0.5×打点/1000)     # 高打点充分区分
  − 被铳轮: −0.5(基础罚) − min(20, 0.5×铳点/1000)     # 与和牌对称
```

### PPO 自对弈（src/model/train_rl_vec.py）
- 向量化并行 8 局；对手池 self 60% / past 25%（历史版本）/ SL 15%
- PPO：clip 0.15、GAE(0.99, 0.95)、KL 早停 0.3、稀有动作加权 ≤4×、熵 0.06
- bias-snap 归中防头冻结/两极分化；head_z_cap 软限幅防失控
- always_win 阶段：荣和/自摸强制（先学"能和就和"）

### 热干预（免重启）
编辑 `logs/rl_hyper.json`（每 ~10s 应用）：奖励 6 参数 / 学习率 / bias / snap_exclude / head_z_cap / always_win / reset_heads / pause / quit / 评估口径。

## 训练进度与表现（截至 2026-08-28）
> ⚠️ **因资源和时间限制，当前仅训练至计划一半**（单 GPU）。

| 项目 | 数值 |
|---|---|
| 计划 / 已完成 | 10,000 局 / 500 epoch；**4,293 局 / 214 epoch（约 43%）** |
| 稳定性 | ratio≈1.000、KL 0.001–0.14、clip <5%（无失控）|

**vs-SL**（vs transfer_final 1v3，滑动评估采样）：pt 加权胜率 0.27–0.79、每百局 pt **−1533~+2433**、rank 0.25–0.53——互有胜负、未稳定收敛。

**当前打法（副露流）**：副露 1.48 次/轮、门清吃 75–85%、碰 43–56%、杠 12–81%、立直 60–90%、门清默和仅 1.5%、和率 19.3%/放铳 14.5%。

## 目录结构
```
src/   → agent/policy.py, env/riichi_game.py, model/(net/features/rl_reward/train_rl_vec/…), tenhou/, riichi/
tools/ → eval_vs_sl, eval_sliding, train_reward_pred, rl_train_dashboard, start_rl_train, rl_watchdog, …
docs/  → 架构/方案/规则差异/数据许可文档
tests/ → 引擎 oracle 回归测试（244 项）
```

## 数据与许可
- 训练数据：天凤（Tenhou）公开对局日志（详见 docs/data_license_check.md）
- 引擎与天凤规则差异（12 处，oracle 验证）：docs/tenhou_rules_authoritative.md

## 致谢与参考
- 架构参考：Suphx（arXiv:2003.13590）、mjai 协议、mahjong 库