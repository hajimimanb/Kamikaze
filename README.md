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

## Project Status（项目状态）

> **Frozen at the engineering-validation milestone.** 当前冻结于**工程验证里程碑**。

- **Engineering Validation：Complete**（环境 / 数据管线 / SL / 奖励建模 / 自对弈 PPO / 评估与可复现基础设施 / D0 回归 均达成）
- **Scientific Evaluation：Not completed**（RL 大规模训练、vs-SL 多 seed 统计评估、ablation、Human 行为分析 **未包含在本次发布中**）
- 原因归因：`deferred due to the available compute budget`（计算预算），而非"环境无法训练"。

> **No claim is made that the RL policy outperforms the SL baseline.**
> README 中任何训练方式与历史表现仅作说明，不构成已证实的科研结论。

## 快速开始

### 方式 A：只使用规则引擎（mjai 协议）

引擎不依赖任何训练产物，单独导入即可驱动对局。引擎在 `src/` 下，示例需在仓库根目录用
项目 venv 运行（或 `PYTHONPATH=src`）：`cd 仓库根目录 && .venv/Scripts/python.exe your_script.py`

```python
from env.riichi_game import RiichiGame, RiichiConfig

game = RiichiGame(RiichiConfig(), seed=42)     # 新建一局（东四局制）
while game.phase != "game_end":
    obs = game.state.get_observation()          # 可观察状态（手牌/牌河/副露/宝牌…）；无参=当前决策者
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
- 牌编码：tile136 整数 0–135，满足 `tile = kind×4 + copy`（每种牌 4 个 id，copy=0..3）：
  - kind 0–33：0-8=1m-9m、9-17=1p-9p、18-26=1s-9s、27-30=E/S/W/N、31-33=白/發/中
  - 例：1m=0-3，2m=4-7，…，9m=32-35，1p=36-39，…，7z(中)=132-135
  - 赤宝牌固定 id：赤5m=16、赤5p=52、赤5s=88（kind 4/13/22 的 copy 0）
  - 与 mjai 牌串互转：`mjai_pai(tile)`（如 16→"5mr"）/ `parse_mjai_pai("1m")`

**事件流**：mjai JSON（`start_kyoku`/`tsumo`/`dahai`/`reach`/`chi`/`pon`/`ankan`/`daiminkan`/`kakan`/`dora`/`hora`/`ryuukyoku`/`end_kyoku`/`end_game`；
杠事件按 `ankan`（暗杠）、`daiminkan`（大明杠）、`kakan`（加杠）区分，鸣牌事件含 `consumed`/`from` 字段），
与 [mjai.app](https://github.com/mjai/mjai) 协议对齐，可直接对接外部对局环境。

**默认规则**（`RiichiConfig()`，与天凤逐条 oracle 验证一致）：
- 赤宝牌 ON、喰断（副露断幺）ON、喰替（换吃）OFF
- **双响（double ron）ON**；三响 = 三家和流局（无结算、庄家连庄、立直棒保留）
- **无切上满贯**（kiriage OFF，天凤无切上）；**役满单倍**（无双重役满，含四暗刻单骑/国士十三面）
- 暗杠不可抢（含国士）；仅加杠（kakan）可被枪杠
- 流局：九种九牌 / 四风连打 / 四杠散 / 四家立直 / 三家和；流局满贯（nagashi）ON
- 西入 ON（南四局后全员低于返し 30000 继续西场，庄家过线即止）；连庄制和（agari-yame）ON
- 起分 25000 / 返し 30000
可按规则集覆盖：`RiichiConfig(kiriage_mangan=True, double_ron=False, atamahane=True, double_yakuman=True, ...)`
（`atamahane` 仅在 `double_ron=False` 时生效）。

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

### 方式 E：使用 RL 模型（Kamikaze 自对弈提升版）

加载强化学习模型决策（与引擎对局 / 接入外部环境）。**模型已随仓库分发**（fp16 版，`models/kamikaze_rl_v1_fp16.pt`，~58MB）：

```python
from agent.policy import RiichiPolicy
from env.riichi_game import RiichiGame, RiichiConfig

# 1) 加载 RL 模型（fp16 分发版，自动转 fp32；启用事件注意力旁支）
policy = RiichiPolicy("models/kamikaze_rl_v1_fp16.pt", seed=0, use_event_attn=True)

# 2) 单步决策：
obs = game.state.get_observation()
action = policy.act(obs)                 # 确定性部署（阈值判定）
# 或采样（探索）：action, logp, value, ent = policy.sample_with_logp(obs, temperature=1.0)

# 3) 整局对战（AI 坐 0 位，其余随机）：
game = RiichiGame(RiichiConfig(), seed=42)
while game.phase != "game_end":
    game.step(policy(game) if game.turn == 0 else game.random_action())

# 4) 事件注意力自动接入：policy(game) 传入真实时间线事件
```

### 方式 F：使用 SL 模型（监督学习基线）

SL 模型是仅用**监督学习**训练于天凤牌谱的基线（无 RL 自对弈提升），是 RL 版的起点：

```python
# 加载 SL 基线模型（无事件注意力旁支）
sl_policy = RiichiPolicy("checkpoints/sl/transfer/transfer_final.pt", seed=0, use_event_attn=False)

# 使用方式与 RL 版相同：
#   action = sl_policy.act(obs)                     # 单步
#   game.step(sl_policy(game) if game.turn == 0 else game.random_action())  # 整局
```

**SL 与 RL 版区别**：
- **SL 版**（transfer_final）：从 Tenhou 职业/高段位牌谱逐决策监督学习（切牌/立直/吃/碰/杠分别建模），
  学习人类动作分布——是"模仿人类"的基线，稳定但无自我提升。
- **RL 版**（Kamikaze）：在 SL 版基础上用 **PPO 自对弈强化学习**提升（Φ 全局奖励 + 对称和牌/放铳奖励），
  并学习出自身的打法风格（当前为副露流）。RL 版为当前主推模型；SL 版可作基线对照（评估用 `--b transfer_final.pt`）。

**说明**：SL 模型（transfer_final.pt）文件较大未随仓库分发，需按"方式 B"的 SL 阶段训练产出，
或自行放置到 `checkpoints/sl/transfer/transfer_final.pt`；RL fp16 版已随仓库分发。

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
- 向量化并行（`--vec`，默认 4 局）；对手池 self 60% / past 25%（历史版本）/ SL 15%
- PPO：clip 0.2、GAE(0.99, 0.95)、KL 早停 0.5、稀有动作加权 ≤8×、熵 0.01
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

**模型表现**（4293 局 / 49,544 轮统计）：

| 指标 | 数值 |
|---|---|
| **和牌分布**（模型家和牌方式）| 副露和 **75.0%** / 立直和（门清）23.5% / 默听和 1.5% |
| **副露** | 平均 1.48 次/轮（副露流打法）|
| **和铳率** | 和牌率 19.3% / 放铳率 14.5%（每轮）|
| **平均和点 / 平均铳点** | **5,164 点 / 4,587 点** |
| **vs-SL 平均顺位** | **2.50**（n=728 局，均势 2.5）|
| vs-历史版本平均顺位 | 2.62（n=1,030 局）|

**vs-SL 滑动评估**（独立评估进程，100 局口径）：pt 加权胜率 0.27–0.79、每百局 pt **−1533~+2433**、rank 0.25–0.53——互有胜负、未稳定收敛。

## 目录结构
```
src/   → agent/policy.py, env/riichi_game.py, model/(net/features/rl_reward/train_rl_vec/…), tenhou/, riichi/
tools/ → eval_vs_sl, eval_sliding, train_reward_pred, rl_train_dashboard, start_rl_train, rl_watchdog, …
docs/  → 架构/方案/规则差异/数据许可文档
tests/ → 引擎 oracle 回归测试（28 文件 / 65 个测试函数）
```

## 数据集

**来源**：天凤（Tenhou）公开对局日志（2026-01 日期段，60 个日期文件）。

**规模与用途**：
| 数据集 | 规模 | 用途 |
|---|---|---|
| transfer_records（处理后的逐决策记录）| **27,579 局** | Φ 奖励预测器训练（tools/train_reward_pred.py）|
| Tenhou 原始日志 | 同源 | 数据校验/规则 oracle 验证/评估参照 |

**记录格式**（data/processed/transfer_records/records-*.jsonl.gz，JSON 行 + gzip）：
每行 = 一个玩家的一个决策时刻的完整可观察状态：`seat / round / honba / scores / oya / riichi_sticks / wall_left / dora_indicators / hand / melds / discards / legal_actions / label`（label 为人类实际采取的动作）。

**处理管线**：src/tenhou/（下载 → 解析 mjlog → 校验 → 提取逐决策记录）；
数据不随仓库分发（体积大），需自行按 docs/data_license_check.md 获取天凤日志后重跑管线。

**许可**：天凤日志为公开数据；使用条款详见 docs/data_license_check.md。

## 数据与许可
- 训练数据：天凤（Tenhou）公开对局日志（详见 docs/data_license_check.md）
- 引擎与天凤规则差异（完整差异清单，高优先级项已对齐、剩余低优先项见文档）：docs/tenhou_rules_authoritative.md

## 致谢与参考
- 架构参考：Suphx（arXiv:2003.13590）、mjai 协议、mahjong 库