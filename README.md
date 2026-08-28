# Kamikaze — Riichi Mahjong AI（神风 · 日麻 AI）

副露流麻将 AI：trained on Tenhou logs with supervised learning (SL) and reinforcement learning (RL)。

基于天凤（Tenhou）真实对局数据训练、自研引擎 + 深度模型的日麻决策系统。
核心路线：**Tenhou 牌谱 → 特征工程 → SL 迁移 → Φ 全局奖励预测 → PPO 自对弈强化学习 → 综合评估**。

## 模型架构

### 主干：MultiHeadRiichiNet（src/model/net.py）
单模型共享主干 + 多决策头设计（约 2100 万参数，fp32 权重 ~122MB）：

```
输入: 284 通道特征 × 34 牌位（含手牌/牌河/副露/宝牌/风/自风/赤宝/危险度等）
  ↓
Stem: Conv1d(284 → 256, kernel=3, padding=1)
  ↓
Trunk: 50 × ResidualBlock(256)         # Conv1d(256,256,3)×2 + ReLU 残差
  ↓  ← ② 事件因果注意力注入（可选，默认开启）
Decoupled Heads:
  ├─ DiscardHead(34)     切牌头: Conv(256→32) + FC(32×34→1024→256→34)
  ├─ BinaryHead × 7      二值决策头（每头独立）: Conv(256→64) + FC(64×34+34→512→2)
  │    ├─ 立直 riichi / 吃 chow / 碰 pon / 杠 kan
  │    └─ 荣和 ron / 自摸 tsumo / 九种九牌 kyushu
  └─ ValueHead(1)        价值头: Conv(256→32) + FC(32×34→512→1)   # PPO critic
```

### ② 事件因果注意力（src/model/attn_modules.py）
- 8 头自注意力，输入**公开事件序列**（64 事件 × 46 维：打牌/副露/立直/宝牌等公开信息）
- **因果掩码**：只看过去事件；取最后 token 隐状态作为全局上下文 (B, 256)
- **门控注入**：h += inj_scale(0.02) × σ(gate) × tanh(ctx)——逐通道门控，防 logit 放大失控

### Φ 全局奖励预测器（src/model/rl_reward.py，Suphx 式）
- GRU 2 层(512) + 2 层 FC，输入**16 维轮级特征**：
  - 记分板 12 维：本轮得分/40000、累计分/100000、庄位、本场、立直棒、场风/半庄、相对均分、得分夹取、和牌/立直/被铳标志
  - 手牌 4 维：向听数/3、是否听牌、宝牌数/8、副露数/4
- 标签 = 终局精算点数 settlement_pt（uma 20/10/−10/−20 + 分数差/1000 + 供托）
- 奖励：r_k = Φ(前缀k) − Φ(前缀k−1)（差分奖励分配，Suphx Eq.4 逐前缀监督）

## 训练方法

### 奖励结构（全部 pt 量纲，热可调）

```
r = Φ差分(局面→最终pt)
  + 和牌轮: +0.5(基础奖) + min(20, 0.5×打点/1000)     ← 高打点充分区分
  − 被铳轮: −0.5(基础罚) − min(20, 0.5×铳点/1000)     ← 与和牌对称
```

### PPO 自对弈（src/model/train_rl_vec.py）
- **向量化并行**：8 局并行，seat0 采样训练、其他座位贪心（当前/历史/SL 对手）
- **对手池**：self 60% / past 25%（历史版本虚构自对弈）/ SL 15%
- **PPO**：clip 0.15、GAE(γ=0.99, λ=0.95)、minibatch 512、KL 早停 0.3、稀有动作加权(≤4×)、熵 0.06
- **bias-snap 归中**：chow/pon/kan 每池 bias=−median(z)（防头冻结/两极分化）；tsumo/ron/riichi 排除可自由饱和
- **软限幅** head_z_cap：riichi 12 / chow 20 / pon 20 / kan 40（|z| 超限梯度衰减）
- **阶段约束** always_win：tsumo/ron 强制和牌（头冻结，先学"能和就和"）

### 热干预（免重启）
编辑 logs/rl_hyper.json（训练每 ~10s 检查应用）：奖励 6 参数 / 学习率 / bias / snap_exclude / head_z_cap / always_win / reset_heads / pause-quit / wr_mode。

## 训练进度与表现（截至 2026-08-28）

> ⚠️ **当前模型因资源和时间限制，仅训练至计划的一半**（单 GPU 环境）。

| 项目 | 数值 |
|---|---|
| 计划总量 | 10,000 局 / 500 epoch（20 局/epoch）|
| 已完成 | **4,293 局 / 214 epoch（约 43%，近半）** |
| 训练稳定性 | ratio≈1.000、KL 0.001–0.14、clip <5%、熵 0.12–0.16（无失控）|

**vs-SL 表现**（vs transfer_final，1v3；滑动评估 100 局口径采样）：
| 指标 | 数值 |
|---|---|
| pt 加权胜率 | 0.27 – 0.79（震荡，均势 0.5）|
| 每百局 pt | **−1533 ~ +2433**（最佳评估点 +2433）|
| rank 胜率 | 0.25 – 0.53（均势 0.375）|

**当前打法（副露流）**：
| 行为 | 数据 |
|---|---|
| 副露率 | **1.48 次/轮**（激进吃碰杠）|
| 门清吃 / 副露吃 | 75–85% / 25–38% |
| 碰 | 43–56% |
| 杠 | 12–81%（近期收敛偏低）|
| 立直 | 60–90% |
| 门清默和 | **仅 1.5%**（几乎不门清默和——副露流的典型特征）|
| 和牌率 / 放铳率 | 19.3% / 14.5% |

**说明**：副露流是"奖励（快和有利）+ 环境（自对弈对手防守弱）+ 阶段（强制和牌）"共同作用的结果，属当前训练阶段的学习产物；vs-SL 已可互有胜负但未稳定收敛，完整训练与后续阶段（解锁见逃/防守读牌）仍在计划中。

## 目录结构

```
src/  → agent/policy.py, env/riichi_game.py, model/(net/features/rl_reward/train_rl_vec/train_rl_mp/train_rl/train_sl*), tenhou/
tools/ → eval_vs_sl, eval_sliding, train_reward_pred, rl_train_dashboard, start_rl_train, rl_watchdog, archive_rl_logs
docs/  → 架构/方案/规则差异文档
requirements.txt / README.md
```

## 安装

```bash
pip install -r requirements.txt
# PyTorch（RTX 50 系需 cu130 构建，见 requirements.txt 注释）
```

## 训练

```bash
# 1) Φ 奖励预测器（16 维特征，Tenhou 27.5K 局）
python tools/train_reward_pred.py

# 2) RL 自对弈（20 局/epoch × 500 epoch = 10K 局）
python tools/start_rl_train.py
```

## 评估

```bash
# 验收（1200 局；判定 rank 均值≥0.66 且 CI 下界≥0.60）
python tools/eval_vs_sl.py --a checkpoints/sl/rl/rl_v1.pt \
    --b checkpoints/sl/transfer/transfer_final.pt --games 100 --seeds 3 --event-attn

# pt 口径（最近 20 局：pt 加权胜率 + 平均每百局 pt）
python tools/eval_vs_sl.py --a checkpoints/sl/rl/rl_v1.pt \
    --b checkpoints/sl/transfer/transfer_final.pt --games 5 --seeds 1 --pt --window 20 --event-attn
```

- **rank 胜率** = P(1位) + 0.5×P(2位)（1v3 均势 0.375，目标 0.66）
- **pt 加权胜率** = Σ胜局pt / (Σ胜局pt + Σ负局pt)（均势 0.5）
- **每百局 pt** = 平均 settlement_pt × 100（零和均势 0）

## 监控面板

```bash
python tools/rl_panel_server.py   # render 循环 + http.server 8090
# 浏览器打开 http://127.0.0.1:8090/rl_train_panel.html
```

## 数据与许可

- 训练数据来自天凤（Tenhou）公开对局日志（详见 docs/data_license_check.md）
- 引擎与 Tenhou 规则差异（12 处，oracle 验证）见 docs/tenhou_rules_authoritative.md

## 致谢与参考

- 架构参考：Suphx（arXiv:2003.13590）、mjai 协议、mahjong 库
