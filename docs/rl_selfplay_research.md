# 自对弈对手设计调研（v1.2 依据）

> 用户决策 2026-09：增加 epoch 数、减少每 epoch 局数；去掉训练期随机对手，
> SL 预热后直接进入正式模型自对弈强化学习。本文档为文献依据与落地记录。

## 1. 文献事实（"自对弈是什么和什么对弈"）

### AlphaGo（Nature 2016, Silver et al.）
- 三阶段：SL 策略网络（人类棋谱 3000 万步）→ RL 策略网络（策略梯度）→ 价值网络。
- RL 自对弈：**当前策略与随机抽取的"过去版本"的自己对弈**
  （"the current policy network plays against a randomly selected previous version of itself"）。
- 随机棋手从未作为正式训练对手（仅最低难度/测试）。

### AlphaGo Zero（2017, DeepMind 博客 / Nature 2017）
- 无人类数据、无 SL 预热：**当前（最强）网络与自己下棋**，4.9M 局从零自对弈。
- "自对弈"最纯粹的定义：同一网络（的最强版本）生成对局双方。

### OpenAI Five（Dota 2）
- 主 agent 与**自己的过去版本**（past versions）对战（fictitious self-play），
  扩展为联赛制（league），防策略循环与灾难性遗忘。

### Suphx（微软，arXiv:2003.13590，日麻）
- SL 预热 2300 万局天凤牌谱 → RL 微调；RL 阶段与**自己的历史版本**对弈
  （与 AlphaGo-Lee 一脉相承）。中文解读: 澎湃 https://www.thepaper.cn/newsDetail_forward_6861899

### 综述/其它
- Distributed DRL survey（arXiv:2212.00253）：naive self-play（相同权重互打）
  是最常见基线；多人博弈成熟系统均加历史版本/对手池防坍缩。
- riichimahjong.net 强 AI 汇总：日麻 AI 均从 SL 权重出发自对弈，无"先打随机"课程。

## 2. 结论
- 随机对手的合法用途：eval sanity 参考线（我们 --b random，0.375）、从零起点（不需要）。
- 训练对手从 SL 权重出发直接自对弈：主=当前策略 θ 坐满 4 座（seat0 采样训练、
  他家阈值贪心 act）＋历史 epoch checkpoint 池（虚构自对弈）＋ SL 基线低权重保留。

## 3. 落地（v1.2）
- src/model/train_rl.py: OpponentPool（self 60% / past 25% / sl 15%，k=5）；
  默认 10 局/epoch × 1000 epoch（总 10,000 局，用户指定）；
  vs-SL 评估每 10 epoch（100 局）× 2 局（座位轮换）；历史检查点每 100 epoch 存档、
  环形保留 5 个并入对手池。
- 验收口径不变：vs SL ≥66% 且 95%CI 下界 ≥0.60。
- 面板对手构成改为 自/史/SL 三列（rl_train_dashboard.py）。

## 4. 验证
- smoke --event-attn: ratio=1.0000 OK（243 步，1 局+PPO 更新）。
- 对手池抽样分布单测：空池 self/sl=63/37；满池 self/past/sl=60.3/25.1/14.7 ✓
- 全部变更 py_compile OK。
