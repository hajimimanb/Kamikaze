# 架构与训练流程优化方案 v2.0（critic 评估后修订）

> 流程: innovator 产出 v1.0 → critic 评估（docs/rl_optimization_critic.md）→ 本版按 §3 修改意见修订 → auditor 终评。

## 0. 决策总览（critic 裁定）
| 候选 | 决策 | 时机 | 工作量（修正后） |
|---|---|---|---|
| T1 奖励尺度对齐+settlement_pt+Phi监控 | 采用（修改） | M0 必修 | 1~1.5 人·天 |
| T2 稀有动作重加权+熵掩码 | 采用（修改） | M0 必修 | 1~1.5 人·天 |
| A1 事件 recency 升级 | 采用（拆分：mask=必修, recency+池化=增强） | M0 | 1~1.5 人·天 |
| T5 阈值再校准循环 | 采用 | M0 | 0.5 人·天 |
| T4 温度/熵课程 | 采用 | M0 | 0.5 人·天 |
| T10 局况条件化 adv 归一化 | 采用 | M0 | 0.5 人·天 |
| T9 评估协议增强 | 采用 | M0 | 0.5~1 人·天 |
| A4 事件 token 完整化（34维OR→~84维） | 采用（修改） | M0（与 A1 合并） | 1.5~2 人·天 |
| T3 对手课程+池+锦标赛 | 采用（拆两段：课程 M1、池+赛后置） | M1 | 0.5~1 + 1 人·天 |
| T6 KL 锚定（人类混合后置） | 采用 | M1 | 0.5~1 人·天 |
| A2 输入相关门控 / A5 ClaimHead / A3 多深度 / T7 整形 / T8 吞吐 | 后置消融 | M3 | — |
| A6 阈值温度学习化 | 放弃（与 T5 二选一选 T5） | — | — |
| T11 hindsight | 暂缓（阶段3 再评估） | 阶段3 | — |
| ③④ 增量（相对座位序等） | 采纳修正清单 | 阶段2 | — |
| 19~23（神谕/反事实/双流/Dueling/league） | 同意放弃 | — | — |
总预算: M0 7~8 + M1 2~2.5 = 9~11 人·天（critic 修正）

## 1. 三个必修正问题（critic §3 裁定，v1.0 错误）
1. 【T1】废除「轮得分差 x 1000」过渡（量级方向错误：正负1e6~8e6 vs value 正负25 差 4 个数量级）。
   改为: 先写尺度核对脚本（--smoke 打印 value/轮奖励/settlement_pt 均值±std），
   |E[r]| 与 E|ΔV| 比值 ∈ [0.2,5]；优先复用已训练 Phi ckpt（reward_predictor.pt），过渡常数仅兜底。
2. 【A4】放弃「3x34 稀疏」（102 维不可行）。改为: 34 维 OR 编码（吃牌 3 张顺子 OR 完全区分 345/456）+
   被叫牌 1 维 + 杠后宝牌翻转 1 维 + 立直时机 1 维 + 一发巡 1 维 → event_dim 46→~84；
   build_events_from_game 与 build_events 双处同一 schema，smoke 断言维度一致。
3. 【T2】熵掩码后归一化分母改为「合法行数」（H_bar = 5 头合法行比例加权），防补零/非法行稀释。

## 2. 采用项实施要点（critic §3 可操作意见摘要）
T1: settlement_pt 原样 ±20~60；Phi 差分=Phi 输出尺度；Phi 监控每 500 局 MSE 对比基线，>30% 触发重训（50%自对弈+50%人类，lr 1e-4 ~500 步）。
T2: loss 加权 w_h=clip(1/稀有度,1,8)（近 2000 局统计，只乘 policy loss 不改采样）；熵掩码+合法行分母；过采样后置。
A1: 提交1 padding mask（key_padding_mask 组合 causal triu + 掩码均值池化，必修）；提交2 巡目/立直后手切数/相对距离通道 + last+mean-pool 拼接（增强）。消融 mask/mask+pool/mask+pool+count 三档。
A4: 34 维 OR 编码 + 4 个标量通道；Linear 46→84 参数 +9.7K；轨迹 events float16 (64,84) 存储 +30%。
T5: 校准集=自对弈自然正负样本；Cfp=5/10；tsumo 正样本<200 保持旧阈值；每 epoch 写 ckpt 元数据；验收降 >2pt 回滚。
T4: 温度只改采样不改 act()；discard 退火到 0.7、binary 保持 1.0；T4③ 作上限约束（主控=alpha 自适应防双重过调）。
T10: 分组键=(顺位,半庄段) 8 组，<50 样本回退全局；位置=GAE 后 policy loss 前；--adv-grouped 默认开。
T9: vs-random 固定 seeds+参考线 0.375；human agreement 固定 200 局面（discard top-1 + 声明一致率），只作趋势。
T3: 两段课程按对 SL 胜率自适应（随机权重=clip(1−wr/0.66,0,1)）；池+锦标赛后置（top-K≤8，mini-tournament 排名，剔除近距 ckpt）。
T6: KL 只对采样头；w=0.1×(1−epoch/E) 衰减；人类混合 BC loss 后置（records→RL action 结构转换 +0.5 人·天）。

## 3. 关键架构发现（critic 裁定为最有价值项）
**③④ 相对座位序**: 日麻鸣牌优先级（下家优先）依赖下家/对家/上家相对序；
纯 3 家全置换等变丢失该结构（Suphx 的 seat invariance 是 cyclic 旋转等变——保持相对顺序，不等价）。
修正: OpponentEquivariant 聚合前加相对座位 one-hot 条件（阶段 2 实施）。
另: ③ 危险度=tenpai×待牌概率分解（可解释+ground truth 直接监督）；④ 加鸣牌反应预测 aux head。

## 4. M0（开训前）拆分：必修正确性 vs 增强
必修正确性（先做，独立验证）: T1 尺度对齐、T2 熵掩码分母、A1-mask padding、T9 vs-random sanity
增强（可消融）: A1-recency+pool、A4 token 完整化、T5/T4/T10、T2 loss 加权、T3 课程
每项独立提交 + 同 seed 验收（100×3 种子，CI 判定），逐项归因到 66% 门槛。

## 5. 创新度诚实评估（critic）
25 候选约 20 个是标准技术正确落地（非研究级新方法）；领域差异化在 ②事件因果注意力（Tjong 无）、
③④麻将表示（相对座位序/副露组成/危险度分解）、阶段3 剥削（麻将空白）。
对 66% 门槛贡献最大的是正确性修复（T1/T2/A1-mask/T9），属「把该做的做对」。

## 6. 放弃项（复核确认）
神谕引导（Suphx 已做）、反事实价值学习（自举风险 D3）、双流/Transformer 主干（Tjong 2024 已做）、
Dueling 头（共享 trunk 已复用）、league 联盟（4 人麻将+66% 门槛不需要）
