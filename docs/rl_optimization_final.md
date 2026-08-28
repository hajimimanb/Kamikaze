# RL 阶段1 优化创新——四人综合最终方案（2026-08-26）

> 四人: captain（综合）· innovator（方案）· critic（评估裁定）· auditor（终评 92/100 通过）
> 依据: docs/rl_optimization_proposal.md（v2.0）/ rl_optimization_critic.md / rl_optimization_acceptance.md

## 1. 综合结论
- 方案成立可执行：critic 裁定 10 采用 / 5 后置消融 / 3 放弃 / 1 暂缓；auditor 终评 92/100 通过（有条件）
- 三轴评分: 可行性 90 / 创新度 70（诚实评估）/ ROI 88
- 核心定位: 25 候选中 ~20 项是标准技术正确落地；领域差异化在 ②事件因果注意力 + ③④ 麻将表示 + 阶段3 剥削

## 2. 最终实施路线
### M0 开训前（必修正确性，先行独立验证）
1. T1 奖励尺度对齐（尺度核对脚本 + 比值 0.2~5 验收 + Phi ckpt 优先 + Phi 监控）
2. T2 熵掩码合法行分母 + loss 加权
3. A1-mask padding mask（+掩码均值池化）
4. T9 vs-random sanity（0.375 参考线）
执行规范: 每子任务引用 critic §3 对应小节号（T1→§3-T1 等），独立提交 + 同 seed 验收（100×3 种子 CI）

### M0 增强（可消融，逐项叠加）
A1-recency+pool、A4 token 完整化（34维OR→~84）、T5 阈值再校准、T4 温度/熵课程、T10 局况条件化 adv 归一化

### M1 训练期
T3 两段对手课程（按对 SL 胜率自适应）→ 池+锦标赛（后置）；T6 KL 锚定（人类混合后置）

### M2 验收
66% 门槛（eval_vs_sl v2: 100局×3种子, CI 下界≥0.60）+ 消融归因（每项 on/off）

### M3 后置消融
A2 输入相关门控 / A5 ClaimHead / A3 多深度注入（先单点扫描）/ T7 潜在整形 / T8 吞吐（4座同批先行）

## 3. 关键架构发现（写入阶段 2 设计）
- ③④ 相对座位序: 全置换等变丢失下家/对家/上家结构（Suphx 是 cyclic 旋转等变）→ 聚合前加相对座位 one-hot
- ③ 危险度 = tenpai x 待牌概率分解（可解释 + ground truth 直接监督）
- ④ 鸣牌反应预测 aux head

## 4. 放弃项（最终确认）
神谕引导（Suphx 已做）/ 反事实价值学习（自举风险）/ 双流 Transformer（Tjong 2024）/ Dueling / league / A6 阈值学习化（T5 替代）

## 5. 工作量与里程碑
M0 7~8 人·天 + M1 2~2.5 人·天 = 9~11 人·天；GPU 训练 5x2000 局（吞吐工程后 3~15 GPU·时）

## 6. 剩余问题（低优先级，不阻塞，从 critic §3 继承）
R1 value scale 兜底 / R2 ratio 修正细节 / R3 ckpt 元数据字段 / R4 人类混合比例 / R5 锦标赛 K 值 / R6 校准集样本量 / R7 T2 过采样后置 / R8 吞吐异步化

## 7. 建议执行方式（auditor）
M0 任务书注明「critic 文档为完整实施规范」，每子任务引用 critic §3 小节号；必修 4 项先行，增强逐项叠加，每项独立归因到 66% 门槛。
