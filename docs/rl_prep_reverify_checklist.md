# RL 开训 Go 条件复验清单（judge · 供 captain 修复对照）

> 用途：captain 按恢复 Go 清单修复后，judge 逐项复验的验收判据（每项给出"证据形态"与"通过标准"）。
> 关联：docs/rl_prep_judgement.md（No-Go 71/100，3 阻塞 + 2 附加）

---

## 阻塞项复验（3 项，全部通过才可 Go）

### B1. T1 Φ 差分奖励接线 + scale 复查
**验收判据（证据形态 = 代码 + 实测输出）：**
- [ ] train_rl.py 奖励路径改用 Φ 差分（每轮 per_round_reward = Φ(x_1..k) − Φ(x_1..k−1)），不再用 (score−prev)/100000（train_rl.py:77 原行删除/替换）
- [ ] rollout 中 Φ 由 reward_predictor.pt 加载并在轮次边界注入（reward 分配时点与轮次对应，逻辑自洽）
- [ ] Φ 缺失/加载失败有明确报错（不静默回退旧奖励）
- [ ] scale_check 更新为按 Φ 差分奖励计算：**ratio_pass=true**（std 比或 |E[r]|/|E[V]| 入 [0.2,5]），logs/scale_check.json 时间戳晚于本次修复
- [ ] 实测（复验时我重跑）：
  - 新 scale_check → ratio ∈ [0.2,5] ✓
  - train_rl --smoke（Φ 路径）→ 有限 loss、无 NaN、ratio≈1 断言通过
- [ ] value/critic 尺度说明：Φ 差分 std≈12.2 vs value_std≈6.3（比值 1.95 已在窗口内，若 value 头未重训需在 m0 进度注明"adv 归一化兜底"）

### B2. C5 合规核查文档
**验收判据（证据形态 = 文档存在 + 内容完备）：**
- [ ] docs/data_license_check.md 存在
- [ ] 含：数据来源（天凤日志？）与许可结论；可再分发性判定（默认保底方案：代码+权重+获取指引+特征级子集）；未过时的切换预案（特征级决策点 / 引擎自对弈数据自有版权）；第三方许可审计状态（mjai.app-main / tenhou-to-mjai / shanten_tmp）；AGPL 防线声明（转换器自写，未引入 mjai.app-main 代码）
- [ ] 结论可执行：写出"合规通过，可发布路径 X"或"暂缓，切保底方案 Y"的明确判定

### B3. oracle 4000 局重跑
**验收判据（证据形态 = 日志 + 数字）：**
- [ ] 4000 局（或 ≥4000）重跑日志存在（含 nagashi 修复后口径）
- [ ] 精度 ≥99.95%（按事件/局/结算粒度，与 rl_final_proposal §4 承诺口径一致）；若低于需写明新数字与处理
- [ ] 结果写入 m0 进度或数据卡草稿

---

## 附加项复验（2 项，不阻塞 Go 但要求在 M2 前闭环/记录）

### A1. eval_vs_sl.py --event-attn 开关
**验收判据：**
- [ ] eval_vs_sl.py 新增 --event-attn 参数并传入 RiichiPolicy（a/b 均可）
- [ ] 加载 event-attn 模型时事件旁支权重被正确加载（非 unexpected 丢弃）——复验时跑 1 局带旁支确认无警告异常
- [ ] （M2 验收 C1 消融的前提，现修复不阻塞开训）

### A2. kan-only 声明态 pass logp 低危项
**验收判据：**
- [ ] 记录到 m0 进度/评审文档（接受为已知近似，注明频率 0.075% 步、初始 ratio=0.20 偏差、影响可忽略）；或顺手修复（kan-only 走 _claim_probs 归一化）
- [ ] 无论修否，注明决策即可（复验时按记录核对）

---

## 复验通过后需同步更新的文档
- [ ] docs/m0_prep_progress.md：T1 行 ⚠️→✅（ratio 新值）、C5 行、oracle 行、待办清空
- [ ] （可选）docs/rl_prep_judgement.md 补记复验结论（Go）

## 我的复验动作（收到完成通知后）
1. 重跑新 scale_check（Φ 路径）→ 核对 ratio_pass
2. 重跑 train_rl --smoke（Φ 路径）→ 有限 loss + ratio≈1
3. 核对 data_license_check.md 内容五要素
4. 核对 oracle 4000 重跑日志数字（≥99.95%）
5. 跑 eval_vs_sl 1 局带 --event-attn 确认事件权重加载
6. 更新完善度评分并给最终 Go/No-Go
