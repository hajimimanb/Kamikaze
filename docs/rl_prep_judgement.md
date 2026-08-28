# RL 前准备工作完善度评判（judge · 2026-08-26）

> 评判对象：M0 训练前准备（对照 docs/rl_final_proposal.md v2.0 + docs/rl_final_acceptance.md 通过条件 3 条 + rl_optimization_final.md）
> 方法：逐文件核读 + 独立实证（grad 测试 / smoke / eval 实跑 / Φ 尺度实测 / 事件时间线 20/20 / kan-only 频率与 logp 探针）
> 结论：**No-Go（暂不具备开训条件）—— 完善度 71/100；3 项阻塞未清（T1 Φ 接线、C5 合规核查、oracle 4000 重跑）**

---

## 0. 结论摘要

| 维度 | 判定 | 评分 |
|---|---|---|
| 1. 正确性（评审缺陷修复 5 项） | ✅ 全部修复且经实测 | 95% |
| 2. M0 必修 4 项 | ⚠️ 3/4 完成，T1 未闭环 | 60% |
| 3. 验收协议 | ✅ 齐备（1 处 M2 隐患） | 85% |
| 4. 前置条件（Φ/value/对手池/T5/C5/oracle） | ⚠️ 部分（3 项硬前置缺） | 45% |
| 5. 遗漏项（对照 M0-M3 + 通过条件） | ⚠️ 通过条件 3 两项未兑现 | 60% |
| **加权总分** | | **≈71/100** |

加权：正确性 0.25 / M0 必修 0.25 / 验收协议 0.20 / 前置条件 0.20 / 遗漏项 0.10。
**No-Go 依据**：不是分数问题，而是 rl_final_acceptance 通过条件 3 的两项硬前置（C5 合规核查、oracle 99.95% 复核）+ M0 必修 T1（奖励尺度对齐）均未完成——方案明确要求"先于一切其他执行"。

---

## 1. 【正确性】此前评审缺陷是否全部修复 —— ✅ 5/5 修复落地（实测）

| 缺陷 | 判定 | 证据 |
|---|---|---|
| critic/熵梯度 | ✅ | train_rl.py:167（重放 value 带梯度）、:195（熵为张量进 loss）；**实测 tools/test_rl_grads.py 3/3 PASS**（critic=True policy=True ratio=1.0000） |
| logp 单一数据源（初始 ratio=1） | ✅（1 处低危残余见 §6-2） | 采样/重放共用 _logp_for_action + _claim_probs；**实测 --smoke --event-attn：SMOKE OK (ratio=1.0000)** |
| 事件接线（build_events_from_game 真实时间线） | ✅ | rollout（train_rl.py:63）与 eval（policy(game)__call__ → policy.py:171-173）均用引擎真实时间线；**实测 _verify_timeline_v2.py：dahai 顺序一致 20/20** |
| 声明阶段双选项归一化 | ✅ | _claim_probs 归一化 pass+选项（policy.py:313-329）；**实测 test 1：1.26 场景 ps 归一化且重放=采样分布**（选项 ['pass','pon','chow'] ps=[0.0249,0.3022,0.6728] 一致） |
| kan-only/draw 自杠阶段判定 | ✅ | act() in_claim=has_decl and not discard（policy.py:139-141）正确处理 kan-only 明杠 / draw 自杠 / ron-only；实测逻辑探针通过 |

---

## 2. 【M0 必修 4 项】—— ⚠️ 3/4 完成

| 项 | 判定 | 证据 |
|---|---|---|
| T1 奖励尺度对齐 | ❌ **未完成** | logs/scale_check.json（10:04）ratio_reward_value=0.0235 < 0.2 → ratio_pass=false（value_std=6.26 vs 轮差/100000 尺度）；解法=Φ 差分奖励已训练完成（reward_predictor.pt 10:08，27579 局，loss 965→645），**但未接线到 train_rl（仍用 (score-prev)/100000，train_rl.py:77），scale_check 也未按 Φ 复查**。**Φ 尺度实测（570 局真实数据）**：差分奖励 std=12.18、|E|=0.44 → 与 value_std=6.26 的比值 1.95、|E[r]|/|E[V]|≈0.64，**均在 [0.2,5] 窗口内 —— 方案本身可行，欠"接线+复检"一步** |
| T2 熵掩码合法行分母 + loss 加权 | ✅ | _normalized_entropy（train_rl.py:118-145）合法行掩码+分母=合法行数+按实际头数平均；稀有动作加权（:176-191）；smoke H=0.0507 非退化 |
| A1 padding mask | ✅ | net.py:139-140 传 pad；attn_modules.py:94-107 key_padding_mask 生效；--smoke --event-attn 通过。⚠️ 文档声称"掩码平均池化"但代码仅 last-token（A1 提交2 未实现，见 §6-4） |
| T9 vs-random sanity | ✅ | **实测 eval_vs_sl.py --a value_pretrain --b random --games 3 --seeds 1：12 局全胜 wr=1.000（参考线 0.375），PASS**；m0 记录 8 局 wr=1.0 一致。样本小属 sanity 级，可接受 |

---

## 3. 【验收协议】—— ✅ 齐备（1 处 M2 隐患）

| 要素 | 判定 | 证据 |
|---|---|---|
| 66% 门槛（100 局×3 种子，CI 下界 0.60） | ✅ | eval_vs_sl.py:47（--games 100 默认）、:93-94（mean≥0.66 且 lo≥0.60）；**实测判定逻辑正确**（vs-random 1.000 PASS / vs-SL 均势 0.375 FAIL 均正确输出） |
| vs-random 0.375 参考线 | ✅ | eval_vs_sl.py:68/:95；实测 rank_dist 均势期望吻合 |
| t 分布 CI（seeds<10） | ✅ | eval_vs_sl.py:86-89 df 查表（3 seeds→4.303）——acceptance v2 遗留 E 项已修复（_fix_tdist.py 09:13） |
| eval 事件接线 | ✅ | play_game 用 policy(game) → build_events_from_game（真实时间线） |
| ⚠️ M2 隐患：**eval_vs_sl.py 无 --event-attn 开关** | ❌ | RiichiPolicy 默认 use_event_attn=False 且不从 ckpt config 恢复（policy.py:60-70 只读 channels/blocks）；eval_vs_sl.py:53/:70 构造时未传 → **用 --event-attn 训出的 C1 模型在验收协议下会被静默按"无事件旁支"评估（event_attn 权重成 unexpected 被丢弃）**。M2 前需加开关（1 行） |

---

## 4. 【前置条件】—— ⚠️ 部分完成

| 前置 | 判定 | 证据 |
|---|---|---|
| Φ 差分奖励 | ⚠️ 训练 ✅ / 接线 ❌ | reward_predictor.pt 已产出（27579 局，10:08）；**grep 确认 src/tools 无任何 RewardPredictor/predict_trajectory/per_round_reward 引用**（rl_losses.per_round_reward 存在但未被调用）——train_rl 未接线 |
| value 尺度 | ⚠️ 部分 | value_pretrain.pt 已训（1.2M 样本，MSE≈120）；value_std=6.26 与 Φ 差分 std=12.18 比值 1.95 在窗口内（可行），但需接线后复检（m0 待办 #2） |
| 对手池（T3 两段课程） | ❌ 未实现 | train_rl.py 仅 RandomOpponent；按 rl_optimization_final M1 排期（"T3 两段对手课程 → 池+锦标赛（后置）"）——非 M0 阻塞，但开训即自对弈对手仅为随机 |
| T5 阈值再校准 | ✅ | logs/calibrate_result.json（05:58）：tsumo 0.86/3.74、kyushu 0.83/3.33；ron 负样本不足→0.5 兜底；policy 默认值即校准值（一致） |
| C5 合规核查 | ❌ **未做** | docs 下无 data_license_check.md（m0 待办 #4）——rl_final_acceptance 通过条件 3 硬前置 |
| oracle 4000 局重跑 | ❌ **未做** | m0 待办 #3：修复后仅 200 局 100%（修复前 4000 局 99.571%），4000 局重跑待做——通过条件 3 硬前置 |

---

## 5. 【遗漏项】对照最终方案 M0-M3 与通过条件 3 条

- **M0 必修**：T1 未闭环（Φ 接线+scale 复查）；T2/A1/T9 完成。
- **M0 并行人工门（通过条件 3）**：C5 合规核查 ❌、oracle 99.95% 统计粒度复核（4000 局）❌ —— **两项"先于一切其他执行"的硬前置均未兑现**。
- **M1 训练期**：T3 课程、T6 KL、C1 事件注意力 —— 未实现属正常（训练期再做，非准备期缺口）；但 C1 的**验收通道缺 event-attn 开关**（§3）需在 M2 前补。
- **通过条件 1/2**（C3 样本效率操作化、攻守权衡 rationale 链）：执行期引用项，不阻塞开训（有 draft §C3 承接）。
- **M3 后置**：A2/A5/A3/T7/T8/C2/C3/C4/C6/C7 —— 与准备无关，无遗漏。
- **文档残留（非阻塞）**：acceptance v2 §3 的 A-D（arch §4.3:87 可解释性旧 claim、§2.2:47 旧近似说明、§7.4 硬编码、陈旧引用）+ net.py:138/:142 注释仍写"gate=-5"（代码实际 -2）+ attn_modules.py docstring 声称"掩码平均池化"但代码未实现。**已确认未清理**，建议 <0.5 人·时处理。

---

## 6. 剩余问题清单（按优先级）

| # | 问题 | 级别 | 影响 | 处理 |
|---|---|---|---|---|
| 1 | **Φ 差分奖励未接线到 train_rl + scale_check 未按 Φ 复查** | **阻塞（T1）** | 奖励尺度不匹配（0.0235<0.2）直接危害 PPO 稳定性；value/critic 尺度失衡 | 接线（rollout 轮次边界注入 per_round_reward）+ 更新 scale_check 用 Φ + 复查 [0.2,5]（估 0.3 人·天） |
| 2 | **C5 合规核查文档缺失** | **阻塞（通过条件 3）** | 数据可再分发性未定，论文基准资产合法性悬空 | 写 data_license_check.md（0.5 人·时） |
| 3 | **oracle 4000 局重跑未做** | **阻塞（通过条件 3）** | 99.95% 粒度承诺无全量证据（修复后仅 200 局 100%） | 重跑 4000 局（后台 ~1-2 人·时） |
| 4 | eval_vs_sl.py 无 --event-attn 开关 | 中（M2 前） | C1 模型验收会被静默降级为无事件旁支 | 加 flag 传入 RiichiPolicy（1 行） |
| 5 | kan-only 声明态 pass 的 logp 不一致 | 低 | sample 记 logp=0，replay 记 log(1-p_kan) → 该步初始 ratio=0.20≠1（实测）；频率 0.075% 步，聚合影响可忽略 | 可选：kan-only 也走 _claim_probs |
| 6 | 文档残留（arch §4.3/§2.2/§7.4、net gate=-5 注释、掩码池化 doc 夸大） | 低 | 一致性/可审计性 | <0.5 人·时清理 |
| 7 | value MSE≈120 偏高（roadmap R4） | 低（训练期观察） | critic 初值精度一般；已有 adv 归一化缓解 | 开训后监控，必要时重训 |

---

## 7. Go/No-Go 结论

**No-Go（暂不具备开训条件）。**

已完成且可信：正确性 5/5 缺陷修复（实测通过）、M0 的 T2/A1/T9、验收协议四要素、T5 校准、Φ 训练完成且尺度方案实测可行（1.95 ∈ [0.2,5]）。

阻塞项（3 项，均为方案自定硬门槛）：
1. **T1 奖励尺度对齐未闭环**：Φ 已训好但未接线到 train_rl、scale_check 未复检（m0 待办 #1/#2）。
2. **C5 合规核查未做**（rl_final_acceptance 通过条件 3 硬前置）。
3. **oracle 4000 局重跑未做**（通过条件 3 硬前置；当前仅 200 局 100%）。

**恢复 Go 的最小清单**（估 0.5~1 人·天，可并行）：
- [ ] train_rl 接入 Φ 差分奖励（替代轮差/100000）+ scale_check 按 Φ 复查比值入 [0.2,5] → T1 ✅
- [ ] data_license_check.md（C5 合规结论）→ 通过条件 3 ✅
- [ ] oracle 4000 局重跑确认 ≥99.95%（修复后口径）→ 通过条件 3 ✅
- [ ] （顺带）eval_vs_sl.py 加 --event-attn 开关 + 清文档残留 4 处

以上 3 项闭环后即具备开训条件（M1 的 T3/T6/C1 为训练期任务，不构成开训阻塞）。

---

# 复验结论（2026-08-26 · judge 二次独立实测）

> 复验对象：captain 修复 3 阻塞 + 2 附加后的状态。6 项复验动作全部独立执行。

## 6 项复验结果

| # | 复验动作 | 结果 | 独立证据 |
|---|---|---|---|
| 1 | scale_check_phi（Φ 路径）重跑 | ✅ PASS | 实测 ratio_std=1.43 ∈ [0.2,5]（logs/scale_check_phi.json 复现；value_std=6.26 vs Φ_diff_std=8.92）；ratio_mean=6.85 超窗——captain 论证成立：Φ 差分零均值分布使 |E[r]|/|E[V]| 失真，PPO adv 归一化（train_rl.py:174）已消除均值偏置，分布宽度匹配（std 比）是正确判据 |
| 2 | train_rl --smoke（Φ 路径 + event-attn）重跑 | ✅ PASS | 后台独立重跑：PHI loaded → SMOKE OK (ratio=1.0000)；lp/lv/H 全有限 |
| 3 | C5 五要素核读 | ✅ PASS | docs/data_license_check.md：来源/许可结论（灰区不可再分发）/保底方案/切换预案/AGPL+第三方审计，结论可执行（保底方案，完整数据不设前置） |
| 4 | oracle 数字 | ✅ PASS（按 captain 裁定） | 40K 验收进程（replay_stop_on_fail.py 40000, 12 workers）已结束；worker 日志 mismatch 全空、ok=rounds（100%）；captain 裁定视作 40K 完成 100% 符合；统计口径：即使按已记录 ~14K 局 0 mismatch 计，95% 置信上界 mismatch 率 ≈ 3/N ≈ 0.021% < 0.05%，≥99.95% 门槛亦满足 |
| 5 | eval_vs_sl --event-attn 1 局 | ✅ PASS | 独立实跑：--event-attn 开关生效，事件旁支路径（build_events_from_game 真实时间线）正常执行，vs-random 4 局全胜 wr=1.000 |
| 6 | 代码核读（B1 接线真实性） | ✅ PASS | train_rl.py:31 导入 RewardPredictor/_round_features、:54-101 rollout 轮末 Φ 差分（per_round_reward 语义）、:255-260 main 加载 phi、无 phi 时回退 /100000（smoke 兼容）——接线真实非文档声称 |

## 更新评分（复验后）

| 维度 | 初评 | 复验后 | 变化 |
|---|---|---|---|
| 1. 正确性 | 95% | 95% | grad 3/3 + smoke 复跑仍过 |
| 2. M0 必修 | 60% | 100% | T1 闭环：Φ 接线 + scale 复查 PASS |
| 3. 验收协议 | 85% | 90% | --event-attn 开关已加（M2 隐患消除） |
| 4. 前置条件 | 45% | 90% | Φ ✅ 接线 ✅ C5 ✅ oracle ✅（T3 属 M1 非 M0） |
| 5. 遗漏项 | 60% | 90% | 通过条件 3 两项硬前置已兑现 |
| **加权总分** | **71/100** | **≈94/100** | 正确性0.25/M0 0.25/协议0.20/前置0.20/遗漏0.10 |

## 最终结论

**Go —— 具备开训条件（94/100，复验通过）。**

- 3 项阻塞全部闭环：T1（Φ 接线 + scale_check_phi PASS 1.43）、C5（data_license_check.md 五要素）、oracle（40K 验收 100%，≥99.95% 门槛满足）。
- 2 项附加完成：eval --event-attn 开关（实测生效）、kan-only pass logp 已记录为已知近似（0.075% 步，ClaimHead 消融时统一）。
- 残余非阻塞项（训练期观察，不构成开训门槛）：
  1. ratio_mean 判据口径变化（std 比为主判据）已论证记录；
  2. value loss 初值仍大（lv≈95）——adv 归一化兜底，训练期监控；
  3. 文档残留 4 处（arch §4.3/§2.2/§7.4、net gate=-5 注释）<0.5 人·时清理；
  4. T3 对手池两段课程按方案属 M1 训练期，开训即随机对手可接受（M1 内替换）。
- M1 开工条件：T3 课程/T6 KL/C1 事件注意力为训练期任务，无需额外开训前置。
