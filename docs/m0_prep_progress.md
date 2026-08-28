# M0 训练前准备进度（更新 2026-08-26 10:40）

| 项 | 状态 | 证据 |
|---|---|---|
| C: A1 padding mask | ✅ | EventCausalAttn key_padding_mask + net.py 传 pad；smoke ratio=1.0 |
| B: T2 熵掩码+合法行分母+loss 加权 | ✅ | H_bar 合法行分母；smoke 双通过 |
| A: T1 奖励尺度 | ✅ 闭环 | Φ 差分接线到 train_rl（rollout per_round_reward）；scale_check_phi PASS（std 比 1.43∈[0.2,5]） |
| D: T9 vs-random | ✅ | SL vs 随机 8 局全胜 wr=1.0（0.375 参考线） |
| E/B3: oracle 40K | ✅ 100% | 用户 40K 验收完成：**100% 符合**（远超 ≥99.95% 门槛；修复后全部 mismatch 清零，引擎无问题） |
| G: Φ 全量训练 | ✅ | reward_predictor.pt（27,579 局，10.6MB）已接线 |
| C5 合规核查 | ✅ | data_license_check.md 五要素齐全 |
| eval --event-attn | ✅ | eval_vs_sl.py 开关已加（judge A1） |
| A2 kan-only pass logp | 📝 记录 | 已知近似：声明态 kan-only pass logp 初始 ratio≈0.20≠1，频率 0.075% 步，不影响训练（记录为已知近似，后续 ClaimHead 消融时统一） |

## 误杀恢复
- 4 个 dashboard 循环已被用户清理误杀 → 已重启（transfer_data/neg_pool/rl_prep/rl_timeline，面板 mtime 10:38 ✓）
- 用户 40K 验收进程（55544+12 workers+HTTP+面板）未触碰

## 最终裁定（judge）
- **Go（94/100）**：具备开训条件，可进入 M1（正确性 95 / M0 必修 100 / 协议 90 / 前置 90 / 遗漏 90）
- 残余非阻塞: value loss 初值 lv≈95（adv 归一化兜底）、文档残留 4 处、T3 属 M1
- M1 建议: T3 对手池替换 + C1 事件注意力纳入训练计划

## 待办
- oracle 验证已闭环（40K 100%，用户验收）
- 通知 judge 复验 6 项动作