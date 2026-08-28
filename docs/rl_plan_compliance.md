# RL 训练计划 vs 文档 vs 代码符合度评估 + 训练漏洞（compliance）

> 评估人: compliance（计划符合度评估官）· 团队: rl-train-monitor
> 评估时间: 2026-08-26 11:30（M1a 已完成之后，基于当前磁盘代码版本）
> 评估对象: docs/rl_stage1_training_plan.md（执行计划）· docs/rl_stage1_arch.md / rl_spec.md / training_plan.md / rl_roadmap.md（规格文档）· src/model/train_rl.py / rl_losses.py / rl_reward.py / agent/policy.py / model/net.py / attn_modules.py / tools/eval_vs_sl.py / tools/rl_train_dashboard.py（代码）· logs/rl_train_metrics.jsonl / rl_train.txt / checkpoints/sl/rl/*.pt（实测产物）

---

## 0. 总评

**综合符合度: 7.0 / 10（B）** — 核心 PPO 训练链路（奖励/损失/熵/梯度/logp 单一数据源）与计划高度一致且经实测验证；但**计划与代码存在 3 项系统性脱节**（T3 对手课程未实现、Φ 漂移监控完全缺失、T5 阈值再校准未接入训练循环），另有文档间矛盾（H_target 1.5 vs 0.35）与若干代码级漏洞。M1a（链路+面板验证）目的达成，但 **M1b（5×2000 局正式训练）开跑前必须补齐 T3 课程与 Φ 漂移监控**。

| 维度 | 得分 | 一句话结论 |
|---|---|---|
| 1. 数据真实性 | **10/10** | metrics 与日志/ckpt 全链路互证，无伪造 |
| 2. 计划-代码符合度 | **6.5/10** | 核心训练链路 ✓；T3/Φ漂移/T5 校准/warmup/分层 lr 未落地 |
| 3. 文档间一致性 | **6/10** | H_target 矛盾、arch 32局池 vs 计划每局更新、卡片 8 项 vs 5 卡 |
| 4. 训练漏洞 | **6/10** | 见 §3：T3 缺失为首要漏洞，gate 未分化观察，双实例风险 |
| 5. 监控与告警 | **7.5/10** | 8 类告警全可触发；P0② NameError 已修复；缺 Φ 漂移上板 |

---

## 1. 数据真实性 — 10/10（复核通过）

- metrics epoch 0：lp=-0.0367117 / lv=42.2143 / H=0.0863551 / ratio=0.9999972 / gate_mean=0.119184 / wr_vs_sl=0.125 — 与任务给定值逐一吻合（evaluator 已证，compliance 复核 jsonl 原字节一致）。
- 与训练日志互证：`[rl] epoch 0 done: 30 games avg_p=-0.0367 avg_v=42.2143 gate=0.119 wr_vs_sl=0.125 346s`（logs/rl_train.txt）— 完全一致。
- epoch 1 真实新增：`H=0.1527 / lv=29.09 / gate=0.11916 / wr=0.125 / elapsed_s=583.6`，随后日志 `RL_V1 DONE -> checkpoints/sl/rl/rl_v1.pt (718s)`，ckpt 已落盘（config: event_attn=True, games=30, epochs=2, seed=3, thresholds/temps/heads 元数据齐）✓。
- 写入链路：train_rl.py 主循环直接 json.dumps 追加（当前版本另含 kl/lp_disc/lp_bin/reward_mean/reward_std/throughput_gph 字段，M1a 运行版为旧 10 字段版）— 数据源可信。

## 2. 计划-代码符合度矩阵（逐项）

### 2.1 符合 ✓
| 计划项（rl_stage1_training_plan §2/§3/§1） | 代码证据 | 判定 |
|---|---|---|
| MultiHeadRiichiNet(284ch, 256, 50块, 7头+value, event_attn=True) | net.py：channels=256/blocks=50/BINARY_HEADS=7/use_event_attn；feature_channels(full)=284 | ✓ |
| ctx_gate 初始 -2（σ≈0.12） | net.py L122 `torch.full((256,), -2.0)`；实测 gate_mean=0.119 ✓ | ✓ |
| 初始权重 transfer_final + value_pretrain 覆盖 | value_pretrain.pt config: value_base=transfer_final.pt（整网同构 157 keys）；strict=False missing=12 | ✓（等效） |
| GRU Φ 差分奖励（reward_predictor.pt 27,579 局） | train_rl.py L262-267 加载 phi；rollout 轮末 `r=Φ(x_1..k)−Φ(x_1..k−1)`；scale_check PASS | ✓ |
| PPO clip 0.2 / GAE γ0.99 λ0.95 / adv 归一化 | ppo_update L195-197 | ✓ |
| 归一化混合熵（合法行分母） | _normalized_entropy L141-168（discard/ln34 + binary/ln2，合法头平均） | ✓ |
| 稀有动作 loss 加权 w=clip(1/freq,1,8) | ppo_update L199-214（Counter 频率 → clip 1..8） | ✓ |
| critic 带梯度 / 熵为张量 / logp 单一数据源 | ppo_update L189-190/229；smoke 断言初始 ratio≈1（实测 0.999997） | ✓ |
| 事件接线 build_events_from_game（引擎真实时间线） | rollout_game L73 / ppo_update events_t；eval_vs_sl 经 policy(game) 传事件 | ✓ |
| AdamW lr 3e-4 / clip_grad 1.0 | main L279 / L232 | ✓ |
| 验收：eval_vs_sl --games 100 --seeds 3，wr≥0.66 且 CI 下界≥0.60 | eval_vs_sl.py：4 座位轮换 × games/seat × seeds；t 分布临界值（seeds<10）；判定 mean≥0.66 且 lo≥0.60 | ✓ |
| metrics JSONL 字段（lp/lv/H/ratio/clip/gate/value/wr） | main L319-325（当前版另加 kl/奖励/吞吐） | ✓ |
| 每 epoch 保存 ckpt（含 thresholds/temps/heads） | main L363-365 | ✓ |
| M1a = 30 局×2 epoch 链路验证 | 实测 30×2 完成，RL_V1 DONE | ✓ |

### 2.2 不符合/未实现 ✗（按严重度排序）
1. **【高】T3 两段对手课程未实现**：计划 §2 明确"epoch 0-1 随机对手混合（探索）→ epoch 2+ SL(transfer_final) 主对手；切换按对 SL 胜率自适应（随机权重=clip(1−wr/0.66,0,1))"。代码 `opp = RandomOpponent()` 恒定，**全程随机对手**；_sl_ref 仅用于 wr_vs_sl 评估，从不作对手。M1a 恰为 2 epoch（随机阶段）未暴露，**正式 M1b 5 epochs 将全部只有随机对手 → 策略学习目标与验收目标（对 SL 66%）脱节**。
2. **【高】Φ 漂移监控完全缺失**：计划 §4 / arch §7.6 要求"每 500 局天凤日志 MSE 监控，>30% 恶化触发重训（50%自对弈+50%人类，lr 1e-4）"，产出 logs/rl_phi_monitor.json。**代码与面板均无此功能**（文件不存在；train_rl.py 无监控逻辑；面板无对应曲线/告警）。
3. **【中】T5 阈值再校准未接入训练循环**：计划 §3.5"每 2 epoch 再校准（自对弈自然样本，Cfp=5/10，tsumo<200 保持旧阈值）"。train_rl.py 无任何再校准调用（tools/auto_gate.py、calibrate_thresholds.py 为 M0 独立脚本，未接线）。
4. **【中】warmup 500 → cosine 未实现**：计划 §2"优化: AdamW lr 3e-4 · clip_grad 1.0 · warmup 500 → cosine"。代码无任何 lr scheduler（grep 无结果）。arch §11 列为"可后置"，但计划正文仍为执行项 → 文档与实现不一致。
5. **【中】分层 lr 未实现**：计划 §2"分层 lr（推荐）: event_attn/ctx_gate 3e-4、trunk/heads 3e-5、value 3e-4"。代码单一 AdamW 3e-4（无 param groups）。arch 标"推荐，消融可选"，可接受为后置，但**与 §2.2-4 叠加导致事件旁支梯度饥饿风险**（见 §3-2）。
6. **【中】熵自适应未实现 + H_target 文档矛盾**：rl_spec §3 写 H_target≈1.5；arch §7.4 写 H_target=min(0.35, H_bar_sl+0.1) — **两文档矛盾**。代码固定 entropy_coef=0.01，adaptive_entropy_coef 函数存在但从未调用。实测 H=0.086→0.153（归一化混合熵，max≈1），远低于任一目标值。

### 2.3 文档间一致性缺口
- H_target：rl_spec(1.5) vs arch(≤0.35) — 矛盾，需 captain 裁定。
- 更新粒度：arch §7.3"32 局池 + 3 inner epoch × minibatch 512" vs 训练计划 §3.2"每局: rollout → ppo_update"（代码按后者）— arch 为"目标设计"、计划为"执行规范"，arch §11 已声明后置，但两文档并存易误导。
- 卡片数：计划 §4"卡片 8 项" vs 实现 5 卡 10 字段（evaluator 已记录）。
- 进度行粒度：计划 §3.3"每 200 局" vs 实现每 10 局（P0③ 改进，符合"数据越详细越好"）。

## 3. 训练漏洞清单（代码级 + 运行级）

| # | 漏洞 | 严重度 | 证据/影响 |
|---|---|---|---|
| L1 | T3 对手课程缺失（见 §2.2-1） | 高 | wr_vs_sl 实测 0.125（16 局点估，低于 1打3 均势期望 0.375）→ 策略未向 SL 技能方向优化；M1b 前必修 |
| L2 | 事件旁支 gate 未分化（梯度饥饿） | 中高 | ctx_gate 初始 σ(-2)=0.119，60 局后 gate_mean 0.11918→0.11916（几乎不动）、gate_std≈0.0002 → 事件注意力旁支可能未实际学习；无分层 lr + 无 warmup 放大风险；面板 gate 锁死阈值(<0.02)检测不到此状态 |
| L3 | Φ 漂移无监控（见 §2.2-2） | 高 | Φ 是差分奖励地基，OOD 漂移无人发现即奖励信号失真 |
| L4 | 双实例同写文件 | 中 | venv + 系统 python 各跑一份训练/渲染（同写 jsonl/txt/HTML last-write-wins）；当前 _dash_guard.py 亦双实例（PID 47856/69544）→ P1-6 未处理 |
| L5 | 训练日志编码 GBK/ISO-8859 | 低 | rl_train.txt 为 GBK（0xd0 字节）；面板已用 errors="ignore" 兼容；建议训练命令加 PYTHONIOENCODING=utf-8 |
| L6 | wr_vs_sl 样本过小且注释不符 | 低 | 每 epoch 16 局（2 sweep×4 seat×2 game），代码注释"10 局"、面板标题"8 局"均错；0.125 无 CI，易误读 |
| L7 | 轮末奖励时间戳偏移（轻微） | 低 | rollout 把轮末奖励累加到 traj[-1]（本方最后一步），若轮末以他家动作结束则奖励挂到较早 step；稀疏奖励下影响小 |
| L8 | elapsed_s 口径不一致 | 低 | jsonl 207.2s（metrics 构建时点）vs 日志 346s（epoch 全流程含 16 局 vs-SL）— 自适应阈值用 jsonl 差分（M1a 下 376s→thr 564s，实测不误报，可接受） |
| L9 | 训练进程 PID 未记录 | 低 | logs/pids.txt 仅 HTTP/DB_BASH/ACC；计划 §6 误杀防护要求记录 train_rl PID |

## 4. 监控与告警复核（2026-08-26 11:29 代码版本）

- 8 类告警全部可触发（evaluator 伪造实测，compliance 复核代码逻辑：ratio 偏离/NaN/clip/gate 锁死/value 尺度/wr 回退/进度停滞/进程死亡 + 无 metrics 兜底）✓
- **P0② NameError 已修复**：当前 check_alerts 签名 `(pts, last, prog, done=False)`，render 中 `_done` 在调用前计算并传参（L138-139）→ 死亡告警可正常发出 ✓（evaluator 11:25 复核时的版本已更新）
- **P0① 守护+心跳**：_dash_guard.py 存在并运行（心跳写入 logs/rl_train_dash.log，11:27-11:29 连续 ok）✓ 但 guard 自身无自愈、无双实例去重（L4）
- P0③ 进度粒度 gi%10 ✓；P0④ 状态判定（DONE 文本 + wmic PID）✓ 面板实测显示"✅ 完成"
- **缺 Φ 漂移曲线/告警**（P1-1 仍未上板）

## 5. 改进清单

### P0 — M1b（5×2000 局）开跑前必须修
- **P0-A** 实现 T3 两段对手课程：epoch 0-1 随机（保持现有）；epoch 2+ 主对手=SL(transfer_final)，随机权重 clip(1−wr/0.66,0,1) 自适应；并在面板/日志显示当前对手构成（否则训练目标与验收脱节）。
- **P0-B** 接入 Φ 漂移监控：每 500 局对天凤日志算 Φ MSE/相关性 → logs/rl_phi_monitor.json；>30% 恶化触发重训（50% 自对弈 + 50% 人类，lr 1e-4）；面板加曲线 + 告警。

### P1 — M1b 期间应补
- P1-A warmup 500→cosine + 分层 lr（event_attn/ctx_gate 3e-4、trunk 3e-5、value 3e-4）— 缓解 L2 梯度饥饿，让事件旁支真正可学。
- P1-B T5 阈值再校准接入训练循环（每 2 epoch，自对弈自然样本）。
- P1-C 熵自适应落地并裁定 H_target（建议按 arch：min(0.35, H_bar_sl+0.1)，先实测 SL 熵）；面板显示目标/系数。
- P1-D 双实例去重（统一启动脚本 + PID 锁；kill_dupes.ps1 已知问题）。
- P1-E wr_vs_sl 附 Wilson/normal CI + 修正注释/标题（16 局）。

### P2 — 体验与健壮性
- P2-A PYTHONIOENCODING=utf-8 统一日志编码；训练 PID 写入 pids.txt。
- P2-B gate_std 显示精度（%.4f，避免 0.000 误导）；面板"卡片 8 项"规格与实现对齐。
- P2-C guard 自愈 + 双实例检测；wmic → PowerShell Get-CimInstance（新版 Windows 兼容）。

## 6. 结论

M1a 验证目标（训练链路 + 面板监控 + 数据真实）**达成**；核心 PPO 实现正确性（critic 梯度/熵张量/logp 单一源/事件接线）经 smoke 与实测 ratio≈1 确认。但**训练计划与代码存在 3 项系统性脱节（T3 课程、Φ 漂移监控、T5 校准）**，其中 T3 与 Φ 漂移直接影响 M1b 的学习目标与奖励可信度，属 P0 级前置。建议：先修 P0-A/P0-B 再启动正式训练；P1 中优先分层 lr（关系到事件注意力创新 ② 能否生效）。## 7. P0 修复复核（2026-08-26 11:37，compliance 二轮）

> 复核方式：代码审读（当前磁盘版本：train_rl.py 11:36 / rl_train_dashboard.py 11:34）+ 实测渲染。

### 复核结论速览

| 项 | 结论 | 说明 |
|---|---|---|
| P0-A T3 两段对手课程 | **✅ 已实现** | train_rl.py 新增 _SLOpponent + _pick_opponent：随机权重 = clip(1−wr/0.66, 0, 1) 与计划 §2 公式一致；每局按概率选随机/SL 对手；metrics 新增 opp_random/opp_sl 字段；wr_cur 逐 epoch 自适应（L312-381） |
| P0-B Φ 漂移监控 | **❌ 仍未实现** | 代码/面板均无；logs/rl_phi_monitor.json 不存在 —— M1b 前仍需补 |
| P1 分层 lr | **✅ 已实现** | AdamW 三组参数（event_attn/ctx_gate 3e-4、trunk/heads 3e-5、value 3e-4，L299-308） |
| P1 面板混合数据崩溃（t3 指出） | **✅ 已修复** | svg_series L122 已加 `p[1] is not None` 过滤；实测 render OK（pts=2 alerts=0） |
| 面板新曲线 | **✅ 已加** | KL / reward_mean / 吞吐 + head 分项 loss（2 组新曲线，共 6 组） |

### 仍缺失（更新后的缺口清单）
- ❌ Φ 漂移监控（P0，M1b 前必修）
- ❌ warmup 500→cosine（P1）
- ❌ T5 阈值再校准接入训练循环（P1）
- ❌ 熵自适应 + H_target 裁定（P1，文档矛盾 1.5 vs 0.35 未裁定）
- ⚠️ 面板未显示对手构成（opp_random/opp_sl 已在 metrics 但未上板/打印）——T3 课程生效后训练者无法看到当前对手混合比
- ⚠️ 双实例去重、日志编码、wr CI、gate_std 精度（P1/P2 同前）

### T3 实现正确性抽检
- `_pick_opponent`: wr=None（首 epoch）→ w_random=1 → 全随机（探索）✓；wr=0.125 → w_random≈0.81（81% 随机）✓ 符合"wr 低→随机多"设计；wr→0.66 → 全 SL ✓ 与验收目标对齐。
- 注意：当前实现按 wr 自适应而非硬性"epoch≥2 全 SL"，更平滑，与计划"切换按对 SL 胜率自适应"一致 ✓。
- 潜在注意：wr=0.125 时仍 81% 随机，SL 占比仅 19% —— 若 wr 长期低，SL 训练量偏少；可考虑加"SL 占比下限"（如 epoch≥2 时 min 20% SL），供 captain 权衡。

## 8. 最终结论（二轮后）

- 核心 PPO 链路 + T3 对手课程 + 分层 lr + 面板健壮性修复已落地，M1a 数据真实（10/10）。
- **唯一剩余 P0：Φ 漂移监控**（计划 §4 明确要求，训练奖励可信度地基）。
- 建议 M1b 开跑前：① 补 Φ 漂移监控（每 500 局天凤日志 MSE + >30% 恶化重训 + 面板曲线）；② 面板上板对手构成；③ 顺带落 warmup/cosine（P1）；其余 P1/P2 可训练中补。
