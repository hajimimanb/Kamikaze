# RL 阶段 1 训练计划（正式 M1）

> 依据: docs/rl_final_proposal.md（四人终稿）+ docs/m0_prep_progress.md（M0 准备 Go 94/100）+ judge 复验
> 状态: M0 完成，具备开训条件；本计划为正式训练执行规范

## 1. 目标与验收
- 目标: 7 头校准策略 + 事件因果注意力② 的自对弈 PPO 强化，产出 rl_v1_final.pt
- 验收（硬门槛）: eval_vs_sl.py --a rl_v1_final.pt --b transfer_final.pt --games 100 --seeds 3
  win_rate（1位率+0.5×2位率）≥ 0.66 且 95% CI 下界 ≥ 0.60（1 RL + 3 SL 座位轮换）
- 附评估: vs-random sanity（参考线 0.375）、human agreement 趋势（固定 200 局面，仅监控）

## 2. 模型与训练配置
- 模型: MultiHeadRiichiNet(284ch, 256, 50块, 7头+value, use_event_attn=True)
  初始权重: transfer_final.pt strict=False（missing 12 随机）+ value_pretrain.pt 覆盖 value
  ctx_gate 初始 -2（sigma≈0.12）逐通道门控；padding mask 已启用（M0-A1）
- 奖励: GRU Φ 差分奖励（reward_predictor.pt，27,579 局训练；轮末 per_round_reward）
  尺度已核对: std 比 1.43 ∈ [0.2,5]（scale_check_phi PASS）
- PPO: clip 0.2 · GAE γ0.99 λ0.95 · 归一化混合熵（合法行分母）· 稀有动作 loss 加权（w=clip(1/freq,1,8)）
- 优化: AdamW lr 3e-4 · clip_grad 1.0 · warmup 500 → cosine
- 分层 lr（推荐）: event_attn/ctx_gate 3e-4、trunk/heads 3e-5、value 3e-4
- 对手: **自对弈对手池（v1.2，替代 T3 随机课程）**——当前策略(主, 60%) /
  历史 epoch 检查点(25%, 最近 5 个) / SL 基线 transfer_final(15%)；
  依据: AlphaGo-Lee 与随机历史版本对弈、AlphaGo Zero 纯自对弈、Suphx/OpenAI Five 历史版本池；
  随机对手不再用于训练（仅 smoke 与 eval sanity 参考线 0.375）
- 规模: **10 局/epoch × 1000 epoch = 10,000 局**（用户指定）；
  每局即 PPO 更新，epoch=10 局训练+1 条指标；
  评估节奏: **--eval-every 10 epoch（=每 100 局）一次 vs-SL 评估（--eval-games 2 局，
  座位轮换 seat=g%4）+ value 统计**；
  存档: --save-every 100 epoch（=每 1000 局）历史 checkpoint，环形保留 5 个入对手池；
  T5 校准 --t5-every 100 epoch（=每 1000 局）；
  GPU ~33h（吞吐 ~300 局/时）

## 3. 训练流程（每 epoch）
1. 采样: 10 局/epoch（4 家引擎，seat0=本方策略 GPU 推理，
   他家=对手池: 当前策略(自对弈)/历史 checkpoint/SL 基线，按 60/25/15 抽样）；
   每 10 epoch（100 局）执行 vs-SL 评估（2 局，座位轮换）+ value 统计；
   T5 校准每 100 epoch（1000 局）
2. 每局: rollout（Φ 差分奖励 + 真实时间线事件 + 轨迹 obs/action/logp/value/ent/events）→ ppo_update
3. 每 200 局: 进度行打印（avg_p/avg_v/H/ratio/clip）
4. epoch 末: 生成 metrics JSONL（lp/lv/H/ratio/clip_rate/gate_mean/std/value 统计/wr_vs_sl）+ 保存 ckpt
5. 每 2 epoch: T5 阈值再校准（自对弈自然样本，Cfp=5/10，tsumo<200 保持旧阈值）

## 4. 监控与异常（rl_train_panel.html）
- 实时: 状态/进度/卡片 8 项/曲线 4 组/详细表（5s 刷新）
- 异常告警: ratio 偏离[0.8,1.2] / loss NaN / gate 锁死(<0.02) / clip 率>10% / value 尺度 / wr 回退 / 进度停滞(5min) / 进程死亡
- Φ 漂移: 每 500 局天凤日志 MSE 监控，>30% 恶化触发重训（50%自对弈+50%人类，lr 1e-4）

## 5. 里程碑
- M0: 准备（完成，Go 94/100）
- M1a: 短验证（30局×2epoch，训练链路+面板验证）✅ 完成
- M1a+修复: P0 全闭环（T3 课程/Φ 漂移监控/T5 校准/分层 lr/面板 8.4 分）→ 集成重跑验证通过
- M1b: 正式训练（10局/epoch × 1000 epoch = 10,000 局，自对弈对手池；
  评估每 10 epoch × 2 局 vs SL）→ 产出 rl_v1_final.pt + 全 metrics
- M2: 验收（66%+CI 下界 0.60）+ 消融归因（② on/off 等）
- M3: 阶段2（防守读牌③④）+ 后置消融

## 6. 资源与进程管理
- 训练进程: nohup python -u src/model/train_rl.py --games 10 --epochs 1000 --event-attn \
    --eval-every 10 --eval-games 2 --save-every 100 ...
- 面板循环: rl_train_dashboard（5s）；其他 4 面板并行
- 误杀防护: 进程 PID 记录（wmic 查询），不触碰用户 40K 验收进程
- 日志: logs/rl_train.txt（进度）+ rl_train_metrics.jsonl（指标）+ ckpt 每 epoch

## 7. 风险与回退
- R1 训练不收敛（wr 持续 <0.5）: 检查熵/对手池构成/Φ 尺度；Plan B 降级口径（CI 下界>0.50 + L3 优于 SL 2/3 项）
- R2 gate 锁死: 分层 lr 检查；A2 输入相关门控后置消融
- R3 Φ 漂移: 监控触发重训
- R4 进度停滞/进程死亡: 面板告警 → 重启（日志断点续跑需重采样，可接受）