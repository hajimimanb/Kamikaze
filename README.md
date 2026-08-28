# Riichi Mahjong AI（日麻 AI）

基于天凤（Tenhou）真实对局数据，自研引擎 + 深度学习模型的日麻 AI 决策系统。
核心路线：**Tenhou 数据 → 特征工程 → SL 迁移 → Φ 全局奖励预测 → PPO 自对弈强化学习 → 综合评估**。

## 特性

- **自研规则引擎**（`src/env/riichi_game.py`）：完整日麻规则（立直/一发/宝牌/杠/流局/高点法），与 Tenhou 真实对局 oracle 验证（12 处规则差异已文档化）
- **深度模型**（`src/model/net.py`）：284 通道特征 → 50 层 Residual 主干 → 切牌头(34) + 7 个二值决策头（立直/吃/碰/杠/荣和/自摸/九种九牌）+ 价值头；可选 **②事件因果注意力旁支**（门控注入，防 logit 放大失控）
- **Φ 全局奖励预测**（Suphx 式）：GRU 预测终局精算点数（settlement_pt），16 维轮级特征（记分板 12 + 手牌 4：向听/听牌/宝牌/副露），差分奖励分配
- **PPO 自对弈**：向量化并行对局、GAE、minibatch、KL 早停、稀有动作加权、动态 bias 归中（防决策头冻结/两极分化）、历史版本对手池（虚构自对弈）
- **奖励结构**：Φ 差分 + 对称事件奖励（基础和牌奖 + 打点缩放封顶 20 / 基础放铳惩罚 + 铳点缩放封顶 20，pt 量纲）
- **热干预系统**：`logs/rl_hyper.json` 实时调整全部超参数（奖励/学习率/温度/bias/软限幅/开关），零重启
- **综合评估**：vs-SL 验收协议（rank 胜率 + pt 加权胜率 + 每百局 pt）、独立滑动评估进程、实时监控面板

## 目录结构

```
├── src/
│   ├── agent/policy.py        # 策略封装（act/采样/logp/软限幅/强制和牌）
│   ├── env/riichi_game.py     # 日麻引擎（用户自有，勿改）
│   ├── model/
│   │   ├── net.py             # 多头残差网络 + 事件注意力
│   │   ├── features.py        # 284 通道特征
│   │   ├── rl_reward.py       # Φ 奖励预测器 + settlement_pt
│   │   ├── train_rl_vec.py    # 向量化 PPO 自对弈训练器（主）
│   │   ├── train_rl_mp.py     # PPO 更新核心（GAE/损失/bias 归中）
│   │   ├── train_rl.py        # 单进程训练器 + 对手池 + 熵
│   │   └── train_sl*.py       # SL 训练/迁移
│   └── tenhou/                # Tenhou 日志解析/数据管线
├── tools/
│   ├── eval_vs_sl.py          # 验收评估（--pt 模式）
│   ├── eval_sliding.py        # 独立滑动评估进程
│   ├── train_reward_pred.py   # Φ 训练
│   ├── rl_train_dashboard.py  # 监控面板
│   ├── start_rl_train.py      # 训练启动脚本
│   ├── rl_watchdog.py         # 训练守护
│   └── archive_rl_logs.py     # 日志归档
├── docs/                      # 架构/方案/规则差异文档
├── requirements.txt
└── README.md
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
#   或手动：
python src/model/train_rl_vec.py --vec 8 --games 20 --epochs 500 --event-attn     --ckpt checkpoints/sl/rl/rl_v1_resume_ep51.pt --win-bonus 0.5 --base-win-bonus 0.5     --base-deal-penalty 0.5 --deal-penalty 0.5 --bias-snap 1 --snap-exclude tsumo,ron,riichi,kyushu
```

### 热干预（免重启）

编辑 `logs/rl_hyper.json`（训练每 ~10s 检查并应用）：
- 奖励：`win_bonus`/`base_win_bonus`/`win_bonus_cap`/`base_deal_penalty`/`deal_penalty`/`deal_penalty_cap`/`riichi_cost`
- 学习率：`lr_head`/`lr_gate`/`lr_value`（`lr_schedule=fixed` 时生效）
- 决策：`bias`/`snap_exclude`/`head_z_cap`/`head_entropy`/`always_win`/`reset_heads`
- 运行：`action`（run/pause/quit）、`wr_mode`（rank/pt）

## 评估

```bash
# 验收（1200 局：--games 100 --seeds 3；判定 rank 均值≥0.66 且 CI 下界≥0.60）
python tools/eval_vs_sl.py --a checkpoints/sl/rl/rl_v1.pt     --b checkpoints/sl/transfer/transfer_final.pt --games 100 --seeds 3 --event-attn

# pt 口径（最近 20 局：pt 加权胜率 + 平均每百局 pt）
python tools/eval_vs_sl.py --a checkpoints/sl/rl/rl_v1.pt     --b checkpoints/sl/transfer/transfer_final.pt --games 5 --seeds 1 --pt --window 20 --event-attn
```

- **rank 胜率** = P(1位) + 0.5×P(2位)（1v3 均势 0.375，目标 0.66）
- **pt 加权胜率** = Σ胜局pt / (Σ胜局pt + Σ负局pt)（均势 0.5）
- **每百局 pt** = 平均 settlement_pt × 100（零和均势 0）
- 独立滑动评估进程（`eval_sliding.py`）每 epoch 自动评估并记录趋势

## 监控面板

```bash
python tools/rl_panel_server.py   # render 循环 + http.server 8090
# 浏览器打开 http://127.0.0.1:8090/rl_train_panel.html
```
面板含：训练指标曲线、决策头执行率、对 SL/历史版本胜率、进步分析、Φ 漂移监控、并行评估（20 点滑动平均）。

## 数据与许可

- 训练数据来自天凤（Tenhou）公开对局日志（详见 `docs/data_license_check.md`、`docs/data_report.md`）
- 引擎与 Tenhou 规则差异（12 处，oracle 验证）见 `docs/tenhou_rules_authoritative.md`

## 致谢与参考

- 架构参考：Suphx（arXiv:2003.13590）、[mjai 协议](https://github.com/mjai/mjai)、mahjong 库
