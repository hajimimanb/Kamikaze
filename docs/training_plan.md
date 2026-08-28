# 模型训练路径规划（SL → RL → 测评）v1

生成 2026-08-25。前置资产：引擎(443 测试+oracle 78/78+30/30 独立验收)、天凤 2026 全年语料(237/237 抓完, 未定稿)、张量化管线(19.2k samples/s)、训练端(5.6k samples/s)、L0-L4 测评工具链。

## 0. 启动前置（corpus final，待用户指令执行）
1. post_finish3 看门狗 → 最终 reexport（MJAI 全知+掩码版, C1/C2 已修代码）→ manifest → 终版 data_report.md → POST_FINISH_DONE；
2. data-engineer 人工核验分区清单（train/val/eval_holdout=20260801 起，零泄漏）；
3. 通知 ml-engineer "corpus final"。

## 1. 监督学习（SL）两阶段课程

### 1.1 张量化（t11 前置）
- preprocess.py：train 分区白名单 → uint8 分片(zstd)+manifest（~6500 万条 / 13 万局，预估 10-20 分钟，10 workers）；
- verify_partition.py：holdout/val 零泄漏审计（已有 smoke 通过）。

### 1.2 阶段a：课程第一阶段（mode=simple，17 通道，仅手牌+宝牌+基础）
- 目标：摸切基线模型；核心指标 tsumogiri_acc（基率 35.2%）、discard top1（随机 9.5%）；
- 建议：blocks=20 轻主干先跑通全流程（~30 分钟），再 blocks=50 全尺寸；
- 超参：AdamW lr=3e-4 warmup 5k + cosine、batch 512-1024、AMP fp16、clip 1.0、epochs 1-2；
- 验收：真实 300 步冒烟已 0.561 top1（历史），正式训练目标 top1 ≥ 55%。

### 1.3 阶段b：课程第二阶段（mode=full，126 通道，全特征多任务）
- 弃牌 CE(34路带掩码) + 立直/吃/碰/杠 4 二元头 BCE（仅在有选择权时监督）；
- epochs 2（先 1 epoch 看收敛）；每 1 万样本打点；每 epoch 后跑 val 评估；
- 验收线：val discard top1 ≥ 50%（目标 55-65% 人类区间）；top5 ≥ 85%；
- 警戒：top1 > 70% 触发 L1 过拟合/泄漏标记（offline_metrics 内置），交 t7 泄漏检查。

### 1.4 SL 期间评估节奏
- 每个 checkpoint（best.pt）：eval/run_eval.py --levels L0,L1（自动，秒-分钟级）；
- 阶段结束：L3 联赛 200 局缩减版 + L2 EV 抽样 100 点；
- 训练看板：checkpoints/train_progress.html（tools/train_dash_loop.py，pythonw 10 秒刷新）实时展示 loss/top1/top5/摸切/各头 Acc/速率/GPU/超参。

## 2. 强化学习（阶段4，SL 基线之后）

1. **自对弈环境**：RiichiEnv 四人半庄（规则已定稿）；向量化多局并行 + GPU 批量推理（目标 ≥1 万局/时）；
2. **算法**：PPO + 熵正则（Suphx 式）；GAMMA 衰减、GAE；对手池=历史 checkpoint 混战防坍缩；
3. **全局奖励预测器**：2 层 GRU(512) 输入当轮+历史轮特征 → 预测终局名次奖励，作密集奖励；
4. **神谕引导（可选开关）**：oracle 可见全牌墙，逐阶段剥离信息迁移到普通代理；
5. **规模**：目标 ≥1 亿环境步；每 2000 万步 checkpoint + 锚定联赛回归（L3, 2000 局/组合）；
6. **监控**：平均顺位/和牌率/放铳率/立直收支/锚定 Elo 曲线（league_history.jsonl → 看板复用）。

## 3. 最终测评（阶段5）
- L0-L3 全自动（t9 工具）；L4 通道A 30 场况 + 通道B 天凤留出集 ≥300 点差异判定（专家打分, 明显失误率 ≤5%）；
- 可解释性：模型输出 top_k+factors+rationale（runner.py 契约, factors 走 riichi.explain）。

## 4. 失败阈值（失败处理协议）
- 连续 3 次 loss NaN/梯度爆炸 → 停；OOM 无法降 batch 解决 → 停；val top1 低于随机基线×2 持续 2 个评估点 → 停；>70% top1 疑似过拟合 → 泄漏审计。

## 5. 可复现性
- 训练 config 随 checkpoint 存 config.json；语料日期/分区规则/依赖锁定（requirements-lock.txt）全记录；种子固定。