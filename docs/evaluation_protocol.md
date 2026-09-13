# 阶段5 测评协议（专家打分版）v1

本文档定义最终测评的两条通道与可解释性输出规范，供模型开发、场景生成与专家评审使用。

## 1. 通道A：人工场况查验（>=20 场况）

由系统随机生成或按类别配额生成 20+ 个局面（每个局面含完整观测 JSON，按 docs/observation_schema.md §2 格式）。类别配额（至少每类 1 题）：

1. 防守判断：对手立直后，手牌无役/低价值，需在危险牌与完全弃和之间选择（含现物/筋/壁/早外分析）。
2. 进攻听牌：两向听 vs 危险度中等，攻守判断。
3. 立直判断：门前听牌但待牌危险/役种价值低（愚形 vs 好形、有无断幺/平和）。
4. 鸣牌判断：碰/吃后手牌价值变化（向听推进 vs 门清价值损失）。
5. 役满/大牌机会：国士/清一色/三色路线取舍。
6. 点数状况：TOP 位保位 vs 末位追分（危险进攻的 EV）。
7. 濒临飞人：负分边缘的对攻判断。
8. 剩余巡数：牌山<=20 张时的巡目与安全度权衡。
9. 宝牌处理：宝牌对子/边张的保留与拆解。
10. 赤宝/门清混合型：染手 vs 断幺速攻。

每题的模型输出格式（可解释性硬性要求）：
```
{
  "action": {"type":"discard","tile":16},
  "top_k": [{"action":{"type":"discard","tile":16},"p":0.31,"factors":{"shanten":2,"ukeire":12,"danger":0.2,"ev":1200,"speed":0.7}}],
  "rationale": "切赤5万: 保持一向听, 进张12张; 对下家为筋牌较安全; 期望~1200点"
}
```
专家按 1-5 分对动作质量与解释质量分别打分。

## 2. 通道B：与天凤专家对局的策略相似度对比

> v1.1（用户决策 2026-08-25）：弃用雀魂数据，通道B 基准改为天凤留出集。

- 从天凤 eval-holdout 分区（C:/agentwork/data/processed/tenhou/records.jsonl 中预留、永不进 SL 训练的凤凰桌留出集，约 10-15 万局）随机抽取 >=300 个决策点（去重玩家/对局，保证多样性；尽量覆盖上述 10 类）。
- 对每个决策点，用模型对相同观测给出策略（action + top_k + rationale + factors）。
- 不要求与人类动作相同。专家对每个差异点判定：模型动作 (a) 更优/等价 (b) 可理解但略差 (c) 明显失误，并记录理由。
- 汇总指标：一致率(完全同动作)、专家净胜率(模型更优数 - 失误数)/总样本、分类得分（按 10 类分别统计）。
- 底线：模型明显失误率 <=5%，否则视为评估不通过（对应失败协议"评估异常"）。

## 3. 可解释性实现约束
- factors 与 rationale 必须是决策的忠实归因（规则因子计算 + 模型注意力辅助），不允许事后拼凑。
- 因子计算器 src/riichi/explain.py（engine-developer 实现）作为唯一因子来源，模型输出引用其键名。
- 模型输出 JSON 结构与本文档一致，测评脚本直接解析。

## 4. 离线附加指标（非专家通道，供迭代参考）
- 留出牌谱决策一致率（弃牌 top-1 / top-5）。
- 自对弈天梯：ELO 曲线、平均顺位、和牌率/放铳率、立直收支。
- 与随机/贪心基线对比胜率。

## 5. 测评脚本（旧版描述，已被 §6/§7 取代）
C:/agentwork/eval/ 下: scenario_gen.py（生成/抽样场况）、expert_report.py（专家打分表模板）、similarity_eval.py（通道B 抽样与差异标注表）。ml-engineer 在阶段5 前实现, 与模型推理接口对接。
## 6. 分层测评框架（用户已批准 · v1.1 扩展，t9 实现）

### L0 规则一致性冒烟（秒级，每次 checkpoint 必跑）
- sanity_smoke.py：模型输出合法动作率必须 100%（掩码生效率=100%）；20+ 手工构造牌型在模型推理下不崩溃、可解释输出与 factors 数值一致（rationale 引用的数字与 factors 一致，自动校验）；输入扰动（随机手牌/空牌河/极端点数）鲁棒。
- 输出：PASS/FAIL + 明细。

### L1 决策一致率（分钟级，迭代主力）
- offline_metrics.py：在天凤留出集（eval-holdout，绝不进训练）上统计：弃牌 top-1 / top-5 一致率，按动作类型拆分（摸切/手切/立直/吃/碰/杠/和牌）；模型概率熵分布。
- 校准基线：先在同集上测"人类 vs 人类"一致率（用留出集中不同时段/不同玩家的同局面重采样近似，或引用公开的 65-70% 参照区间），报告时同时给出模型值与该区间。
- 警戒：模型弃牌 top-1 明显高于 70% 时自动标记"疑似过拟合/泄漏"，交由 t7 泄漏检查复核。

### L2 反事实 EV 评估（小时级，与人类标签无关的客观尺）
- ev_eval.py：从留出集随机抽 1000 个决策点；对每个候选合法动作，用当前最强模型作为 rollouter 模拟 N 次（默认 100）到终局，比较各动作的期望最终顺位/点数 EV。
- 输出：EV 最优动作与人类动作一致率、按动作 EV 差的分布、模型动作相对人类动作的平均 EV 增益（可为负）。
- 用途：模型与专家意见相左时，用 EV 判"谁更优"的自动代理（最终以专家判定为准）。

### L3 锚定联赛（小时级，强度硬指标）
- league_eval.py：固定种子 × 大样本 4 人对局（每个对手组合默认 2000 局），对手池锚定：random / tsumogiri / 贪心向听规则 bot（shanten_greedy，规则基线实现放 eval/baselines/）/ 历史 checkpoint。
- 指标：平均顺位、和牌率、放铳率、立直局收支、局收支、锚定 Elo（以固定对手池为锚换算，保证跨版本可比）、置信区间。
- 引擎用 C:/agentwork/src/env/riichi_game.py；GPU 批量推理（多局并行）目标 >=1 万局/小时；运行写日志、结果存 JSON 历史曲线。
- 频率：里程碑跑；每次训练 2 万步快照可只跑 200 局/对手的缩减版。

### L4 专家终评（最终门，人工）
- 即本文 §1/§2：通道A 20+ 场况（动作质量+解释质量各 1-5 分）；通道B 天凤留出集 >=300 决策点差异判定（更优/等价/可理解略差/明显失误），明显失误率 <=5% 及格。
- 客观性保障：双盲、随机打乱、留出集零泄漏（t7 自动检查）。

## 7. 测评脚本清单（t9 交付物，C:/agentwork/eval/）
- sanity_smoke.py（L0）、offline_metrics.py（L1）、ev_eval.py（L2）、league_eval.py（L3，含 eval/baselines/random.py / tsumogiri.py / shanten_greedy.py）、scenario_gen.py + similarity_eval.py + expert_report.py（L4）。
- 统一入口 eval/run_eval.py：一条命令按层跑并汇总 JSON/HTML 报告；模型推理统一接口 model_runner.predict(observations_json_list) -> [{action, top_k, factors, rationale}, ...]（ml-engineer 实现推理侧适配）。

## 8. 专家打分细则与通道B 抽样说明（t17 补充, v1.2）

### 8.1 通道A 打分细则（动作质量 / 解释质量 各 1-5 分）

动作质量 (以天凤凤凰/雀魂王座级别人类基准为参照):
- 5 分: 最优解或等价最优 (多解并列时任一); 攻守判断与人类专家一致
- 4 分: 次优但差距很小 (EV 损失 <= 约 300 点或等待质量仅略差), 可接受
- 3 分: 可理解但明显略差 (损失向听/待牌质量/安全度明显, EV 损失约 300-1500 点)
- 2 分: 明显失误 (无谓放铳风险、拆好形留愚形、该和不和/不该攻强攻, EV 损失 >1500 点)
- 1 分: 严重失误 (直接放铳现物外的危险牌、自毁好形听牌、放弃确定和牌)
评分不看模型是否与人类动作一致, 只看该局面下的客观优劣; 有分歧时以 L2 反事实 EV 为自动参考, 最终以专家判定为准。

解释质量 (protocol §3: factors/rationale 必须忠实归因):
- 5 分: 引用的因子与规则计算完全一致, 论证覆盖 向听/进张/安全度/役价值 中的关键维度, 无编造
- 4 分: 忠实且基本充分, 缺一个次要维度或表述不精确
- 3 分: 忠实但不充分 (只给出结论或单一因子), 或个别数字与 factors 有出入
- 2 分: 部分归因错误 (把无关因子当主因) 或 rationale 与 factors 明显不一致
- 1 分: 事后拼凑、与决策无关、或引用不存在的因子数值
自动校验 (L0 sanity_smoke) 只保证最低线 (数字一致、键名合法), 专家评分覆盖"是否抓住真正的决策理由"。

### 8.2 通道B 差异判定细则（判定值: 更优 / 等价 / 可理解略差 / 明显失误）

- 更优: 模型动作在该局面下 EV 明显高于人类动作 (以 L2 反事实 EV 或专家判断为准)
- 等价: 与人类动作同档 (同为最优解, 或 EV 差在噪声范围内)
- 可理解略差: 非最优但理由成立 (如 愚形立直 vs 好形默听), 符合某一流派思路
- 明显失误: 无合理理由支持 (放铳风险与收益严重不匹配、放弃明显更优的路线)
底线: 明显失误率 = 明显失误数 / 总判定数 <= 5%, 否则评估不通过 (失败协议"评估异常")。

### 8.3 通道B 抽样说明 (eval/similarity_eval.py 实现口径)

- 数据源: 天凤留出集 eval-holdout 分区 (game_id 日期 >= 2026-08-01; 清单
  data/processed/tenhou/splits/eval_holdout_games.txt, 由 data-engineer 维护)。
- 抽样量: >=300 决策点 (默认 300); 去重: 每局至多 2 点、每(局,座位)至多 1 点,
  数据不足时自动放宽每局上限 (每(局,座位)去重保持)。
- 类别覆盖: 按 §1 的 10 类配额抽样 (每类至多 ceil(n/10) 点), 不足时以"其他"类
  补齐; 数据中缺失的类别 (如濒临飞人) 在 summary 中注明, 由通道A 场况补齐。
- 可复现: 固定 --seed 与 --records 文件列表即完全可复现; 输出 sample.jsonl /
  compare.csv / diff_annotations.csv / summary.json|md。
- 差异标注表: 专家在 diff_annotations.csv 的 判定/理由 列按 8.2 细则回填后,
  汇总 expert_diff 与 5% 底线检查生效 (expert_report.py 生成打分表)。
- 双盲与零泄漏: 抽样与标注分离; 留出集绝不进 SL 训练, t7 泄漏检查自动复核。

### 8.4 L1 人类-人类基线口径 (t17 实测)

- 官方校准基线: 公开参照区间 65-70% (弃牌 top-1), 常量 HUMAN_HUMAN_BASELINE;
  模型 top-1 > 70% 自动标记"疑似过拟合/泄漏"。
- 同集实测 (eval/human_baseline.py, t17, 留出集 1.2M 条扫描):
  因单标注数据中"完全同局面"几乎不重复, 采用情境指纹分组的重采样代理。
  实测区间: hand 级 (同手牌 34 计数, n=1075 对) 一致率 78.9% [76.3%, 81.2%];
  hand_wall 级 (n=212 对) 78.8% [72.8%, 83.7%];
  medium 级 (同向听/同摸牌/同宝牌/同壁桶/同位次/同副露数, n=500 万对) 17.5%;
  coarse 级 3.3% (接近随机下界)。
  结论: 同手牌条件下人类弃牌一致率实测约 79%, 高于公开参照 65-70%
  (后者口径为完整同局面含牌河/场况, 约束更多但选择空间更小);
  官方校准基线维持保守的 65-70% (>70% 告警), 实测区间作为补充证据,
  由 offline_metrics 输出 human_human_baseline.measured_proxy 字段携带。

---

## 9. 统一独立评估协议（协议 2.2 · RL 阶段 vs-SL 对比）

> 本章与 §1–§8 的「专家打分 / 可解释性」终极测评（阶段5）互补：§9 定义 RL 模型与基线之间
> 的**可复现、多 seed、带置信区间**的独立强度对比，是 README 主结果表与 D4 评估脚本的契约。
> 口径依据 feedback1（2026-09-13）§4/§5/§6/§12–§17 修正。

### 9.1 基线与定位

| 层级 | 基线 | 用途（feedback1 §7） |
|---|---|---|
| Sanity | Random | 环境/评估 sanity check，非正式竞争基准 |
| Sanity | Rule-based（向听贪心等规则 bot） | 环境/评估 sanity baseline |
| Learning | SL（`transfer_final.pt`） | **核心 baseline** |
| Learning | SL + PPO（RL v1） | **核心目标** |

正式结论只取 Learning 层对比；Random/Rule 仅用于逐级验证（先跑通 Random → Rule → SL → RL）。

### 9.2 主指标与辅助指标

- **主指标：Mean Rank（平均顺位）**。麻将是四人排序博弈，只用第一名率会丢失信息。
- 辅助指标：Rank 1 Rate（胜率）、Placement Score、pt/100 局、和牌率、放铳率。
- **术语口径（feedback1 §6）**：
  - `Win Rate` = **Rank 1 Rate** = P(rank=1)；
  - `Placement Score` = P(rank=1) + 0.5·P(rank=2)（**不再叫 Win Rate**，rank2 不是 win）。

### 9.3 指标定义（精确公式）

设 agent 在每个 seat s ∈ {1,2,3,4} 下分别各打 N 局（共 4N 局），`rank(s,g)` ∈ {1,2,3,4}：

$$
\mathrm{MeanRank} = \frac{1}{4N}\sum_{s=1}^{4}\sum_{g=1}^{N}\mathrm{rank}(s,g)
$$

- `Rank 1 Rate` = `#(rank=1) / 4N`
- `Placement Score` = `#(rank=1)/4N + 0.5·#(rank=2)/4N`
- `pt/100 局` = 100 × 平均每局 `settlement_pt`（`rl_reward.settlement_pt(..., "tenhou")`）
- `和牌率` = 我方和牌局数 / 我局数（从 `game.step` 返回的 `hora` 事件 actor 提取）
- `放铳率` = 我方放铳局数 / 我局数（`hora` 事件 target 提取，`eval_sliding.py` 已有提取模式）

**要求**：同时报告「每 seat 的结果」与「4 seat 汇总结果」（feedback1 §5）。

### 9.4 seed 与座次协议

| 项 | 值 |
|---|---|
| seed 数 | pilot 3，正式 5 |
| 每 seed 局数 | pilot 500，正式 1000 |
| 座次轮换 | 4 座位均衡轮换（agent 依次坐 seat 0/1/2/3，`eval_vs_sl.py` 已实现骨架） |
| 总规模 | pilot 1500 局；正式 5000 局（目标 3000–5000） |
| 混随机对手 | 默认**不混**；`--b random` 仅作 sanity 参照 |

### 9.5 两阶段评估（feedback1 §16/§17）

- **Stage A（pilot）**：`3 seeds × 500 局`，快速判断是否收敛、方差量级。
- **Stage B（full）**：仅当 Stage A 结果接近/有争议时，扩展到 `5 seeds × 1000 局` 出正式结果。
- 避免一开始 `5×1000` 才发现 RL 未收敛，浪费 GPU 时间。

### 9.6 统计报告规范（feedback1 §14/§15）

- `std` = **seed 间离散程度**（cross-seed SD），是分布宽度，不是估计精度。
- `95% CI` = **估计量的不确定性**，统一用 **Bootstrap 95% CI**（resample 10000 次）。
- **两个不要混**。最终报告两栏：`Mean ± SD across seeds` 与 `Bootstrap 95% CI`。
- 不再同时出 t-CI 与 bootstrap CI 双轨；主体统一 bootstrap，seed 级 std 作附加信息。

### 9.7 results.json 结构（feedback1 §13 · D4 输出契约）

```json
{
  "experiment": "rl_vs_sl_v1",
  "git_commit": "...",
  "config_hash": "...",
  "checkpoint_a": "...",
  "checkpoint_b": "...",
  "games_per_seed": 1000,
  "seeds": [
    {"seed": 1, "seat": 0, "games": 250,
     "mean_rank": 2.41, "rank1_rate": 0.31, "placement_score": 0.46,
     "win_rate": 0.31, "pt_per_100": 321, "win_rate_ci": [0.25, 0.37]}
  ],
  "aggregate": {
    "mean_rank": {"mean": 2.40, "sd": 0.06, "ci95": [2.28, 2.52]},
    "rank1_rate": {"mean": 0.31, "sd": 0.03, "ci95": [0.25, 0.37]}
  }
}
```

### 9.8 评估种子与轨迹再生（feedback1 §12）

- 对局本身是随机环境，**不必固定「完整 trajectory」**。
- 真正冻结的是：**模型权重 / seed protocol / benchmark config**。
- 原则：`Evaluation seeds are fixed and independently generated from training seeds; game trajectories are generated online by the frozen policies.`

---

## 10. 防泄漏硬规则（协议 6.2）

> 本规则为评估可信度的底线（feedback1 §10/§11 扩展）。任何违反即触发失败协议「评估异常」。

1. **train/eval 分区不重叠**：同一牌局（以 `game_id` 维度切分）不同时进入训练与测试，沿用
   `data/processed/tenhou/splits/` 约定；天凤留出集 `eval_holdout_games.txt` 永不进 SL 训练。
2. **预处理统计量只在训练集拟合后冻结**（feedback1 §10）：
   `All data-derived preprocessing statistics (normalization statistics, feature statistics,
   reward-model preprocessing) are fitted on training data only and then frozen for evaluation.`
   禁止用全量数据（含 eval/留出集）计算归一化/特征/奖励统计量。
3. **RL 对手池不含独立测试模型权重**：对手池（self / past / sl）只用训练侧产物，评估集模型永不出现在训练对手池。
4. **评估期模型不可变**（feedback1 §11）：
   `Evaluation models must be immutable during evaluation.` 评估过程冻结参数，只 forward、不 backward、不更新权重。
5. **评估种子独立**：评估种子固定且独立于训练种子；轨迹由冻结 policy 在线生成（见 §9.8）。
