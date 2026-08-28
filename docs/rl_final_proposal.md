# 麻将 AI 最终方案 v2.0（四人共同修订·captain 终稿）

> 流程: innovator draft v1.1 → critic 评估（P0x4/P1x6）→ auditor 裁定（Q6/Q7/Q8）→ captain 拍板 → 本终稿
> 统一论点: 麻将决策的结构化表示学习（事件因果推理② / 对手信念③ / 座位对称④ = 对局结构三侧面显式建模）

## 0. 三目标与优先级
- 创新 > 效果 > 解释性 > 工作量；创新必须建立在效果大幅提升上（量化门槛见 §3）
- 论文（主锚 RiichiBench 基准 + C1-C4 参考 agent 模块）/ 应用（蒸馏部署 + mjai bot）/ 开源（协议+引擎+leaderboard）

## 1. 论文定位
- 主推标题: RiichiBench: Benchmarking Event-Causal Reasoning for Riichi Mahjong AI
- venue: NeurIPS/ICLR D&B（基准主锚，与开源资产强匹配）; 模型主锚备选 AAAI/IJCAI
- 贡献（写明主次，防 kitchen-sink）:
  贡献4（主锚）= RiichiBench 决策级标准化基准（协议+oracle 引擎+可复现 leaderboard）
  贡献1-3 = 在统一基准上评估的参考 agent 模块: C1 事件因果注意力 / C2 危险度分解 / C3 相对座位序等变

## 2. 创新点（C1-C8，核心 4+1）
- C1 事件因果注意力【真新·组合级】: 公开事件时间线显式建模为因果序列（事件 token + 因果掩码 + recency 计数 + 逐通道门控）。
  表述精确化: '麻将 AI 中首次将公开事件时间线显式建模为因果序列（据我们所知）'（非 attention 首次）
- C2 信念危险度分解【真新·方法级】: 危险度 = P(听牌) x P(待牌) 显式分解（可解释 + ground truth 直接监督）
- C3 相对座位序等变【真新·分析级】: 全置换等变丢失下家/对家/上家相对序（Suphx 为 cyclic 旋转等变）；命题级信息结构论证 + 样本效率操作化
- C4 阶段3 剥削【真新·领域空白】: bonus section，best-response 验证优先（收益不确定不赌主线）
- C5 RiichiBench【资产真实】: '首个决策级标准化基准（据我们所知）'；related work 对照 mjai.app/riichi.dev（mjai.app 2026-04-30 退役迁移，生态空窗）
- C6 蒸馏 8 块【工程交付】: 与 C2 共享解释链、单列 1 人·天
- C7 mjai bot 包装【应用】: 端到端对局冒烟（附录外部锚点）
- C8 复现工程【开源】: 种子确定性/脚本/LICENSE/模型卡/demo

## 3. 效果铁律与量化门槛（critic R5 + auditor Q6）
1. 效果指认: 每创新点声明主效果域（C1=SL top-1+解释质量，RL 期为间接收益不作主主张）
2. 独立消融: 每项 on/off + CI + 置换检验；同预算同 eval
3. 最小可报告效应量: top-1 +0.3pt / 放铳率 -2%（CI 排除 0），低于自动降级附录
4. 门槛（auditor Q6 统计口径）: 胜率/收支 相对≥10% 且配对 CI 不含 0（300 局可分辨）；
   放铳率 相对降≥5%（critic 裁定，JongMaster 锚点）+ 配对置换检验与 CI 标注；top-1 0.5pt 评估集 ≥5K 决策点；
   信念头 top-1≥85% 且 ECE≤0.05 同时满足
5. C2 表2 增'分解 vs 单一整体危险度头'消融行（方法贡献关键证据）
6. C4 池外泛化守门: holdout 风格池 + ≥1 外部 bot 胜率 CI 下界 ≥0.48

## 4. 效果证据与基线
- R2 口径声明: 77.28% 为自家 holdout 合法弃牌 top-1，高于 Suphx 报告 76.7%（原协议 50K 测试集），口径不同仅供参考；
  表1 补'本工作统一协议下可复现锚定值'行；复现 Suphx 型架构于 RiichiBench 切分（P1 可选）
- R1 外部锚点: mjai bot 对打强开源 bot（Mortal/kanachan/akochan），矩阵进附录
- 99.95% oracle 口径（按事件/局/结算）写入数据卡前复核钉死

## 5. RiichiBench（C5）落地
- 资产: 55.9M 决策点 + oracle 99.95% Python 引擎（Suphx/Tjong 不公开）+ 分层协议 L0-L4 + 指标族 + 种子管理
- 许可（auditor Q7 裁定）: 默认保底方案（代码+权重+获取指引+特征级子集），完整数据可再分发不作前置假设；
  合规核查列为 M0 并行人工门，未过切'特征级决策点'或'引擎自对弈数据（自有版权）'；发布前 0.5 人·天条款核查写数据卡
- AGPL 防线: 不引入 mjai.app-main 代码（转换器自写）；第三方许可证审计（mjai.app-main/tenhou-to-mjai/shanten_tmp）发布前完成
- 工作量: C5 打包 1.5~2 人·天（含合规+数据卡+leaderboard）

## 6. 预算与里程碑
- 总预算 13~15 人·天（9~11 效果达标 + 4~5 三目标包装）；封顶 15
- M0 开训前: T1 尺度对齐/T2 熵掩码/A1 mask/T9 sanity（必修）+ C5 合规核查（并行）→ 7~8 人·天
- M1 训练: T3 课程/T6 KL/C1 事件注意力/C5 基准 → 2~3 人·天
- M2 验收: 66%+CI 下界 0.60 + 消融归因（C1-C4 逐项）
- M3 后置: A2/A5/A3/T7/T8 消融 + C2/C3/C4 阶段2/3 + C6 蒸馏 + C7 bot

## 7. Go/No-Go 与 Plan B
- 阶段1: 对 SL 基线 win_rate ≥0.66 且配对 CI 下界 ≥0.60（1 RL + 3 SL 座位轮换写死）
- 阶段2: 放铳率相对降 ≥5%（对齐 JongMaster -5.12% 领域锚点，critic 裁定）+ 胜率非劣效（配对 CI 下界 ≥0.50）
  + 信念头 top-1≥85% 且 ECE≤0.05；统计条款: 配对准换检验 + 95% CI 报告（CI 跨 0 但点估计达标 → 降级'方向性证据'不 fail 但标注）
- 阶段3: 对风格化池 66% + 池外泛化守门 + Nash-gap 报告
- Plan B: RL 未达 66% → 降级口径 '配对 CI 下界 >0.50' + 'L3 优于 SL'（3 项中 ≥2 项配对显著占优）；触发不影响 C1-C3 消融报告

## 8. 开源与复现清单
- RiichiBench repo: eval harness + L0-L4 协议 + holdout 清单 + README leaderboard（基线/CI/命令）+ demo 回放
- 数据卡: 许可结论/来源/统计粒度/获取指引；模型卡: 权重/推理脚本/延迟指标
- LICENSE（自写代码 MIT/Apache-2.0）+ 第三方许可审计

## 9. 待执行前提（auditor 核对清单）
- P0: C5 声称修正 / 合规闸门 / C2 分解消融行 / C4 池外守门 — 终评硬门槛
- P1: C1 RL 期指认 / C3 样本效率操作化 / C6 预算修正 / 贡献主次 / C1 表述精确化 / oracle 粒度钉死
- P2: mjai 冒烟 / 许可审计 / 表1 锚定值