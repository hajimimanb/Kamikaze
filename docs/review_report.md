# 阶段1 质量审计报告（t7 · reviewer）

审计日期: 2026-08-25 | 审计人: reviewer | 依据: docs/rules_spec.md（用户确认版）+ docs/observation_schema.md

## 总览

| 审计项 | 结论 |
|---|---|
| 1. 引擎验收（t1，244 测试复跑全绿） | 条件通过（4 项缺陷待修，见 1.2） |
| 2. 天凤解析器验收（t3） | 通过（3/5 局 0 errors；其余为已知残留告警） |
| 3. 数据集验收（t3） | 通过（标签/legal 全合法；格式混跑期注意项） |
| 4. 数据泄漏检查 | 通过（train 37588 / eval 5451 零重叠） |

---

## 1. 引擎验收（RiichiEnv, src/env/riichi_game.py）

### 1.1 测试规模与覆盖
- 全套件 244 测试复跑全绿（pytest tests/ -q）。
- 覆盖无重大空白：和牌/役满/振听/鸣牌/流局/牌山/观测/立直/本场/头跳/双响/四杠/切上两模式均有专项测试。
- 缺口：役牌雀头、tobi（负分终局）、鸣牌解除同巡振听、暗杠抢杠（单骑国士）四类无测试（与下方缺陷对应）。

### 1.2 独立结算验收（19 PASS / 5 FAIL，手算期望值；脚本 review/_t7_engine_check.py）

PASS：国士13面 han=26、四暗刻单骑 han=26、大三元 han=13、抢杠（加杠任意役可抢，han=4 fu=40 含抢来赤5m 赤宝=8000 结算）、
舍牌振听、立直见逃永久振听、头跳队列序与结算、本场+立直棒结算（3900+600+3000）、切上满贯（4番30符=8000，spec §4）、
西入继续/不继续两模式；四杠流局引用其 test_four_kans_ryuukyoku_with_two_owners。

FAIL（缺陷清单）：

| # | 缺陷 | 证据 | 影响 |
|---|---|---|---|
| P6 | 庄家自摸役满点数翻倍（mahjong 2.0.0 库 scores.py 对役满 tsumo 取 double_rounded 致 2 倍） | 国士13面庄家自摸=32000all（正确16000all）；大三元=16000all（正确8000all） | 所有模拟/自对弈得分 |
| P2 | 负分终局（tobi）缺失 | 荣和后 -3600 仍继续对局 | 与 rules_spec §3 冲突 |
| P0 | 役牌雀头未计分——但天凤实测不计（见 1.3），引擎与天凤数据一致；若用户要求计数则需补 | 复现 review/engine_pair_yakuhai_repro.py | 取决于口径裁决 |
| t10 | 同巡振听无役形状见逃不置振听（未合入，任务仍 pending） | 无役2z见逃后5z荣和仍提供 | 与 rules_spec §2 冲突 |
| P5 | 鸣牌不解除同巡振听（spec §2: 鸣牌时即解除） | 代码审查: temp_furiten 仅在摸牌时清除 | 振听玩家鸣牌后错误禁和/禁抢杠 |

### 1.3 役牌雀头：天凤数据实证（重要口径结论）
- 扫描 5 局真实天凤 MJAI（8 个带字牌雀头的和牌）：天凤一律不计役牌雀头/连风对子
  （庄家双东对子单骑自摸，天凤 yaku 列表无自风/场风役，4番30符=11700 而非 6番）。
- mahjong 库（1.2.1/2.0.0）同样不实现 pair yakuhai，引擎现行为与天凤训练数据一致。
- 结论：建议维持现状（不计雀头役）；captain 指令 #4 的 supplement 对齐口径与天凤数据相悖，请 captain/用户裁决。
  若维持天凤口径，P0 不再是缺陷；mjaicheck 的 supplement 已默认关闭。

### 1.4 已与 engine-developer 同步的修复清单
- P6（役满庄家自摸翻倍）、P2（tobi）、P5（鸣牌清振听）、t10（合并+测试）。
- kiriage 默认 True 与 spec §4 一致（撤回早前异议）；抢杠实现符合新口径（暗杠任意国士可抢、非国士不可）。

---## 2. 天凤解析器验收（t3, src/tenhou/）

### 2.1 MJAI 重放（mjaicheck，随机 5 局稳定副本 review/_audit_games/）
- 3/5 局 0 errors；2/5 局 2-3 个分数对账告警（供托结转语义，见 2.3）。
- 牌种预算/回合/摸切/鸣牌/立直/和牌役种/点数逐项重放通过。
- 全知版输出（tehais 全可见）——早期建议已被采纳，完整严格校验可执行。

### 2.2 发现（需 data-engineer 修复，2 项）
| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| C1 | 事件类型拼写 ryuukyoku（规范为 ryukyoku） | src/tenhou/mjai_export.py | 官方 mjai 解析器（serde）会拒绝；外部重放工具不兼容 |
| C2 | 部分 dahai 的 tsumogiri 标记与刚摸的牌不一致（约 1%） | mjai_export.py 的 tsumogiri 推导 | 摸切基线指标（schema §2.1）精度 |

### 2.3 已知残留（验证器侧，非转换器缺陷）
- 荒牌流局时天凤存在供托结转（实测 2 家听牌仍结转立直棒）——重放器按教科书规则清供托，
  导致 2/5 局分数对账 ±1000 告警；已降级为告警。转换器忠实复制天凤数据，无过错。
- karaten（空听）不建模：tenpai 判定差异→连庄派生告警（已降级）。

---

## 3. 数据集验收（records-*.jsonl.gz）

- 规模: 200 分片（2026-01 至 2026-08-24），抽样 3 分片 = 824,942 条 / 1,646 局。
- 每局样本数: min 58 / max 1039 / 均值 501 —— 符合预期（含和牌/鸣牌机会样本）。
- legal_actions 非空 100%；label 类型 100% 合法；discard 标签 100% 在手且在合法列表；
  riichi/pon/chow/kan/ron/tsumo 标签 100% 与 legal_actions 一致。
- 手牌张数: 92% 通过（13/14 或 13-3×副露±杠调整）；8% 异常集中在三副露+杠组合与新旧格式混跑期，
  与 data-engineer 预告的格式过渡一致——需 post_finish 看门狗收口后复查。
- 年份兼容: 仅 2026 年数据可用（2009-2024 天凤存档 404，README 已说明）；
  旧格式兼容由 tests/test_tenhou_parser.py 的 14 个 pre-2012 + 8 个 2025 边界样本覆盖（测试全过）。
- 西入不采样（round>7 = 0 条）。

---

## 4. 数据泄漏检查
- check_leakage.py: train 37588 / eval 5451 game_id 零重叠；划分=game_id 纯函数（日期窗口+hash）。
- tests/test_splits.py 2/2（注意: 分片批跑期间 splits.json 原地重写，曾捕获一次瞬时不一致，
  建议 post_finish 用原子替换收口）。
- 评估脚本只读 holdout 的静态审计：待 t9（eval-tooling-developer）交付 eval/ 目录后执行。

---

## 5. 风险清单（按优先级）

1. 高 P6 役满庄家自摸翻倍——影响所有模拟/自对弈得分。
2. 高 P2 tobi 缺失——影响长期对局（RL 自对弈）终局条件。
3. 高 t10 未合入——与用户强制口径（rules_spec §2）冲突。
4. 中 P5 鸣牌不解除同巡振听。
5. 中 C1 ryuukyoku 拼写——协议合规性（对内训练无影响，对外部重放工具有）。
6. 中 C2 tsumogiri 标记误差约 1%——摸切基线指标。
7. 低 records 8% 手牌张数异常（格式过渡期）——post_finish 复查。
8. 低 splits.json 原地重写瞬时不一致——建议原子替换。
9. 裁决项 役牌雀头口径（引擎 vs 天凤数据）——建议维持天凤口径（不计），请 captain/用户确认。

## 6. 复现与工具
- 引擎验收脚本: review/_t7_engine_check.py（19P/5F）
- 役牌雀头复现: review/engine_pair_yakuhai_repro.py
- MJAI 重放: review/mjaicheck（57 测试全绿）; 抽样局: review/_audit_games/
- 泄漏检查: review/check_leakage.py
---

## 7. t15 修复复核（2026-08-25 第二轮，用户口径已定）

用户裁决：役牌雀头不计番、仅符（连风对子 4 符、单役牌对子 2 符）。

| 项 | 结论 | 证据 |
|---|---|---|
| t10 同巡振听（无役形状见逃置振听） | ✅ 已合入 | 独立验收 F3 转 PASS；tests/test_rule_fixes.py::TestShapeFuriten |
| P5 鸣牌解除同巡振听 | ✅ 已修 | test_call_lifts_temp_furiten_immediately |
| P2 负分终局（tobi） | ✅ 已修 | 独立验收 J3 转 PASS；TestWestExtension::test_negative_score_ends_game_tobi |
| 抢杠三态 | ✅ | 加杠任意役（独立验收 D 组 PASS）；暗杠仅国士（test_ankan_robbed_only_by_kokushi）；振听者不得抢（2 条测试） |
| 雀头口径一致性 | ✅ | 不计番：双东对子无他役时无荣和选项（独立探测）；仅符：有役时连风对子 4 符计入（独立探测 fu=50 与天凤一致） |
| 全量 pytest | ✅ 432 passed（冻结时 2 失败 TestAgariYame/TestChankan 已清零） |
| 独立手算验收 _t7_engine_check.py | ⚠️ 21 PASS / 3 FAIL —— 仅 P6 残留 | A2/B2/C2 庄家自摸役满仍翻倍 |

P6 状态：已向 engine-developer 提供精确修法（_evaluate_win 返回前修正庄家自摸役满 cost 为 8000×n 每家）；
修复后复跑 _t7_engine_check.py 应 24/24。
---

## 8. v1.6 引擎复核（2026-08-25 第三轮，天凤口径对齐）

engine-developer v1.6+v1.7 复核：全仓 pytest 443 passed；独立验收 **27/27 PASS**。

| 项 | 结论 | 证据 |
|---|---|---|
| double_yakuman=False（天凤单倍役满） | ✅ | A1/B1 han=13；牌谱实证四暗刻単騎=48000 |
| double_ron=True（天凤双响） | ✅ | G4/G5：两家和牌、放铳者付两家（本场/供托上家取り待专项验证） |
| kiriage_mangan=False（天凤段位战无切上） | ✅ | 默认值确认；I 组用例（6番）不受影响 |
| tobi/西入/t10/P5/抢杠（加杠任意役） | ✅ | J1/J2/J3、F3 均 PASS |
| P6 庄家自摸役满「翻倍」 | ✅ 撤销误报 | §5.1 定稿：庄家自摸每家付 2a；牌谱实证四暗刻庄家自摸=16000all（deltas[-16000,-16000,+48000,-16000]）——引擎/库/数据三方一致，P6 不成立 |
| v1.7 暗杠任何人不可抢（含国士） | ✅ 已合入 | chankan_kokushi 默认 False、ankan 分支移除；TestChankanRules 5/5（含 test_ankan_cannot_be_robbed_even_by_kokushi、开关 noop） |

备注：kuitan=True 保留（天凤凤凰桌实际为喰断なし，详见 docs/tenhou_rules_authoritative.md §1-4，待用户裁决）。

## 8.1 补充复核（P6 争议终裁后）

- captain 终裁：§5.1 庄家自摸每家 2a（役满=16000all）为正确口径；8000all 是子家自摸的闲家档。
- 验收脚本已补 K2 对照：子家自摸大三元 = 庄家 16000 / 闲家 8000（[9000,57000,17000,17000] ✓）。
- 最终：_t7_engine_check.py **30/30 PASS**；v1.7 暗杠不可抢（TestChankanRules 5/5）；全仓 443 passed。
