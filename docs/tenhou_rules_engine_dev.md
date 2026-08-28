# 天凤规格独立调研清单（engine-developer 版）

调研日期: 2026-08-25 | 调研人: engine-developer | 用途: 与 reviewer 的 docs/tenhou_rules_authoritative.md 交叉验证

## 调研方法与独立性说明
本清单独立完成，交叉验证渠道：
1. 【联网检索】web_search 命中的公开规则页（链接见各条【来源链接】），但检索工具只返回标题/URL 无正文，
   正文级结论无法直接抓取 —— 故正文级核实以第 2、3 渠道为主；
2. 【本地权威源码】docs/mjai.app-main/（Rust 版 mjai.app，天凤规格的权威实现；源码多处直接引用
   tenhou.net/man 作为规则依据，如连风牌 4 符）—— 本清单逐条读源码核验；
3. 【对照】reviewer 的 docs/tenhou_rules_authoritative.md（含官方手册中文全译与凤凰桌牌谱实证）；
4. 我方已有实证：mjai.app 原生引擎 oracle 交叉验证 78 局 78/78（tests/test_oracle.py）。

标注约定：结论后有【核验方式】，凡与 reviewer 结论一致处注明「与 reviewer 一致」，不一致处重点说明。

---

## 1. 切り上げ満貫（切上满贯）

【天凤口径】段位战（含凤凰桌）无切上满贯：4番30符/3番60符按牌型表计（4番30符=7700/11600），
不升满贯。切上满贯仅出现在雀荘战模式的差异清单中。
【来源链接】
- tenhou.net/man/（官方手册，段位战规则清单不含切り上げ満貫；雀荘战小节列出为差异项——reviewer 摘录）
- 本地源码核验：docs/mjai.app-main/src/algo/point.rs —— 4番30符/3番60符 不进入 mangan 分支；
  mangan_up 表按 han 档位，无切上逻辑。
【与当前 rules_spec/引擎默认的差异】rules_spec §4 写「切り上げ満貫あり」，引擎 kiriage_mangan=True 默认。
与天凤口径相反。
【建议修正项】训练语料=凤凰桌 → 建议 kiriage_mangan 默认 False（或至少训练/评估配置显式 False），
雀魂口径 True 保留为开关。优先级：高（与 reviewer 一致）。

## 2. 連風牌（东场东家双东）符/番口径

【天凤口径】役牌/连风对子不计番（役牌番只认刻子/杠子）；雀头只计符：役牌雀头 +2 符，
连风雀头（自风=场风）+4 符（官方手册「連風牌は4符」，源码注释直接引用 tenhou.net/man）。
【来源链接】
- tenhou.net/man/（官方手册符计算节）
- 本地源码核验：docs/mjai.app-main/src/algo/agari.rs —— 役牌番仅从 all_kotsu_and_kantsu() 计；
  符计算对 bakaze +2、对 jikaze +2 → 连风雀头 4 符。
- mahjong 库实测：2.0.0 对连风雀头只给 2 符（valued_tiles 双风只传一次的 bug），引擎已在
  _evaluate_win 补偿 +2 符并按修正符数重算点数（TestWindPairAndTripletScoring 测试全绿）。
【与当前 rules_spec/引擎默认的差异】rules_spec §5.5 已定稿「对子不计番、连风雀头 +4 符」✓ 与天凤一致；
引擎已实现。无差异。
【建议修正项】无（已闭环）。优先级：-

## 3. 一発/裏ドラ/杠宝牌时序

【天凤口径】
- 杠宝牌：暗杠即乗り（翻牌立即生效）；明杠/加杠後めくり（打牌时翻，或若连续加杠则在下一次
  岭上摸牌直前翻）。抢杠成立则加杠的杠宝牌不再翻开。
- 一発：立直后至自己第一次打牌前（无人鸣牌）成立；与枪杠（连续杠第 1 回）复合；连续加杠第 2 回
  起不附一発；与岭上不复合（自己的暗杠也会打断一発）。
- 裏ドラ：立直和牌时按已翻表宝牌指示牌数翻开对应数量的裏指示牌。
【来源链接】
- tenhou.net/man/（reviewer 摘录：カンドラ/一発/槍槓/裏ドラ各节）
- 本地源码核验（与 reviewer 一致，另有两点补充）：
  - board.rs：ankan 后立即 add_new_dora；daiminkan/kakan 置 need_new_dora_at_discard（打牌时翻）/
    need_new_dora_at_tsumo（续杠后次回自摸前翻）；枪杠 Hora 处理在翻牌前 → 抢杠成立不翻杠宝牌 ✓
    与引擎现实现一致（引擎已在 _dora_at_tsumo/_dora_at_discard 双槽位实现，oracle 78/78 验证）。
  - update.rs Ankan 分支：at_ippatsu = false（对所有玩家，含杠者自己）→ 岭上不与一発复合 ✓。
  - update.rs 枪杠分支：chankan_chance 保留可抢者的一発；但 Hora 事件处理顶部 chankan_chance.take()
    后置 at_ippatsu=false —— 即 mjai.app 在枪杠荣和时打断一発，与官方手册「一発と槍槓(1回目)は複合する」
    不一致（源码级偏离，非天凤口径）。
【与当前 rules_spec/引擎默认的差异】
- 杠宝牌时序：引擎已按天凤口径实现 ✓ 无差异。
- 一発×岭上：引擎 _execute_kan（ankan 分支）不打断杠者自己的一発 —— 与天凤口径不一致，
  需改：ankan 也应置 _ippatsu_broken[seat]=True。
- 一発×枪杠：引擎保留可抢者一発（按官方手册口径）✓；与 mjai.app 实现不一致（极端场景，78 局未触发）。
【建议修正项】① ankan 打断杠者自己一発（1 行改动+1 测试）；② 枪杠一発保留维持手册口径并在文档
标注 mjai.app 偏离。优先级：低。

## 4. 振听三类细则

【天凤口径】
- 舍牌振听：所听之牌（形状可和，含无役）出现在自家牌河 → 永久振听（荣和不可，自摸可）。
- 同巡振听：放弃「可和的牌」（有役）后成立，至自己下一次打牌时解除；鸣牌（逆巡副露）不解除
  （reviewer 摘录：「同巡振聴の解消は打牌時、逆巡副露では解消しない」）。
- 立直后见逃：永久立直振听。
【来源链接】
- tenhou.net/man/（振聴节，reviewer 摘录）
- baike.baidu.com/item/振听（百度百科词条，引 riichi.wiki 与雀魂教程 —— 注意：该词条是雀魂口径，
  与天凤同巡振听触发条件不同）
- 本地源码核验：update.rs —— can_ron_agari 仅在 waits[pai] 且有役(或立直/河底)时提供；
  to_mark_same_cycle_furiten 仅随 can_ron 提供而置 → 同巡振听=放弃有役可和牌才置 ✓ 与天凤一致。
【与当前 rules_spec/引擎默认的差异】rules_spec §2 采用雀魂口径（无役形状见逃也置同巡振听、
鸣牌即解除）—— 与天凤不一致，但系用户明确指定（t10 已按此实现并有测试）。
【建议修正项】保留用户口径（已裁决），本文档仅标注差异。优先级：低（已裁决）。

## 5. 西入条件与连庄

【天凤口径】南4（オーラス）结束：分数 <0 者出现 → 立即终局；否则未达 30000 返点时进入西入，
之后每局按同一条件；庄家连庄（和了/荒牌听牌）正常延续；オーラス和了止め = 庄家在南4 和了且
30000 点以上且 top 时终局（reviewer 摘录）。
【来源链接】
- tenhou.net/man/
- moegirl.icu/zh-hans/麻将/游戏长度/延长战（延长战/西入规则）
- 本地源码核验：arena/game.rs —— kyoku>=length+4(W4) 硬上限；kyoku>=length 且非连庄且有人>=30000
  终局；agari-yame（renchan owari）= kyoku>=length-1 且 scores[oya]>=30000 且 top；负分立即终局 ✓
  与天凤一致。
【与当前 rules_spec/引擎默认的差异】rules_spec §3 与引擎 _advance_round（负分立即终局 + 全员
<30000 继续 + W4 上限 + 和了止め 30000+top）✓ 已一致。无差异。
【建议修正项】无。优先级：-

## 6. 頭跳ね/ダブロン

【天凤口径】允许双响（ダブロンあり）：两家各自结算，本场费/供託（立直棒）上家取り
（离放铳者最近的和牌者全取）；亲家含其中则连庄；三家荣和 → 三家和途中流局（连庄）。
【来源链接】
- tenhou.net/man/（reviewer 摘录）
- queji.com 头跳词条（多家同时荣和只判一家的规则——注意该词条讲的是头跳规则本身，天凤实际采用双响）
- 本地源码核验：board.rs 多荣和处理（cycle().skip(target+1).take(3)，第一和牌者全取本场/供托）✓；
  但 board.rs 顶部注释 No triple-ron ryukyoku —— mjai.app 不实现三家和流局（三家全部正常结算），
  与天凤官方口径不一致。
【与当前 rules_spec/引擎默认的差异】rules_spec §4「頭跳ね（双响默认 OFF）」；引擎 atamahane=True /
double_ron=False 默认。与天凤口径相反。
【建议修正项】若训练语料对齐天凤：双响默认 ON（double_ron=True，上家取全棒本场已实现）+ 三家和
流局（引擎目前无 ron3；mjai.app 亦无，需自研实现）。优先级：高（双响默认）；中（三家和流局）。

## 7. 流局四类与流局满贯；本场费/供託/立直棒分配

【天凤口径】流局：九种九牌（可选，第一自摸时 9 种以上幺九）、四风连打（4 人第一打同一风牌）、
四杠散了（4 杠且非同一人）、四家立直（4 人立直成立）、荒牌（牌山耗尽）听牌结算（3000 点分摊）、
流局满贯（牌河全幺九且未被鸣，mangan 结算）。本场费：荣和 300×N 由放铳者付、自摸每家 100×N；
立直棒 1000 归和牌者，流局时保留至下局。
【来源链接】
- tenhou.net/man/；zh.wikipedia.org/wiki/流局满贯
- 本地源码核验：board.rs —— four wind（同一风牌 4 人第一打）、four kan（kans==4 且无人 4 杠）、
  four riichi（accepted_riichis==4）、nagashi（can_nagashi_mangan）、kyushu（9 种以上幺九）、
  exhaustive（tiles_left==0）+ 听牌结算 ✓；引擎已全部实现（oracle 78/78 覆盖荒牌/听牌结算）。
【与当前 rules_spec/引擎默认的差异】rules_spec §4 列出的四类+流局满贯与引擎默认一致 ✓ 无差异。
【建议修正项】无。优先级：-

## 8. 役满多重/数え役満/双倍役满

【天凤口径】役满可複合（不同役满累乘）；无双倍役满：四暗刻单骑、国士无双十三面、大四喜均按
单倍计（官方手册役一览附注）；数え役満（13番以上）按单倍役满。凤凰桌牌谱实证：四暗刻単騎 =
48000（单倍）。
【来源链接】
- tenhou.net/man/（役一覧附注，reviewer 摘录）
- riichi.wiki Multiple yakuman 页面
- 本地源码核验：agari.rs —— 四暗刻 yakuman += 1（不区分单骑）、kokushi = Yakuman(1)（不区分
  十三面）、大四喜/大三元等各 +1，无任何 ×2 路径 ✓ 与 reviewer 结论一致。
【与当前 rules_spec/引擎默认的差异】rules_spec §4「双重役满」；引擎 double_yakuman=True 默认
（mahjong 库 has_double_yakuman）。与天凤口径相反。
【建议修正项】double_yakuman 默认改 False（训练/评估配置显式 False）。优先级：高（与 reviewer 一致）。

## 9. 包牌（責任払い）/赤宝牌

【天凤口径】
- 包牌：有。大三元/大四喜成立时，打出役满确定牌的玩家承担：复合役满得点全部由包者支付
  （自摸=全額、荣和=折半），本场费亦由包者承担；四杠子不包。
- 赤宝牌：5m/5p/5s 各 1 枚（共 3 枚）。
【来源链接】
- tenhou.net/man/（パオ节，reviewer 摘录）
- mahjong.huijiwiki.com/wiki/包牌（灰机麻将维基）
- 本地源码核验：board.rs 有完整 pao 实现（paos 追踪 + deltas 中包者全額/折半支付 + 本场亦包）✓
  与 reviewer 一致。
【与当前 rules_spec/引擎默认的差异】引擎未实现包牌（_apply_ron_payments/_settle_tsumo 无 pao
路径）。rules_spec 未提包牌。
【建议修正项】实现大三元/大四喜包牌（含複合役满全包、荣和折半、本场亦包）；赤宝牌 3 枚引擎已
实现 ✓。优先级：中（罕见但影响结算正确性，oracle 78 局未触发）。

---

## 汇总：建议修正项与优先级

| 优先级 | 项 | 动作 |
|---|---|---|
| 高 | 1 切上满贯 | 默认 kiriage_mangan=False（或训练配置显式关） |
| 高 | 6 双响 | 默认 double_ron=True / atamahane=False（上家取棒已实现） |
| 高 | 8 双倍役满 | 默认 double_yakuman=False |
| 中 | 9 包牌 | 实现大三元/大四喜 pao（全包/折半/本场亦包） |
| 中 | 6 三家和流局 | 双响开启后补 ron3 途中流局（mjai.app 亦缺，需自研） |
| 低 | 3 暗杠打断自己一発 | ankan 时 _ippatsu_broken[seat]=True（1 行+测试） |
| 低 | 3 枪杠一発 | 维持手册口径（保留可抢者一発），标注 mjai.app 源码级偏离 |
| 低 | 9 立直宣言牌放铳不成立供托 | 引擎现先扣 1000；改延后至放铳窗口结束 |
| 低（已裁决） | 4 同巡振听雀魂口径 | 保留用户口径，标注天凤差异 |
| 待数据确认 | 喰断（凤凰桌） | reviewer 称凤凰桌喰断なし；mjai.app 源码恒 kuitan ON 无法佐证 —— 建议 data-engineer 用凤凰桌牌谱统计副露断幺出现率定论 |

## 与 reviewer 文档的交叉验证结论
reviewer 的 docs/tenhou_rules_authoritative.md 第 1、2(符)、5、7、8、9、10(枪杠一発)、11、14 项与
我方独立核验（本地 mjai.app 源码 + 检索链接）一致；差异点仅两处：
- 第 10 项补充：mjai.app 源码实际在枪杠荣和时打断一発（与官方手册相反），我方建议按手册口径保留引擎现行为；
- 第 13 项（同巡振听解除时机）：天凤为打牌时解除、rules_spec 为鸣牌时解除 —— 功能上等价（鸣牌后必打牌），保留用户口径。

## 附：本轮检索到的公开来源 URL 清单
- https://tenhou.net/man/ （官方手册）
- http://wiki.lingshangkaihua.com/mediawiki/index.php?title=天鳳規則 （官方手册中文译）
- https://wiki.queji.com/mediawiki/index.php?title=天凤规则 （另一中文译版）
- https://riichi.wiki/index.php?title=Comparison_of_popular_rulesets （各平台规则对比）
- https://riichi.wiki/index.php?title=Multiple_yakuman
- https://zh.wikipedia.org/wiki/日本麻将规则 （半庄战/オーラス/西入）
- https://baike.baidu.com/item/振听 （振听词条，雀魂口径）
- https://moegirl.icu/zh-hans/麻将/游戏长度/延长战 （延长战/西入）
- https://queji.com/zh-hant/glossary/atamahane/ （头跳）
- https://mahjong.huijiwiki.com/wiki/包牌 （包牌）