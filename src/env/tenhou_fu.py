# -*- coding: utf-8 -*-
"""天凤精确符数计算（替代 mahjong-lib 的符数，后者在待牌判定/拆解上与天凤不一致）。

规则来源：天凤规则 + queji 符计算词条（2026-08 快照）：
- 底符 20；门清荣和 +10；自摸 +2
- 刻子：中张 明2/暗4/明杠8/暗杠16；幺九 ×2
- 雀头：三元/自风/场风 +2；连风（自风=场风）天凤 +4
- 待牌：两面/双碰 0；坎张/边张/单骑 +2
- 切上（合计非 10 倍数向上取整）
- 特殊：七对子 25（不切上）；平和自摸 20；平和荣和 30；副露合计 20 -> 30
- 高点法：多拆解取符数最高者
"""
from collections import Counter


def is_terminal_or_honor(kind):
    return kind >= 27 or kind % 9 in (0, 8)


def _is_chiitoitsu(counts):
    return all(v == 2 for v in counts.values()) and sum(counts.values()) == 14


def _is_kokushi(counts):
    kinds = list(counts)
    return (sum(counts.values()) == 14 and len(kinds) == 13
            and all(is_terminal_or_honor(k) for k in kinds))


def _sets_dfs(counts, remaining, acc, out):
    """枚举面子拆解（含顺子/刻子/杠）。counts: Counter(kind)。"""
    if remaining == 0:
        if all(v == 0 for v in counts.values()):
            out.append(list(acc))
        return
    k = next((i for i in range(34) if counts.get(i, 0) > 0), None)
    if k is None:
        return
    # 刻子
    if counts.get(k, 0) >= 3:
        c = dict(counts); c[k] -= 3
        _sets_dfs(c, remaining - 1, acc + [[k, k, k]], out)
    # 顺子（数牌且可衔接）
    if k < 27 and k % 9 <= 6 and counts.get(k + 1, 0) > 0 and counts.get(k + 2, 0) > 0:
        c = dict(counts); c[k] -= 1; c[k + 1] -= 1; c[k + 2] -= 1
        _sets_dfs(c, remaining - 1, acc + [[k, k + 1, k + 2]], out)
    # 杠（手握 4 张未开杠）
    if counts.get(k, 0) >= 4:
        c = dict(counts); c[k] -= 4
        _sets_dfs(c, remaining - 1, acc + [[k, k, k, k]], out)


def _concealed_set_fu(s, win_kind=None, tsumo=True, win_in_run=False):
    """手牌面子符：顺子 0；刻子（自摸=暗刻 4/8；双碰荣和完成=明刻 2/4）；
    杠（仅自摸可能，暗杠 16/32）。

    明刻判定（双碰荣和）要求和牌张真正"加入"该刻子：
    若和牌张同时出现在某个顺子中（高点法下进顺子），刻子仍按暗刻计。
    （oracle-verified: 5808fd49 E1, 222m 234m 99m 111p 567p 荣和2m：
    天凤 2m 暗刻4+1p暗刻8=42->50符；引擎曾误判 2m 明刻2 -> 40符）"""
    k = s[0]
    if len(s) == 4:
        return 16 if not is_terminal_or_honor(k) else 32
    if s[0] == s[1] == s[2]:
        if not tsumo and win_kind == k and not win_in_run:
            return 2 if not is_terminal_or_honor(k) else 4  # 双碰荣和=明刻
        return 4 if not is_terminal_or_honor(k) else 8
    return 0


def _wait_fu(sets, win_kind):
    """和牌张的待牌符（高目：和牌张可能同时在多个顺子中，取最大符；
    双碰 0 / 两面 0 / 边张 2 / 坎张 2；不在任何面子里=单骑 2）。"""
    in_set = False
    best = 0
    for s in sets:
        if win_kind not in s:
            continue
        in_set = True
        if len(s) == 4:
            continue  # 杠非和牌结构
        if len(s) == 3 and s[0] == s[1] == s[2]:
            continue  # 双碰：待牌 0 符（刻子符已另计）
        a = s[0]
        if win_kind == a + 1:
            fu = 2  # 坎张（缺中张）
        elif win_kind == a and a % 9 == 6:
            fu = 2  # 边张（789 等 7）
        elif win_kind == a + 2 and a % 9 == 0:
            fu = 2  # 边张（123 等 3）
        else:
            fu = 0  # 两面
        if fu > best:
            best = fu
    return best if in_set else 2  # 单骑（和牌张为雀头）


def calculate_fu(concealed, melds, win_tile, tsumo, player_wind, round_wind,
                 yaku_names=()):
    """concealed: 13/14 张手牌 tile136（自摸时含摸牌；荣和时不含和牌张）。
    melds: 引擎副露 dict 列表。win_tile: tile136。返回天凤符数。"""
    hand = list(concealed)
    if not tsumo:
        hand.append(win_tile)
    win_kind = win_tile // 4
    counts = Counter(t // 4 for t in hand)
    if sum(counts.values()) != 14:
        return None  # 非法
    if _is_chiitoitsu(counts):
        # 高点法（永远取点数更高者）：七对子(25符 2番) 与 4面子1雀头
        # （二杯口 3番 等）是两种拆分，分别计算取高者。
        # 二杯口(3番) 点数高于七对子(2番) -> 按普通形计符；
        # 一杯口(1番) 低于七对子 -> 保持七对子 25 符。
        # （oracle-verified: f7a2693a E2, 778899m 334455p 88p 听7m单骑,
        #   天凤按二杯口 4番40符=8000满贯，而非七对子 25符=6400）
        if "Ryanpeikou" in yaku_names:
            pass  # 继续普通形（4面子1雀头）计算
        else:
            return 25
    if _is_kokushi(counts):
        return 25
    # 平和特殊处理
    if "Pinfu" in yaku_names:
        return 20 if tsumo else 30
    # 面子符（顺子 0；刻/杠）
    meld_fu = 0
    for m in melds:
        k = m["tiles"][0] // 4
        closed = not m.get("open")
        if m["type"] in ("kan", "shouminkan"):
            base = 16 if closed else 8   # 暗槓16/明槓8（中张），幺九×2
        elif m["type"] == "pon":
            base = 4 if closed else 2    # 暗刻4/明刻2（中张），幺九×2
        else:
            continue
        meld_fu += base if not is_terminal_or_honor(k) else base * 2
    # 底符与和牌方式
    base = 20
    open_melds = any(m["type"] in ("chi", "pon", "shouminkan") or m.get("open")
                     for m in melds)
    if not tsumo and not open_melds:
        base += 10  # 门清荣和
    if tsumo:
        base += 2
    # 拆解（高点法：取最大符）
    need_sets = 4 - len(melds)
    best = 0
    found = False
    for pk in range(34):
        if counts.get(pk, 0) < 2:
            continue
        c = dict(counts); c[pk] -= 2
        out = []
        _sets_dfs(c, need_sets, [], out)
        # 高点法：番优先于符。一杯口（Iipeiko）依赖"两组相同顺子"的拆法；
        # 只比符会偏向刻子拆法导致番数下降（oracle-verified: 6081421gm S3,
        # 333p444p555p456sNN 荣和3p：刻子拆法 2番50符 vs 一杯口拆法 3番40符，
        # 天凤取后者）。yaku 声明 Iipeiko 时拆解必须支持两组同顺。
        if "Iipeiko" in yaku_names:
            from collections import Counter as _RunC
            run_counts = _RunC(tuple(s) for s in out
                               if len(s) == 3 and s[0] != s[1])
            out = [sets for sets in out
                   if max(_RunC(tuple(s) for s in sets
                                if len(s) == 3 and s[0] != s[1]).values(),
                           default=0) >= 2]
        for sets in out:
            # 手牌面子符（双碰荣和完成的三张按明刻）+ 雀头符 + 待牌符
            win_in_run = any(len(s) == 3 and s[0] != s[1] and win_kind in s
                              for s in sets)
            sf = sum(_concealed_set_fu(s, win_kind, tsumo, win_in_run) for s in sets)
            pf = 0
            if pk in (31, 32, 33) or pk == player_wind or pk == round_wind:
                pf = 4 if (player_wind == round_wind and pk == player_wind) else 2
            # 高点法：和牌张同时可作雀头缺张时按单骑 +2
            # （oracle-verified: 4932fbac S4 E, 6p77p88p999p 234s WWW 自摸 9p:
            # 9p 在 789p 两面与 99p 单骑中，天凤取单骑 40 符而非两面 30 符）
            wf = 2 if win_kind == pk else _wait_fu(sets, win_kind)
            total = base + meld_fu + sf + pf + wf
            if total > best:
                best = total
            found = True
    if not found:
        return None
    # 副露（或特殊）合计 20 -> 30
    if best == 20:
        best = 30
    # 切上
    return (best + 9) // 10 * 10
