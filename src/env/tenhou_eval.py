# -*- coding: utf-8 -*-
"""综合高点法番符计算（重构版，2026-08-26）：按拆法枚举，每个拆法独立计算
【该拆法的番 + 该拆法的符】→ 查点数表得打点 → 取打点最高者。
番与符永远来自同一拆法（oracle 实证：333p444p555p456sNN 荣和3p，
一杯口拆法 3番40符 > 刻子拆法 2番50符；lib 的番/符可能来自不同拆法）。

拆法相关役（本模块按拆法判定；lib 役类名全集见 DECOMP_YAKU）：
  Pinfu/Iipeiko/Ryanpeikou/Sanshoku/Ittsu/Toitoi/Sanankou/SanKantsu/
  SanshokuDoukou/Chinitsu/Honitsu/Chanta/Junchan/Tanyao/Chun/Haku/Hatsu/
  SeatWind*/RoundWind*/Chiitoitsu/Kokushi/Honroto
拆法无关役（lib 判定，番固定）：
  立直/一发/自摸/宝牌/里宝/赤宝/枪杠/海底/河底/岭上/天和/地和/双立直/
  小三元(Shosangen 番数固定，役牌刻子由本模块重算，合计一致)

固定番 = lib_han - lib 拆法相关番（lib 拆法相关役名由 DECOMP_YAKU 识别，
副露减番按 lib 规则修正）。
"""
from collections import Counter
from env.tenhou_fu import (is_terminal_or_honor, _sets_dfs, _concealed_set_fu,
                           _wait_fu, _is_chiitoitsu, _is_kokushi)


def _is_terminal(k):
    return k < 27 and k % 9 in (0, 8)


def _is_honor(k):
    return k >= 27


def _is_open_meld(m):
    return m.get("open", False) or m["type"] in ("chi", "pon", "shouminkan")


def _meld_kinds(melds):
    """返回副露面子 kind 列表（[[k,k,k] 刻子 / [a,b,c] 顺子 / [k]*4 杠]）。"""
    out = []
    for m in melds or []:
        kinds = sorted(t // 4 for t in m["tiles"])
        if m["type"] in ("kan", "shouminkan"):
            out.append([kinds[0]] * 4)
        elif len(set(kinds)) == 1:
            out.append(kinds[:3])
        else:
            out.append(kinds)
    return out


# lib 判定的"拆法相关役"（本模块重新按拆法计算，从固定番中剔除）
DECOMP_YAKU = frozenset({
    "Pinfu", "Iipeiko", "Ryanpeikou", "Sanshoku", "Ittsu", "Toitoi",
    "Sanankou", "SanKantsu", "SanshokuDoukou", "Chinitsu", "Honitsu",
    "Chanta", "Chantai", "Junchan", "Tanyao", "Chiitoitsu", "Kokushi",
    "KokushiMusou", "Honroto", "Chun", "Haku", "Hatsu",
    "SeatWindEast", "SeatWindSouth", "SeatWindWest", "SeatWindNorth",
    "RoundWindEast", "RoundWindSouth", "RoundWindWest", "RoundWindNorth",
    # 兼容旧名（lib 不会返回，保留无害）
    "Yakuhai", "Sankantsu", "Sanseitou",
})

_LIB_HAN = {
    "Pinfu": 1, "Iipeiko": 1, "Ryanpeikou": 3, "Sanshoku": 2, "Ittsu": 2,
    "Toitoi": 2, "Sanankou": 2, "SanKantsu": 2, "SanshokuDoukou": 2,
    "Chinitsu": 6, "Honitsu": 3, "Chanta": 2, "Chantai": 2, "Junchan": 3,
    "Tanyao": 1, "Chiitoitsu": 2, "Kokushi": 13, "KokushiMusou": 13,
    "Honroto": 2, "Chun": 1, "Haku": 1, "Hatsu": 1,
    "SeatWindEast": 1, "SeatWindSouth": 1, "SeatWindWest": 1,
    "SeatWindNorth": 1, "RoundWindEast": 1, "RoundWindSouth": 1,
    "RoundWindWest": 1, "RoundWindNorth": 1,
    "Yakuhai": 1, "Sankantsu": 2, "Sanseitou": 2,
}

# 副露时减 1 番的役（lib 的 han_open 比 han_closed 少 1）
_OPEN_DECREMENT = frozenset({"Sanshoku", "Ittsu", "Chinitsu", "Honitsu",
                             "Chanta", "Chantai", "Junchan"})


def han_for_decomposition(sets, pk, win_kind, tsumo, melds, player_wind,
                          round_wind, kuitan):
    """按给定拆法计算【拆法相关役】的 (han, yaku_names)。
    sets: 手牌面子（kind 列表）；pk: 雀头 kind；win_kind: 和牌张 kind。"""
    meld_sets = _meld_kinds(melds)
    all_sets = list(sets) + meld_sets
    n_melds = len(melds or [])
    open_any = any(_is_open_meld(m) for m in (melds or []))
    closed = n_melds == 0
    yaku = []
    han = 0

    def add(name, h):
        nonlocal han
        han += h
        yaku.append((name, h))

    runs = [s for s in all_sets if len(s) == 3 and s[0] != s[1]]
    trips = [s for s in all_sets if len(s) == 3 and s[0] == s[1] == s[2]]
    kans = [s for s in all_sets if len(s) == 4]

    # ---- 役牌刻子（三元/自风/场风；连风算两个役牌）----
    yakuhai = 0
    for s in trips + kans:
        k = s[0]
        if k in (31, 32, 33):
            yakuhai += 1
        elif k == player_wind and k == round_wind:
            yakuhai += 2
        elif k == player_wind or k == round_wind:
            yakuhai += 1
    if yakuhai:
        add("Yakuhai", yakuhai)

    # ---- 断幺 ----
    if (all(not is_terminal_or_honor(k) for s in all_sets for k in s)
            and not is_terminal_or_honor(pk)):
        add("Tanyao", 1)

    # ---- 平和（门清、全顺子、两面听、非役牌雀头）----
    if closed and not trips and not kans and len(runs) == 4:
        pk_is_yakuhai = (pk in (31, 32, 33) or pk == player_wind
                         or pk == round_wind)
        if not pk_is_yakuhai:
            # 两面听：和牌张必须在顺子端部（a 或 a+2）且非边张。
            # 和牌张在顺子中张（a+1）= 坎张，Pinfu 不成立
            # （oracle-verified: 2026070217gm S? 789p789p789p 55p 234s 荣和8p：
            # 8p 是 789p 的坎张，天凤无 Pinfu）。
            ryanmen = False
            for s in runs:
                if win_kind not in s:
                    continue
                a = s[0]
                if ((win_kind == a and a % 9 != 6) or
                        (win_kind == a + 2 and a % 9 != 0)):
                    ryanmen = True
                    break
            if ryanmen:
                add("Pinfu", 1)

    # ---- 一杯口 / 二杯口（门清）----
    if closed:
        rc = Counter(tuple(s) for s in runs)
        pairs = [v // 2 for v in rc.values()]
        n_iipeiko = sum(pairs)
        if n_iipeiko >= 2:
            add("Ryanpeikou", 3)
        elif n_iipeiko == 1:
            add("Iipeiko", 1)

    # ---- 三色同顺（2番门清 / 1番副露）----
    seq = {}
    for s in runs:
        base = s[0] % 9
        suit = s[0] // 9
        seq.setdefault(base, set()).add(suit)
    if any(len(v) == 3 for v in seq.values()):
        add("Sanshoku", 2 if closed else 1)

    # ---- 一气通贯（2番门清 / 1番副露）----
    for suit in range(3):
        kinds = {s[0] % 9 for s in runs if s[0] // 9 == suit}
        if {0, 3, 6} <= kinds:
            add("Ittsu", 2 if closed else 1)
            break

    # ---- 对对和 ----
    if len(trips) + len(kans) == 4:
        add("Toitoi", 2)

    # ---- 三暗刻（荣和时和牌张完成的刻子=明刻；但若和牌张同时进顺子
    #      （高点法取顺子，坎张/两面 +2符），刻子未被和牌完成 → 仍为暗刻。
    #      oracle-verified: 89f831e0 E4, 111m11p222p123s222s 荣和2s：
    #      天凤三暗刻 2番50符，2s 进 123s 顺子）
    n_ankan = 0
    win_in_run = any(len(s) == 3 and s[0] != s[1] and win_kind in s for s in sets)
    for s in sets:
        if len(s) == 3 and s[0] == s[1] == s[2]:
            if tsumo:
                n_ankan += 1
            elif win_kind != s[0] or win_in_run:
                n_ankan += 1
    for m in (melds or []):
        if m["type"] == "ankan":
            n_ankan += 1
        elif m["type"] in ("kan", "shouminkan") and not _is_open_meld(m):
            n_ankan += 1
    if n_ankan >= 3:
        add("Sanankou", 2)

    # ---- 三杠子 ----
    if len(kans) >= 3:
        add("SanKantsu", 2)

    # ---- 三色同刻（仅数牌刻子；字牌 %9 会与数牌撞号，须排除
    #      oracle-verified: 375e5d80 S1, 77m 111p 111s 777s EEE 荣和1s：
    #      E(27)%9=0 与 1p/1s 同组被误判，天凤无三色同刻）----
    tri_kind = {}
    for s in trips + kans:
        if s[0] >= 27:
            continue
        tri_kind.setdefault(s[0] % 9, set()).add(s[0] // 9)
    if any(len(v) == 3 for v in tri_kind.values()):
        add("SanshokuDoukou", 2)

    # ---- 混一色 / 清一色 ----
    has_honor = pk >= 27 or any(k >= 27 for s in all_sets for k in s)
    suits = set()
    for s in all_sets:
        for k in s:
            if k < 27:
                suits.add(k // 9)
    if pk < 27:
        suits.add(pk // 9)
    if not has_honor and len(suits) == 1 and suits:
        add("Chinitsu", 6 if closed else 5)
    elif has_honor and len(suits) == 1 and suits:
        add("Honitsu", 3 if closed else 2)

    # ---- 混全带幺 / 纯全带幺 / 混老头 ----
    n_chi = sum(1 for s in all_sets if len(s) == 3 and s[0] != s[1])
    sets_term_hon = all(is_terminal_or_honor(s[0]) or is_terminal_or_honor(s[-1])
                        for s in all_sets)
    if sets_term_hon and is_terminal_or_honor(pk):
        has_term_grp = any(_is_terminal(s[0]) or _is_terminal(s[-1])
                           for s in all_sets) or _is_terminal(pk)
        has_honor_grp = _is_honor(pk) or any(_is_honor(s[0]) or _is_honor(s[-1])
                                             for s in all_sets)
        if not has_honor and n_chi and has_term_grp:
            # 纯全带幺：无字牌、含顺子、每面子（含雀头）端为 1/9
            add("Junchan", 3 if closed else 2)
        elif has_honor and n_chi and has_term_grp and has_honor_grp:
            # 混全带幺：lib 要求含顺子、有 terminal 组也有 honor 组
            add("Chantai", 2 if closed else 1)
    # 混老头：全部牌都是幺九或字牌
    if (all(is_terminal_or_honor(k) for s in all_sets for k in s)
            and is_terminal_or_honor(pk)):
        add("Honroto", 2)

    return han, yaku


def _points_table(han, fu, tsumo, is_dealer, kiriage=False):
    """用 lib 点数表计算打点（用于拆法间比较；与 riichi_game 结算口径一致）。"""
    from mahjong.hand_calculating.scores import ScoresCalculator
    from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
    from mahjong.constants import EAST, SOUTH
    cfg = HandConfig(is_tsumo=tsumo,
                     player_wind=EAST if is_dealer else SOUTH,
                     options=OptionalRules(kiriage=kiriage))
    c = ScoresCalculator().calculate_scores(han=han, fu=fu, config=cfg,
                                            is_yakuman=False)
    return c["total"]


def _fu_for_decomposition(sets, pk, win_kind, tsumo, melds, player_wind,
                          round_wind, yaku, counts):
    """按给定拆法计算该拆法的符（天凤口径）。

    - 平和：自摸 20 符 / 荣和 30 符（自摸符不加、门清荣和 10 符）
    - 底符 20；门清荣和 +10；自摸 +2（非平和）
    - 面子符（手牌刻子按暗刻/明刻，杠按暗杠/明杠）+ 雀头符 + 待牌符
    - 单骑：和牌张为雀头且手中恰 2 张该牌
    - 切上：合计非 10 倍数向上取整；合计 20 -> 30（副露/非平和时）
    """
    names = {n for n, _ in yaku}
    if "Pinfu" in names:
        return 20 if tsumo else 30

    open_any = any(_is_open_meld(m) for m in (melds or []))
    base = 20
    if not tsumo and not open_any:
        base += 10
    if tsumo:
        base += 2
    meld_fu = 0
    for m in melds or []:
        k = m["tiles"][0] // 4
        closed = not _is_open_meld(m)
        if m["type"] in ("kan", "shouminkan"):
            meld_fu += (16 if closed else 8) * (2 if is_terminal_or_honor(k) else 1)
        elif m["type"] == "pon":
            meld_fu += (4 if closed else 2) * (2 if is_terminal_or_honor(k) else 1)
    win_in_run = any(len(s) == 3 and s[0] != s[1] and win_kind in s for s in sets)
    sf = sum(_concealed_set_fu(s, win_kind, tsumo, win_in_run) for s in sets)
    pf = 0
    if pk in (31, 32, 33) or pk == player_wind or pk == round_wind:
        pf = 4 if (player_wind == round_wind and pk == player_wind) else 2
    # 单骑：和牌张可作雀头即 +2（高点法，即便和牌张同时进顺子
    # oracle-verified: 4932fbac S4 E, 6p77p88p999p 234s WWW 自摸 9p:
    # 9p 在 789p 两面与 99p 单骑中，天凤取单骑 40 符而非两面 30 符）
    wf = 2 if win_kind == pk else _wait_fu(sets, win_kind)
    fu = base + meld_fu + sf + pf + wf
    if fu == 20:
        fu = 30
    return (fu + 9) // 10 * 10


def evaluate_hand(concealed, melds, win_tile, tsumo, player_wind, round_wind,
                  is_dealer, lib_yaku_objs, lib_han, kiriage=False):
    """按拆法枚举，返回打点最高的 (han, fu, yaku_names)。

    concealed: 13/14 张手牌 tile136（自摸时含摸牌；荣和时不含和牌张）。
    melds: 引擎副露 dict 列表。win_tile: tile136。
    lib_yaku_objs: mahjong-lib 判定的役对象列表（拆法无关番由此固定）。
    lib_han: lib 总番（含宝牌/立直等拆法无关役）。
    kiriage: 切上满贯（cfg.kiriage_mangan）。
    """
    hand = list(concealed)
    if not tsumo:
        hand.append(win_tile)
    win_kind = win_tile // 4
    counts = Counter(t // 4 for t in hand)
    if sum(counts.values()) != 14:
        return None
    open_any = any(_is_open_meld(m) for m in (melds or []))
    # 拆法无关番（固定）：lib 总番 - lib 拆法相关番
    lib_decomp = 0
    fixed_names = []
    for y in lib_yaku_objs or []:
        name = type(y).__name__
        if name in DECOMP_YAKU:
            h = _LIB_HAN.get(name, 0)
            if open_any and name in _OPEN_DECREMENT:
                h = max(1, h - 1)
            lib_decomp += h
        else:
            fixed_names.append(name)
    fixed_han = lib_han - lib_decomp
    if fixed_han < 0:
        fixed_han = 0

    best = None  # (points, han, fu, yaku_names)

    # 七对子 / 国士 特殊形（25 符，不切上）
    if _is_chiitoitsu(counts):
        # 七对子拆法相关役：断幺/混一色/清一色/混老头（lib 已判，fixed_han
        # 已剔除，此处按拆法重加；二杯口等普通形在 4面子枚举中点数更高者胜）
        yaku7 = ["Chiitoitsu"]
        h7 = 2
        if all(not is_terminal_or_honor(k) for k in counts):
            yaku7.append("Tanyao"); h7 += 1
        suits7 = {k // 9 for k in counts if k < 27}
        if all(is_terminal_or_honor(k) for k in counts):
            yaku7.append("Honroto"); h7 += 2
        elif len(suits7) == 1:
            if any(k >= 27 for k in counts):
                yaku7.append("Honitsu"); h7 += 3
            else:
                yaku7.append("Chinitsu"); h7 += 6
        han = h7 + fixed_han
        fu = 25
        pts = _points_table(han, fu, tsumo, is_dealer, kiriage)
        best = (pts, han, fu, yaku7 + fixed_names)
    if _is_kokushi(counts):
        han = 13 + fixed_han
        fu = 25
        pts = _points_table(han, fu, tsumo, is_dealer, kiriage)
        if best is None or pts > best[0]:
            best = (pts, han, fu, ["Kokushi"] + fixed_names)

    # 4面子1雀头枚举（每个拆法独立番+符）
    need_sets = 4 - len(melds or [])
    for pk in range(34):
        if counts.get(pk, 0) < 2:
            continue
        c = dict(counts)
        c[pk] -= 2
        out = []
        _sets_dfs(c, need_sets, [], out)
        for sets in out:
            h, y = han_for_decomposition(sets, pk, win_kind, tsumo, melds,
                                         player_wind, round_wind, False)
            total_han = h + fixed_han
            yn = [n for n, _ in y] + fixed_names
            fu = _fu_for_decomposition(sets, pk, win_kind, tsumo, melds,
                                       player_wind, round_wind, y, counts)
            pts = _points_table(total_han, fu, tsumo, is_dealer, kiriage)
            if best is None or pts > best[0]:
                best = (pts, total_han, fu, yn)
    if best is None:
        return None
    return {"han": best[1], "fu": best[2], "yaku": best[3], "points": best[0]}
