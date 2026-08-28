# -*- coding: utf-8 -*-
"""天凤精确点数表（非役满）。

mahjong-lib 的 ScoresCalculator 用 "ceil100(base*2)" 统一公式，与天凤实际表值
有两处偏差（真实天凤日志 oracle 实证）：
  1. 30符4番 亲家自摸: 天凤表值 = 4000 all（lib 公式 = 3900）
     证据: a6985d96 kyoku4 E 4番30符自摸 oracle deltas=[-4000,-4000,-4000,+13000]
           （13000 = 3*4000 + 1 立直棒 1000）
  2. 20符1番 自摸: 天凤表值 子 400/700、亲 700 all（公式 160->200/400 错误，
     天凤表对低基本点有最低档：ツモ 1番 最低 400/700）
其余格与标准日麻表一致（雀姬速查表互验，公式 base=fu*2^(2+han) 逐格核对）。

返回值结构与 mahjong ScoresCalculator 相同，可直接替换。
"""

# 子家自摸: (子付, 亲付)  [符][番]
CHILD_TSUMO = {
    20: {1: (400, 700), 2: (700, 1300), 3: (1300, 2600), 4: (2600, 5200)},
    25: {2: (800, 1600), 3: (1600, 3200), 4: (3200, 6400)},
    30: {1: (300, 500), 2: (500, 1000), 3: (1000, 2000), 4: (2000, 3900)},
    40: {1: (400, 700), 2: (700, 1300), 3: (1300, 2600), 4: (2600, 5200)},
    50: {1: (400, 800), 2: (800, 1600), 3: (1600, 3200), 4: (3200, 6400)},
    60: {1: (500, 1000), 2: (1000, 2000), 3: (2000, 3900), 4: (2000, 4000)},
    70: {2: (1200, 2300), 3: (2000, 4000), 4: (2000, 4000)},
}
# 亲家自摸: 每家支付 [符][番]
DEALER_TSUMO = {
    20: {1: 700, 2: 1300, 3: 2600, 4: 5200},
    25: {2: 1600, 3: 3200, 4: 6400},
    30: {1: 500, 2: 1000, 3: 2000, 4: 4000},   # 4番 = 4000（天凤表值）
    40: {1: 700, 2: 1300, 3: 2600, 4: 5200},
    50: {1: 800, 2: 1600, 3: 3200, 4: 6400},
    60: {1: 1000, 2: 2000, 3: 3900, 4: 4000},
    70: {2: 2300, 3: 4000, 4: 4000},
}
# 子家荣和 [符][番]
CHILD_RON = {
    20: {1: 700, 2: 1300, 3: 2600, 4: 5200},
    25: {2: 1600, 3: 3200, 4: 6400},
    30: {1: 1000, 2: 2000, 3: 3900, 4: 7700},
    40: {1: 1300, 2: 2600, 3: 5200, 4: 8000},
    50: {1: 1600, 2: 3200, 3: 6400, 4: 8000},
    60: {1: 2000, 2: 3900, 3: 7700, 4: 8000},
    70: {2: 4500, 3: 8000, 4: 8000},
}
# 亲家荣和 [符][番]
DEALER_RON = {
    20: {1: 1000, 2: 2000, 3: 3900, 4: 7700},
    25: {2: 2400, 3: 4800, 4: 9600},
    30: {1: 1500, 2: 2900, 3: 5800, 4: 11600},
    40: {1: 2000, 2: 3900, 3: 7700, 4: 12000},
    50: {1: 2400, 2: 4800, 3: 9600, 4: 12000},
    60: {1: 2900, 2: 5800, 3: 11600, 4: 12000},
    70: {2: 6800, 3: 12000, 4: 12000},
}

MAN_TIERS = {  # han -> 基本点（天凤: 双重役满按单倍 13番起算）
    13: 8000, 11: 6000, 8: 4000, 6: 3000, 5: 2000,
}


def _level(han, capped, kiriage=False):
    if han >= 13:
        return "yakuman"
    if han >= 11:
        return "sanbaiman"
    if han >= 8:
        return "baiman"
    if han >= 6:
        return "haneman"
    if han >= 5 or capped:
        return "kiriage mangan" if kiriage else "mangan"
    return ""


def _fallback(han, fu, tsumo, is_dealer):
    """80+ 符等表外格：base=fu*2^(2+han)（cap 2000）公式。"""
    base = fu * (1 << (2 + han))
    if base > 2000:
        base = 2000
    r100 = lambda x: (x + 99) // 100 * 100
    if not tsumo:
        return (r100(base * (6 if is_dealer else 4)), 0)
    if is_dealer:
        return (r100(base * 2), r100(base * 2))
    return (r100(base * 2), r100(base))


def tenhou_cost(han, fu, tsumo, is_dealer, honba=0, kyoutaku=0, kiriage=False):
    """天凤精确点数 dict（结构与 mahjong ScoresCalculator 相同）。

    约定与引擎 _settle_* 一致: 子家自摸时 main = 亲家支付, additional = 子家支付；
    荣和时 main = 总支付。表值仅在非满贯（han<5 且基本点<=2000）时使用，
    满贯以上统一公式（与天凤一致）。
    """
    capped = False
    kiriage_level = False
    if han >= 5:
        if han >= 13:
            base = 8000; capped = True
        elif han >= 11:
            base = 6000; capped = True
        elif han >= 8:
            base = 4000; capped = True
        elif han >= 6:
            base = 3000; capped = True
        else:
            base = 2000; capped = True
    else:
        base = fu * (1 << (2 + han))
        if kiriage and ((han == 4 and fu == 30) or (han == 3 and fu == 60)):
            base = 2000; capped = True; kiriage_level = True
        elif base > 2000:
            base = 2000; capped = True

    r100 = lambda x: (x + 99) // 100 * 100
    if not tsumo:
        main = r100(base * (6 if is_dealer else 4))
        main_bonus = 300 * honba
        additional = 0
        additional_bonus = 0
    elif is_dealer:
        if not capped and DEALER_TSUMO.get(fu, {}).get(han) is not None:
            main = DEALER_TSUMO[fu][han]
        else:
            main = r100(base * 2)
        main_bonus = 100 * honba
        additional = main
        additional_bonus = 100 * honba
    else:
        pair = CHILD_TSUMO.get(fu, {}).get(han)
        if not capped and pair is not None:
            child_pay, dealer_pay = pair   # 表: (子付, 亲付)
            main = dealer_pay
            additional = child_pay
        else:
            main = r100(base * 2)
            additional = r100(base)
        main_bonus = 100 * honba
        additional_bonus = 100 * honba

    kyoutaku_bonus = 1000 * kyoutaku
    total = (main + main_bonus) + 2 * (additional + additional_bonus) + kyoutaku_bonus
    return {
        "main": main, "additional": additional,
        "main_bonus": main_bonus, "additional_bonus": additional_bonus,
        "kyoutaku_bonus": kyoutaku_bonus,
        "total": total,
        "yaku_level": _level(han, capped, kiriage_level),
    }

