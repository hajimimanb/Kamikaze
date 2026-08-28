# -*- coding: utf-8 -*-
"""Precise shanten features for the SL feature tensor (48M-sample budget).

Design (researched 2026-08-25; alternatives: shanten-dp / riichi-tools-rs /
per-suit DP — see docs; we keep the battle-tested 'mahjong' cmj3 as oracle):

- Decision-point hand sizes are always 14-3m (discard/riichi: drawn then
  discard) or 13-3m (call decisions). Both are valid inputs for
  calculate_shanten, which assumes (14-n)//3 open melds — exactly our
  semantics (closed tiles only; melds are a separate record field).
- Per-tile shanten-after-discard (切X后向听数) is computed only on 14-3m
  hands (resulting size 13-3m, valid). 12-tile inputs never occur at
  decision points (the package rejects 12), so no special casing is needed.
- ukeire (进张数) applies to 13-3m hands only (adding a tile -> 14-3m),
  and is gated to shanten<=2 for cost (34 add-tile evaluations).
"""
from __future__ import annotations

from mahjong.shanten import Shanten
from tenhou.shanten_tmp import shanten34


def counts34(tiles) -> list:
    """tile136 list -> 34-length kind counts."""
    c = [0] * 34
    for t in tiles:
        c[t // 4] += 1
    return c


def shanten_of(counts) -> int:
    """Regular+chiitoi+kokushi shanten of closed kind-counts (-1 = agari)."""
    return shanten34(counts)


def shanten_regular(counts) -> int:
    """一般形向听数 (4 面子 + 1 雀头), -1 = 和了."""
    return Shanten.calculate_shanten_for_regular_hand(list(counts))


def shanten_chiitoi(counts) -> int:
    """七对子向听数, -1 = 和了."""
    return Shanten.calculate_shanten_for_chiitoitsu_hand(list(counts))


def shanten_kokushi(counts) -> int:
    """国士无双向听数, -1 = 和了."""
    return Shanten.calculate_shanten_for_kokushi_hand(list(counts))


def per_tile_shanten(counts) -> list:
    """34-vector: shanten after discarding one tile of kind k.

    Kinds not present in hand keep the current shanten (neutral, they are
    not discard candidates). Caller must only pass 14-3m hands.
    """
    base = list(counts)
    cur = shanten34(base)
    out = [cur] * 34
    for k in range(34):
        if base[k] > 0:
            c = list(base)
            c[k] -= 1
            out[k] = shanten34(c)
    return out


def ukeire(counts) -> int:
    """进张数: distinct kinds (with <4 copies available) whose addition
    lowers the shanten. Only meaningful for 13-3m hands."""
    cur = shanten34(counts)
    if cur > 2:  # cost gate: ukeire matters near tenpai
        return 0
    base = list(counts)
    n = 0
    for k in range(34):
        if base[k] >= 4:
            continue
        c = list(base)
        c[k] += 1
        if shanten34(c) < cur:
            n += 1
    return n


def tenpai_waits(counts) -> list:
    """34 0/1: kinds completing the hand when tenpai (shanten == 0).
    Non-tenpai hands return all zeros. Only for 13-3m hands."""
    out = [0] * 34
    if shanten34(counts) != 0:
        return out
    base = list(counts)
    for k in range(34):
        if base[k] >= 4:
            continue
        c = list(base)
        c[k] += 1
        if shanten34(c) == -1:
            out[k] = 1
    return out
