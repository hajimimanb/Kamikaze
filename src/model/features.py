# -*- coding: utf-8 -*-
"""Suphx-style observation feature encoder with mode switch.

Captain supplement (three-stage curriculum learning):
  mode="simple"  -> curriculum stage 1: hand + dora + basic scalar channels ONLY
                    (no discards / melds / field situation), for the tsumogiri
                    baseline model. 17 channels.
  mode="full"    -> expanded Suphx-style baseline (~120 channels): per-seat
                    discards (count n1..n4 + tsumogiri/riichi/red/early/late),
                    per-seat melds (count n1..n4 + red + type), dora indicators
                    and dora kinds (counts + binary tiers), visible tile
                    counts, score buckets/point diffs, winds/rounds/honba
                    one-hots, last-event encoding, etc.
                    Look-ahead / 838-958 channel Suphx features are NOT
                    implemented yet, but the registry is designed to extend.

Input contract: docs/observation_schema.md v1.1
  - tile136 = kind*4 + copy, kind 0..33 (0-8 man, 9-17 pin, 18-26 sou,
    27-30 E/S/W/N, 31-33 haku/hatsu/chun)
  - red5 fixed: 5m=16, 5p=52, 5s=88 (kind 4/13/22 copy 0)
  - record keys: hand(tile136 list), melds[4], discards[4], dora_indicators,
    scores[4], oya, round, honba, riichi_sticks, wall_left, n_kan,
    riichi_declared[4], last_event, legal_actions, label, game_id

Output: (C, 34, 1) float32 tensor (Suphx layout: 34 tile-kind columns).

Channel registry is data-driven: every channel has (group, modes, name,
description, fill_fn); the layout (names + descriptions + quantization scale)
is exported to JSON for explainability (docs/channel_layout.json).

Tile helpers come from the shared src/riichi/tiles.py (schema §1 contract);
MJAI-notation strings can be parsed with riichi.tiles.parse_mpsz for the
MJAI-protocol compatibility checks (schema §7.5).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from riichi.tiles import (
    RED_FIVES as _RED_FIVES,
    RED_FIVE_KINDS as _RED_FIVE_KINDS,
    counts34 as _counts34_list,
    kind_of,
    next_dora_kind,
)
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld
from model.shanten_feat import (
    counts34 as _shanten_counts34,
    per_tile_shanten as _per_tile_shanten,
    shanten_of as _shanten_of,
    shanten_chiitoi as _shanten_chiitoi,
    shanten_kokushi as _shanten_kokushi,
    shanten_regular as _shanten_regular,
    tenpai_waits as _tenpai_waits,
    ukeire as _shanten_ukeire,
)

N_KINDS = 34
RED5_TILES = set(_RED_FIVES)
RED5_KINDS = set(_RED_FIVE_KINDS)
WIND_KINDS = (27, 28, 29, 30)          # E S W N
DRAW_EVENT_TYPES = ("draw", "tsumo", "dahai")  # last_event types carrying a drawn tile
VALID_MODES = ("simple", "full")
# multi-task binary action heads. ron/tsumo/kyushu are bool-valued
# decision heads (win / tsumo / nine-terminals abortive draw): they learn
# BOTH "take the action" (label matches) and "decline" (label is a different
# action while the option was legal) from the existing records.
BINARY_HEADS = ("riichi", "chow", "pon", "kan", "ron", "tsumo", "kyushu")

# ---------------------------------------------------------------------------
# tile utilities — reuse shared src/riichi/tiles.py (schema §1 contract)
# ---------------------------------------------------------------------------


def tile_to_kind(t: int) -> int:
    """tile136 -> kind 0..33 (shared riichi.tiles.kind_of)."""
    return kind_of(int(t))


def counts34(tiles) -> np.ndarray:
    """list of tile136 -> (34,) float32 count vector."""
    return np.asarray(_counts34_list(tiles), dtype=np.float32)


def indicator_to_dora_kind(kind: int) -> int:
    """Kind of dora indicated by an indicator tile of that kind."""
    return next_dora_kind(int(kind))


def score_bucket_col(score: int) -> int:
    """10k-point bucket -> column for one-hot (0,3,6,...,30; overflow -> 33)."""
    b = min(max(int(score) // 10000, 0), 11)
    return min(b * 3, 33)


# ---------------------------------------------------------------------------
# channel registry
# ---------------------------------------------------------------------------

# Each entry: (group, modes, name, description, fill_fn(obs, out34))
# fill_fn writes one 34-vector into out34 (np.float32 array).
CHANNELS: list = []


def _reg(group: str, modes: tuple, name: str, desc: str, fill) -> None:
    CHANNELS.append((group, tuple(modes), name, desc, fill))


# ================= SIMPLE (curriculum stage 1) =================

# --- hand (simple + full) ---
for _n in range(1, 5):
    _reg("hand", ("simple", "full"), f"hand_count{_n}",
         f"手牌中该牌种持有张数 >= {_n}（0/1）",
         lambda obs, out, n=_n: np.add(
             out, (counts34(obs["hand"]) >= n).astype(np.float32), out=out))


def _fill_hand_red(obs, out):
    for t in obs["hand"]:
        if t in RED5_TILES:
            out[tile_to_kind(t)] = 1.0


_reg("hand", ("simple", "full"), "hand_red5",
     "手牌含赤5（万/筒/索）标记", _fill_hand_red)


# --- dora (simple + full) ---
def _fill_dora_indicators(obs, out):
    np.add(out, counts34(obs.get("dora_indicators", [])), out=out)


def _fill_dora_kinds(obs, out):
    for t in obs.get("dora_indicators", []):
        k = indicator_to_dora_kind(tile_to_kind(t))
        out[k] = min(out[k] + 1.0, 4.0)


_reg("dora", ("simple", "full"), "dora_indicators",
     "可见宝牌指示牌计数（含杠宝指示，最大5）", _fill_dora_indicators)
_reg("dora", ("simple", "full"), "dora_kinds",
     "指示牌对应的宝牌种类计数（上限4）", _fill_dora_kinds)


# --- basic scalar context (simple + full) ---
def _fill_last_draw(obs, out):
    ev = obs.get("last_event") or {}
    if ev.get("type") in DRAW_EVENT_TYPES and "tile" in ev:
        out[tile_to_kind(ev["tile"])] = 1.0


_reg("basic", ("simple", "full"), "last_draw",
     "刚摸的牌标记（摸后决策点）", _fill_last_draw)


def _fill_seat_wind(obs, out):
    out[WIND_KINDS[(obs["seat"] - obs["oya"]) % 4]] = 1.0


_reg("basic", ("simple", "full"), "seat_wind",
     "自风 one-hot（列 27-30 = 东南西北）", _fill_seat_wind)


def _fill_prevalent_wind(obs, out):
    out[WIND_KINDS[(obs.get("round", 0) // 4) % 2]] = 1.0


_reg("basic", ("simple", "full"), "prevalent_wind",
     "场风 one-hot（东/南）", _fill_prevalent_wind)


def _fill_round(obs, out):
    out[:] = min(obs.get("round", 0), 7) / 7.0


_reg("basic", ("simple", "full"), "round",
     "局数归一化（0..7 -> 0..1）", _fill_round)


def _fill_honba(obs, out):
    out[:] = min(obs.get("honba", 0), 4) / 4.0


_reg("basic", ("simple", "full"), "honba", "本场数归一化", _fill_honba)


def _fill_riichi_sticks(obs, out):
    out[:] = min(obs.get("riichi_sticks", 0), 4) / 4.0


_reg("basic", ("simple", "full"), "riichi_sticks", "立直棒数归一化", _fill_riichi_sticks)


def _fill_wall_left(obs, out):
    out[:] = min(max(obs.get("wall_left", 0), 0), 70) / 70.0


_reg("basic", ("simple", "full"), "wall_left", "牌山剩余归一化（0..70）", _fill_wall_left)


def _fill_own_score(obs, out):
    out[score_bucket_col(obs["scores"][obs["seat"]])] = 1.0


_reg("basic", ("simple", "full"), "own_score_bucket",
     "自身点数 10k 分桶 one-hot", _fill_own_score)


def _fill_rank(obs, out):
    s = list(obs["scores"])
    rank = 1 + sum(1 for x in s if x > s[obs["seat"]])  # 1-based, ties share top
    out[:] = (rank - 1) / 3.0


_reg("basic", ("simple", "full"), "rank", "自身顺位归一化（1位=0）", _fill_rank)


def _fill_oya_flag(obs, out):
    out[:] = 1.0 if obs["seat"] == obs.get("oya", 0) else 0.0


_reg("basic", ("simple", "full"), "is_oya", "是否亲家", _fill_oya_flag)


# ================= FULL-ONLY extensions =================

# --- visible tile counts (full): hand + melds + discards + dora indicators ---
def _visible_counts(obs):
    cnt = counts34(obs["hand"]) + counts34(obs.get("dora_indicators", []))
    for s in range(4):
        for m in obs["melds"][s]:
            for t in m["tiles"]:
                cnt[tile_to_kind(t)] += 1.0
        for d in obs["discards"][s]:
            cnt[tile_to_kind(d["tile"])] += 1.0
    return cnt


def _fill_visible_count(obs, out):
    out[:] = np.minimum(_visible_counts(obs), 4.0) / 4.0


_reg("visible", ("full",), "visible_count",
     "全桌可见该牌种张数/4（手牌+副露+牌河+宝牌指示）", _fill_visible_count)


# --- discards per seat (full): ordered river (24 pos) + aggregated marks ---
MAX_RIVER = 24


def _mk_river(seat_i: int, pos: int):
    def _f(obs, out, s=seat_i, p=pos):
        ds = obs["discards"][s]
        if p < len(ds):
            out[tile_to_kind(ds[p]["tile"])] = 1.0
    return _f


def _mk_river_handcut(seat_i: int, pos: int):
    def _f(obs, out, s=seat_i, p=pos):
        ds = obs["discards"][s]
        if p < len(ds) and not ds[p].get("tsumogiri"):
            out[tile_to_kind(ds[p]["tile"])] = 1.0
    return _f


for _s in range(4):
    for _i in range(MAX_RIVER):
        _reg("river", ("full",), f"river_s{_s}_pos{_i}",
             f"座位{_s} 牌河第{_i + 1}张的牌种（one-hot；未切=0）", _mk_river(_s, _i))
        _reg("river", ("full",), f"river_handcut_s{_s}_pos{_i}",
             f"座位{_s} 牌河第{_i + 1}张是否手切（1=手切/0=摸切）", _mk_river_handcut(_s, _i))


def _mk_discard_count(seat_i: int):
    def _f(obs, out, s=seat_i):
        cnt = counts34([d["tile"] for d in obs["discards"][s]])
        out[:] = np.minimum(cnt, 4.0) / 4.0
    return _f


def _mk_discard_mark(seat_i: int, kind: str):
    def _f(obs, out, s=seat_i):
        for d in obs["discards"][s]:
            hit = ((kind == "riichi" and d.get("riichi")) or
                   (kind == "red" and d["tile"] in RED5_TILES))
            if hit:
                out[tile_to_kind(d["tile"])] = 1.0
    return _f


for _s in range(4):
    _reg("discards", ("full",), f"discard_count_s{_s}",
         f"座位{_s} 牌河该牌种张数/4", _mk_discard_count(_s))
    for _k in ("riichi", "red"):
        _reg("discards", ("full",), f"discard_{_k}_s{_s}",
             f"座位{_s} 牌河{_k}标记", _mk_discard_mark(_s, _k))


# --- melds per seat (full): 6 channels each ---
def _mk_meld_count(seat_i: int):
    def _f(obs, out, s=seat_i):
        cnt = np.zeros(N_KINDS, dtype=np.float32)
        for m in obs["melds"][s]:
            for t_ in m["tiles"]:
                cnt[tile_to_kind(t_)] += 1.0
        out[:] = np.minimum(cnt, 4.0) / 4.0
    return _f


def _mk_meld_total(seat_i: int):
    def _f(obs, out, s=seat_i):
        out[:] = min(len(obs["melds"][s]), 4) / 4.0
    return _f


def _mk_meld_red(seat_i: int):
    def _f(obs, out, s=seat_i):
        for m in obs["melds"][s]:
            if m.get("red"):
                for t in m["tiles"]:
                    if t in RED5_TILES:
                        out[tile_to_kind(t)] = 1.0
    return _f


def _mk_meld_type(seat_i: int):
    def _f(obs, out, s=seat_i):
        vals = {"chow": 1.0 / 3.0, "chi": 1.0 / 3.0, "pon": 2.0 / 3.0}
        for m in obs["melds"][s]:
            v = vals.get(m.get("type"), 1.0)  # kan (daiminkan/ankan/kakan) -> 1
            for t in m["tiles"]:
                out[tile_to_kind(t)] = max(out[tile_to_kind(t)], v)
    return _f


for _s in range(4):
    _reg("melds", ("full",), f"meld_count_s{_s}",
         f"座位{_s} 副露该牌种张数/4", _mk_meld_count(_s))
    _reg("melds", ("full",), f"meld_total_s{_s}",
         f"座位{_s} 副露组数/4", _mk_meld_total(_s))
    _reg("melds", ("full",), f"meld_red_s{_s}",
         f"座位{_s} 副露赤5标记", _mk_meld_red(_s))
    _reg("melds", ("full",), f"meld_type_s{_s}",
         f"座位{_s} 副露类型（吃1/3、碰2/3、杠1）", _mk_meld_type(_s))


# --- global state (full) ---
def _fill_n_kan(obs, out):
    out[:] = min(obs.get("n_kan", 0), 4) / 4.0


_reg("state", ("full",), "n_kan", "场上总杠数归一化", _fill_n_kan)


def _mk_riichi_declared(seat_i: int):
    def _f(obs, out, s=seat_i):
        if obs.get("riichi_declared", [False] * 4)[s]:
            out[:] = 1.0
    return _f


for _s in range(4):
    _reg("state", ("full",), f"riichi_declared_s{_s}",
         f"座位{_s} 是否已立直", _mk_riichi_declared(_s))


def _mk_score_all(seat_i: int):
    def _f(obs, out, s=seat_i):
        out[score_bucket_col(obs["scores"][s])] = 1.0
    return _f


for _s in range(4):
    _reg("state", ("full",), f"score_bucket_s{_s}",
         f"座位{_s} 点数 10k 分桶 one-hot", _mk_score_all(_s))


def _mk_point_diff(i: int):
    def _f(obs, out, idx=i):
        other = (obs["seat"] + 1 + idx) % 4
        diff = (obs["scores"][obs["seat"]] - obs["scores"][other] + 40000.0) / 80000.0
        out[:] = min(max(diff, 0.0), 1.0)
    return _f


for _i in range(3):
    _reg("state", ("full",), f"point_diff_{_i}",
         f"与下家/对家/上家的点差归一化（+40000 平移至 0..1）",
         _mk_point_diff(_i))


_EVENT_TYPE_COLS = {
    "draw": 0, "tsumo": 0, "dahai": 3, "discard": 3,
    "chi": 6, "chow": 6, "pon": 9, "daiminkan": 12, "ankan": 12,
    "kakan": 12, "riichi": 15, "ron": 18, "hora": 18,
}


def _fill_last_event_type(obs, out):
    ev = obs.get("last_event") or {}
    col = _EVENT_TYPE_COLS.get(ev.get("type"), 21)
    out[min(col, 33)] = 1.0


def _fill_last_event_seat(obs, out):
    ev = obs.get("last_event") or {}
    if ev.get("seat") is not None:
        out[:] = ev["seat"] / 3.0


def _fill_last_event_tile(obs, out):
    ev = obs.get("last_event") or {}
    if "tile" in ev:
        out[tile_to_kind(ev["tile"])] = 1.0


_reg("state", ("full",), "last_event_type",
     "上一事件类型列编码（摸/切/吃/碰/杠/立直/和）", _fill_last_event_type)
_reg("state", ("full",), "last_event_seat",
     "上一事件来源座位/3", _fill_last_event_seat)
_reg("state", ("full",), "last_event_tile",
     "上一事件涉及牌标记", _fill_last_event_tile)


def _fill_riichi_sticks_onehot(obs, out):
    out[min(obs.get("riichi_sticks", 0), 4)] = 1.0


_reg("state", ("full",), "riichi_sticks_onehot",
     "立直棒数 one-hot（列=棒数）", _fill_riichi_sticks_onehot)


def _fill_round_onehot(obs, out):
    out[min(obs.get("round", 0), 7)] = 1.0


_reg("state", ("full",), "round_onehot",
     "局数 one-hot（列 0..7 = E1..E4,S1..S4）", _fill_round_onehot)


def _fill_honba_onehot(obs, out):
    out[min(obs.get("honba", 0), 4)] = 1.0


_reg("state", ("full",), "honba_onehot", "本场数 one-hot", _fill_honba_onehot)


def _fill_visible_red5(obs, out):
    n = 0
    for t in obs["hand"]:
        if t in RED5_TILES:
            n += 1
    for s in range(4):
        for m in obs["melds"][s]:
            for t in m["tiles"]:
                if t in RED5_TILES:
                    n += 1
        for d in obs["discards"][s]:
            if d["tile"] in RED5_TILES:
                n += 1
    out[:] = min(n, 3) / 3.0


_reg("state", ("full",), "visible_red5",
     "可见赤5总数/3（手牌+副露+牌河）", _fill_visible_red5)


def _mk_discards_total(seat_i: int):
    def _f(obs, out, s=seat_i):
        out[:] = min(len(obs["discards"][s]), 24) / 24.0
    return _f


for _s in range(4):
    _reg("state", ("full",), f"discards_total_s{_s}",
         f"座位{_s} 已切牌数/24", _mk_discards_total(_s))


# ================= SHANTEN (precise; full-only) =================

def _closed_counts(obs):
    return _counts34_list(obs["hand"])


def _fill_shanten_current(obs, out):
    out[:] = (_shanten_of(_closed_counts(obs)) + 1) / 10.0


_reg("shanten", ("full",), "shanten_current",
     "当前向听数（-1..7 归一化到 0..1）", _fill_shanten_current)


def _fill_shanten_discard(obs, out):
    cnt = _closed_counts(obs)
    cur = _shanten_of(cnt)
    if len(obs["hand"]) % 3 == 2:  # 14-3m discard/riichi point
        pts = _per_tile_shanten(cnt)
        for k, s in enumerate(pts):
            out[k] = (s + 1) / 10.0
    else:  # 13-3m call point: no per-tile discard; neutral = current
        out[:] = (cur + 1) / 10.0


_reg("shanten", ("full",), "shanten_discard",
     "切X后的向听数（34列；不在手中的牌=当前向听数）", _fill_shanten_discard)


def _fill_shanten_ukeire(obs, out):
    cnt = _closed_counts(obs)
    if len(obs["hand"]) % 3 == 1:  # 13-3m only
        out[:] = _shanten_ukeire(cnt) / 34.0
    else:
        out[:] = 0.0


_reg("shanten", ("full",), "shanten_ukeire",
     "进张数/34（13张点；向听>2 时返回 0）", _fill_shanten_ukeire)


def _fill_shanten_waits(obs, out):
    cnt = _closed_counts(obs)
    if len(obs["hand"]) % 3 == 1:
        ws = _tenpai_waits(cnt)
        for k, v in enumerate(ws):
            out[k] = float(v)


_reg("shanten", ("full",), "shanten_waits",
     "听牌时和了牌 0/1（34列；非听牌全 0）", _fill_shanten_waits)


def _fill_waits_count(obs, out):
    if len(obs["hand"]) % 3 != 1:  # 13-3m 点才有听牌概念
        return
    cnt = _closed_counts(obs)
    if _shanten_of(cnt) != 0:
        return
    out[:] = sum(_tenpai_waits(cnt)) / 34.0


_reg("shanten", ("full",), "shanten_waits_count",
     "听牌时和了牌种数/34（非听牌=0）", _fill_waits_count)


def _fill_waits_total(obs, out):
    if len(obs["hand"]) % 3 != 1:
        return
    cnt = _closed_counts(obs)
    if _shanten_of(cnt) != 0:
        return
    vis = _visible_counts(obs)  # 手牌+副露+牌河+指示牌
    ws = _tenpai_waits(cnt)
    tot = 0
    for k in range(34):
        if ws[k]:
            tot += max(4 - vis[k], 0)
    out[:] = min(tot, 24) / 24.0


_reg("shanten", ("full",), "shanten_waits_total",
     "听牌时剩余和了张数/24（按全桌可见扣减；非听牌=0）", _fill_waits_total)


# ================= FIELD-SITUATION completeness (full-only) =================

def _fill_uradora_expected(obs, out):
    riichi_declared = obs.get("riichi_declared", [False] * 4)
    n_riichi = sum(1 for s in range(4) if riichi_declared[s])
    inds = obs.get("dora_indicators", [])
    visible = [0] * N_KINDS
    for t in obs["hand"]:
        visible[t // 4] += 1
    for s in range(4):
        for m in obs["melds"][s]:
            for t in m["tiles"]:
                visible[t // 4] += 1
        for d in obs["discards"][s]:
            visible[d["tile"] // 4] += 1
    wall = max(obs.get("wall_left", 0), 1)
    exp = 0.0
    for ind in inds:
        nk = next_dora_kind(ind // 4)
        unseen = max(4 - visible[nk], 0)
        exp += unseen / wall
    out[:] = min(n_riichi * exp / 4.0, 1.0)


_reg("state", ("full",), "uradora_expected",
     "里宝期望：立直人数 × Σ(指示牌宝牌剩余张/牌山剩余)/4", _fill_uradora_expected)


# --- per-hand-type shanten (full; appended so earlier indexes are stable) ---
def _fill_shanten_regular(obs, out):
    out[:] = (_shanten_regular(_closed_counts(obs)) + 1) / 10.0


_reg("shanten", ("full",), "shanten_regular",
     "一般形向听数（-1..8 归一化 0..1）", _fill_shanten_regular)


def _fill_shanten_chiitoi(obs, out):
    out[:] = (_shanten_chiitoi(_closed_counts(obs)) + 1) / 10.0


_reg("shanten", ("full",), "shanten_chiitoi",
     "七对子向听数（-1..6 归一化 0..1）", _fill_shanten_chiitoi)


def _fill_shanten_kokushi(obs, out):
    out[:] = (_shanten_kokushi(_closed_counts(obs)) + 1) / 14.0


_reg("shanten", ("full",), "shanten_kokushi",
     "国士无双向听数（-1..13 归一化 0..1）", _fill_shanten_kokushi)




# ================= FINAL v5 additions (appended; prefix 0-16 untouched) =================

# --- A. hand_dora: which kinds in hand are actual dora tiles ---
def _dora_kind_set(obs):
    s = set()
    for t in obs.get("dora_indicators", []):
        s.add(next_dora_kind(t // 4))
    return s


def _fill_hand_dora(obs, out):
    ds = _dora_kind_set(obs)
    for t in obs["hand"]:
        if (t // 4) in ds:
            out[t // 4] = 1.0


_reg("hand", ("full",), "hand_dora",
     "手牌中是宝牌的牌种标记", _fill_hand_dora)


# --- B. per-indicator dora: what each of the 5 indicators is + its dora ---
def _mk_ind_tile(i: int):
    def _f(obs, out, idx=i):
        inds = obs.get("dora_indicators", [])
        if idx < len(inds):
            out[tile_to_kind(inds[idx])] = 1.0
    return _f


def _mk_ind_dora(i: int):
    def _f(obs, out, idx=i):
        inds = obs.get("dora_indicators", [])
        if idx < len(inds):
            out[next_dora_kind(tile_to_kind(inds[idx]))] = 1.0
    return _f


for _i in range(5):
    _reg("dora", ("full",), f"dora_ind{_i}_tile",
         f"第{_i + 1}个宝牌指示牌是什么（one-hot；不足则全0）", _mk_ind_tile(_i))
    _reg("dora", ("full",), f"dora_ind{_i}_dora",
         f"第{_i + 1}个指示牌对应的宝牌是什么（one-hot）", _mk_ind_dora(_i))


# --- D. total visible dora count (all table) ---
def _fill_visible_dora(obs, out):
    ds = _dora_kind_set(obs)
    n = 0
    for t in obs["hand"]:
        n += (t // 4) in ds
    for s in range(4):
        for m in obs["melds"][s]:
            for t in m["tiles"]:
                n += (t // 4) in ds
        for d in obs["discards"][s]:
            n += (d["tile"] // 4) in ds
    out[:] = min(n, 8) / 8.0


_reg("visible", ("full",), "visible_dora_count",
     "全桌可见宝牌总张数/8（手牌+副露+牌河）", _fill_visible_dora)


# --- I. hand_value_discard: engine-scored expected value when discarding X ---
_CALC = HandCalculator()


def _to_meld_objs(seat_melds):
    objs = []
    for m in seat_melds:
        tiles = sorted(m["tiles"])
        mt = m.get("type")
        if mt == "chi":
            objs.append(Meld(meld_type=Meld.CHI, tiles=tiles, opened=True,
                             called_tile=m.get("called") or tiles[0]))
        elif mt == "pon":
            objs.append(Meld(meld_type=Meld.PON, tiles=tiles, opened=True,
                             called_tile=m.get("called") or tiles[0]))
        else:
            objs.append(Meld(meld_type=Meld.KAN, tiles=tiles, opened=(mt != "ankan"),
                             called_tile=m.get("called") if mt != "ankan" else None))
    return objs


def _counts_to_tiles(c):
    out = []
    for k in range(34):
        for _ in range(c[k]):
            out.append(k * 4)  # copy 0 (red5 handled by aka-dora rule)
    return out


def _win_value(closed_counts, win_kind, meld_objs, dora_inds, seat, round_idx, riichi):
    """Engine ron points + han for a tenpai hand winning on win_kind."""
    try:
        tiles = _counts_to_tiles(closed_counts) + [win_kind * 4]  # include win tile
        res = _CALC.estimate_hand_value(
            tiles, win_kind * 4, melds=meld_objs, dora_indicators=list(dora_inds),
            config=HandConfig(is_tsumo=False, is_riichi=riichi,
                              player_wind=(seat - (round_idx % 4)) % 4,
                              round_wind=(round_idx // 4) % 2,
                              options=OptionalRules(has_open_tanyao=True, has_aka_dora=True)))
        if res is None or res.error is not None:
            return 0.0, 0.0
        ron_points = res.cost["main"] if (res.cost and "main" in res.cost) else 0
        return float(ron_points or 0), float(res.han or 0)
    except Exception:
        return 0.0, 0.0


def _fill_value_discard(obs, out):
    hand = obs["hand"]
    if len(hand) % 3 != 2:  # 仅在 14-3m 弃牌点有意义；13-3m 副露点无切牌决策
        return
    cnt = _counts34_list(hand)
    seat = obs["seat"]
    round_idx = obs.get("round", 0)
    meld_objs = _to_meld_objs(obs["melds"][seat])
    dora_inds = obs.get("dora_indicators", [])
    closed = len(obs["melds"][seat]) == 0
    visible = [0] * N_KINDS
    for t in hand:
        visible[t // 4] += 1
    for s in range(4):
        for m in obs["melds"][s]:
            for t in m["tiles"]:
                visible[t // 4] += 1
        for d in obs["discards"][s]:
            visible[d["tile"] // 4] += 1
    for t in dora_inds:
        visible[t // 4] += 1
    wall = max(obs.get("wall_left", 0), 1)
    ura_han = 0.0
    if closed:
        for ind in dora_inds:
            nk = next_dora_kind(ind // 4)
            ura_han += max(4 - visible[nk], 0) / wall
    for k in range(34):
        if cnt[k] == 0:
            continue
        c2 = list(cnt)
        c2[k] -= 1
        if _shanten_of(c2) != 0:  # 仅听牌才计算预期打点
            out[k] = 0.0
            continue
        waits = []
        for w in range(34):
            if c2[w] >= 4:
                continue
            c3 = list(c2)
            c3[w] += 1
            if _shanten_of(c3) == -1:
                waits.append(w)
        num = 0.0
        den = 0.0
        han_ref = 0.0
        for w in waits:
            avail = max(4 - visible[w], 0)
            if avail <= 0:
                continue
            V, H = _win_value(c2, w, meld_objs, dora_inds, seat, round_idx, closed)
            num += V * avail
            den += avail
            han_ref = H
        val = (num / den) if den > 0 else 0.0
        if closed and ura_han > 0 and val > 0 and han_ref > 0:
            val += ura_han * (val / han_ref)  # 里宝期望番 -> 点数（每番均值近似）
        out[k] = min(val / 32000.0, 1.0)


_reg("state", ("full",), "hand_value_discard",
     "切X后预期打点/32000（引擎计分；仅听牌；立直含里宝期望）", _fill_value_discard)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def channel_scale_max(names) -> np.ndarray:
    """Per-channel quantization max (single source of truth for preprocess).

    All channels are one-hot (0/1) or normalized [0,1] -> max 1, except the
    count-valued dora channels (dora_indicators max 5, dora_kinds max 4).
    """
    scale = np.ones(len(names), dtype=np.float32)
    for i, n in enumerate(names):
        if n == "dora_indicators":
            scale[i] = 5.0
        elif n == "dora_kinds":
            scale[i] = 4.0
    return scale


def channel_names(mode: str = "full") -> list:
    """Names of channels active for MODE (order == output channel order)."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    return [name for (_g, modes, name, _d, _f) in CHANNELS if mode in modes]


def feature_channels(mode: str = "full") -> int:
    return len(channel_names(mode))


def group_summary(mode: str = "full") -> dict:
    """{group: n_channels} for the active mode (explainability / docs)."""
    from collections import Counter
    return dict(Counter(g for (g, modes, _n, _d, _f) in CHANNELS if mode in modes))


def channel_layout(mode: str = "full") -> list:
    """Full channel layout for explainability:
    [{index, group, name, description, quant_max}]."""
    names = channel_names(mode)
    scale = channel_scale_max(names)
    out = []
    idx = 0
    for (g, modes, name, desc, _f) in CHANNELS:
        if mode not in modes:
            continue
        out.append({"index": idx, "group": g, "name": name,
                    "description": desc, "quant_max": float(scale[idx])})
        idx += 1
    return out


def save_channel_layout(path: str = "docs/channel_layout.json") -> None:
    """Write the channel layout JSON (names + semantics + scale) for both modes."""
    doc = {
        "contract": "docs/observation_schema.md v1.1",
        "modes": {
            m: {"n_channels": feature_channels(m), "groups": group_summary(m),
                "channels": channel_layout(m)}
            for m in VALID_MODES
        },
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def build_features(obs: dict, mode: str = "full") -> np.ndarray:
    """Encode one observation record -> (C, 34, 1) float32 array."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    specs = [(name, f) for (_g, modes, name, _d, f) in CHANNELS if mode in modes]
    out = np.zeros((len(specs), N_KINDS, 1), dtype=np.float32)
    for i, (_name, f) in enumerate(specs):
        f(obs, out[i, :, 0])
    return out


def discard_mask(obs: dict) -> np.ndarray:
    """(34,) bool mask of legal discard kinds (False = illegal)."""
    mask = np.zeros(N_KINDS, dtype=bool)
    legal = (obs.get("legal_actions") or {}).get("discard")
    if legal is None:  # unknown -> all legal (inference fallback)
        mask[:] = True
    else:
        for t in legal:
            mask[tile_to_kind(t)] = True
    return mask


def action_availability(obs: dict) -> dict:
    """Binary availability per action head, from legal_actions."""
    la = obs.get("legal_actions") or {}
    return {
        "discard": bool(la.get("discard")),
        "riichi": bool(la.get("riichi")),
        "chow": bool(la.get("chow")),
        "pon": bool(la.get("pon")),
        "kan": bool(la.get("kan")),
        "ron": bool(la.get("ron")),
        "tsumo": bool(la.get("tsumo")),
    }


def label_kind(record: dict) -> int:
    """Kind of the human discard label (for discard-head supervision)."""
    lab = record["label"]
    if lab.get("type") in ("discard", "riichi"):
        return tile_to_kind(lab["tile"])
    # chow/pon/kan: focal tile = first of tiles
    if lab.get("type") in ("chow", "pon", "kan") and lab.get("tiles"):
        return tile_to_kind(lab["tiles"][0])
    raise KeyError(f"label has no discard-able tile: {lab}")
