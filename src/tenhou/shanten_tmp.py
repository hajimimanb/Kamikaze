"""Temporary shanten / agari / tenpai helpers for the tenhou pipeline.

TODO(riichi-ai): engine-developer is implementing src/riichi/tiles.py /
explain.py; once those expose shanten/tenpai/agari functions, migrate this
module's call sites to them and delete this file. Until then this wraps the
battle-tested 'mahjong' package (pip: mahjong, classic cmj3 shanten algorithm,
MIT license).

Tile conventions per docs/observation_schema.md section 1:
- kind 0..33, tile136 = kind*4 + copy, red5 = 16 / 52 / 88.
- counts34: 34-length count array.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter

# ---------------------------------------------------------------------------
# basic shanten / completeness
# ---------------------------------------------------------------------------

_SHANTEN = Shanten()


def shanten34(counts: Sequence[int]) -> int:
    """Regular+chiitoi+kokushi shanten for a closed hand given as 34-counts.

    Hands with open melds should pass only the *closed* tile counts; the
    classic algorithm treats missing melds as open (init_mentsu = (14-n)//3),
    which is exactly the semantics we need.

    Works with mahjong 1.2.1 (instance method) and 2.0.0 (static method).
    """
    return _SHANTEN.calculate_shanten(list(counts))


def is_complete(counts: Sequence[int]) -> bool:
    """True if the closed counts (with missing melds implied open) are agari."""
    n = sum(counts)
    if n % 3 != 2:
        return False
    return shanten34(counts) == -1


def tenpai_discards(hand14: Sequence[int]) -> List[int]:
    """For a 14-tile closed hand, return sorted distinct tile136 whose discard
    leaves the hand tenpai (used for the legal riichi mask)."""
    out: List[int] = []
    for t in sorted(set(hand14)):
        counts = TilesConverter.to_34_array([x for x in hand14 if x != t])
        if shanten34(counts) == 0:
            out.append(t)
    return out


def is_tenpai_13(hand13: Sequence[int]) -> bool:
    return shanten34(TilesConverter.to_34_array(list(hand13))) == 0


def winning_tiles(hand13: Sequence[int], melds: Sequence[dict] = ()) -> List[int]:
    """All distinct tile136 that complete hand13(+melds) into agari.

    Slow (34 shanten evaluations); use sparingly (stats, not per-record masks).
    """
    base = list(hand13)
    used = set(base)
    for m in melds:
        used.update(m["tiles"])
    out: List[int] = []
    for kind in range(34):
        copy = 0
        while kind * 4 + copy in used:
            copy += 1
        if copy >= 4:
            continue
        t = kind * 4 + copy
        counts = TilesConverter.to_34_array(base + [t])
        if shanten34(counts) == -1:
            out.append(t)
    return out

# ---------------------------------------------------------------------------
# agari validity with yaku (ron/tsumo masks)
# ---------------------------------------------------------------------------

def _to_meld_objects(melds: Sequence[dict]):
    from mahjong.meld import Meld
    objs = []
    for m in melds:
        tiles = sorted(m["tiles"])
        mtype = m["type"]
        if mtype == "chi":
            objs.append(Meld(meld_type=Meld.CHI, tiles=tiles, opened=True,
                             called_tile=m.get("called") or tiles[0]))
        elif mtype == "pon":
            objs.append(Meld(meld_type=Meld.PON, tiles=tiles, opened=True,
                             called_tile=m.get("called") or tiles[0]))
        else:  # ankan / minkan / kakan -> KAN
            opened = mtype != "ankan"
            objs.append(Meld(meld_type=Meld.KAN, tiles=tiles, opened=opened,
                             called_tile=m.get("called") if opened else None))
    return objs


def is_winning_hand(closed_tiles: Sequence[int], win_tile: int,
                    melds: Sequence[dict], dora_indicators: Sequence[int],
                    seat: int, round_index: int, is_tsumo: bool,
                    is_riichi: bool = False, chankan: bool = False,
                    rinshan: bool = False) -> bool:
    """Full yaku validity check of a winning hand (phoenix rules:
    kuitan allowed, aka-dora present). Used only when the shanten check says
    the hand is complete, so cost is irrelevant."""
    from mahjong.hand_calculating.hand import HandCalculator
    from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules

    calculator = HandCalculator()
    tiles = list(closed_tiles)  # includes win tile, excludes meld tiles
    win_tile = int(win_tile)
    config = HandConfig(
        is_tsumo=is_tsumo,
        is_riichi=is_riichi,
        is_chankan=chankan,
        is_rinshan=rinshan,
        is_haitei=False,
        is_houtei=False,
        is_daburu_riichi=False,
        player_wind=(seat - (round_index % 4)) % 4,
        round_wind=(round_index // 4) % 4,
        options=OptionalRules(has_open_tanyao=True, has_aka_dora=True),
    )
    try:
        result = calculator.estimate_hand_value(
            tiles, win_tile,
            melds=_to_meld_objects(melds),
            dora_indicators=list(dora_indicators),
            config=config,
        )
    except Exception:
        return False
    return bool(result.yaku) if result is not None else False
