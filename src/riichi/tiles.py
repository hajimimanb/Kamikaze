"""Shared tile utilities — encoding contract in docs/observation_schema.md §1.

Encoding:
- kind 0..33: 0-8 = man 1-9, 9-17 = pin 1-9, 18-26 = sou 1-9,
  27-30 = winds E/S/W/N, 31-33 = dragons Haku/Hatsu/Chun.
- tile136 0..135: tile136 = kind*4 + copy, copy in 0..3.
- Red fives are fixed ids: 5m=16, 5p=52, 5s=88 (kind 4/13/22, copy 0).

These helpers are shared by engine-developer, data-engineer and ml-engineer.
"""
from __future__ import annotations

# startup dependency version assertion (task t16): every module importing
# tiles.py — i.e. all ML/engine entrypoints — asserts mahjong==2.0.0 at
# import time; mismatch raises with remediation instructions.
from riichi.version_guard import assert_versions as _assert_versions

_assert_versions()

RED_FIVE_MAN = 16
RED_FIVE_PIN = 52
RED_FIVE_SOU = 88
RED_FIVES = frozenset((RED_FIVE_MAN, RED_FIVE_PIN, RED_FIVE_SOU))
RED_FIVE_KINDS = (4, 13, 22)

_KIND_RED = {4: RED_FIVE_MAN, 13: RED_FIVE_PIN, 22: RED_FIVE_SOU}
# MJAI tile strings: honors are E/S/W/N/P/F/C, red fives get an r suffix.
_MJAI_HONORS = {27: "E", 28: "S", 29: "W", 30: "N", 31: "P", 32: "F", 33: "C"}
_MJAI_HONOR_RANK = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}
def mjai_pai(tile: int) -> str:
    """MJAI-protocol tile string for a 136-format tile.
    e.g. 1m..9m, 1p..9p, 1s..9s, E/S/W/N, P(White)/F(Green)/C(Red dragon),
    red fives as 5mr/5pr/5sr.
    """
    kind = tile // 4
    if kind < 27:
        suit = "mps"[kind // 9]
        rank = kind % 9 + 1
        s = str(rank) + suit
        if tile in RED_FIVES:
            s += "r"
        return s
    return _MJAI_HONORS[kind]
def parse_mjai_pai(s: str) -> int:
    """Inverse of mjai_pai: an MJAI tile string -> one 136-format tile id.
    Honors map to copy 0; 5mr yields the red five (16).
    """
    if s in _MJAI_HONOR_RANK:
        return _MJAI_HONOR_RANK[s] * 4
    red = s.endswith("r")
    if red:
        s = s[:-1]
    suit = s[-1]
    rank = int(s[:-1])
    base = {"m": 0, "p": 9, "s": 18}[suit]
    kind = base + rank - 1
    if red:
        if kind not in _KIND_RED:
            raise ValueError("only fives can be red")
        return _KIND_RED[kind]
    t = kind * 4
    if t in RED_FIVES:
        t += 1
    return t

_SUIT_LETTERS = ("m", "p", "s", "z")


def kind_of(tile: int) -> int:
    return tile // 4


def copy_of(tile: int) -> int:
    return tile % 4


def tile136(kind: int, copy: int = 0) -> int:
    return kind * 4 + copy


def is_red_five(tile: int) -> bool:
    return tile in RED_FIVES


def count_red_fives(tiles) -> int:
    return sum(1 for t in tiles if t in RED_FIVES)


def counts34(tiles136) -> list:
    counts = [0] * 34
    for t in tiles136:
        counts[t // 4] += 1
    return counts


def tiles136_from_counts(counts34) -> list:
    out = []
    for kind, n in enumerate(counts34):
        out.extend(kind * 4 + i for i in range(n))
    return out


def is_suited(kind: int) -> bool:
    return kind < 27


def is_honor(kind: int) -> bool:
    return 27 <= kind <= 33


def is_wind(kind: int) -> bool:
    return 27 <= kind <= 30


def is_dragon(kind: int) -> bool:
    return 31 <= kind <= 33


def is_terminal(kind: int) -> bool:
    return kind in (0, 8, 9, 17, 18, 26)


def is_terminal_or_honor(kind: int) -> bool:
    return is_terminal(kind) or is_honor(kind)


def suit_of(kind: int) -> int:
    return kind // 9


def rank_of(kind: int) -> int:
    return kind % 9 + 1


def kind_str(kind: int) -> str:
    return "%d%s" % (rank_of(kind), _SUIT_LETTERS[suit_of(kind)])


def next_dora_kind(kind: int) -> int:
    if kind < 27:
        base = (kind // 9) * 9
        return base + (kind + 1 - base) % 9
    if 27 <= kind <= 30:
        return 27 + (kind - 27 + 1) % 4
    return 31 + (kind - 31 + 1) % 3


def sort_tiles136(tiles) -> list:
    return sorted(tiles, key=lambda t: (t // 4, t % 4))


def tiles136_to_str(tiles, aka_marker: bool = True) -> str:
    tiles = sort_tiles136(tiles)
    parts = []
    for suit in range(4):
        base = suit * 9
        group = [t for t in tiles if base * 4 <= t < (base + 9) * 4]
        if not group:
            continue
        chars = []
        for t in group:
            rank = t // 4 % 9 + 1
            if aka_marker and t in RED_FIVES:
                chars.append("0")
            else:
                chars.append(str(rank))
        parts.append("".join(chars) + _SUIT_LETTERS[suit])
    return "".join(parts)


def parse_mpsz(s: str) -> list:
    out = []
    cur = ""
    for ch in s:
        if ch in "mpszh":
            base = {"m": 0, "p": 9, "s": 18, "z": 27}[ch]
            counts = {}
            plain5 = {}
            for d in cur:
                if d in ("0", "r", "R"):
                    if base > 18:
                        raise ValueError("red five only exists in suited groups")
                    out.append(_KIND_RED[base + 4])
                    counts[base + 4] = counts.get(base + 4, 0) + 1
                else:
                    rank = int(d)
                    if base == 27:
                        if not 1 <= rank <= 7:
                            raise ValueError("honor rank must be 1..7")
                    kind = base + rank - 1
                    if kind in _KIND_RED:
                        # plain digit '5' takes copies 1,2,3 (copy 0 is red);
                        # the 4th plain 5 falls back to copy 0
                        n = plain5.get(kind, 0)
                        if n >= 4:
                            raise ValueError("too many 5s of one suit")
                        t = kind * 4 + (n + 1 if n + 1 < 4 else 0)
                        plain5[kind] = n + 1
                    else:
                        copy = counts.get(kind, 0)
                        t = kind * 4 + copy
                        counts[kind] = copy + 1
                    out.append(t)
            cur = ""
        elif ch.isdigit() or ch in ("r", "R"):
            cur += ch
        else:
            raise ValueError("unexpected character %r in mpsz string" % ch)
    if cur:
        raise ValueError("mpsz string must end with a suit letter")
    return out


def all_tiles136() -> list:
    return list(range(136))
