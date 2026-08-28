# -*- coding: utf-8 -*-
"""Shared pytest fixtures/helpers for the riichi engine tests.

Adds C:/agentwork/src to sys.path so tests can import riichi / env packages.
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from riichi.tiles import RED_FIVES, parse_mpsz, sort_tiles136  # noqa: E402


# Canonical filler hands: total copies per kind never exceed 4.
DUMMY_HANDS = [
    "123m123p123s456m9m",
    "2468m99p1357s112z",
    "2244m6688p99s112z",
    "111333555m22p44s",
]


def make_wall(draws, dora, ura=None, rinshan=None, seed=0):
    """Build a full 136-tile wall with a controlled draw order and dead wall.

    Layout: indices 0..len(draws)-1 are drawn in order (dealer draws first);
    the rest of the live wall is filled with the remaining tiles shuffled with
    Random(seed). Dead wall (122..135):
      122..125 rinshan pool (drawn back-to-front, i.e. rinshan[-1] first)
      126..130 dora indicators (initial + up to 4 kan dora)
      131..135 ura indicators
    """
    ura = list(ura or [])
    rinshan = list(rinshan or [])
    used = set(draws) | set(dora) | set(ura) | set(rinshan)
    assert len(used) == len(draws) + len(dora) + len(ura) + len(rinshan), "overlapping tiles"
    rem = [t for t in range(136) if t not in used]
    dora = list(dora)
    while len(dora) < 5:
        dora.append(rem.pop())
    while len(ura) < 5:
        ura.append(rem.pop())
    while len(rinshan) < 4:
        rinshan.append(rem.pop())
    used = set(draws) | set(dora) | set(ura) | set(rinshan)
    filler = [t for t in range(136) if t not in used]
    random.Random(seed).shuffle(filler)
    assert len(draws) + len(filler) == 122, "draws+filler must fill 0..121"
    wall = [None] * 136
    for i, t in enumerate(draws):
        wall[i] = t
    pos = len(draws)
    for t in filler:
        wall[pos] = t
        pos += 1
    for i, t in enumerate(rinshan):
        wall[125 - i] = t
    for i, t in enumerate(dora):
        wall[126 + i] = t
    for i, t in enumerate(ura):
        wall[131 + i] = t
    assert None not in wall
    assert sorted(wall) == list(range(136))
    return wall


def make_game(hands, draws, dora, dealer=0, seed=0, ura=None, rinshan=None,
              discards=None, melds=None, scores=None, honba=0, kyoutaku=0,
              riichi=None, n_kan=0, round_idx=0, draw_ptr=0, wall_left=None,
              config=None, game_seed=1, first_draw=True):
    """Create a RiichiGame with a fully deterministic state.

    hands: 13-tile hands (mpsz strings or 136 lists) per seat; the dealer then
    draws the first tile of `draws`. Every hand/meld/discard/dora tile gets a
    unique physical copy id, so scenarios stay physically consistent.
    """
    from env.riichi_game import RiichiGame

    if hands is None:
        hands = DUMMY_HANDS

    held = set()

    def take(t):
        """Assign a unique physical copy for the kind of tile t;
        red fives keep their red copy id."""
        base = t // 4 * 4
        if t in RED_FIVES:
            if t not in held:
                held.add(t)
                return t
            raise AssertionError("red five already held")
        for c in range(4):
            cand = base + c
            if cand not in held and cand not in RED_FIVES:
                held.add(cand)
                return cand
        for c in range(4):
            cand = base + c
            if cand not in held:
                held.add(cand)
                return cand
        raise AssertionError("all 4 copies of tile already held")

    melds_remapped = melds is not None
    hands136 = []
    for i, h in enumerate(hands):
        tiles = parse_mpsz(h) if isinstance(h, str) else sort_tiles136(list(h))
        n_melds = len(melds[i]) if melds_remapped and i < len(melds) else 0
        expected = 13 - 3 * n_melds
        assert len(tiles) == expected, "seat %d hand must be %d tiles" % (i, expected)
        hands136.append([take(t) for t in tiles])
    discards136 = None
    if discards:
        discards136 = []
        for river in discards:
            new_river = []
            for d in river:
                d = dict(d)
                t = d["tile"]
                t = parse_mpsz(t)[0] if isinstance(t, str) else t
                d["tile"] = take(t)
                new_river.append(d)
            discards136.append(new_river)
    melds136 = None
    if melds:
        melds136 = []
        for ms in melds:
            new_ms = []
            for m in ms:
                m = dict(m)
                tiles = []
                for t in m["tiles"]:
                    t2 = parse_mpsz(t)[0] if isinstance(t, str) else t
                    tiles.append(take(t2))
                m["tiles"] = tiles
                if m.get("called"):
                    c = m["called"]
                    ck = parse_mpsz(c)[0] // 4 if isinstance(c, str) else c // 4
                    m["called"] = next(t for t in tiles if t // 4 == ck)
                new_ms.append(m)
            melds136.append(new_ms)
    draws136 = []
    for d in draws:
        tiles = parse_mpsz(d) if isinstance(d, str) else [d]
        for t in tiles:
            draws136.append(take(t))
    dora136 = []
    for d in dora:
        for t in (parse_mpsz(d) if isinstance(d, str) else [d]):
            dora136.append(take(t))
    ura136 = []
    for u in (ura or []):
        for t in (parse_mpsz(u) if isinstance(u, str) else [u]):
            ura136.append(take(t))
    rinshan136 = []
    for x in (rinshan or []):
        for t in (parse_mpsz(x) if isinstance(x, str) else [x]):
            rinshan136.append(take(t))
    wall = make_wall(draws=draws136, dora=dora136, ura=ura136, rinshan=rinshan136,
                     seed=seed)
    g = RiichiGame(config=config, seed=game_seed)
    if wall_left is None:
        wall_left = 70 - draw_ptr
    g.debug_setup(
        hands=hands136, dealer=dealer, wall=wall, discards=discards136, melds=melds136,
        scores=scores, honba=honba, kyoutaku=kyoutaku, riichi=riichi, n_kan=n_kan,
        round_idx=round_idx, draw_ptr=draw_ptr, wall_left=wall_left)
    if not first_draw:
        # suppress tenhou/chiihou eligibility for scenario tests
        g._draw_counter = 2
        g._draws_made = [2, 2, 2, 2]
    return g


@pytest.fixture
def simple_game():
    """A game where seat 0 is one tile from tsumo on the first draw."""
    hands = [
        "123m123p123s456m9m",   # waits on 9m
        "2468m99p1357s112z",
        "2244m6688p99s112z",
        "111333555m22p44s",
    ]
    return make_game(hands, draws=["9m", "3z", "3z", "3z"], dora=["7z"], dealer=0)
