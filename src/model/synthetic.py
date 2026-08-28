# -*- coding: utf-8 -*-
"""Synthetic decision-point record generator (schema v1.1) for smoke tests.

Generates tsumogiri-dominated data: human discards the just-drawn tile with
probability p_tsumogiri (the stage-1 curriculum baseline target), plus a few
pon samples so the stage-2 binary heads see data.
"""
import json
import random


def make_record(rng, i, p_tsumogiri=0.85):
    deck = list(range(136))
    rng.shuffle(deck)
    seats = [0, 1, 2, 3]
    seat = rng.choice(seats)
    oya = rng.choice(seats)

    hands = [[deck.pop() for _ in range(13)] for _ in range(4)]
    drawn = deck.pop()
    hand = hands[seat] + [drawn]

    dora_indicators = [deck.pop()]

    discards = []
    for s in range(4):
        ds = []
        for k in range(2 if s != seat else 3):
            t = hands[s].pop() if hands[s] else deck.pop()
            ds.append({"tile": t, "tsumogiri": bool(rng.random() < 0.4),
                       "riichi": False})
        discards.append(ds)

    scores = [25000 + rng.randint(-8000, 8000) for _ in range(4)]
    legal_discards = sorted({t for t in hand})

    # label: usually tsumogiri of the drawn tile, else a random hand tile
    others = sorted({t for t in hand if t != drawn} or [drawn])
    if rng.random() < p_tsumogiri:
        label = {"type": "discard", "tile": drawn}
    else:
        label = {"type": "discard", "tile": rng.choice(others)}

    legal_actions = {
        "discard": legal_discards,
        "riichi": [], "chow": [], "pon": [], "kan": [],
        "ron": False, "tsumo": False,
    }
    # occasionally offer a pon so the pon head sees data (stage 2)
    if rng.random() < 0.05 and hands[(seat + 1) % 4]:
        t = hands[(seat + 1) % 4][0]
        legal_actions["pon"] = [{"type": "pon", "tiles": [t] * 3,
                                 "from": (seat + 1) % 4, "red": False}]
        if rng.random() < 0.5:
            label = {"type": "pon", "tiles": [t] * 3}

    return {
        "game_id": "synth-" + format(i // 1000, "04d"), "source": "synthetic",
        "seat": seat, "round": rng.randint(0, 7), "honba": 0,
        "riichi_sticks": rng.randint(0, 2),
        "wall_left": rng.randint(20, 70),
        "dora_indicators": dora_indicators,
        "scores": scores, "oya": oya, "hand": hand, "melds": [[], [], [], []],
        "discards": discards, "n_kan": 0,
        "riichi_declared": [False, False, False, False],
        "last_event": {"type": "draw", "seat": seat, "tile": drawn},
        "legal_actions": legal_actions, "label": label,
    }


def write_records(path, n=2000, seed=0, p_tsumogiri=0.85):
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps(make_record(rng, i, p_tsumogiri)) + "\n")
    return path
