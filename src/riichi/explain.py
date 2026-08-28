"""Explainability helpers (docs/observation_schema.md §4).

Shared by the whole team: shanten, ukeire (visible-tile deduction), per-river
safety heuristics (genbutsu / suji / kabe) and a rough yaku expectation.
All functions are pure: they take plain data and return plain dicts.
"""
from collections import Counter

from mahjong.agari import Agari
from mahjong.shanten import Shanten

from riichi.tiles import (
    RED_FIVES,
    counts34,
    is_honor,
    is_terminal_or_honor,
    kind_of,
    next_dora_kind,
    sort_tiles136,
)


def calc_shanten(hand136, melds=None):
    """Shanten number of a hand (concealed portion + closed-kan tiles).

    Returns -1 for a complete hand, 0 for tenpai, positive otherwise.
    Open melds are treated as complete melds and are not counted.
    """
    counts = counts34(hand136)
    for m in (melds or []):
        if m["type"] == "kan" and not m.get("open", True):
            for t in m["tiles"]:
                counts[t // 4] += 1
    n = sum(counts)
    if n == 0:
        return 8
    if n % 3 == 0:
        # degenerate (should not happen in a real game); pad with a dummy pair
        return 8
    return Shanten.calculate_shanten(counts)


def calc_ukeire(hand136, melds=None, visible136=None, shanten=None):
    """Effective tiles (ukeire): every tile kind that reduces the shanten,
    with counts of copies remaining after subtracting visible tiles.

    Returns {"kinds": [...], "count": total, "per_kind": {kind: n},
             "tiles136": [...]} (tiles136 = one representative copy per kind).
    """
    visible = Counter(t // 4 for t in (visible136 or []))
    hand_counts = counts34(hand136)
    for k in range(34):
        visible[k] += hand_counts[k]
    if melds:
        for m in melds:
            for t in m["tiles"]:
                visible[t // 4] += 1
    base = calc_shanten(hand136, melds)
    if base < 0:
        return {"kinds": [], "count": 0, "per_kind": {}, "tiles136": []}
    uke = {}
    for k in range(34):
        if visible[k] >= 4:
            continue
        if base == 0:
            # tenpai: effective tiles are the winning tiles
            ok = calc_shanten(hand136 + [k * 4], melds) == -1
        else:
            ok = calc_shanten(hand136 + [k * 4], melds) < base
        if ok:
            uke[k] = 4 - visible[k]
    return {
        "kinds": sorted(uke),
        "count": sum(uke.values()),
        "per_kind": uke,
        "tiles136": sorted(k * 4 for k in uke),
    }


def _suji_kinds(river_kinds):
    """Kinds protected by the suji rule for one discard river."""
    out = set()
    for k in river_kinds:
        if k >= 27:
            continue
        base = k // 9 * 9
        r = k % 9
        if r - 3 >= 0:
            out.add(base + r - 3)
        if r + 3 <= 8:
            out.add(base + r + 3)
    return out


EARLY_DISCARDS = 6  # first row of the river counts as "early"


def tile_safety(discards_by_seat, visible136=None):
    """Per-opponent safety heuristics.

    Returns a dict {seat: {"genbutsu": [...], "suji": [...], "kabe": [...],
    "early": [...], "danger": [...]}} with kind lists. danger = kinds with no
    hard safety marker against that opponent. "early" is a weaker marker:
    kinds the opponent discarded within the first six discards (not treated
    as safe by itself, exposed for downstream risk scoring).
    """
    visible = Counter(t // 4 for t in (visible136 or []))
    kabe_walls = {k for k in range(27) if visible[k] >= 4}
    kabe_protected = set()
    for k in kabe_walls:
        base = k // 9 * 9
        r = k % 9
        if r - 1 >= 0:
            kabe_protected.add(base + r - 1)
        if r + 1 <= 8:
            kabe_protected.add(base + r + 1)
    kabe_safe = kabe_walls | kabe_protected
    out = {}
    for seat, river in enumerate(discards_by_seat):
        river_kinds = {d["tile"] // 4 for d in river}
        genbutsu = set(river_kinds)
        suji = _suji_kinds(river_kinds) - genbutsu
        # "early": kinds the opponent let go within the first six discards
        # (a weak safety signal; NOT part of the hard-safe set)
        early = {d["tile"] // 4 for d in river[:EARLY_DISCARDS]}
        safe = genbutsu | suji | kabe_safe
        danger = [k for k in range(34) if k not in safe]
        out[seat] = {
            "genbutsu": sorted(genbutsu),
            "suji": sorted(suji),
            "kabe": sorted(kabe_safe),
            "early": sorted(early),
            "danger": danger,
        }
    return out


def count_dora(hand136, melds=None, dora_indicators=None, aka=True):
    """Dora count (indicator dora + red fives) for a hand incl. melds."""
    dora_kinds = {next_dora_kind(t // 4) for t in (dora_indicators or [])}
    tiles = list(hand136)
    for m in (melds or []):
        tiles.extend(m["tiles"])
    dora = sum(1 for t in tiles if t // 4 in dora_kinds)
    aka_n = sum(1 for t in tiles if t in RED_FIVES) if aka else 0
    return {"dora": dora, "aka": aka_n}


def _runs(counts):
    """All runs found greedily in one suit region of a 34-count array."""
    out = []
    for base in (0, 9, 18):
        for r in range(7):
            if counts[base + r] and counts[base + r + 1] and counts[base + r + 2]:
                out.append((base + r, base + r + 1, base + r + 2))
    return out


def yaku_expectation(hand136, melds=None, dora_indicators=None, seat=0,
                     round_idx=0, aka=True, kuitan=True):
    """Rough hand-value expectation (heuristic, not a full evaluation).

    Detects structure yaku candidates (tanyao / yakuhai / iipeiko / sanshoku /
    ittsu / honitsu / chinitsu / toitoi / chiitoitsu / sanankou / sanshoku
    doukou / shosangen) plus dora / aka / closed-hand bonus, and sums a
    rough han estimate. Not authoritative — use the engine for exact scores.
    """
    melds = melds or []
    all_tiles = list(hand136)
    for m in melds:
        all_tiles.extend(m["tiles"])
    counts = counts34(all_tiles)
    open_hand = any(m.get("open", True) for m in melds)
    yaku = []
    han = 0

    def add(name, h_closed, h_open=None):
        nonlocal han
        h = h_closed if not open_hand else (h_open if h_open is not None else h_closed)
        if h > 0:
            yaku.append(name)
            han += h

    # tanyao-able: no terminals / honors anywhere
    kinds = {k for k, n in enumerate(counts) if n}
    tanyao = all(k < 27 and k % 9 not in (0, 8) for k in kinds)
    if tanyao and (not open_hand or kuitan):
        add("tanyao", 1, 1)

    # yakuhai triplets (incl. kan)
    seat_wind = 27 + seat
    round_wind = 27 + min(round_idx // 4, 2)
    valued = (31, 32, 33, seat_wind, round_wind)
    for k in range(27, 34):
        if counts[k] >= 3:
            if k in valued:
                add("yakuhai_%d" % k, 1, 1)

    # chiitoitsu
    if len([1 for n in counts if n >= 2]) >= 6 and sum(1 for n in counts if n) >= 13:
        add("chiitoitsu", 2)

    # toitoi: no runs possible (all sets/pairs) — rough check
    runs = _runs(counts)
    if not runs and sum(1 for n in counts if n >= 2) >= 5:
        add("toitoi", 2, 2)

    # iipeiko / sanshoku / ittsu over the suited part
    run_ids = set(runs)
    for base in (0, 9, 18):
        # ittsu 123+456+789
        if all((base + r, base + r + 1, base + r + 2) in run_ids for r in (0, 3, 6)):
            add("ittsu", 2, 1)
    for r in range(7):
        trip = [(base + r, base + r + 1, base + r + 2) for base in (0, 9, 18)]
        if all(t in run_ids for t in trip):
            add("sanshoku", 2, 1)
    if not open_hand:
        for base in (0, 9, 18):
            for r in range(7):
                run = (base + r, base + r + 1, base + r + 2)
                c = sum(1 for m in (melds or []) if tuple(t // 4 for t in m["tiles"]) == run)
                if counts[run[0]] >= 2 and counts[run[1]] >= 2 and counts[run[2]] >= 2:
                    add("iipeiko", 1)
                    break
            else:
                continue
            break

    # sanshoku doukou
    for r in range(9):
        if all(counts[base + r] >= 3 for base in (0, 9, 18)):
            add("sanshoku_doukou", 2, 2)

    # sanankou (3 concealed triplets — closed pon/kans)
    concealed_sets = sum(1 for k, n in enumerate(counts) if n >= 3)
    if concealed_sets >= 3:
        add("sanankou", 2, 2)

    # honitsu / chinitsu
    suit_kinds = {k // 9 for k in kinds if k < 27}
    has_honor = any(k >= 27 for k in kinds)
    if len(suit_kinds) == 1:
        if has_honor:
            add("honitsu", 3, 2)
        else:
            add("chinitsu", 6, 5)

    # shosangen: two dragon triplets + dragon pair
    dragon_triplets = sum(1 for k in (31, 32, 33) if counts[k] >= 3)
    dragon_pair = any(counts[k] == 2 for k in (31, 32, 33))
    if dragon_triplets == 2 and dragon_pair:
        add("shosangen", 2, 2)

    dora = count_dora(hand136, melds, dora_indicators, aka)
    han += dora["dora"] + dora["aka"]
    if not open_hand:
        yaku.append("menzen")
        han += 1  # riichi potential
    return {
        "yaku": yaku,
        "han_est": han,
        "dora": dora["dora"],
        "aka": dora["aka"],
        "closed": not open_hand,
        "shanten": calc_shanten(hand136, melds),
    }


def explain_decision(game, seat):
    """Full §4 explain block for an engine observation (convenience wrapper)."""
    hand = list(game.hands[seat])
    melds = game.melds[seat]
    visible = []
    for s in range(4):
        visible.extend(game.hands[s])
        for m in game.melds[s]:
            visible.extend(m["tiles"])
        for d in game.discards[s]:
            visible.append(d["tile"])
    visible.extend(game.dora_indicators)
    safety = tile_safety(game.discards, visible)
    others = {s: safety[s] for s in range(4) if s != seat}
    return {
        "shanten": calc_shanten(hand, melds),
        "ukeire": calc_ukeire(hand, melds, visible),
        "safety": others,
        "yaku_expectation": yaku_expectation(
            hand, melds, game.dora_indicators, seat=seat,
            round_idx=game.round_idx, aka=game.cfg.aka, kuitan=game.cfg.kuitan),
    }
