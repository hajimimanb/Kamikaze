"""Decision-point extraction: parsed mjlog games -> records.jsonl.

Record contract: docs/observation_schema.md sections 2 and 3.

Sampling rules implemented (section 3):
- discard decisions: every draw by a non-riichi player (real choice)
- riichi: the declaration moment (label type=riichi carries the tile)
- calls (chow/pon/kan): sampled only when the player actually called
- ron/tsumo: sampled at the win (natural 1:1 weighting, no downsampling)
- ryuukyoku rounds produce no labels (stats only)
- rounds beyond index 7 (west rounds) are never sampled
- forced tsumogiri after riichi is not sampled (no real choice)

legal_actions:
- shanten/tenpai via tenhou.shanten_tmp (temporary; TODO unify with riichi/)
- ron/tsumo yaku validity via mahjong HandCalculator (open tanyao + aka dora)
- kuikae (喰い替え) restrictions applied to the discard right after a call
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from . import mjlog_parser as mp
from .shanten_tmp import (is_winning_hand, tenpai_discards)
from riichi.tiles import is_terminal_or_honor
from mahjong.tile import TilesConverter

RED_FIVES = (16, 52, 88)

# schema 2 uses "chow" for the chi label; the internal/mjai name is "chi"
_RECORD_MELD_TYPE = {"chi": "chow"}

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _counts34(tiles: List[int]) -> List[int]:
    c = [0] * 34
    for t in tiles:
        c[t // 4] += 1
    return c


def _is_complete(counts: List[int]) -> bool:
    from .shanten_tmp import shanten34
    n = sum(counts)
    if n % 3 != 2:
        return False
    return shanten34(counts) == -1


def _meld_record(m: dict, from_abs: Optional[int]) -> dict:
    return {"type": _RECORD_MELD_TYPE.get(m["type"], m["type"]),
            "tiles": sorted(m["tiles"]),
            "from": from_abs, "red": any(t in RED_FIVES for t in m["tiles"])}


# chow/pon/kan enumeration on a discarded tile
def _chow_options(hand: List[int], called: int) -> List[List[int]]:
    kind = called // 4
    if kind >= 27:
        return []
    counts = _counts34(hand)
    out = []
    suit = kind // 9
    low = suit * 9
    for base in (kind - 2, kind - 1, kind):
        if base < low or base + 2 >= low + 9:
            continue
        need = [base, base + 1, base + 2]
        need.remove(kind)
        a, b = need
        if counts[a] >= 1 and counts[b] >= 1:
            # enumerate concrete copies (first available copy of each kind;
            # red5 (copy 0) naturally included)
            ca = [t for t in hand if t // 4 == a]
            cb = [t for t in hand if t // 4 == b]
            for x in ca:
                for y in cb:
                    out.append(sorted([called, x, y]))
    # dedup
    seen = set()
    res = []
    for o in out:
        key = tuple(o)
        if key not in seen:
            seen.add(key)
            res.append(o)
    return res


def _pon_options(hand: List[int], called: int) -> List[List[int]]:
    kind = called // 4
    copies = [t for t in hand if t // 4 == kind]
    if len(copies) < 2:
        return []
    out = []
    for i in range(len(copies)):
        for j in range(i + 1, len(copies)):
            out.append(sorted([called, copies[i], copies[j]]))
    return out


def _minkan_options(hand: List[int], called: int) -> List[List[int]]:
    kind = called // 4
    copies = [t for t in hand if t // 4 == kind]
    if len(copies) < 3:
        return []
    return [sorted([called] + copies[:3])]


# ---------------------------------------------------------------------------
# round-level extractor
# ---------------------------------------------------------------------------

class RoundState:
    def __init__(self, rnd: dict):
        self.round = rnd["round"]
        self.honba = rnd["honba"]
        self.riichi_sticks = rnd["riichi_sticks"]
        self.dora_indicators = [rnd["dora_ind"]]
        self.scores = [s * 100 for s in rnd["scores"]]  # ten attr is pts/100
        self.oya = rnd["oya"]
        self.hands = [list(h) for h in rnd["hands"]]
        self.melds: List[List[dict]] = [[], [], [], []]
        self.discards: List[List[dict]] = [[], [], [], []]
        self.riichi = [False] * 4
        self.n_kan = 0
        self.wall_left = 70
        self.draws_made: List[int] = [0] * 4
        self.last_draw: List[Optional[int]] = [None] * 4
        self.pending_riichi: List[bool] = [False] * 4
        self.kuikae_forbidden: List[set] = [set(), set(), set(), set()]
        self.last_meld_type: List[str] = ["none"] * 4
        self.pending: Dict[int, Optional[dict]] = {0: None, 1: None, 2: None, 3: None}
        self.last_discard: Optional[Tuple[int, int]] = None
        self.last_kakan: Optional[Tuple[int, int]] = None  # (seat, tile)
        self.last_event: Optional[dict] = None

    # -- helpers -----------------------------------------------------------
    def snapshot(self, seat: int, hand: List[int], last_event: dict,
                 discards_override: Optional[List] = None) -> dict:
        melds = [[_meld_record(m, self._meld_from(m, p))
                  for m in self.melds[p]] for p in range(4)]
        discards = (discards_override
                    if discards_override is not None
                    else [[dict(d) for d in self.discards[p]] for p in range(4)])
        return {
            "seat": seat,
            "round": self.round,
            "honba": self.honba,
            "riichi_sticks": self.riichi_sticks,
            "wall_left": self.wall_left,
            "dora_indicators": list(self.dora_indicators),
            "scores": list(self.scores),
            "oya": self.oya,
            "hand": sorted(hand),
            "melds": melds,
            "discards": discards,
            "n_kan": self.n_kan,
            "riichi_declared": list(self.riichi),
            "last_event": last_event,
        }

    def _meld_from(self, m: dict, seat: int) -> Optional[int]:
        if m["type"] in ("ankan", "nuki", "kakan"):
            return None
        return (seat + m["from"]) % 4

    def emit(self, seat: int, rec: dict, label: dict, legal: dict, out: list,
             game_id: str) -> None:
        if self.round > 7:
            return
        full = self.snapshot(seat, rec["hand"], rec["last_event"],
                              rec.get("discards"))
        full.update({"game_id": game_id, "source": "tenhou",
                     "legal_actions": legal, "label": label, "explain": {}})
        out.append(full)


class GameStats:
    def __init__(self):
        self.rounds = 0
        self.agari = 0
        self.ryuukyoku = 0
        self.ryuukyoku_types: Dict[str, int] = {}
        self.anomalies = 0

    def as_dict(self) -> dict:
        return {"rounds": self.rounds, "agari": self.agari,
                "ryuukyoku": self.ryuukyoku,
                "ryuukyoku_types": self.ryuukyoku_types,
                "anomalies": self.anomalies}


def extract_game(game: dict, records: list, stats: GameStats) -> None:
    game_id = game["log_id"] or "?"
    for rnd in game["rounds"]:
        st = RoundState(rnd)
        stats.rounds += 1
        for ev in rnd["events"]:
            kind = ev[0]
            if kind == "draw":
                _on_draw(st, ev[1], ev[2], game_id, records, stats)
            elif kind == "discard":
                _on_discard(st, ev[1], ev[2], game_id, records, stats)
            elif kind == "meld":
                _on_meld(st, ev[1], ev[2], game_id, records, stats)
            elif kind == "reach":
                if ev[2] == 1:
                    st.pending_riichi[ev[1]] = True
                else:
                    st.riichi_sticks += 1
                    if ev[3]:
                        st.scores = [s * 100 for s in ev[3]]
            elif kind == "dora":
                st.dora_indicators.append(ev[1])
            elif kind == "agari":
                _on_agari(st, ev[1], game_id, records, stats)
            elif kind == "ryuukyoku":
                # 九种九牌 (abortive): the declarer's first draw ends the round
                # before any discard; emit their pending draw as a kyushu label.
                if (ev[1] or {}).get("type") == "yao9":
                    for p in range(4):
                        if st.pending[p] is not None:
                            rec = st.pending[p]
                            st.pending[p] = None
                            legal = _draw_legal(st, p, rec["drawn"], rec["hand"])
                            legal["kyushu"] = True
                            st.emit(p, rec, {"type": "kyushu"}, legal,
                                    records, game_id)
                            break
                st.pending = {k: None for k in st.pending}
                stats.ryuukyoku += 1
                t = ev[1]["type"] or "?"
                stats.ryuukyoku_types[t] = stats.ryuukyoku_types.get(t, 0) + 1
        if rnd["result"] is not None and rnd["result"][0] == "agari":
            stats.agari += 1
        # drop any dangling pending records (should not happen)
        for p in range(4):
            if st.pending[p] is not None:
                stats.anomalies += 1
                st.pending[p] = None


def _draw_legal(st: RoundState, seat: int, tile: int, hand: List[int]) -> dict:
    # schema 2.1: the discard list must be the COMPLETE tile136 list of the
    # hand (duplicates listed separately; the encoder builds the 34-dim mask)
    legal = {"discard": sorted(hand), "riichi": [], "chow": [],
             "pon": [], "kan": [], "ron": False, "tsumo": False}
    # kuikae: remove forbidden kinds from legal discards
    if st.kuikae_forbidden[seat]:
        legal["discard"] = [t for t in legal["discard"]
                            if t // 4 not in st.kuikae_forbidden[seat]]
    # tsumo (only on a real draw)
    counts = _counts34(hand)
    if tile is not None and _is_complete(counts):
        legal["tsumo"] = is_winning_hand(
            hand, tile, st.melds[seat], st.dora_indicators, seat,
            st.round, is_tsumo=True, is_riichi=st.riichi[seat])
    # kyushu (九种九牌): first draw of the round, no calls, 9+ terminal kinds
    if (st.draws_made[seat] == 1 and not any(st.melds)
            and tile is not None
            and len({t // 4 for t in hand
                     if is_terminal_or_honor(t // 4)}) >= 9):
        legal["kyushu"] = True
    # kan (ankan / kakan)
    if st.wall_left > 0 and st.n_kan < 4:
        c34 = counts
        for kind in range(34):
            if c34[kind] == 4:
                if not st.riichi[seat]:  # riichi+ankan: rare, skipped (approx)
                    tiles = sorted(kind * 4 + i for i in range(4))
                    legal["kan"].append({"type": "ankan", "tiles": tiles})
        for m in st.melds[seat]:
            if m["type"] == "pon":
                kind = m["tiles"][0] // 4
                for t in hand:
                    if t // 4 == kind:
                        legal["kan"].append(
                            {"type": "kakan", "tiles": sorted(m["tiles"] + [t])})
    # riichi: menzen (open melds break menzen; ankan does not),
    # score >= 1000, wall >= 4
    menzen = not any(m["type"] in ("chi", "pon", "minkan", "kakan")
                     for m in st.melds[seat])
    if (menzen and not st.riichi[seat]
            and st.scores[seat] >= 1000 and st.wall_left >= 4):
        legal["riichi"] = tenpai_discards(hand)
    return legal


def _on_draw(st: RoundState, seat: int, tile: int, game_id: str,
             records: list, stats: GameStats) -> None:
    st.wall_left -= 1
    st.draws_made[seat] += 1
    st.last_draw[seat] = tile
    st.hands[seat].append(tile)
    st.last_event = {"type": "draw", "seat": seat, "tile": tile}
    st.last_discard = None
    if st.riichi[seat]:
        return  # forced tsumogiri: no discard sample; tsumo win is sampled
               # separately at the AGARI event
    rec = {"hand": sorted(st.hands[seat]),
           "drawn": tile,
           "last_event": dict(st.last_event),
           "discards": [[dict(d) for d in st.discards[p]] for p in range(4)]}
    st.pending[seat] = rec


def _finalize_draw(st: RoundState, seat: int, label: dict, records: list,
                   game_id: str, stats: GameStats) -> None:
    rec = st.pending[seat]
    if rec is None:
        stats.anomalies += 1
        return
    st.pending[seat] = None
    legal = _draw_legal(st, seat, rec["drawn"], rec["hand"])
    st.emit(seat, rec, label, legal, records, game_id)


def _call_legal(st: RoundState, seat: int, called: int,
                 hand: Optional[List[int]] = None) -> dict:
    hand = st.hands[seat] if hand is None else hand
    legal = {"discard": [], "riichi": [], "chow": [], "pon": [], "kan": [],
             "ron": False, "tsumo": False}
    legal["pon"] = [{"tiles": o} for o in _pon_options(hand, called)]
    if st.last_discard is not None and (seat + 3) % 4 == st.last_discard[0]:
        legal["chow"] = [{"tiles": o} for o in _chow_options(hand, called)]
    legal["kan"] = [{"tiles": o} for o in _minkan_options(hand, called)]
    # ron: hand + called complete and has yaku
    counts = _counts34(hand + [called])
    if _is_complete(counts):
        legal["ron"] = is_winning_hand(
            hand + [called], called, st.melds[seat], st.dora_indicators,
            seat, st.round, is_tsumo=False, is_riichi=st.riichi[seat])
    return legal


def _on_discard(st: RoundState, seat: int, tile: int, game_id: str,
                records: list, stats: GameStats) -> None:
    tsumogiri = st.last_draw[seat] == tile
    st.last_draw[seat] = None
    was_riichi_decl = st.pending_riichi[seat]
    # post-call discard (no draw): a real choice point, sampled too
    if (st.pending[seat] is None and not st.riichi[seat]
            and st.kuikae_forbidden[seat]):
        st.pending[seat] = {
            "hand": sorted(st.hands[seat]),
            "drawn": None,
            # schema 2.1: the trigger is the meld the player just made
            # ("chow" per schema label naming)
            "last_event": {"type": _RECORD_MELD_TYPE.get(
                st.last_meld_type[seat], st.last_meld_type[seat]),
                "seat": seat},
            "discards": [[dict(d) for d in st.discards[p]] for p in range(4)],
        }
    if tile in st.hands[seat]:
        st.hands[seat].remove(tile)
    else:
        stats.anomalies += 1
    st.discards[seat].append({"tile": tile, "tsumogiri": tsumogiri,
                              "riichi": was_riichi_decl})
    st.last_discard = (seat, tile)
    st.last_kakan = None
    st.last_event = {"type": "discard", "seat": seat, "tile": tile}
    # finalize the discarder's pending record (draw or post-call)
    if st.pending[seat] is not None and not st.riichi[seat]:
        # captain enhancement 2026-08-25: explicit tsumogiri flag on the
        # label. discard: true iff the tile equals the just-drawn tile of a
        # draw decision (last_event.type == "draw"); riichi declarations are
        # always recorded as hand-cut (false); post-call discards are false.
        is_draw_decision = st.pending[seat]["last_event"]["type"] == "draw"
        label_tsumogiri = bool(
            is_draw_decision and st.pending[seat]["last_event"]["tile"] == tile)
        if was_riichi_decl:
            label = {"type": "riichi", "tile": tile, "tsumogiri": False}
        else:
            label = {"type": "discard", "tile": tile,
                     "tsumogiri": label_tsumogiri}
        _finalize_draw(st, seat, label, records, game_id, stats)
    elif st.pending[seat] is not None:
        st.pending[seat] = None  # riichi'd forced discard: dropped
    st.kuikae_forbidden[seat] = set()
    if was_riichi_decl:
        st.riichi[seat] = True
        st.pending_riichi[seat] = False


def _on_meld(st: RoundState, seat: int, meld: dict, game_id: str,
             records: list, stats: GameStats) -> None:
    if meld is None:
        stats.anomalies += 1
        return
    mtype = meld["type"]
    pre_hand = sorted(st.hands[seat])
    # remove used tiles from hand
    used = []
    if mtype == "chi":
        used = [t for t in meld["tiles"] if t != meld["called"]]
    elif mtype == "pon":
        used = [t for t in meld["tiles"] if t != meld["called"]]
    elif mtype == "minkan":
        used = [t for t in meld["tiles"] if t != meld["called"]]
    elif mtype == "ankan":
        used = list(meld["tiles"])
    elif mtype == "kakan":
        used = [meld["added"]] if meld.get("added") is not None else []
    elif mtype == "nuki":
        used = list(meld["tiles"])
    for t in used:
        if t in st.hands[seat]:
            st.hands[seat].remove(t)
        else:
            stats.anomalies += 1
    from_abs = st._meld_from(meld, seat)
    new_meld = _meld_record(meld, from_abs)
    if mtype != "kakan":
        st.melds[seat].append(new_meld)
    st.last_draw[seat] = None

    st.last_meld_type[seat] = mtype
    if mtype in ("minkan", "ankan", "kakan"):
        st.n_kan += 1
        st.wall_left += 1
    if mtype in ("ankan", "kakan"):
        # self-declared kan at draw time: label the pending draw record
        if st.pending[seat] is not None:
            label = {"type": "kan", "tiles": sorted(meld["tiles"])}
            _finalize_draw(st, seat, label, records, game_id, stats)
    if mtype == "kakan":
        # update the existing pon into a kakan for future state
        added_t = [meld["added"]] if meld.get("added") is not None else []
        for m in st.melds[seat]:
            if m["type"] == "pon" and m["tiles"][0] // 4 == meld["tiles"][0] // 4:
                m["type"] = "kakan"
                m["tiles"] = sorted(m["tiles"] + added_t)
                break
        st.last_kakan = (seat, meld.get("added") or meld["tiles"][-1])
        st.last_event = {"type": "kakan", "seat": seat,
                         "tile": st.last_kakan[1]}
        st.last_discard = None
        return  # chankan (if any) is sampled at the AGARI event

    # call decision record (chow / pon / minkan from a discard)
    if mtype in ("chi", "pon", "minkan"):
        if st.last_discard is None:
            stats.anomalies += 1
            return
        (dseat, dtile) = st.last_discard
        rec = {"hand": pre_hand,
               "last_event": {"type": "discard", "seat": dseat, "tile": dtile}}
        legal = _call_legal(st, seat, meld["called"], hand=pre_hand)
        # schema 2 label types: chow (not "chi"); mjai stream keeps "chi"
        label = {"type": "chow" if mtype == "chi" else
                 ("pon" if mtype == "pon" else "kan"),
                 "tiles": sorted(meld["tiles"])}
        st.emit(seat, rec, label, legal, records, game_id)
        # kuikae for the next discard
        kind = meld["called"] // 4
        if mtype == "chi":
            tiles = sorted(meld["tiles"])
            pos = tiles.index(meld["called"])
            forbidden = {kind}
            if pos == 0 and kind % 9 <= 5:
                forbidden.add(kind + 3)
            elif pos == 2 and kind % 9 >= 3:
                forbidden.add(kind - 3)
            st.kuikae_forbidden[seat] = forbidden
        elif mtype == "pon":
            st.kuikae_forbidden[seat] = {kind}


def _on_agari(st: RoundState, agari: dict, game_id: str,
              records: list, stats: GameStats) -> None:
    who = agari["who"]
    if agari["type"] == "tsumo":
        # finalize the pending draw record as a tsumo win; riichi'd winners
        # have no pending record, so build the record from current state
        if st.pending[who] is not None:
            rec = st.pending[who]
            st.pending[who] = None
            legal = _draw_legal(st, who, agari["machi"], rec["hand"])
            legal["tsumo"] = True
            st.emit(who, rec, {"type": "tsumo"}, legal, records, game_id)
        else:
            rec = {"hand": sorted(st.hands[who]),
                   "drawn": agari["machi"],
                   "last_event": {"type": "draw", "seat": who,
                                  "tile": agari["machi"]},
                   "discards": [[dict(d) for d in st.discards[p]]
                                for p in range(4)]}
            legal = _draw_legal(st, who, agari["machi"], rec["hand"])
            legal["tsumo"] = True
            legal["discard"] = [agari["machi"]]  # forced tsumogiri in riichi
            st.emit(who, rec, {"type": "tsumo"}, legal, records, game_id)
    else:
        from_who = agari["from_who"]
        tile = agari["machi"]
        chankan = (st.last_kakan is not None and st.last_kakan[0] == from_who)
        last_event = ({"type": "kakan", "seat": from_who, "tile": tile}
                      if chankan
                      else {"type": "discard", "seat": from_who, "tile": tile})
        rec = {"hand": sorted(st.hands[who]),
               "last_event": last_event}
        legal = {"discard": [], "riichi": [], "chow": [], "pon": [], "kan": [],
                 "ron": True, "tsumo": False}
        if not chankan:
            legal["pon"] = [{"tiles": o} for o in _pon_options(st.hands[who], tile)]
            if (who + 3) % 4 == from_who:
                legal["chow"] = [{"tiles": o}
                                 for o in _chow_options(st.hands[who], tile)]
            legal["kan"] = [{"tiles": o}
                            for o in _minkan_options(st.hands[who], tile)]
        st.emit(who, rec, {"type": "ron", "tile": tile}, legal, records, game_id)


def extract_game_file(path: str, records: list, stats: GameStats,
                      game_id_override: str = None) -> None:
    game = mp.parse_game_file(path)
    if game_id_override:
        game["log_id"] = game_id_override
    elif game["log_id"] is None:
        game["log_id"] = os.path.basename(path).split(".")[0]
    extract_game(game, records, stats)


def write_records(path: str, records: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
