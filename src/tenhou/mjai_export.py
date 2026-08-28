"""MJAI protocol event-stream export (schema section 7.5).

One JSONL per game at data/processed/tenhou/mjai/{game_id}.jsonl with the
canonical mjai.app event protocol:
  start_game / start_kyoku / tsumo / dahai / chi / pon / daiminkan / ankan /
  kakan / dora / reach / hora / ryukyoku / end_kyoku / end_game

Tile notation per docs/mjai_ref/tile.rs: 1-9m/p/s, honors
E,S,W,N,P,F,C (東南西北白發中), red fives 5mr/5pr/5sr, "?" unknown.
Field names per docs/mjai_ref/event.rs (hora uses ura_markers; riichi deposit
is reach_accepted). The events derive from the same parsed game stream as
extract.py, so records.jsonl and the mjai stream stay consistent.
"""
from __future__ import annotations

import json
import os
import urllib.parse
from typing import Dict, List, Optional

from . import mjlog_parser as mp

RED = {16: "5mr", 52: "5pr", 88: "5sr"}

# Honors per docs/mjai_ref/tile.rs MJAI_PAI_STRINGS: E S W N P F C
HONORS = "ESWNPFC"

YAKU_NAMES = [
    "menzenchin tsumohou", "riichi", "ippatsu", "chankan", "rinshan kaihou",
    "haitei raoyue", "houtei raoyui", "pinfu", "tanyao", "iipeikou",
    "yakuhai:1", "yakuhai:2", "yakuhai:3", "yakuhai:4",
    "yakuhai:5", "yakuhai:6", "yakuhai:7", "yakuhai:8",
    "yakuhai:9", "yakuhai:10", "yakuhai:11",
    "daburu riichi", "chiitoitsu", "chanta", "ittsu", "sanshoku doujun",
    "sanshoku doukou", "sankantsu", "toitoi", "sanankou", "shousangen",
    "honroutou", "ryanpeikou", "junchan", "honitsu", "chinitsu", "renhou",
    "tenhou", "chihou", "daisangen", "suuankou", "suuankou tanki",
    "tsuuiisou", "ryuuiisou", "chinroutou", "chuuren poutou",
    "chuuren poutou 9-wait", "kokushi musou", "kokushi musou 13-wait",
    "daisuushi", "shousuushi", "suukantsu", "dora", "uradora", "akadora",
]

RYUUKYOKU_REASONS = {
    "yao9": "kyuushukyuuhai",
    "reach4": "suuchariichi",
    "ron3": "hule3",
    "kan4": "suukaikan",
    "kaze4": "suufonrenda",
    "nm": "nagashi mangan",
    "nagashi": "nagashi mangan",
}


def tile_str(tile: int) -> str:
    if tile in RED:
        return RED[tile]
    kind = tile // 4
    if kind < 27:
        return "%d%s" % (kind % 9 + 1, "mps"[kind // 9])
    return HONORS[kind - 27]


def _xor3(t: int) -> int:
    # classic converter ordering: sorted(..., key=lambda x: x ^ 3) puts the
    # red five (copy 0) AFTER the plain copies of the same kind
    return t ^ 3


def tiles_str(tiles, key=None) -> List[str]:
    return [tile_str(t) for t in sorted(tiles, key=key)]


def _name(s: Optional[str]) -> str:
    if s is None:
        return "NoName"
    try:
        return urllib.parse.unquote(s)
    except Exception:
        return "NoName"


def build_events(game: dict) -> List[dict]:
    events: List[dict] = []
    names = game.get("players") or [None, None, None, None]
    gtype = game.get("type") or 0
    # classic converter start_game extras (tolerated by event.rs):
    # kyoku_first = 0 for hanchan, 4 for tonpu; aka_flag = red fives enabled
    events.append({"type": "start_game",
                   "names": [_name(n) for n in names],
                   "kyoku_first": 0 if (gtype & 0x08) else 4,
                   "aka_flag": not bool(gtype & 0x02)})

    for rnd in game["rounds"]:
        round_i = rnd["round"]
        bakaze = "ESWN"[min(round_i // 4, 3)]
        events.append({
            "type": "start_kyoku",
            "bakaze": bakaze,
            "kyoku": round_i % 4 + 1,
            "honba": rnd["honba"],
            "kyotaku": rnd["riichi_sticks"],
            "dora_marker": tile_str(rnd["dora_ind"]),
            "oya": rnd["oya"],
            "scores": [s * 100 for s in rnd["scores"]],
            # x^3 ordering keeps 5mr/5pr/5sr after plain 5s (classic converter)
            "tehais": [tiles_str(h, key=_xor3) for h in rnd["hands"]],
        })
        last_draw = [None] * 4
        round_ura: List[str] = []   # shared across horas of the round
        round_horas: List[dict] = []
        for ev in rnd["events"]:
            kind = ev[0]
            if kind == "draw":
                s, t = ev[1], ev[2]
                last_draw[s] = t
                events.append({"type": "tsumo", "actor": s, "pai": tile_str(t)})
            elif kind == "discard":
                s, t = ev[1], ev[2]
                tsumogiri = last_draw[s] == t
                last_draw[s] = None
                events.append({"type": "dahai", "actor": s, "pai": tile_str(t),
                               "tsumogiri": tsumogiri})
            elif kind == "meld":
                s, m = ev[1], ev[2]
                _emit_meld(events, s, m)
                # C2 fix: a chi/pon/minkan consumes a discard, not a draw —
                # the caller's next discard can never be tsumogiri; clear the
                # stale drawn tile (~1% of dahai flags were wrong)
                last_draw[s] = None
            elif kind == "reach":
                # step 1 declares (before the dahai); step 2 is the deposit,
                # emitted as reach_accepted right after the riichi discard
                # (docs/mjai_ref/event.rs + conv.rs reference behavior)
                if ev[2] == 1:
                    events.append({"type": "reach", "actor": ev[1]})
                else:
                    events.append({"type": "reach_accepted", "actor": ev[1]})
            elif kind == "dora":
                events.append({"type": "dora", "dora_marker": tile_str(ev[1])})
            elif kind == "agari":
                hora = _make_hora(ev[1])
                if hora["ura_markers"] and not round_ura:
                    round_ura = hora["ura_markers"]
                    # backfill earlier horas of this round (double ron)
                    for h in round_horas:
                        h["ura_markers"] = round_ura
                hora["ura_markers"] = round_ura
                round_horas.append(hora)
                events.append(hora)
            elif kind == "ryuukyoku":
                _emit_ryuukyoku(events, ev[1])
        events.append({"type": "end_kyoku"})

    owari = game.get("owari")
    if owari:
        try:
            vals = [float(x) for x in owari.split(",")]
            final = [int(x * 100) for x in vals[::2]]
            events.append({"type": "end_game", "scores": final})
        except ValueError:
            events.append({"type": "end_game"})
    else:
        events.append({"type": "end_game"})
    return events


def _emit_meld(events: List[dict], actor: int, m: dict) -> None:
    mtype = m["type"]
    if mtype == "chi":
        events.append({
            "type": "chi", "actor": actor, "target": (actor + m["from"]) % 4,
            "pai": tile_str(m["called"]),
            "consumed": tiles_str([t for t in m["tiles"] if t != m["called"]],
                                  key=_xor3),
        })
    elif mtype == "pon":
        events.append({
            "type": "pon", "actor": actor, "target": (actor + m["from"]) % 4,
            "pai": tile_str(m["called"]),
            "consumed": tiles_str([t for t in m["tiles"] if t != m["called"]],
                                  key=_xor3),
        })
    elif mtype == "minkan":
        events.append({
            "type": "daiminkan", "actor": actor,
            "target": (actor + m["from"]) % 4,
            "pai": tile_str(m["called"]),
            "consumed": tiles_str([t for t in m["tiles"] if t != m["called"]],
                                  key=_xor3),
        })
    elif mtype == "ankan":
        # classic converter: copies in descending order (red five last)
        base = m["tiles"][0] // 4 * 4
        events.append({"type": "ankan", "actor": actor,
                       "consumed": [tile_str(base + 3 - i)
                                    for i in range(4)]})
    elif mtype == "kakan":
        added = m.get("added")
        events.append({
            "type": "kakan", "actor": actor, "pai": tile_str(added),
            "consumed": tiles_str([t for t in m["tiles"] if t != added],
                                  key=_xor3),
        })
    elif mtype == "nuki":
        events.append({"type": "dahai", "actor": actor,
                       "pai": tile_str(m["tiles"][0]), "tsumogiri": False})


def _make_hora(a: dict) -> dict:
    who = a["who"]
    from_who = a["from_who"]
    hand_tiles = list(a["hand"])
    for m in a["melds"]:
        hand_tiles.extend(m["tiles"])
    yaku_pairs = []
    if a.get("yaku"):
        vals = a["yaku"]
        for i in range(0, len(vals) - 1, 2):
            yaku_pairs.append([YAKU_NAMES[vals[i]], vals[i + 1]])
    if a.get("yakuman"):
        for yid in a["yakuman"]:
            yaku_pairs.append([YAKU_NAMES[yid], 13])
    ura = []
    if a.get("uradora"):
        # tenhou writes "0" as a null placeholder for doraHaiUra
        ura = [tile_str(t) for t in a["uradora"] if t != 0]
    ten = a.get("ten") or [0, 0, 0]
    sc = a.get("sc") or []
    scores_before = [x * 100 for x in sc[::2]] if sc else [0, 0, 0, 0]
    gains = [x * 100 for x in sc[1::2]] if sc else [0, 0, 0, 0]
    return {
        "type": "hora",
        "actor": who,
        "target": from_who,
        "pai": tile_str(a["machi"]) if a.get("machi") is not None else "?",
        # event.rs field name is ura_markers; the extra fields below are
        # tolerated by the official serde parser (unknown fields ignored)
        "ura_markers": ura,
        "hora_tehais": tiles_str(hand_tiles, key=_xor3),
        "yakus": yaku_pairs,
        "fu": ten[0],
        "hora_points": ten[1],
        "deltas": gains,
        "scores": [b + g for b, g in zip(scores_before, gains)],
    }


def _emit_ryuukyoku(events: List[dict], r: dict) -> None:
    rtype = r.get("type")
    reason = RYUUKYOKU_REASONS.get(rtype, "hule4")
    hands = r.get("hands") or [[], [], [], []]
    tehais = [tiles_str(h) for h in hands]
    tenpais = [len(h) > 0 for h in hands]
    sc = r.get("sc") or []
    scores_before = [x * 100 for x in sc[::2]] if sc else [0, 0, 0, 0]
    gains = [x * 100 for x in sc[1::2]] if sc else [0, 0, 0, 0]
    events.append({
        # C1 fix: canonical snake_case of event.rs's Ryukyoku is "ryukyoku"
        # (Hepburn, single u) — external mjai tooling (official parser,
        # classic converter) expects this spelling
        "type": "ryukyoku",
        "reason": reason,
        "tehais": tehais,
        "tenpais": tenpais,
        "deltas": gains,
        "scores": [b + g for b, g in zip(scores_before, gains)],
    })


def mask_events(events: List[dict], seat: int) -> List[dict]:
    """Masked view from the given seat's perspective (mjai convention, cf.
    the official DockerMjaiLogEngine): other players' start_kyoku tehais
    become "?" and other players' tsumo pai become "?"; everything else
    (dahai, melds, dora, hora, ryuukyoku) is public and stays visible."""
    out = []
    for e in events:
        e = dict(e)
        t = e["type"]
        if t == "start_kyoku":
            tehais = []
            for p in range(4):
                row = e["tehais"][p]
                tehais.append(row if p == seat else ["?"] * len(row))
            e["tehais"] = tehais
        elif t == "tsumo" and e["actor"] != seat:
            e["pai"] = "?"
        out.append(e)
    return out


def _write_events(path: str, events: List[dict]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return path


def export_game(game: dict, out_dir: str, masked: bool = True,
                mask_seat: int = 0) -> str:
    """Write the MJAI stream.

    - {out_dir}/{logid}.jsonl        : omniscient (all four tehais visible;
      mjlog XML contains full hands so this costs nothing extra)
    - {out_dir}/{logid}.masked.jsonl : masked from mask_seat's perspective
      (the reviewer's strict replay validator uses the omniscient one for
      full checks; the masked one mirrors the training-side observable state)
    """
    events = build_events(game)
    path = _write_events(os.path.join(out_dir, game["log_id"] + ".jsonl"),
                         events)
    if masked:
        _write_events(os.path.join(out_dir, game["log_id"] + ".masked.jsonl"),
                      mask_events(events, mask_seat))
    return path


def export_game_file(path: str, out_dir: str, masked: bool = True,
                     mask_seat: int = 0) -> str:
    game = mp.parse_game_file(path)
    if game["log_id"] is None:
        game["log_id"] = os.path.basename(path).split(".")[0]
    return export_game(game, out_dir, masked, mask_seat)
