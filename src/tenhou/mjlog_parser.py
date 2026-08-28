"""State-machine parser for tenhou.net mjlog XML.

Format notes (verified 2026-08 against tenhou.net/3 client JS, the
MahjongRepository / shinkuan / NikkeTryHard parsers and real logs):

- Root: <mjloggm ver="2.3">. Sequence: SHUFFLE, GO, UN, [BYE], TAIKYOKU,
  then per round: INIT, events..., AGARI|RYUUKYOKU, optionally owari attr.
- Draws: T/U/V/W = seats 0/1/2/3. Discards: D/E/F/G = seats 0/1/2/3.
  mjlog files contain UPPERCASE discard tags only (lowercase exists in the
  realtime protocol, not in mjlog); tsumogiri is COMPUTED by comparing the
  discard tile with the seat's most recently drawn tile. A lowercase tag, if
  ever seen, is additionally treated as tsumogiri.
- INIT seed = "round,honba,riichi_sticks,dice0,dice1,dora_ind".
- N m attribute (meld): official bit layout (confirmed against tenhou client
  function Ac and independent parsers):
    m & 0x3   = from-who (0=self/ankan, 1=shimocha, 2=toimen, 3=kamicha)
    bit 0x4   = chi:   base kind*4 from ((m>>10)//3 mapped suit*7), copies
                from (m>>3)&3, (m>>5)&3, (m>>7)&3; called = (m>>10)%3 index.
    bits 0x8/0x10 (0x18) = pon (bit 0x8) / kakan (bit 0x10):
                t4 = (m>>5)&3 (the unused/added copy), called index = (m>>9)%3,
                kind = (m>>9)//3; pon tiles = the 3 copies excluding t4.
    else (0x20 nuki or kan): kind*4 = (m & 0xFF00) >> 8 & ~3, copy = low bits.
  Tile values are kind*4+copy; copy 0 of 5m/5p/5s is the red five (16/52/88).
- REACH step=1 declares (before the discard), step=2 deposits the stick.
- RYUUKYOKU type: yao9 | reach4 | ron3 | kan4 | kaze4 | nm | nagashi;
  hai0-3 (tenpai hands) only for nm; sc always present.
- AGARI: who==fromWho is tsumo; multiple AGARI = double/triple ron.
  paoWho appears for sekinin-barai. owari attr on the last AGARI/RYUUKYOKU.
"""
from __future__ import annotations

import gzip
import io
import re
import xml.etree.ElementTree as ET
from typing import Dict, Iterator, List, Optional, Tuple

TILE_RE = re.compile(r"^([TUVWDEFGtuvwdefg])([0-9]+)$")

DRAW_LETTERS = {"T": 0, "U": 1, "V": 2, "W": 3}
DISCARD_LETTERS = {"D": 0, "E": 1, "F": 2, "G": 3}
DRAW_OF = {v: k for k, v in DRAW_LETTERS.items()}


class ParseError(Exception):
    pass


# ---------------------------------------------------------------------------
# meld decoding (official tenhou encoding)
# ---------------------------------------------------------------------------

def decode_meld(m: int) -> dict:
    """Decode the N tag's m attribute into
    {"type": chi|pon|minkan|ankan|kakan|nuki,
     "tiles": [tile136...] (pon: 3 tiles; kan: 4; chi: 3),
     "called": tile136 or None,
     "from": 0-3 relative-to-caller or None}."""
    g = m & 0x3
    if m & 0x4:  # chi
        p = (m & 0xFC00) >> 10
        c = p % 3
        p = p // 3
        base = 4 * (9 * (p // 7) + p % 7)
        tiles = [base + ((m & 0x18) >> 3),
                 base + 4 + ((m & 0x60) >> 5),
                 base + 8 + ((m & 0x180) >> 7)]
        called = tiles[c]
        return {"type": "chi", "tiles": tiles, "called": called, "from": g}
    if m & 0x18:  # pon / kakan
        t4 = (m & 0x60) >> 5
        p = (m & 0xFE00) >> 9
        c = p % 3
        base = 4 * (p // 3)
        tiles = [base + i for i in range(4)]
        added = tiles.pop(t4)  # for pon: unused copy; for kakan: added tile
        called = tiles[c]
        if m & 0x10:  # kakan
            return {"type": "kakan", "tiles": tiles + [added],
                    "called": called, "from": g, "added": added}
        return {"type": "pon", "tiles": tiles, "called": called, "from": g,
                "unused": added}
    if m & 0x20:  # nuki (sanma north)
        t = (m & 0xFF00) >> 8
        return {"type": "nuki", "tiles": [t], "called": None, "from": None}
    # kan
    base = ((m & 0xFF00) >> 8) & ~3
    copy = (m & 0xFF00) >> 8 & 3
    tiles = [base + i for i in range(4)]
    if g == 0:
        return {"type": "ankan", "tiles": tiles, "called": None, "from": None}
    return {"type": "minkan", "tiles": tiles, "called": base + copy, "from": g}


def meld_from_abs(meld: dict, who: int) -> Optional[int]:
    """Absolute seat the called tile came from (None for ankan/nuki/kakan)."""
    if meld["type"] in ("ankan", "nuki", "kakan"):
        return None
    return (who + meld["from"]) % 4


# ---------------------------------------------------------------------------
# game parsing
# ---------------------------------------------------------------------------

def _int_or_none(attrs: dict, key: str) -> Optional[int]:
    v = attrs.get(key)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _csv_int(attrs: dict, key: str) -> Optional[List[int]]:
    v = attrs.get(key)
    if v is None:
        return None
    if v == "":
        return []
    try:
        return [int(x) for x in v.split(",")]
    except ValueError:
        return None


def parse_game(xml_bytes: bytes) -> dict:
    """Parse one mjlog game into a dict of events (see module docstring)."""
    # strip a possible leading BOM / trailing garbage, feed iterparse
    if xml_bytes[:3] == b"\xef\xbb\xbf":
        xml_bytes = xml_bytes[3:]
    try:
        tree = ET.iterparse(io.BytesIO(xml_bytes), events=("start",))
    except ET.ParseError as e:
        raise ParseError("XML parse error: %s" % e) from e

    game: dict = {
        "log_id": None, "ver": None, "type": None, "lobby": None,
        "players": None, "dan": None, "rate": None, "sx": None,
        "oya": None, "rounds": [], "owari": None,
    }
    cur_round: Optional[dict] = None

    for _, elem in tree:
        tag = elem.tag
        if not isinstance(tag, str):
            continue
        if tag == "mjloggm":
            game["ver"] = elem.get("ver")
        elif tag == "SHUFFLE":
            pass
        elif tag == "GO":
            game["type"] = _int_or_none(elem.attrib, "type")
            game["lobby"] = elem.attrib.get("lobby")
        elif tag == "UN":
            # full UN carries all four names; a reconnect UN2 has a single
            # n0 and must not overwrite the player list (classic converter
            # behavior: names come from the full UN)
            if all("n%d" % i in elem.attrib for i in range(4))                     or game["players"] is None:
                names = [elem.attrib.get("n%d" % i) for i in range(4)]
                game["players"] = names
                game["dan"] = _csv_int(elem.attrib, "dan")
                game["rate"] = _csv_int(elem.attrib, "rate")
                game["sx"] = elem.attrib.get("sx")
        elif tag == "BYE":
            pass  # disconnected player; not needed for decisions
        elif tag == "TAIKYOKU":
            game["oya"] = _int_or_none(elem.attrib, "oya")
        elif tag == "INIT":
            seed = _csv_int(elem.attrib, "seed") or [0, 0, 0, 0, 0, 0]
            cur_round = {
                "round": seed[0],
                "honba": seed[1],
                "riichi_sticks": seed[2],
                "dora_ind": seed[5],
                "oya": _int_or_none(elem.attrib, "oya"),
                "scores": _csv_int(elem.attrib, "ten") or [0, 0, 0, 0],
                "hands": [_csv_int(elem.attrib, "hai%d" % i) or [] for i in range(4)],
                "events": [],
                "result": None,
            }
            game["rounds"].append(cur_round)
        elif tag == "REACH":
            if cur_round is not None:
                ten = _csv_int(elem.attrib, "ten")
                cur_round["events"].append(
                    ("reach", _int_or_none(elem.attrib, "who"),
                     _int_or_none(elem.attrib, "step"), ten))
        elif tag == "DORA":
            if cur_round is not None:
                cur_round["events"].append(
                    ("dora", _int_or_none(elem.attrib, "hai")))
        elif tag == "N":
            if cur_round is not None:
                m = _int_or_none(elem.attrib, "m")
                who = _int_or_none(elem.attrib, "who")
                cur_round["events"].append(
                    ("meld", who, decode_meld(m) if m is not None else None))
        elif tag == "AGARI":
            if cur_round is not None:
                ba = _csv_int(elem.attrib, "ba") or [0, 0]
                meld_codes = _csv_int(elem.attrib, "m") or []
                agari = {
                    "type": "tsumo" if elem.attrib.get("who") == elem.attrib.get("fromWho") else "ron",
                    "who": _int_or_none(elem.attrib, "who"),
                    "from_who": _int_or_none(elem.attrib, "fromWho"),
                    "pao_who": _int_or_none(elem.attrib, "paoWho"),
                    "hand": _csv_int(elem.attrib, "hai") or [],
                    "melds": [decode_meld(c) for c in meld_codes],
                    "machi": _int_or_none(elem.attrib, "machi"),
                    "ten": _csv_int(elem.attrib, "ten"),
                    "yaku": _csv_int(elem.attrib, "yaku"),
                    "yakuman": _csv_int(elem.attrib, "yakuman"),
                    "dora": _csv_int(elem.attrib, "doraHai"),
                    "uradora": _csv_int(elem.attrib, "doraHaiUra"),
                    "sc": _csv_int(elem.attrib, "sc"),
                    "ba": ba,
                    "owari": elem.attrib.get("owari"),
                }
                cur_round["events"].append(("agari", agari))
                cur_round["result"] = ("agari", len(cur_round["events"]) - 1)
                if agari["owari"] is not None:
                    game["owari"] = agari["owari"]
        elif tag == "RYUUKYOKU":
            if cur_round is not None:
                ryu = {
                    "type": elem.attrib.get("type"),
                    "sc": _csv_int(elem.attrib, "sc"),
                    "ba": _csv_int(elem.attrib, "ba") or [0, 0],
                    "hands": [_csv_int(elem.attrib, "hai%d" % i) or [] for i in range(4)],
                    "owari": elem.attrib.get("owari"),
                }
                cur_round["events"].append(("ryuukyoku", ryu))
                cur_round["result"] = ("ryuukyoku", len(cur_round["events"]) - 1)
                if ryu["owari"] is not None:
                    game["owari"] = ryu["owari"]
        else:
            m = TILE_RE.match(tag)
            if m and cur_round is not None:
                letter = tag[0]
                if letter in DRAW_LETTERS:
                    cur_round["events"].append(
                        ("draw", DRAW_LETTERS[letter], int(tag[1:])))
                elif letter in DISCARD_LETTERS:
                    cur_round["events"].append(
                        ("discard", DISCARD_LETTERS[letter], int(tag[1:])))
                # lowercase tags are tolerated but, per the format, never occur

    return game


def parse_game_file(path: str) -> dict:
    """Read a (possibly gzipped) mjlog file and parse it."""
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except OSError as e:
            raise ParseError("gzip error in %s: %s" % (path, e)) from e
    return parse_game(raw)


def iter_parse_dir(paths: Iterator[str]) -> Iterator[dict]:
    """Yield parsed games; malformed files are skipped (caller should count)."""
    for p in paths:
        try:
            yield parse_game_file(p)
        except ParseError:
            continue
