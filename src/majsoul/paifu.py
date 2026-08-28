# -*- coding: utf-8 -*-
"""Official Majsoul paifu download + parse into decision records (schema §2).

Paifu source (tried in order):
  1. POST https://game.maj-soul.com/1/fetchGameRecord {"game_uuid": uuid}
  2. GET  https://game.maj-soul.com/1/?paipu={uuid} -> HTML w/ embedded JSON
  3. Mirror fallback: {mirror}api/v2/pl4/game_record/{uuid}

Raw paifu JSONs are cached to data/raw/majsoul/paifu/{uuid}.json.

Game JSON format (liqi): {"log": [[seat, {"name": "RecordX", "data": {...}}], ...],
"accounts": [...], "head": {...}}. Tiles are tile136 ids (kind*4+copy) matching
observation_schema.md §1. Red fives: 16 / 52 / 88.

Replay produces one JSON record per human decision point (schema §2):
  - after DealTile to seat s: next action of s (discard/riichi/kan/tsumo)
  - after DiscardTile: each actual call by other players (chow/pon/kan/ron)
"""
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zlib
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIFU_DIR = "C:/agentwork/data/raw/majsoul/paifu"
PAIFU_PAGE = "https://game.maj-soul.com/1/?paipu="
FETCH_RECORD = "https://game.maj-soul.com/1/fetchGameRecord"
MIRRORS = [
    "https://5-data.amae-koromo.com/",
    "https://1.data.amae-koromo.com/",
    "https://4.data.amae-koromo.com/",
]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0")

RED5 = {16, 52, 88}
ORPHANS = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}


def _to_ints(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return [int(x) for x in str(v).split(",") if x != ""]


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def _get(url, data=None, headers=None, timeout=30):
    hdrs = {"User-Agent": UA, "Accept": "*/*", "Referer": "https://game.maj-soul.com/1/"}
    hdrs.update(headers or {})
    body = json.dumps(data).encode("utf-8") if data is not None else None
    if body is not None:
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def _http_json(url, payload, timeout=30):
    status, body = _get(url, payload, timeout=timeout)
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError:
        return None


def extract_paifu_json(html):
    """Extract the game JSON from a paipu page HTML."""
    candidates = []
    for m in re.finditer(r'window\.\w+\s*=\s*(\{.*?\})\s*;?\s*</script>', html, re.S):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and ("log" in obj or "records" in obj):
                return obj
            candidates.append(obj)
        except ValueError:
            pass
    idx = html.find('"RecordNewRound"')
    if idx < 0:
        idx = html.find('RecordNewRound')
    if idx >= 0:
        start = max(html.rfind("{", 0, idx), html.rfind("[", 0, idx))
        if start >= 0:
            depth = 0
            for j in range(start, len(html)):
                c = html[j]
                if c in "{[":
                    depth += 1
                elif c in "}]":
                    depth -= 1
                    if depth == 0:
                        blob = html[start:j + 1]
                        try:
                            obj = json.loads(blob)
                            if isinstance(obj, dict):
                                return obj
                            candidates.append(obj)
                        except ValueError:
                            candidates.append(blob)
                        break
    return candidates[0] if candidates else None


def download_paifu_json(uuid, token=None, force=False, dest_dir=PAIFU_DIR):
    """Download + cache the raw paifu JSON. Returns the parsed dict or raises."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, uuid + ".json")
    if os.path.exists(dest) and os.path.getsize(dest) > 100 and not force:
        with io.open(dest, "r", encoding="utf-8") as f:
            return json.load(f)

    errors = []
    try:
        obj = _http_json(FETCH_RECORD, {"game_uuid": uuid})
        if isinstance(obj, dict) and ("log" in obj or "records" in obj or "head" in obj):
            _cache(dest, obj)
            return obj
        errors.append("fetchGameRecord: unexpected %s" % str(obj)[:120])
    except Exception as e:
        errors.append("fetchGameRecord: %s" % e)
    try:
        status, html = _get(PAIFU_PAGE + uuid)
        text = html.decode("utf-8", "replace")
        obj = extract_paifu_json(text)
        if obj is not None:
            _cache(dest, obj)
            return obj
        errors.append("paipu page: no JSON extracted (%d bytes, status %d)" % (len(text), status))
    except Exception as e:
        errors.append("paipu page: %s" % e)
    for mirror in MIRRORS:
        try:
            headers = {}
            if token:
                headers["authorization"] = "Bearer " + token
            status, body = _get(mirror + "api/v2/pl4/game_record/" + uuid, headers=headers)
            obj = json.loads(body.decode("utf-8"))
            if isinstance(obj, dict) and "log" in obj:
                _cache(dest, obj)
                return obj
            errors.append("mirror game_record: unexpected shape")
        except Exception as e:
            errors.append("mirror game_record: %s" % e)
    raise RuntimeError("failed to fetch paifu %s: %s" % (uuid, " | ".join(errors)[:400]))


def _cache(dest, obj):
    with io.open(dest, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


# --------------------------------------------------------------------------
# Mahjong primitives (34-kind counts)
# --------------------------------------------------------------------------

def tile_kind(t):
    return t // 4


def is_red5(t):
    return t in RED5


def counts34(tiles):
    ct = [0] * 34
    for t in tiles:
        ct[t // 4] += 1
    return ct


def counts_tuple(tiles):
    return tuple(counts34(tiles))


@lru_cache(maxsize=1 << 16)
def _win_rec(ct, melds, pair):
    i = 0
    while i < 34 and ct[i] == 0:
        i += 1
    if i == 34:
        return melds == 0 and pair == 0
    if melds > 0:
        if ct[i] >= 3:
            c2 = ct[:i] + (ct[i] - 3,) + ct[i + 1:]
            if _win_rec(c2, melds - 1, pair):
                return True
        if i < 27 and i % 9 <= 6 and ct[i + 1] >= 1 and ct[i + 2] >= 1:
            c2 = ct[:i] + (ct[i] - 1, ct[i + 1] - 1, ct[i + 2] - 1) + ct[i + 3:]
            if _win_rec(c2, melds - 1, pair):
                return True
    if pair and ct[i] >= 2:
        c2 = ct[:i] + (ct[i] - 2,) + ct[i + 1:]
        if _win_rec(c2, melds, 0):
            return True
    return False


def _kokushi(ct):
    for k in range(34):
        if k in ORPHANS:
            if ct[k] < 1:
                return False
        elif ct[k] != 0:
            return False
    return True


def _chiitoi(ct):
    pairs = 0
    for c in ct:
        if c % 2 == 1:
            return False
        if c:
            pairs += c // 2
    return pairs == 7


def win_shape(ct_tuple):
    """True if 34-counts form a winning shape (4 melds+pair / chiitoi / kokushi)."""
    return _win_rec(ct_tuple, 4, 1) or _chiitoi(ct_tuple) or _kokushi(ct_tuple)


def _normal_shanten(ct):
    """Exact normal-hand shanten (classic 8-2m-d-t decomposition DFS)."""
    best = [9]

    def dfs(i, m, p, q):
        if m > 4 or p > 1 or q > 4:
            return
        while i < 34 and ct[i] == 0:
            i += 1
        if i == 34:
            qq = min(q, 4 - m)
            sh = 8 - 2 * m - min(p, 1) - qq
            if sh < best[0]:
                best[0] = sh
            return
        c = ct[i]
        if c >= 2:
            ct[i] -= 2
            dfs(i, m, p + 1, q)
            dfs(i, m, p, q + 1)
            ct[i] += 2
        if c >= 3:
            ct[i] -= 3
            dfs(i, m + 1, p, q)
            ct[i] += 3
        if i < 27 and i % 9 <= 6 and ct[i + 1] >= 1 and ct[i + 2] >= 1:
            ct[i] -= 1
            ct[i + 1] -= 1
            ct[i + 2] -= 1
            dfs(i, m + 1, p, q)
            ct[i] += 1
            ct[i + 1] += 1
            ct[i + 2] += 1
        if i < 27 and i % 9 <= 7 and ct[i + 1] >= 1:
            ct[i] -= 1
            ct[i + 1] -= 1
            dfs(i, m, p, q + 1)
            ct[i] += 1
            ct[i + 1] += 1
        if i < 27 and i % 9 <= 6 and ct[i + 2] >= 1:
            ct[i] -= 1
            ct[i + 2] -= 1
            dfs(i, m, p, q + 1)
            ct[i] += 1
            ct[i + 2] += 1
        dfs(i + 1, m, p, q)

    dfs(0, 0, 0, 0)
    return best[0]


def calc_shanten(ct):
    """Exact shanten (-1 agari / 0 tenpai): normal + chiitoi + kokushi."""
    normal = _normal_shanten(list(ct))
    d = sum(1 for c in ct if c >= 2)
    quads = sum(1 for c in ct if c >= 4)
    chiitoi = 6 - d + quads
    o = sum(1 for k in ORPHANS if ct[k] >= 1)
    p = 1 if any(ct[k] >= 2 for k in ORPHANS) else 0
    kokushi = 13 - o - p
    return min(normal, chiitoi, kokushi)


def riichi_candidates(hand14, scores_seat, riichi_declared, wall_left, melds):
    """Tiles of hand14 whose discard leaves tenpai and riichi is declarable."""
    if melds or riichi_declared or scores_seat < 1000 or wall_left < 4:
        return []
    out = []
    seen = set()
    for t in hand14:
        if t in seen:
            continue
        seen.add(t)
        ct = counts34(hand14)
        ct[t // 4] -= 1
        if calc_shanten(ct) == 0:
            out.append(t)
    return out


def has_yaku(hand_counts, melds, closed, riichi_declared, is_tsumo, win_tile_kind,
             seat, round_wind):
    """Minimal majsoul yaku gate for ron/tsumo legality (approximation; missing
    pinfu/ittsu/sanshoku/etc. -> occasional false negatives, never used to
    contradict an actual win label)."""
    kinds = set(k for k, c in enumerate(hand_counts) if c > 0)
    for m in melds:
        kinds.update(tile_kind(t) for t in m["tiles"])
    all_simple = all(k < 27 and k % 9 != 0 and k % 9 != 8 for k in kinds)
    yakuhai_kinds = {31, 32, 33, 27 + seat, round_wind}
    has_yakuhai_triplet = any(hand_counts[k] >= 3 for k in yakuhai_kinds) or \
        any(tile_kind(m["tiles"][0]) in yakuhai_kinds and m["type"] in ("pon", "kan") for m in melds)
    honors_used = any(k >= 27 for k in kinds)
    suits_used = {k // 9 for k in kinds if k < 27}
    honitsu = len(suits_used) == 1 and honors_used
    chinitsu = len(suits_used) == 1 and not honors_used
    if closed:
        if riichi_declared or is_tsumo:
            return True
        return all_simple or has_yakuhai_triplet or honitsu or chinitsu or \
            _chiitoi(tuple(hand_counts)) or _kokushi(tuple(hand_counts))
    if all_simple or has_yakuhai_triplet or honitsu or chinitsu:
        return True
    if melds and all(m["type"] in ("pon", "kan") for m in melds):
        nonz = [c for c in hand_counts if c > 0]
        twos = [c for c in nonz if c == 2]
        if len(twos) == 1 and all(c == 3 for c in nonz if c != 2):
            return True
    return False


# --------------------------------------------------------------------------
# Legal actions at a decision point
# --------------------------------------------------------------------------

def _kan_options(hand, melds):
    out = []
    ct = counts34(hand)
    for k, c in enumerate(ct):
        if c == 4:
            out.append({"tiles": [k * 4 + i for i in range(4)]})
    for m in melds:
        if m["type"] == "pon" and ct[tile_kind(m["tiles"][0])] >= 1:
            k = tile_kind(m["tiles"][0])
            cands = [t for t in hand if t // 4 == k and t not in m["tiles"]]
            if cands:
                out.append({"tiles": sorted(m["tiles"] + [cands[0]])})
    return out


def _response_options(hand, melds, discard_tile, disc_seat, my_seat):
    k = discard_tile // 4
    ct = counts34(hand)
    pon, chow, kan = [], [], []
    if ct[k] >= 2:
        copies = [t for t in hand if t // 4 == k][:2]
        pon.append({"tiles": sorted(copies + [discard_tile])})
    if ct[k] >= 3:
        copies = [t for t in hand if t // 4 == k][:3]
        kan.append({"tiles": sorted(copies + [discard_tile])})
    if k < 27 and (disc_seat - my_seat) % 4 == 3:
        r = k % 9
        base = k - r
        for lo in (r - 2, r - 1, r):
            if lo < 0 or lo > 6:
                continue
            need = [base + lo + i for i in range(3)]
            if k in need:
                missing = [n for n in need if n != k]
                if all(ct[n] >= 1 for n in missing):
                    tiles = sorted([discard_tile] + [n * 4 + 0 for n in missing])
                    # use actual hand copies
                    used = {k: False for k in missing}
                    picked = []
                    for t in hand:
                        if t // 4 in used and not used[t // 4]:
                            used[t // 4] = True
                            picked.append(t)
                    tiles = sorted([discard_tile] + picked)
                    chow.append({"tiles": tiles})
    return pon, chow, kan


class RoundState:
    def __init__(self):
        self.hands = [[], [], [], []]
        self.melds = [[], [], [], []]
        self.discards = [[], [], [], []]
        self.scores = [25000, 25000, 25000, 25000]
        self.riichi_declared = [False, False, False, False]
        self.riichi_sticks = 0
        self.round = 0
        self.honba = 0
        self.oya = 0
        self.dora = []
        self.wall_left = 70
        self.n_kan = 0
        self.round_wind = 27


# --------------------------------------------------------------------------
# Game replay
# --------------------------------------------------------------------------

class _Replay:
    def __init__(self, game_id):
        self.st = RoundState()
        self.records = []
        self.game_id = game_id
        self.pending = None

    def peek(self):
        if self.pending is None:
            try:
                self.pending = next(self.events)
            except StopIteration:
                self.pending = None
        return self.pending

    def take(self):
        if self.pending is None:
            try:
                return next(self.events)
            except StopIteration:
                return None
        v = self.pending
        self.pending = None
        return v

    def base_record(self, seat, label, last_event, hand, legal):
        st = self.st
        return {
            "game_id": self.game_id,
            "source": "majsoul",
            "seat": seat,
            "round": st.round,
            "honba": st.honba,
            "riichi_sticks": st.riichi_sticks,
            "wall_left": st.wall_left,
            "dora_indicators": list(st.dora),
            "scores": list(st.scores),
            "oya": st.oya,
            "hand": sorted(hand),
            "melds": [list(m) for m in st.melds],
            "discards": [list(d) for d in st.discards],
            "n_kan": st.n_kan,
            "riichi_declared": list(st.riichi_declared),
            "last_event": last_event,
            "legal_actions": legal,
            "label": label,
        }

    def handle_responses(self, disc_seat, dt):
        """Consume + sample immediate responses to disc_seat's discard of dt."""
        st = self.st
        while True:
            nxt = self.peek()
            if nxt is None:
                break
            nidx2, (nseat, naction) = nxt
            ndata = naction.get("data") or {}
            nname = naction.get("name")
            if nname == "RecordChiPengGang" and int(ndata.get("type", -1)) in (0, 1, 2):
                self.take()
                rs = int(nseat)
                mtype = int(ndata["type"])
                tiles = _to_ints(ndata.get("tiles"))
                froms = _to_ints(ndata.get("from"))
                from_seat = disc_seat
                if froms:
                    try:
                        called_pos = tiles.index(dt)
                        from_seat = froms[called_pos] if called_pos < len(froms) else disc_seat
                    except ValueError:
                        pass
                st.n_kan += 1 if mtype == 2 else 0
                hand = list(st.hands[rs])
                pon, chow, kan = _response_options(hand, st.melds[rs], dt, disc_seat, rs)
                legal = {
                    "discard": sorted(set(hand)),
                    "riichi": [],
                    "chow": [c["tiles"] for c in chow],
                    "pon": [p["tiles"] for p in pon],
                    "kan": [k["tiles"] for k in kan],
                    "ron": False,
                    "tsumo": False,
                }
                ltype = {0: "chow", 1: "pon", 2: "kan"}[mtype]
                self.records.append(self.base_record(
                    rs, {"type": ltype, "tiles": tiles},
                    {"type": "discard", "seat": disc_seat, "tile": dt}, hand, legal))
                meld = {"type": ltype, "tiles": tiles, "from": int(from_seat),
                        "red": any(t in RED5 for t in tiles)}
                st.melds[rs].append(meld)
                for t in tiles:
                    if t != dt and t in st.hands[rs]:
                        st.hands[rs].remove(t)
                # the caller's post-call discard is also a sampled decision point
                nxt2 = self.peek()
                if nxt2 is not None:
                    n2seat, n2action = nxt2[1]
                    n2data = n2action.get("data") or {}
                    if n2action.get("name") == "RecordDiscardTile" and int(n2seat) == rs:
                        self.take()
                        dt2 = int(n2data["tile"])
                        if n2data.get("left_count") is not None:
                            st.wall_left = int(n2data["left_count"])
                        hand = list(st.hands[rs])
                        legal = {
                            "discard": sorted(set(hand)),
                            "riichi": [],
                            "chow": [],
                            "pon": [],
                            "kan": [],
                            "ron": False,
                            "tsumo": False,
                        }
                        self.records.append(self.base_record(
                            rs, {"type": "discard", "tile": dt2},
                            {"type": ltype, "seat": rs, "tiles": tiles}, hand, legal))
                        st.hands[rs].remove(dt2)
                        st.discards[rs].append({"tile": dt2,
                                                "tsumogiri": bool(n2data.get("moqie")),
                                                "riichi": bool(n2data.get("liqi"))})
                        self.handle_responses(rs, dt2)
                continue
            if nname == "RecordHule" and _is_ron(ndata):
                self.take()
                for hule in ndata.get("hules") or []:
                    hs = int(hule.get("seat", nseat))
                    legal = {
                        "discard": sorted(set(st.hands[hs])),
                        "riichi": [],
                        "chow": [],
                        "pon": [],
                        "kan": [],
                        "ron": True,
                        "tsumo": False,
                    }
                    self.records.append(self.base_record(
                        hs, {"type": "ron"},
                        {"type": "discard", "seat": disc_seat, "tile": dt},
                        list(st.hands[hs]), legal))
                _apply_hule(st, ndata)
                break
            break

    def run(self, log):
        st = self.st
        self.events = iter(enumerate(log))
        while True:
            ev = self.take()
            if ev is None:
                break
            idx, (seat, action) = ev
            name = action.get("name")
            data = action.get("data") or {}

            if name == "RecordNewRound":
                st.round = int(data.get("ju", 0))
                st.honba = int(data.get("ben", 0))
                st.oya = int(data.get("seat", 0))
                st.wall_left = int(data.get("left_count", 70))
                st.dora = [int(data["dora"])] if data.get("dora") is not None else []
                st.hands = [list(data.get("tile%d" % i, [])) for i in range(4)]
                st.melds = [[], [], [], []]
                st.discards = [[], [], [], []]
                st.riichi_declared = [False, False, False, False]
                st.n_kan = 0
                st.round_wind = 27 if st.round < 4 else 28
                if "liqibang" in data:
                    st.riichi_sticks = int(data["liqibang"])
                continue

            if name == "RecordDealTile":
                s = int(seat)
                tiles = data.get("tile") or []
                t = int(tiles[0]) if tiles else None
                if t is not None:
                    st.hands[s].append(t)
                if data.get("left_count") is not None:
                    st.wall_left = int(data["left_count"])
                nxt = self.peek()
                if nxt is None:
                    continue
                nidx, (nseat, naction) = nxt
                ndata = naction.get("data") or {}
                nname = naction.get("name")
                if nname == "RecordDiscardTile" and int(nseat) == s:
                    self.take()
                    dt = int(ndata["tile"])
                    is_riichi = bool(ndata.get("liqi"))
                    if ndata.get("left_count") is not None:
                        st.wall_left = int(ndata["left_count"])
                    hand = list(st.hands[s])
                    closed = not st.melds[s]
                    legal = {
                        "discard": sorted(set(hand)),
                        "riichi": riichi_candidates(hand, st.scores[s], st.riichi_declared[s],
                                                   st.wall_left, st.melds[s]) if closed else [],
                        "chow": [],
                        "pon": [],
                        "kan": [k["tiles"] for k in _kan_options(hand, st.melds[s])],
                        "ron": False,
                        "tsumo": False,
                    }
                    ct = counts_tuple(hand)
                    if win_shape(ct) and has_yaku(
                            counts34(hand), st.melds[s], closed, st.riichi_declared[s],
                            True, tile_kind(dt), s, st.round_wind):
                        legal["tsumo"] = True
                    label = {"type": "riichi", "tile": dt} if is_riichi else {"type": "discard", "tile": dt}
                    self.records.append(self.base_record(
                        s, label, {"type": "deal", "seat": s, "tile": t}, hand, legal))
                    st.hands[s].remove(dt)
                    st.discards[s].append({"tile": dt, "tsumogiri": bool(ndata.get("moqie")),
                                           "riichi": is_riichi})
                    if is_riichi:
                        st.riichi_declared[s] = True
                        st.riichi_sticks += 1
                        st.scores[s] -= 1000
                    self.handle_responses(s, dt)
                    continue
                if nname == "RecordAnGangAddGang" and int(nseat) == s and int(ndata.get("type", -1)) in (2, 3):
                    self.take()
                    tiles = _to_ints(ndata.get("tiles"))
                    st.n_kan += 1
                    hand = list(st.hands[s])
                    legal = {
                        "discard": sorted(set(hand)),
                        "riichi": [],
                        "chow": [],
                        "pon": [],
                        "kan": [k["tiles"] for k in _kan_options(hand, st.melds[s])],
                        "ron": False,
                        "tsumo": False,
                    }
                    self.records.append(self.base_record(
                        s, {"type": "kan", "tiles": tiles},
                        {"type": "deal", "seat": s, "tile": t}, hand, legal))
                    _apply_kan(st, s, tiles, int(ndata.get("type")))
                    continue
                if nname == "RecordHule" and int(nseat) == s and _is_tsumo(ndata, s):
                    self.take()
                    hand = list(st.hands[s])
                    legal = {
                        "discard": sorted(set(hand)),
                        "riichi": [],
                        "chow": [],
                        "pon": [],
                        "kan": [],
                        "ron": False,
                        "tsumo": True,
                    }
                    self.records.append(self.base_record(
                        s, {"type": "tsumo"},
                        {"type": "deal", "seat": s, "tile": t}, hand, legal))
                    _apply_hule(st, ndata)
                    continue
                continue

            if name == "RecordDiscardTile":
                s = int(seat)
                dt = int(data["tile"])
                if data.get("left_count") is not None:
                    st.wall_left = int(data["left_count"])
                self.handle_responses(s, dt)
                if dt in st.hands[s]:
                    st.hands[s].remove(dt)
                st.discards[s].append({"tile": dt, "tsumogiri": bool(data.get("moqie")),
                                       "riichi": bool(data.get("liqi"))})
                if data.get("liqi"):
                    st.riichi_declared[s] = True
                    st.riichi_sticks += 1
                    st.scores[s] -= 1000
                continue

            if name == "RecordHule":
                _apply_hule(st, data)
                continue

            if name == "RecordAnGangAddGang":
                tiles = _to_ints(data.get("tiles"))
                st.n_kan += 1
                _apply_kan(st, int(seat), tiles, int(data.get("type", 2)))
                continue

            if name == "RecordNewDora":
                if data.get("dora") is not None:
                    st.dora.append(int(data["dora"]))
                continue

            if name == "RecordNoTile":
                _apply_hule(st, data)
                continue

            if name == "RecordLiuJu":
                _apply_hule(st, data)
                continue

            continue
        return self.records


def _is_tsumo(data, seat):
    for hule in data.get("hules") or []:
        if int(hule.get("seat", -1)) == int(seat) and bool(hule.get("zimo")):
            return True
    return len(data.get("hules") or []) == 1 and not _is_ron(data)


def _is_ron(data):
    return any(not bool(h.get("zimo")) for h in data.get("hules") or [])


def _apply_kan(st, seat, tiles, kan_type):
    if kan_type == 3:
        for m in st.melds[seat]:
            if m["type"] == "pon" and tile_kind(m["tiles"][0]) == tile_kind(tiles[0]):
                m["type"] = "kan"
                m["tiles"] = tiles
                break
        for t in tiles:
            if t in st.hands[seat] and tile_kind(t) == tile_kind(tiles[0]):
                st.hands[seat].remove(t)
                break
    else:
        meld = {"type": "kan", "tiles": tiles, "from": seat, "red": any(t in RED5 for t in tiles)}
        st.melds[seat].append(meld)
        for t in tiles:
            if t in st.hands[seat]:
                st.hands[seat].remove(t)


def _apply_hule(st, data):
    if isinstance(data.get("scores"), list) and len(data["scores"]) == 4:
        st.scores = [int(x) for x in data["scores"]]
    if data.get("liqibang") is not None:
        st.riichi_sticks = int(data["liqibang"])
    else:
        st.riichi_sticks = 0


def parse_game(game, game_id=None, max_records=None):
    """Replay a paifu JSON into decision records (schema §2). Returns list."""
    if not isinstance(game, dict):
        raise ValueError("game JSON is not a dict")
    log = game.get("log")
    if log is None:
        raise ValueError("game JSON has no 'log'")
    if isinstance(game.get("head"), dict):
        game_id = game_id or game.get("head", {}).get("uuid")
    return _Replay(game_id).run(log)


def parse_file(path, game_id=None):
    with io.open(path, "r", encoding="utf-8") as f:
        return parse_game(json.load(f), game_id=game_id)


# --------------------------------------------------------------------------
# Batch driver
# --------------------------------------------------------------------------

def run_batch(uuid_file, out_file="C:/agentwork/data/processed/majsoul/records.jsonl",
              stats_file="C:/agentwork/data/processed/majsoul/parse_stats.json",
              token=None, limit=None, delay=0.35, skip_existing=True,
              max_retries=3):
    """Download + parse all uuids; append records to out_file."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    uuids = [u.strip() for u in io.open(uuid_file, encoding="utf-8") if u.strip()]
    if limit:
        uuids = uuids[:limit]
    stats = {"total": len(uuids), "downloaded": 0, "parsed": 0, "failed": 0,
             "records": 0, "failures": []}
    done = set()
    if skip_existing and os.path.exists(out_file):
        for line in io.open(out_file, encoding="utf-8"):
            try:
                gid = json.loads(line).get("game_id")
                if gid:
                    done.add(gid)
            except ValueError:
                pass
    t0 = time.time()
    with io.open(out_file, "a", encoding="utf-8") as out:
        for i, uid in enumerate(uuids):
            if uid in done:
                continue
            try:
                game = None
                for attempt in range(max_retries):
                    try:
                        game = download_paifu_json(uid, token=token)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(1 + 2 * attempt)
                stats["downloaded"] += 1
                recs = parse_game(game, game_id=uid)
                n = 0
                for rec in recs:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                stats["parsed"] += 1
                stats["records"] += n
                if i % 50 == 0:
                    print("[paifu] %d/%d parsed=%d records=%d failed=%d elapsed=%.0fs"
                          % (i + 1, len(uuids), stats["parsed"], stats["records"],
                             stats["failed"], time.time() - t0), flush=True)
            except Exception as e:
                stats["failed"] += 1
                stats["failures"].append({"uuid": uid, "error": str(e)[:300]})
                if len(stats["failures"]) <= 30 or stats["failed"] % 100 == 0:
                    print("[paifu] FAIL %s: %s" % (uid, str(e)[:200]), flush=True)
            time.sleep(delay)
    stats["elapsed_s"] = round(time.time() - t0, 1)
    stats["fail_rate"] = round(stats["failed"] / max(1, stats["total"]), 4)
    with io.open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print("[paifu] DONE %s" % json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fetch", "batch"])
    ap.add_argument("uuid", nargs="?")
    ap.add_argument("--uuid-file", default="C:/agentwork/data/raw/majsoul/uuid_list.txt")
    ap.add_argument("--out", default="C:/agentwork/data/processed/majsoul/records.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=0.35)
    args = ap.parse_args()
    if args.command == "fetch":
        obj = download_paifu_json(args.uuid)
        print(json.dumps(obj, ensure_ascii=False)[:2000])
    else:
        run_batch(args.uuid_file, out_file=args.out, limit=args.limit, delay=args.delay)
