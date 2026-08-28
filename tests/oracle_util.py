# -*- coding: utf-8 -*-
"""Oracle cross-validation helpers.

Replays full hanchan logs produced by mjai's native engine (mjai.app /
mlibriichi, the Rust reference implementation) with the local RiichiGame
engine and compares round-by-round outcomes: winners, score deltas and
draw results. This is the independent oracle channel for the rule engine.
"""
import gzip
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from mahjong.shanten import Shanten  # noqa: E402
from mjai.engine import BaseMjaiLogEngine  # noqa: E402
from mjai.mlibriichi.arena import Match  # noqa: E402

from conftest import make_game  # noqa: E402
from env.riichi_game import RiichiConfig  # noqa: E402
from riichi.tiles import parse_mjai_pai, RED_FIVES, mjai_pai  # noqa: E402

KIND_STR = (["%dm" % (i + 1) for i in range(9)] + ["%dp" % (i + 1) for i in range(9)]
            + ["%ds" % (i + 1) for i in range(9)] + ["E", "S", "W", "N", "P", "F", "C"])
HONOR_TO_MPSZ = {"E": "1z", "S": "2z", "W": "3z", "N": "4z", "P": "5z", "F": "6z", "C": "7z"}
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "oracle")


def _plain(s):
    return s[:-1] if s.endswith("r") else s


def _consumed(kawa, n, akas):
    """Tile strings physically consistent with the single red five per suit."""
    plain = _plain(kawa)
    if kawa.endswith("r") or plain[0] != "5":
        return [plain] * n
    idx = {"m": 0, "p": 1, "s": 2}.get(plain[1])
    if idx is not None and akas[idx]:
        return [plain] * (n - 1) + [plain + "r"]
    return [plain] * n


class GreedyAgent(BaseMjaiLogEngine):
    """Deterministic shanten-greedy agent used to generate oracle games.

    Decisions use ONLY the native engine state (last_cans flags, tehai,
    candidates), so the produced log is a legal mjai.app game by
    construction. It riichis, calls pon/chi, kanks and declares wins.
    """

    def __init__(self, name):
        super().__init__(name)
        self.pending = {}  # idx -> riichi discard tile string
        self.rng = random.Random(int(name) + 7)  # stable across processes

    def react_batch(self, game_states):
        res = []
        for gs in game_states:
            res.append(self._react(gs))
        return res

    def _react(self, gs):
        idx = gs.game_index
        pid = self.player_ids[idx]
        st = gs.state
        cans = st.last_cans
        drawn = st.last_self_tsumo()
        # 1. pending riichi discard (engine re-asks after reach)
        pend = self.pending.get(idx)
        if cans.can_discard and pend is not None:
            self.pending[idx] = None
            return json.dumps({"type": "dahai", "actor": pid, "pai": pend,
                               "tsumogiri": pend == drawn})
        # 2. tsumo win
        if cans.can_tsumo_agari:
            return json.dumps({"type": "hora", "actor": pid, "target": pid, "pai": drawn})
        # 3. riichi when the engine allows it; discard the tenpai-keeping tile
        if cans.can_discard and cans.can_riichi:
            best = self._best_discard(st.tehai, drawn, st.akas_in_hand)
            self.pending[idx] = best
            return json.dumps({"type": "reach", "actor": pid})
        # 4. own-turn actions
        if cans.can_discard:
            if st.shanten >= 1 and cans.can_ankan:
                k = st.ankan_candidates()[0]
                s = k if isinstance(k, str) else KIND_STR[k]
                return json.dumps({"type": "ankan", "actor": pid,
                                   "consumed": _consumed(s, 4, st.akas_in_hand)})
            if st.shanten >= 1 and cans.can_kakan:
                k = st.kakan_candidates()[0]
                s = k if isinstance(k, str) else KIND_STR[k]
                s2 = s + "r" if s[0] == "5" and st.akas_in_hand[{"m": 0, "p": 1, "s": 2}[s[1]]] else s
                return json.dumps({"type": "kakan", "actor": pid, "pai": s2,
                                   "consumed": [s2, s2, s2]})
            if getattr(st, "self_riichi_accepted", False):
                return json.dumps({"type": "dahai", "actor": pid, "pai": drawn,
                                   "tsumogiri": True})
            best = self._best_discard(st.tehai, drawn, st.akas_in_hand)
            return json.dumps({"type": "dahai", "actor": pid, "pai": best,
                               "tsumogiri": best == drawn})
        # 5. reactions to other players
        events = json.loads(gs.events_json)
        last = events[-1] if events else {}
        if last.get("type") == "dahai":
            kawa = st.last_kawa_tile()
            if cans.can_ron_agari:
                return json.dumps({"type": "hora", "actor": pid, "target": last["actor"],
                                   "pai": kawa})
            if cans.can_daiminkan:
                return json.dumps({"type": "daiminkan", "actor": pid, "target": last["actor"],
                                   "pai": kawa,
                                   "consumed": _consumed(kawa, 3, st.akas_in_hand)})
            if st.shanten >= 1 and cans.can_pon:
                return json.dumps({"type": "pon", "actor": pid, "target": last["actor"],
                                   "pai": kawa,
                                   "consumed": _consumed(kawa, 2, st.akas_in_hand)})
            if st.shanten >= 1 and (cans.can_chi_low or cans.can_chi_mid or cans.can_chi_high):
                n, s2 = int(kawa[0]), kawa[1]
                if cans.can_chi_low:
                    a, b = "%d%s" % (n + 1, s2), "%d%s" % (n + 2, s2)
                elif cans.can_chi_mid:
                    a, b = "%d%s" % (n - 1, s2), "%d%s" % (n + 1, s2)
                else:
                    a, b = "%d%s" % (n - 2, s2), "%d%s" % (n - 1, s2)
                # physically consistent: when the only copy of a
                # consumed five is the red one, name it as such
                a = self._chi_consumed(a, st)
                b = self._chi_consumed(b, st)
                return json.dumps({"type": "chi", "actor": pid, "target": last["actor"],
                                   "pai": kawa, "consumed": [a, b]})
                n, s2 = int(kawa[0]), kawa[1]
                if cans.can_chi_low:
                    a, b = "%d%s" % (n + 1, s2), "%d%s" % (n + 2, s2)
                elif cans.can_chi_mid:
                    a, b = "%d%s" % (n - 1, s2), "%d%s" % (n + 1, s2)
                else:
                    a, b = "%d%s" % (n - 2, s2), "%d%s" % (n - 1, s2)
                return json.dumps({"type": "chi", "actor": pid, "target": last["actor"],
                                   "pai": kawa, "consumed": [a, b]})
        return json.dumps({"type": "none"})

    def _chi_consumed(self, s, st):
        if s[0] == "5":
            idx = {"m": 0, "p": 1, "s": 2}[s[1]]
            kind = 4 + 9 * idx
            if st.akas_in_hand[idx] and st.tehai[kind] == 1:
                return s + "r"
        return s

    def _best_discard(self, tehai, drawn, akas):
        best_kind, best_s = None, 99
        kinds = [k for k in range(34) if tehai[k] > 0]
        self.rng.shuffle(kinds)
        for k in kinds:
            c = list(tehai)
            c[k] -= 1
            s = Shanten.calculate_shanten(c)
            if s < best_s:
                best_s, best_kind = s, k
        if best_kind in (4, 13, 22):
            suit = "m" if best_kind == 4 else ("p" if best_kind == 13 else "s")
            red = "5" + suit + "r"
            if drawn == red:
                return red
            if akas[{"m": 0, "p": 1, "s": 2}[suit]] and tehai[best_kind] == 1:
                return red
        return KIND_STR[best_kind]


def run_oracle_game(seed):
    """Play a full hanchan with 4 greedy agents against mjai.app; return the
    complete event log (list of dicts)."""
    with tempfile.TemporaryDirectory() as d:
        env = Match(log_dir=d)
        env.py_match(*[GreedyAgent(str(i)) for i in range(4)], seed_start=seed)
        events = []
        for f in sorted(os.listdir(d)):
            with gzip.open(os.path.join(d, f), "rt", encoding="utf-8") as fh:
                content = fh.read()
            events.extend(json.loads(x) for x in content.strip().split(chr(10)))
    return events


_LOGS = {}


def get_oracle_log(seed, live=False):
    """Oracle log with a checked-in gzip cache (tests/fixtures/oracle).

    mjai.app is deterministic for fixed seeds and our agents are
    deterministic, so cached logs are safe; pass live=True (or set
    ORACLE_LIVE=1) to regenerate against the real arena."""
    key = (int(seed[0]), int(seed[1]))
    if key in _LOGS:
        return _LOGS[key]
    path = os.path.join(FIXTURE_DIR, "oracle_%d_%d.json.gz" % key)
    if not live and not os.environ.get("ORACLE_LIVE") and os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            events = json.load(fh)
    else:
        events = run_oracle_game(key)
        if not os.environ.get("ORACLE_LIVE"):
            os.makedirs(FIXTURE_DIR, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(events, fh)
    _LOGS[key] = events
    return events


def split_rounds(events):
    rounds, cur = [], None
    for e in events:
        t = e["type"]
        if t == "start_kyoku":
            cur = [e]
        elif cur is not None:
            cur.append(e)
            if t == "end_kyoku":
                rounds.append(cur)
                cur = None
    return rounds


def mjai_to_mpsz(s):
    if s in HONOR_TO_MPSZ:
        return HONOR_TO_MPSZ[s]
    if s.endswith("r"):
        return "0" + s[1]
    return s


def _oracle_outcome(revents):
    horas = [e for e in revents if e["type"] == "hora"]
    ryus = [e for e in revents if e["type"] == "ryukyoku"]
    if horas:
        deltas = [0, 0, 0, 0]
        for h in horas:
            for i, d in enumerate(h["deltas"]):
                deltas[i] += d
        return {"kind": "hora", "actors": [h["actor"] for h in horas],
                "targets": [h["target"] for h in horas], "deltas": deltas}
    if ryus:
        ry = ryus[-1]
        return {"kind": "ryukyoku", "deltas": ry.get("deltas"),
                "reason": ry.get("reason")}
    return {"kind": "unknown"}


def analyze_round(revents):
    sk = revents[0]
    assert sk["type"] == "start_kyoku"
    hands = [[mjai_to_mpsz(t) for t in tehais] for tehais in sk["tehais"]]
    dealer = sk["oya"]
    bakaze = sk["bakaze"]
    round_idx = (sk["kyoku"] - 1) + (4 if bakaze == "S" else 0) + (8 if bakaze == "W" else 0)
    dora = [mjai_to_mpsz(sk["dora_marker"])]
    draws, rinshan, ura, actions = [], [], [], []
    last_kan = None
    for e in revents[1:]:
        t = e["type"]
        if t == "tsumo":
            if last_kan is not None and last_kan["actor"] == e["actor"]:
                rinshan.append(mjai_to_mpsz(e["pai"]))
            else:
                draws.append(mjai_to_mpsz(e["pai"]))
        elif t in ("dahai", "reach", "reach_accepted", "chi", "pon", "daiminkan", "kakan", "ankan"):
            actions.append(e)
        elif t == "hora":
            actions.append(e)
            # ura markers: the classic parser propagates the same list to
            # every hora event of a double ron, so take the LAST event's
            # full marker list (duplicates within it are distinct wall
            # indicators and must be kept for the ura count).
            mk = e.get("ura_markers") or []
            if mk:
                ura[:] = [mjai_to_mpsz(u) for u in mk]
        elif t == "dora":
            dora.append(mjai_to_mpsz(e["dora_marker"]))
        if t in ("daiminkan", "kakan", "ankan"):
            last_kan = e
        elif t != "dora":
            last_kan = None
    return {"hands": hands, "dealer": dealer, "round_idx": round_idx,
            "draws": draws, "rinshan": rinshan, "dora": dora, "ura": ura,
            "actions": actions, "sk": sk, "outcome": _oracle_outcome(revents)}


ORACLE_CONFIG = RiichiConfig(
    double_ron=True,           # mjai.app handles multi-ron
    atamahane=False,
    kuikae=True,               # mjai.app does not restrict kuikae
    chankan_kokushi=False,     # mjai.app: ankan can never be robbed
    kiriage_mangan=False,      # mjai.app does not round up 4han30fu/3han60fu
    aka=True,
    kuitan=True,
    agari_yame=True,
    west_extension=True,       # mjai.app west rounds (hard cap West-4)
    nagashi_mangan=True,
    kyushu_kyuhai=True,
    four_winds_ryuukyoku=True,
    four_kans_ryuukyoku=True,
)


def _resolve_held(g, seat, consumed_strings):
    """Map oracle consumed tile strings to the engine's physical hand ids."""
    ids, used = [], set()
    hand = sorted(g.hands[seat])
    for s in consumed_strings:
        want = parse_mjai_pai(s)
        kind = want // 4
        candidates = [t for t in hand if t not in used and t // 4 == kind]
        if not candidates:
            raise ValueError("no held tile for consumed %s (seat %d)" % (s, seat))
        exact = [t for t in candidates if t == want]
        if exact:
            t = exact[0]
        else:
            # keep the red/plain identity of a five when the exact copy is
            # absent (a plain "5m" must not consume the red five 16 and
            # vice versa)
            is_red = want in RED_FIVES
            same = [t for t in candidates if (t in RED_FIVES) == is_red]
            t = same[0] if same else candidates[0]
        ids.append(t)
        used.add(t)
    return ids


def replay_round(revents, config=None):
    """Replay one oracle round with the local engine and compare outcomes.

    Returns a dict: ok (bool), kind, details, ours/oracle deltas, error.
    """
    info = analyze_round(revents)
    sk = info["sk"]
    g = make_game(
        hands=["".join(h) for h in info["hands"]],
        draws=info["draws"],
        dora=info["dora"],
        dealer=info["dealer"],
        ura=info["ura"],
        rinshan=info["rinshan"],
        scores=sk["scores"],
        honba=sk["honba"],
        kyoutaku=sk["kyotaku"],
        round_idx=info["round_idx"],
        config=config or ORACLE_CONFIG,
        game_seed=1,
    )
    start_scores = list(sk["scores"])
    # merge reach+dahai pairs into single riichi actions
    acts = info["actions"]
    # A REACH step=2 (mjai "reach_accepted") marks the riichi as ESTABLISHED
    # (deposit charged). A step=1-only reach is a VOIDED declaration (the
    # declaration tile was ron'd/claimed or the round aborted): Tenhou charges
    # no deposit, so feed a plain discard instead of a riichi action.
    accepted = {a["actor"] for a in acts if a["type"] == "reach_accepted"}
    oc0 = info["outcome"]
    if oc0["kind"] == "ryukyoku" and oc0.get("reason") == "reach4":
        # 四家立直: the 4th riichi declaration aborts the round immediately,
        # so Tenhou's log carries no step=2 for it — but the deposit IS
        # charged. Treat the last declaration as an established riichi.
        last_reach = [a for a in acts if a["type"] == "reach"]
        if last_reach:
            accepted.add(last_reach[-1]["actor"])
    feed = []
    i = 0
    while i < len(acts):
        e = acts[i]
        if e["type"] == "reach_accepted":
            # informational only; the deposit is charged by the reach action
            i += 1
            continue
        if e["type"] == "reach":
            nxt = acts[i + 1] if i + 1 < len(acts) else None
            assert nxt and nxt["type"] == "dahai" and nxt["actor"] == e["actor"], "reach not followed by own dahai"
            if e["actor"] in accepted:
                feed.append({"kind": "reach", "actor": e["actor"], "pai": nxt["pai"],
                             "tsumogiri": nxt.get("tsumogiri", False)})
            else:
                feed.append({"kind": "dahai", "actor": e["actor"], "event": nxt})
            i += 2
            continue
        feed.append({"kind": e["type"], "actor": e["actor"], "event": e})
        i += 1

    # 九种九牌 (abortive ryuukyoku, mjlog reason "yao9"): the declarer's
    # first draw ends the round before any discard. Inject a kyushu action
    # right before the declarer's first dahai so the engine ends the round.
    oc = info["outcome"]
    ron3_tile = None
    if oc["kind"] == "ryukyoku" and oc.get("reason") == "ron3":
        for e in reversed(revents):
            if e["type"] == "dahai":
                ron3_tile = e["pai"]
                break

    if oc["kind"] == "ryukyoku" and oc.get("reason") == "yao9":
        declarer = None
        for ev in revents:
            if ev["type"] == "tsumo":
                declarer = ev["actor"]
            elif ev["type"] == "ryukyoku":
                break
        inserted = False
        for j, it in enumerate(feed):
            if it["kind"] == "dahai" and it["actor"] == declarer:
                feed.insert(j, {"kind": "kyushu", "actor": declarer, "event": None})
                inserted = True
                break
        if not inserted:
            # the declarer never discards: their first draw ends the round
            feed.append({"kind": "kyushu", "actor": declarer, "event": None})

    result, err = None, None
    try:
        pi = 0
        steps = 0
        while g.phase not in ("round_end", "game_end"):
            sr = None
            if os.environ.get("ORACLE_TRACE"):
                print("T", steps, g.phase, "turn", g.turn, "wall", g.wall_left,
                      "queue", [(q["seat"], q["kind"]) for q in (g.claim_queue or [])],
                      "pi", pi)
            steps += 1
            item = feed[pi] if pi < len(feed) else None
            seat = g.turn
            if g.phase == "draw":
                if (item is None
                        or item["kind"] not in ("reach", "dahai", "hora", "kyushu", "kakan", "ankan")
                        or item["actor"] != seat):
                    raise ValueError(
                        "desync/exhausted in draw phase: engine turn=%d, oracle next=%s"
                        % (seat, item["kind"] if item else None))
                if item["kind"] == "kyushu":
                    sr = g.step({"type": "kyushu", "actor": seat})
                elif item["kind"] == "reach":
                    sr = g.step({"type": "reach", "actor": seat, "pai": item["pai"],
                                "tsumogiri": item["tsumogiri"]})
                elif item["kind"] == "dahai":
                    sr = g.step({"type": "dahai", "actor": seat, "pai": item["event"]["pai"],
                                "tsumogiri": item["event"].get("tsumogiri", False)})
                elif item["kind"] in ("kakan", "ankan"):
                    ev = item["event"]
                    if item["kind"] == "kakan":
                        kind = parse_mjai_pai(ev["pai"]) // 4
                        meld = next(m for m in g.melds[seat]
                                    if m["type"] == "pon" and m["tiles"][0] // 4 == kind)
                        added = _resolve_held(g, seat, [ev["pai"]])
                        sr = g.step({"type": "kakan",
                                    "tiles": sorted(list(meld["tiles"]) + added)})
                    else:
                        hand_ids = _resolve_held(g, seat, ev["consumed"])
                        sr = g.step({"type": "ankan", "tiles": sorted(hand_ids)})
                else:
                    sr = g.step({"type": "hora", "actor": seat, "target": seat,
                                "pai": mjai_pai(g.drawn_tile)})
                pi += 1
            elif g.phase in ("claim", "chankan"):
                # Ron3 (三家和) abortive draw: the log has NO hora events
                # (only RYUUKYOKU type=ron3), so the feed contains no ron
                # actions. Feed a ron for each ron queue head until the
                # engine's ron3 ryuukyoku fires (round_end check needed:
                # the engine advances into the next round immediately).
                if (oc["kind"] == "ryukyoku" and oc.get("reason") == "ron3"
                        and g.phase == "claim"):
                    # 仅在"该 claim 正是天凤 ron3 触发点"时喂 ron：
                    # 天凤日志中 ron3 流局紧跟在触发打牌之后（下一 feed 事件
                    # 即 ryukyoku）；若天凤在更早的 claim 见逃（日志继续），
                    # 引擎必须 pass 跟随日志（oracle-verified: 3c9627b9 E4,
                    # 座位1 见逃 9s -> 8p 三家和）。
                    from riichi.tiles import mjai_pai as _mp
                    if (ron3_tile is not None and _mp(g._claim_tile) == ron3_tile
                            and any(q["kind"] == "ron" for q in (g.claim_queue or []))):
                        queue = g.claim_queue
                        while any(q["kind"] == "ron" for q in (queue or [])):
                            ron_seats = [q["seat"] for q in (queue or [])
                                         if q["kind"] == "ron"]
                            sr = g.step({"type": "ron", "actor": ron_seats[0]})
                            steps += 1
                            if sr is not None and sr["round_end"] is not None:
                                result = dict(sr["round_end"])
                                break
                            if g.phase == "draw" and g.round_result is not None:
                                # ron3 流局后引擎立即开新局（phase 已变 draw）
                                result = dict(g.round_result)
                                break
                            queue = g.claim_queue
                        if result is not None:
                            break
                        continue
                # A tsumo hora (target == actor) can never fire in a claim
                # window: the oracle player passed the claim and the hora
                # belongs to their LATER draw. Feed a pass instead.
                is_tsumo_hora = (item is not None and item["kind"] == "hora"
                                 and item["event"].get("target") == item["event"].get("actor"))
                queue = g.claim_queue if g.phase == "claim" else g.chankan_queue
                head = queue[0] if queue else None
                # chankan queue items carry only {"seat": ...}; their
                # implicit head kind is "ron"
                head_kind = head.get("kind") if head else None
                if head_kind is None and g.phase == "chankan":
                    head_kind = "ron"
                if (item is not None and not is_tsumo_hora
                        and item["kind"] in ("chi", "pon", "daiminkan", "hora")
                        and item["actor"] == seat):
                    want = {"chi": "chow", "hora": "ron",
                            "pon": "pon", "daiminkan": "daiminkan"}[item["kind"]]
                    if head is not None and head_kind != want:
                        # The oracle declined the current claim head to take a
                        # LATER call of the same seat (e.g. passes a ron to
                        # pon the same tile): skip the head with a pass.
                        sr = g.step({"type": "pass"})
                        continue
                    ev = item["event"]
                    if ev["type"] == "hora":
                        sr = g.step({"type": "hora", "actor": seat, "target": ev["target"]})
                    elif ev["type"] == "chi":
                        hand_ids = _resolve_held(g, seat, ev["consumed"])
                        sr = g.step({"type": "chi", "tiles": sorted([g._claim_tile] + hand_ids)})
                    elif ev["type"] == "pon":
                        hand_ids = _resolve_held(g, seat, ev["consumed"])
                        sr = g.step({"type": "pon", "tiles": sorted([g._claim_tile] + hand_ids)})
                    else:  # daiminkan
                        hand_ids = _resolve_held(g, seat, ev["consumed"])
                        sr = g.step({"type": "daiminkan",
                                "tiles": sorted([g._claim_tile] + hand_ids)})
                    pi += 1
                else:
                    sr = g.step({"type": "pass"})
            else:
                raise ValueError("unexpected engine phase %s" % g.phase)
            if sr is not None and sr["round_end"] is not None:
                result = dict(sr["round_end"])
                break
            if sr is not None and sr["game_end"] is not None:
                raise ValueError("engine ended the game instead of the round")
        if result is None and g.phase == "round_end":
            result = dict(g.round_result)
        if result is None:
            raise ValueError("round never ended (phase=%s)" % g.phase)
    except ValueError as exc:
        err = str(exc)

    oc = info["outcome"]
    our_deltas = [g.scores[s] - start_scores[s] for s in range(4)]
    # Tenhou mjlog sc-gain conventions differ by round kind (verified against
    # raw logs): AGARI deltas are settlement-only (payments+honba+sticks, the
    # -1000 riichi deposits are NOT included), while RYUUKYOKU deltas are the
    # full net change (deposit -1000 AND stick return +1000 both included, so
    # they cancel for riichi players). Our engine deltas are always the full
    # net change, so: hora -> subtract deposits; ryuukyoku -> compare direct.
    n_reach = [0, 0, 0, 0]
    for e in revents:
        if e["type"] == "reach_accepted":
            n_reach[e["actor"]] += 1
    if oc["kind"] == "hora":
        expected_ours = [oc["deltas"][s] - 1000 * n_reach[s] for s in range(4)]
    else:
        expected_ours = list(oc["deltas"])
    out = {"kind": oc["kind"], "oracle_deltas": oc["deltas"],
           "our_deltas": our_deltas, "our_result": result,
           "our_winners": [w["seat"] for w in result["winners"]] if result and "winners" in result else None,
           "our_loser": result.get("loser") if result else None,
           "our_han_fu": [(w["seat"], w["han"], w["fu"]) for w in result["winners"]]
           if result and "winners" in result else None,
           "kyoku": sk["kyoku"], "bakaze": sk["bakaze"],
           "honba": sk["honba"], "kyotaku": sk["kyotaku"]}
    if err is not None:
        out["ok"] = False
        out["error"] = err
        return out
    if oc["kind"] == "hora":
        if result["type"] not in ("ron", "tsumo"):
            out["ok"] = False
            out["error"] = "engine result %s vs oracle hora" % result["type"]
            return out
        if our_deltas != expected_ours:
            out["ok"] = False
            out["error"] = "score delta mismatch (ours=%s oracle_settlement=%s)" % (our_deltas, oc["deltas"])
            return out
        if oc["actors"] != out["our_winners"]:
            out["ok"] = False
            out["error"] = "winner mismatch (ours=%s oracle=%s)" % (out["our_winners"], oc["actors"])
            return out
        if result["type"] == "ron" and result["loser"] != oc["targets"][0]:
            out["ok"] = False
            out["error"] = "loser mismatch (ours=%s oracle=%s)" % (result["loser"], oc["targets"][0])
            return out
    elif oc["kind"] == "ryukyoku":
        if not result["type"].startswith("ryuukyoku:"):
            out["ok"] = False
            out["error"] = "engine result %s vs oracle ryuukyoku" % result["type"]
            return out
        if our_deltas != expected_ours:
            out["ok"] = False
            out["error"] = "ryuukyoku delta mismatch (ours=%s oracle_settlement=%s)" % (our_deltas, oc["deltas"])
            return out
    else:
        out["ok"] = False
        out["error"] = "oracle outcome unknown"
        return out
    out["ok"] = True
    return out
