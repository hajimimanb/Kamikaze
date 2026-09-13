"""Cross-consistency between records.jsonl labels and the MJAI event stream.

Schema section 7.5: records and mjai events derive from the same parsed game
stream and must agree. Checks (per game):

- every discard/riichi label has a matching dahai event (actor + pai);
- every call label (chi/pon/kan) has a matching chi/pon/daiminkan/ankan/kakan
  event (actor + pai/consumed tiles);
- every ron/tsumo label has a matching hora event (actor);
- riichi declaration counts must be equal (they are always sampled);
- tsumogiri flags in the mjai stream agree with the records' river markers.
"""
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from utils.paths import tmp_dir

from tenhou.mjlog_parser import parse_game_file
from tenhou.mjai_export import build_events, tile_str
from tenhou.extract import extract_game, GameStats


def ts2tile(s):
    # inverse of tile_str (docs/mjai_ref/tile.rs notation):
    # "5mr" -> 16; honors "E","S","W","N","P","F","C" -> kinds 27..33
    if s.endswith("r"):
        return {"m": 16, "p": 52, "s": 88}[s[1]]
    suit = s[-1]
    if suit in "ESWNPFC":
        kind = 27 + "ESWNPFC".index(suit)
    else:
        num = int(s[:-1])
        kind = {"m": 0, "p": 9, "s": 18}[suit] + num - 1
    return kind * 4


def _check_game(path, game_id=None):
    g = parse_game_file(path)
    g["log_id"] = game_id or "x"
    records = []
    stats = GameStats()
    extract_game(g, records, stats)
    events = build_events(g)

    dahai = [e for e in events if e["type"] == "dahai"]
    calls = [e for e in events if e["type"] in ("chi", "pon", "daiminkan",
                                                "ankan", "kakan")]
    horas = [e for e in events if e["type"] == "hora"]
    reaches = [e for e in events if e["type"] == "reach"]
    kan_events = [e for e in events if e["type"] in ("daiminkan", "ankan",
                                                     "kakan")]

    n_riichi_labels = 0
    for r in records:
        lt = r["label"]["type"]
        seat = r["seat"]
        if lt in ("discard", "riichi"):
            t = tile_str(r["label"]["tile"])
            assert any(e["actor"] == seat and e["pai"] == t for e in dahai), (
                "no dahai for %s" % r["label"])
            if lt == "riichi":
                n_riichi_labels += 1
                pos_dahai = [i for i, e in enumerate(events)
                             if e["type"] == "dahai" and e["actor"] == seat
                             and e["pai"] == t]
                assert any(any(e["type"] == "reach" and e["actor"] == seat
                               for e in events[:p]) for p in pos_dahai), (
                    "riichi label without preceding reach event")
        elif lt == "pon":
            tset = set(t // 4 for t in r["label"]["tiles"])
            assert any(e["type"] == "pon" and e["actor"] == seat
                       and set(ts2tile(p) // 4
                               for p in [e["pai"]] + e["consumed"]) == tset
                       for e in calls), r["label"]
        elif lt == "chow":
            tset = set(t // 4 for t in r["label"]["tiles"])
            assert any(e["type"] == "chi" and e["actor"] == seat
                       and set(ts2tile(p) // 4
                               for p in [e["pai"]] + e["consumed"]) == tset
                       for e in calls), r["label"]
        elif lt == "kan":
            tset = set(t // 4 for t in r["label"]["tiles"])
            assert any(e["actor"] == seat
                       and set(ts2tile(p) // 4 for p in e["consumed"]) == tset
                       for e in kan_events), r["label"]
        elif lt == "ron":
            assert any(e["type"] == "hora" and e["actor"] == seat
                       and e["target"] != seat for e in horas), r["label"]
        elif lt == "tsumo":
            assert any(e["type"] == "hora" and e["actor"] == seat
                       and e["target"] == seat for e in horas), r["label"]

    # riichi declarations are always sampled (in rounds <= 7): count only
    # reach events of sampled rounds (west rounds produce no records)
    bakaze_idx0 = {"E": 0, "S": 1, "W": 2, "N": 3}
    n_reach_sampled = 0
    cur_round0 = 0
    for e in events:
        if e["type"] == "start_kyoku":
            cur_round0 = bakaze_idx0[e["bakaze"]] * 4 + e["kyoku"] - 1
        elif e["type"] == "reach" and cur_round0 <= 7:
            n_reach_sampled += 1
    assert n_reach_sampled == n_riichi_labels, (
        "reach events %d != riichi labels %d" % (n_reach_sampled,
                                                 n_riichi_labels))

    # tsumogiri agreement per kyoku: the mjai dahai sequence of a seat in
    # a kyoku segment must match the longest record river of
    # (round, honba, seat). Renchan reuses the round number with honba+1, so
    # honba is part of the key.
    bakaze_idx = {"E": 0, "S": 1, "W": 2, "N": 3}
    round_dahai = {}   # (round, honba, seat) -> [dahai events]
    cur_round = None
    cur_honba = None
    for e in events:
        if e["type"] == "start_kyoku":
            cur_round = bakaze_idx[e["bakaze"]] * 4 + e["kyoku"] - 1
            cur_honba = e["honba"]
        elif e["type"] == "dahai" and cur_round is not None:
            round_dahai.setdefault((cur_round, cur_honba, e["actor"]),
                                   []).append(e)
    rivers = {}
    for r in records:
        key = (r["round"], r["honba"], r["seat"])
        river = r["discards"][r["seat"]]
        if len(river) > len(rivers.get(key, [])):
            rivers[key] = river
    for key, mj in round_dahai.items():
        if key[0] > 7:
            continue  # records never sample west rounds
        rv = rivers.get(key, [])
        # the record river is a snapshot before the seat's own discard, so
        # the round-ending discard may be missing: mj >= rv, and the common
        # prefix must agree pairwise
        assert len(mj) >= len(rv), (
            "round/honba/seat %s: %d dahai vs %d river" % (key, len(mj), len(rv)))
        for e, d in zip(mj, rv):
            assert ts2tile(e["pai"]) // 4 == d["tile"] // 4, (e, d)
            assert e["tsumogiri"] == d["tsumogiri"], (e, d)
    return len(records), stats.as_dict()


def test_label_tsumogiri_field():
    """discard/riichi labels carry an explicit tsumogiri flag (captain
    enhancement); riichi declarations are always recorded as hand-cut."""
    g = parse_game_file(str(tmp_dir() / "log0.html"))
    g["log_id"] = "x"
    records = []
    stats = GameStats()
    extract_game(g, records, stats)
    seen = 0
    for r in records:
        lt = r["label"]["type"]
        if lt in ("discard", "riichi"):
            assert "tsumogiri" in r["label"], r["label"]
            assert isinstance(r["label"]["tsumogiri"], bool)
            if lt == "riichi":
                assert r["label"]["tsumogiri"] is False
            if lt == "discard" and r["last_event"]["type"] == "draw":
                want = r["label"]["tile"] == r["last_event"]["tile"]
                assert r["label"]["tsumogiri"] == want, r["label"]
            seen += 1
    assert seen > 100


def test_sample_consistency():
    n, stats = _check_game(str(tmp_dir() / "log0.html"))
    assert n > 0
    assert stats["anomalies"] == 0


def test_edge_fixtures_consistency():
    import glob
    files = sorted(glob.glob(str(tmp_dir() / "edge_fixtures" / "*.xml")))
    assert files
    for f in files[:4]:
        n, stats = _check_game(f)
        assert stats["anomalies"] == 0, f


def test_old_fixtures_consistency():
    import glob
    files = sorted(glob.glob(str(tmp_dir() / "old_fixtures" / "*.xml")))
    assert files
    for f in files[:6]:
        n, stats = _check_game(f)
        assert stats["anomalies"] == 0, f
