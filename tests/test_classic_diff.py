"""Differential audit against the classic mjlog2mjai converter
(docs/mjai_ref_classic/mjlog2mjai-master/parse.py).

Runs both converters on the same XML and compares the emitted MJAI events on
all COMMON fields (my stream carries a few documented extras on hora /
ryuukyoku / start_game which the classic one lacks or vice versa).
"""
import importlib.util
import json
import sys

sys.path.insert(0, "C:/agentwork/src")
sys.path.insert(0, "C:/agentwork")

from tenhou.mjlog_parser import parse_game_file
from tenhou.mjai_export import build_events

spec = importlib.util.spec_from_file_location(
    "parse_classic",
    "C:/agentwork/docs/mjai_ref_classic/mjlog2mjai-master/parse.py")
classic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classic)


def _classic_events(xml_path):
    root = classic.load_mjlog(xml_path)
    text = classic.parse_mjlog_to_mjai(root)
    # both converters now emit the canonical "ryukyoku" (Hepburn, single
    # u; event.rs snake_case of Ryukyoku) — no normalization needed
    return [json.loads(line) for line in text.splitlines() if line.strip()]


COMMON_FIELDS = {
    "start_game": ("names",),
    "start_kyoku": ("bakaze", "dora_marker", "kyoku", "honba", "kyotaku",
                    "oya", "scores", "tehais"),
    "tsumo": ("actor", "pai"),
    "dahai": ("actor", "pai", "tsumogiri"),
    "chi": ("actor", "target", "pai", "consumed"),
    "pon": ("actor", "target", "pai", "consumed"),
    "daiminkan": ("actor", "target", "pai", "consumed"),
    "kakan": ("actor", "pai", "consumed"),
    "ankan": ("actor", "consumed"),
    "dora": ("dora_marker",),
    "reach": ("actor",),
    "reach_accepted": ("actor",),
    "hora": ("actor", "target", "deltas", "ura_markers"),
    "ryukyoku": ("deltas",),
    "end_kyoku": (),
    "end_game": (),
}


def _strip(e, t):
    out = {"type": t}
    for f in COMMON_FIELDS[t]:
        if f in e:
            out[f] = e[f]
    return out


def _diff_events(mine, theirs):
    """Two-pointer comparison.

    Documented classic-converter bugs we tolerate:
    - it drops the 4th reach_accepted of a round (off-by-one cap), which is
      wrong for a 4-riichi ryukyoku (our export emits every deposit);
    - it does not filter the "0" null placeholder of doraHaiUra, so a winner
      without ura gets the bogus ura marker "1m" (our export emits []).
    """
    j = 0
    n_bad = 0
    for i, a in enumerate(mine):
        if j >= len(theirs):
            n_bad += 1
            print("DIFF event %d: classic stream ended early" % i)
            break
        b = theirs[j]
        if (a["type"] == "reach_accepted"
                and b["type"] != "reach_accepted"):
            # classic off-by-one: a reach_accepted it dropped
            continue
        if a["type"] != b["type"]:
            n_bad += 1
            print("DIFF event %d type: mine=%s classic=%s"
                  % (i, a["type"], b["type"]))
            break
        ta = _strip(a, a["type"])
        tb = _strip(b, b["type"])
        if ta != tb:
            if (a["type"] == "hora" and ta.get("ura_markers") == []
                    and tb.get("ura_markers") == ["1m"]):
                pass
            else:
                n_bad += 1
                print("DIFF event %d %s:" % (i, a["type"]))
                print("  mine   :", json.dumps(ta, ensure_ascii=False))
                print("  classic:", json.dumps(tb, ensure_ascii=False))
                if n_bad > 4:
                    raise AssertionError("too many diffs")
        j += 1
    if j != len(theirs):
        n_bad += 1
        print("DIFF: classic stream has %d trailing events" % (len(theirs) - j))
    assert n_bad == 0, "%d differing events" % n_bad


def test_sample_diff():
    _diff_events(build_events(parse_game_file(
        "C:/agentwork/data/tmp/log0.html")),
        _classic_events("C:/agentwork/data/tmp/log0.html"))


def test_edge_fixtures_diff():
    import glob
    for f in sorted(glob.glob("C:/agentwork/data/tmp/edge_fixtures/*.xml"))[:3]:
        _diff_events(build_events(parse_game_file(f)),
                     _classic_events(f))


def test_old_fixtures_diff():
    import glob
    for f in sorted(glob.glob("C:/agentwork/data/tmp/old_fixtures/*.xml"))[:4]:
        _diff_events(build_events(parse_game_file(f)),
                     _classic_events(f))
