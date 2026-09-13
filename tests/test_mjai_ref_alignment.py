"""Conformance of the exported MJAI stream with the authoritative reference
(docs/mjai_ref/event.rs event structures + tile.rs notation).

Replicates the serde constraints of the official parser:
- tile strings must be in the MJAI_PAI_STRINGS set (honors E S W N P F C,
  reds 5mr/5pr/5sr, "?" unknown);
- actor/target in 0..3, kyoku in 1..4, scores/tehais shapes;
- field names per event type.
"""
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from utils.paths import raw_dir, tmp_dir

from tenhou.mjlog_parser import parse_game_file
from tenhou.mjai_export import build_events, mask_events

ALLOWED_TILES = set(
    ["%d%s" % (n, s) for s in "mps" for n in range(1, 10)]
    + list("ESWNPFC")
    + ["5mr", "5pr", "5sr", "?"]
)

EVENT_FIELDS = {
    # kyoku_first/aka_flag are classic-converter extras (event.rs serde
    # ignores unknown fields; they keep byte-compat with old mjai tooling)
    "start_game": {"type", "names", "kyoku_first", "aka_flag"},
    "start_kyoku": {"type", "bakaze", "dora_marker", "kyoku", "honba",
                    "kyotaku", "oya", "scores", "tehais"},
    "tsumo": {"type", "actor", "pai"},
    "dahai": {"type", "actor", "pai", "tsumogiri"},
    "chi": {"type", "actor", "target", "pai", "consumed"},
    "pon": {"type", "actor", "target", "pai", "consumed"},
    "daiminkan": {"type", "actor", "target", "pai", "consumed"},
    "kakan": {"type", "actor", "pai", "consumed"},
    "ankan": {"type", "actor", "consumed"},
    "dora": {"type", "dora_marker"},
    "reach": {"type", "actor"},
    "reach_accepted": {"type", "actor"},
    "hora": {"type", "actor", "target", "pai", "ura_markers",
             "hora_tehais", "yakus", "fu", "hora_points", "deltas",
             "scores"},  # pai/tehais/yakus/... are documented extensions
    "ryukyoku": {"type", "deltas", "reason", "tehais", "tenpais",
                 "scores"},  # tehais/tenpais/scores are extensions
    "end_kyoku": {"type"},
    "end_game": {"type", "scores"},
}

STR_FIELDS = {"bakaze", "dora_marker", "pai"}
STR_LIST_FIELDS = {"consumed", "ura_markers", "hora_tehais"}
STR_LIST_LIST_FIELDS = {"tehais"}
INT_LIST_FIELDS = {"scores", "deltas"}


def _check_events(events):
    for e in events:
        t = e["type"]
        assert t in EVENT_FIELDS, "unknown event type %s" % t
        fields = set(e) - {"type"}
        known = EVENT_FIELDS[t] - {"type"}
        # fixed required fields per variant
        if t in ("tsumo", "dahai"):
            assert {"actor", "pai"} <= fields, e
        elif t in ("chi", "pon", "daiminkan"):
            assert {"actor", "target", "pai", "consumed"} <= fields, e
        elif t == "kakan":
            assert {"actor", "pai", "consumed"} <= fields, e
        elif t == "ankan":
            assert {"actor", "consumed"} <= fields, e
        elif t == "dora":
            assert {"dora_marker"} <= fields, e
        elif t in ("reach", "reach_accepted"):
            assert {"actor"} <= fields, e
        elif t == "start_kyoku":
            assert {"bakaze", "dora_marker", "kyoku", "honba",
                    "kyotaku", "oya", "scores", "tehais"} <= fields, e
        assert fields <= known, "extra unknown fields %s in %s" % (
            fields - known, t)
        if "actor" in e:
            assert 0 <= e["actor"] <= 3, e
        if "target" in e:
            assert 0 <= e["target"] <= 3, e
        if "kyoku" in e:
            assert 1 <= e["kyoku"] <= 4, e
        if "bakaze" in e:
            assert e["bakaze"] in "ESWN", e
        for f in STR_FIELDS & fields:
            assert e[f] in ALLOWED_TILES, (t, f, e[f])
        for f in STR_LIST_FIELDS & fields:
            assert all(x in ALLOWED_TILES for x in e[f]), (t, f, e[f])
        for f in STR_LIST_LIST_FIELDS & fields:
            for row in e[f]:
                assert all(x in ALLOWED_TILES for x in row), (t, f, row)
        for f in INT_LIST_FIELDS & fields:
            assert len(e[f]) == 4, (t, f, e[f])
        if "tehais" in e:
            assert len(e["tehais"]) == 4
            if t == "start_kyoku":
                assert all(len(row) == 13 for row in e["tehais"])
        if "consumed" in e:
            assert len(e["consumed"]) in (2, 3, 4), e
        if t == "hora" and "yakus" in e:
            for name, han in e["yakus"]:
                assert isinstance(name, str) and isinstance(han, int)


def test_sample_alignment():
    g = parse_game_file(str(tmp_dir() / "log0.html"))
    g["log_id"] = "x"
    _check_events(build_events(g))


def test_edge_fixtures_alignment():
    import glob
    for f in sorted(glob.glob(str(tmp_dir() / "edge_fixtures" / "*.xml")))[:4]:
        g = parse_game_file(f)
        g["log_id"] = "x"
        _check_events(build_events(g))


def test_old_fixtures_alignment():
    import glob
    for f in sorted(glob.glob(str(tmp_dir() / "old_fixtures" / "*.xml")))[:6]:
        g = parse_game_file(f)
        g["log_id"] = "x"
        _check_events(build_events(g))


def test_real_day_sample_alignment():
    import glob as g
    files = sorted(g.glob(str(raw_dir() / "mjlog" / "2026" / "20260824*.xml.gz")))
    assert files
    for f in files[:20]:
        game = parse_game_file(f)
        game["log_id"] = "x"
        _check_events(build_events(game))


def test_masked_version():
    """The masked view (seat 0) hides other players' tehais and tsumo tiles
    and keeps everything else identical; it must still pass the schema check
    ("?" is an allowed tile string)."""
    g = parse_game_file(str(tmp_dir() / "log0.html"))
    g["log_id"] = "x"
    events = build_events(g)
    masked = mask_events(events, 0)
    _check_events(masked)
    # own hand visible, others masked
    kyoku = [e for e in masked if e["type"] == "start_kyoku"][0]
    assert all(x != "?" for x in kyoku["tehais"][0])
    for p in (1, 2, 3):
        assert kyoku["tehais"][p] == ["?"] * 13
    # other actors' tsumo masked, own visible
    for e in masked:
        if e["type"] == "tsumo":
            assert e["pai"] == "?" if e["actor"] != 0 else e["pai"] != "?"
    # public events identical to the omniscient stream
    pub = lambda ev, t: [x for x in ev if x["type"] == t]  # noqa: E731
    for t in ("dahai", "chi", "pon", "daiminkan", "ankan", "kakan", "dora",
              "reach", "reach_accepted", "hora", "ryukyoku", "end_kyoku"):
        assert pub(masked, t) == pub(events, t), t
    # omniscient stream has no "?" (every tile is real in a full log)
    for e in events:
        if "pai" in e:
            assert e["pai"] != "?", e
        if e["type"] == "start_kyoku":
            for row in e["tehais"]:
                assert all(x != "?" for x in row)
