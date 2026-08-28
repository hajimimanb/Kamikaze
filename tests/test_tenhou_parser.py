import glob
import os
import sys

sys.path.insert(0, "C:/agentwork/src")
sys.path.insert(0, "C:/agentwork")

from tenhou.mjlog_parser import parse_game_file, decode_meld


def _load(name):
    return parse_game_file(os.path.join("C:/agentwork/data/tmp", name))


def test_sample_2026_structure():
    g = _load("log0.html")
    assert g["ver"] == "2.3"
    assert g["type"] == 169  # 0xA9: hanchan + no-sanma
    assert g["lobby"] == "0"
    assert len(g["rounds"]) == 11
    for r in g["rounds"]:
        assert r["result"] is not None
    # every round ends in AGARI in this log
    agari_count = sum(1 for r in g["rounds"] if r["result"][0] == "agari")
    assert agari_count == 11


def test_meld_decode_sample():
    # N who="1" m="48746" right after G126: seat1 pons seat3's chun(126)
    m = decode_meld(48746)
    assert m["type"] == "pon"
    assert sorted(m["tiles"]) == [124, 125, 126]
    assert m["called"] == 126
    assert m["from"] == 2  # toimen
    # N who="0" m="43594" after F113: pon of north, called 113
    m = decode_meld(43594)
    assert m["type"] == "pon"
    assert sorted(m["tiles"]) == [112, 113, 115]
    assert m["called"] == 113
    # N who="3" m="44106" after E115: pon of north, called 115
    m = decode_meld(44106)
    assert m["type"] == "pon"
    assert sorted(m["tiles"]) == [112, 113, 115]
    assert m["called"] == 115


def test_agari_meld_codes():
    # AGARI hai=25,29,35,48,49,51,61,62 m="586,50731" -> pon 1m (0,1,3) + pon hatsu
    g = _load("log0.html")
    agari = [ev for r in g["rounds"] for ev in r["events"] if ev[0] == "agari"]
    target = [a for a in agari if a[1]["machi"] == 35]
    assert target
    a = target[0][1]
    kinds = sorted(set(t // 4 for m in a["melds"] for t in m["tiles"]))
    assert kinds == [0, 33]  # 1-man pon and chun (中) pon; yaku index 20 = chun
    assert sorted(a["melds"][0]["tiles"]) == [0, 1, 3]
    assert sorted(a["melds"][1]["tiles"]) == [132, 134, 135]


def test_draw_discard_letters_and_tsumogiri():
    g = _load("log0.html")
    r0 = g["rounds"][0]
    ev = r0["events"]
    # first 8 events: T D U E V F W G
    assert [e[0] for e in ev[:8]] == ["draw", "discard", "draw", "discard",
                                      "draw", "discard", "draw", "discard"]
    assert [e[1] for e in ev[:8]] == [0, 0, 1, 1, 2, 2, 3, 3]
    # compute tsumogiri: draw 131 then discard 131 by seat 0 (tsumogiri);
    # T100 then D118 (hand cut)
    last_draw = {}
    tsumo = []
    for e in ev:
        if e[0] == "draw":
            last_draw[e[1]] = e[2]
        elif e[0] == "discard":
            tsumo.append((e[1], e[2], last_draw.get(e[1]) == e[2]))
    assert (0, 131, True) in tsumo
    assert (0, 118, False) in tsumo
    # 40-50% of discards should be tsumogiri in a normal game
    frac = sum(1 for x in tsumo if x[2]) / len(tsumo)
    assert 0.2 < frac < 0.8, frac


def test_old_fixtures_parse():
    files = sorted(glob.glob("C:/agentwork/data/tmp/old_fixtures/*.xml"))
    assert len(files) >= 13
    ok = 0
    for f in files:
        g = parse_game_file(f)
        assert len(g["rounds"]) >= 1, f
        ok += 1
    assert ok == len(files)


def test_old_fixture_no_lobby():
    # bug1.xml: GO without lobby attr (pre-2012 era)
    g = parse_game_file("C:/agentwork/data/tmp/old_fixtures/bug1.xml")
    assert g["lobby"] is None
    assert g["type"] == 169


def test_reach_events():
    g = _load("log0.html")
    reaches = [e for r in g["rounds"] for e in r["events"] if e[0] == "reach"]
    assert reaches, "log0 should contain reach events"
    assert all(e[2] in (1, 2) for e in reaches)


def test_kan_dora_from_fixture():
    import urllib.request
    url = ("https://raw.githubusercontent.com/LiuZJ2019/"
           "mjlog-parser-and-statistics/master/tests/data/"
           "kan4-case-2025012500gm-00a9-0000-c294f3d0.xml")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=30).read()
    open("C:/agentwork/data/tmp/kan4.xml", "wb").write(data)
    g = parse_game_file("C:/agentwork/data/tmp/kan4.xml")
    dora = [e for r in g["rounds"] for e in r["events"] if e[0] == "dora"]
    kans = [e for r in g["rounds"] for e in r["events"]
            if e[0] == "meld" and e[2] and e[2]["type"] in ("ankan", "minkan", "kakan")]
    assert dora, "kan4 fixture should have DORA tags"
    assert kans, "kan4 fixture should have kan melds"


def test_edge_fixtures_all_parse_and_extract():
    # 8 edge-case fixtures (2025, double-ron / 4-kan / 4-wind / 4-reach /
    # nagashi / 9-9 / yakuman / multiple yakuman)
    import glob as _glob
    from tenhou.extract import extract_game, GameStats
    files = sorted(_glob.glob("C:/agentwork/data/tmp/edge_fixtures/*.xml"))
    if not files:
        import pytest
        pytest.skip("edge fixtures not downloaded")
    assert len(files) == 8
    for f in files:
        g = parse_game_file(f)
        recs = []
        stats = GameStats()
        g["log_id"] = f.split("/")[-1].split(".")[0]
        extract_game(g, recs, stats)
        assert stats.anomalies == 0, f
        assert len(recs) > 0, f


def test_gzip_roundtrip():
    import gzip
    raw = open("C:/agentwork/data/tmp/log0.html", "rb").read()
    gz = gzip.compress(raw)
    open("C:/agentwork/data/tmp/log0.test.xml.gz", "wb").write(gz)
    g = parse_game_file("C:/agentwork/data/tmp/log0.test.xml.gz")
    assert len(g["rounds"]) == 11
    os.remove("C:/agentwork/data/tmp/log0.test.xml.gz")
