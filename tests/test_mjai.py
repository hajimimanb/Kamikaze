"""MJAI-protocol alignment: action aliases, pai strings, event names (§supplement)."""
import pytest

from riichi.tiles import mjai_pai, parse_mjai_pai

from conftest import make_game


D1 = "99p99s112233z4z5z6z"
D2 = "77p88p99s1122z33z4z"
D3B = "99p556677z9m5z7z7z6z"


class TestPaiStrings:
    def test_suits(self):
        assert mjai_pai(0) == "1m"
        assert mjai_pai(36) == "1p"
        assert mjai_pai(72) == "1s"
        assert mjai_pai(8) == "3m"

    def test_honors(self):
        assert mjai_pai(108) == "E"
        assert mjai_pai(112) == "S"
        assert mjai_pai(116) == "W"
        assert mjai_pai(120) == "N"
        assert mjai_pai(124) == "P"
        assert mjai_pai(128) == "F"
        assert mjai_pai(132) == "C"

    def test_red_fives(self):
        assert mjai_pai(16) == "5mr"
        assert mjai_pai(52) == "5pr"
        assert mjai_pai(88) == "5sr"
        assert mjai_pai(17) == "5m"

    def test_parse_roundtrip(self):
        for t in (0, 4, 8, 12, 16, 36, 52, 72, 88, 108, 112, 116, 120, 124, 128, 132):
            assert parse_mjai_pai(mjai_pai(t)) == t

    def test_parse_red(self):
        assert parse_mjai_pai("5mr") == 16
        assert parse_mjai_pai("5pr") == 52
        assert parse_mjai_pai("5sr") == 88

    def test_parse_honors(self):
        assert parse_mjai_pai("E") == 108
        assert parse_mjai_pai("C") == 132


class TestActionAliases:
    def test_dahai_alias(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3B],
                      draws=["1p"], dora=["4z"], first_draw=False)
        drawn = g.drawn_tile
        g.step({"type": "dahai", "pai": mjai_pai(drawn), "actor": 0, "tsumogiri": True})
        assert g.discards[0][-1]["tile"] == drawn
        assert g.discards[0][-1]["tsumogiri"] is True

    def test_reach_alias(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3B],
                      draws=["1p"], dora=["4z"], first_draw=False)
        drawn = g.drawn_tile
        g.step({"type": "reach", "pai": mjai_pai(drawn), "actor": 0})
        assert g.riichi_declared[0] is True
        assert g.scores[0] == 24000

    def test_pon_alias_with_pai(self):
        hands = ["123m123p123s456m9m", "2468m99p1357s112z",
                 "99m2467m234p67s9p9p", "111333555m22p44s"]
        g = make_game(hands, draws=["9m", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "dahai", "pai": mjai_pai(g.drawn_tile)})  # 9m
        g.step({"type": "pon", "pai": "9m", "actor": 2, "target": 0})
        assert g.melds[2][0]["type"] == "pon"

    def test_chi_alias_with_pai(self):
        hands = ["123m123p123s456m9m", "12m9p9s1122z5z5z6z6z4s",
                 "5566m9p9s1122z9s5z9m", "333555777p22s88s"]
        g = make_game(hands, draws=["3m", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "dahai", "pai": mjai_pai(g.drawn_tile)})  # 3m
        g.step({"type": "chi", "pai": "3m", "actor": 1, "target": 0})
        assert g.melds[1][0]["type"] == "chi"

    def test_daiminkan_alias_with_pai(self):
        hands = ["123m123p123s456m7s", "9m9m9m99p99s1222z3z4z",
                 "2266m6688p99s112z", "111333555p22s44m"]
        g = make_game(hands, draws=["9m", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "dahai", "pai": mjai_pai(g.drawn_tile)})  # 9m
        g.step({"type": "daiminkan", "pai": "9m", "actor": 1, "target": 0})
        assert g.melds[1][0]["type"] == "kan"
        assert g.melds[1][0]["open"] is True

    def test_ankan_alias_with_pai(self):
        hands = ["1111m234567p99s4z", "2468m99p1357s112z",
                 "2244m6688p99s112z", "333555777m22p44s"]
        g = make_game(hands, draws=["3z"], dora=["7z"], first_draw=False)
        g.step({"type": "ankan", "pai": "1m", "actor": 0})
        assert g.melds[0][0]["type"] == "kan"
        assert g.melds[0][0]["open"] is False

    def test_kakan_alias_with_pai(self):
        melds = [[{"type": "pon", "tiles": ["5m", "5m", "5m"], "from": 1,
                   "open": True, "called": "5m"}], [], [], []]
        hands = ["234m567p99s12z", "99p99s1357s3z4z5z6z6z",
                 "2288m6688p22s2z2z7z", "333555777p22s88s"]
        g = make_game(hands, draws=["5m"], dora=["7z"], first_draw=False,
                      melds=melds)
        g.step({"type": "kakan", "pai": "5m", "actor": 0})
        assert g.melds[0][0]["type"] == "shouminkan"

    def test_hora_alias(self):
        hands = ["123m123p123s456m1z", "11z99p99s567m567m9m",
                 "2288m6688p99s22z4z", "111333444m22p55s"]
        g = make_game(hands, draws=["3z", "1z"], dora=["7z"], first_draw=False)
        g.step({"type": "dahai", "pai": mjai_pai(g.drawn_tile)})
        g.step({"type": "dahai", "pai": mjai_pai(g.drawn_tile)})
        r = g.step({"type": "hora", "actor": 0, "target": 1, "pai": "E"})
        assert r["round_end"]["type"] == "ron"

    def test_ryuukyoku_alias_for_kyushu(self):
        hands = ["19m19p19s1234567z", "99p99s234m567p33s4s",
                 "77p88p234s567m99m5s", "88s234p567m9m44m8p5s"]
        g = make_game(hands, draws=["7z"], dora=["4z"])
        g.step({"type": "ryuukyoku", "actor": 0})
        assert g.round_result["type"] == "ryuukyoku:kyushu"

    def test_none_alias_for_pass(self):
        hands = ["123m123p123s456m9m", "2468m99p1357s112z",
                 "99m2467m234p67s9p9p", "111333555m22p44s"]
        g = make_game(hands, draws=["9m", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "dahai", "pai": mjai_pai(g.drawn_tile)})
        g.step({"type": "none"})  # decline the pon
        assert g.phase == "draw"


class TestMjaiEvents:
    def test_start_kyoku_event_fields(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3B],
                      draws=["1p"], dora=["4z"], first_draw=False)
        e = g.events[0]
        assert e["type"] == "start_kyoku"
        assert e["bakaze"] == "E"
        assert e["kyoku"] == 1
        assert e["oya"] == 0
        assert e["dora_marker"] == mjai_pai(g.dora_indicators[0])
        assert len(e["tehais"]) == 4
        assert all(len(h) == 13 for h in e["tehais"])

    def test_tsumo_and_dahai_event_fields(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3B],
                      draws=["1p"], dora=["4z"], first_draw=False)
        t = g.events[1]
        assert t["type"] == "tsumo" and t["actor"] == 0 and t["pai"] == "1p"
        g.step({"type": "dahai", "pai": mjai_pai(g.drawn_tile), "tsumogiri": True})
        d = [e for e in g.events if e["type"] == "dahai"][0]
        assert d["actor"] == 0 and d["tsumogiri"] is True and d["pai"] == "1p"

    def test_reach_event(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3B],
                      draws=["1p"], dora=["4z"], first_draw=False)
        g.step({"type": "reach", "pai": mjai_pai(g.drawn_tile)})
        assert any(e["type"] == "reach" and e["actor"] == 0 for e in g.events)

    def test_hora_event(self):
        hands = ["123m123p123s456m1z", "11z99p99s567m567m9m",
                 "2288m6688p99s22z4z", "111333444m22p55s"]
        g = make_game(hands, draws=["3z", "1z"], dora=["7z"], first_draw=False)
        g.step({"type": "dahai", "pai": mjai_pai(g.drawn_tile)})
        g.step({"type": "dahai", "pai": mjai_pai(g.drawn_tile)})
        g.step({"type": "hora", "actor": 0, "target": 1, "pai": "E"})
        e = [x for x in g.events if x["type"] == "hora"][-1]
        assert e["actor"] == 0 and e["target"] == 1 and e["pai"] == "E"
        assert any(x["type"] == "end_kyoku" for x in g.events)

    def test_ryuukyoku_event_fields(self):
        hands = ["19m19p19s1234567z", "99p99s234m567p33s4s",
                 "77p88p234s567m99m5s", "88s234p567m9m44m8p5s"]
        g = make_game(hands, draws=["7z"], dora=["4z"])
        g.step({"type": "ryuukyoku", "actor": 0})
        e = [x for x in g.events if x["type"] == "ryuukyoku"][-1]
        assert e["reason"] == "kyushu"

    def test_kan_event_kinds(self):
        hands = ["1111m234567p99s4z", "2468m99p1357s112z",
                 "2244m6688p99s112z", "333555777m22p44s"]
        g = make_game(hands, draws=["3z"], dora=["7z"], first_draw=False)
        g.step({"type": "ankan", "pai": "1m", "actor": 0})
        assert any(e["type"] == "ankan" and e["consumed"] == ["1m", "1m", "1m", "1m"]
                   for e in g.events)
        assert any(e["type"] == "dora" and "dora_marker" in e for e in g.events)

    def test_ambiguous_chi_requires_tiles(self):
        # seat1 could chi 3m as 123 or 234 -> pai alone must raise
        hands = ["123m123p123s456m9m", "112244m9p9s1122z5z",
                 "5566m9p9s1122z9s5z9m", "333555777p22s88s"]
        g = make_game(hands, draws=["3m", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "dahai", "pai": mjai_pai(g.drawn_tile)})
        with pytest.raises(ValueError):
            g.step({"type": "chi", "pai": "3m", "actor": 1, "target": 0})
