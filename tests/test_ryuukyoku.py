"""Ryuukyoku: kyushu, four winds, four kans, exhaustive + settlement, nagashi."""
import pytest

from env.riichi_game import RiichiConfig

from conftest import make_game


# wind-free dummy hands (for kokushi/kyushu/wind scenarios)
D1 = "99p99s234m567p33s4s"
D2 = "77p88p234s567m99m5s"
D3B = "88s234p567m9m44m8p5s"


class TestKyushu:
    def test_kyushu_offered_on_first_draw(self):
        hands = ["19m19p19s1234567z", D1, D2, D3B]
        g = make_game(hands, draws=["7z"], dora=["4z"])  # first_draw=True
        assert g.legal_actions(0)["kyushu"] is True

    def test_kyushu_declared(self):
        hands = ["19m19p19s1234567z", D1, D2, D3B]
        g = make_game(hands, draws=["7z"], dora=["4z"])
        g.step({"type": "kyushu"})
        assert g.round_result["type"] == "ryuukyoku:kyushu"
        assert g.honba == 1  # renchan
        assert g.round_idx == 0 and g.dealer == 0

    def test_kyushu_not_offered_with_8_terminals(self):
        hands = ["19m19p19s11z235m7m8m", "99p99s234m567p33s4s",
                 "77p88p234s567m99m5s", "88s234p567m9m44m8p5s"]
        g = make_game(hands, draws=["7z"], dora=["4z"])
        assert g.legal_actions(0).get("kyushu") is not True

    def test_kyushu_not_offered_after_a_call(self):
        melds = [[{"type": "pon", "tiles": ["2z", "2z", "2z"], "from": 1,
                   "open": True, "called": "2z"}], [], [], []]
        hands = ["19m19p19s1z2z3z4z", D1, D2, D3B]
        g = make_game(hands, draws=["7z"], dora=["4z"], melds=melds)
        assert g.legal_actions(0).get("kyushu") is not True


class TestFourWinds:
    def test_four_winds_ryuukyoku(self):
        hands = ["123m123p123s456m9m",
                 "99p99s234m567p33s4s",
                 "77p88p234s567m99m5s",
                 "99p88s234p567m9m44m"]
        g = make_game(hands, draws=["4z", "4z", "4z", "4z"], dora=["6z"],
                      first_draw=False)
        for _ in range(4):
            g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.round_result["type"] == "ryuukyoku:four_winds"
        assert g.honba == 1 and g.round_idx == 0

    def test_four_winds_not_triggered_by_mixed_winds(self):
        hands = ["123m123p123s456m9m", D1, D2, D3B]
        g = make_game(hands, draws=["4z", "5z", "4z", "4z"], dora=["6z"],
                      first_draw=False)
        for _ in range(4):
            g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.phase != "round_end"


class TestFourKans:
    def test_four_kans_ryuukyoku_with_two_owners(self):
        hands = ["1111m234567p99s4z", D1, D2, D3B]
        g = make_game(hands, draws=["9m"], dora=["4z"], first_draw=False)
        g.n_kan = 3
        g.kan_owners = [0, 1, 2]
        g.step({"type": "kan", "tiles": g.legal_actions(0)["kan"][0]["tiles"]})
        assert g.round_result["type"] == "ryuukyoku:four_kans"

    def test_four_kans_same_owner_continues(self):
        hands = ["1111m234567p99s4z", D1, D2, D3B]
        g = make_game(hands, draws=["9m"], dora=["4z"], first_draw=False)
        g.n_kan = 3
        g.kan_owners = [0, 0, 0]
        g.step({"type": "kan", "tiles": g.legal_actions(0)["kan"][0]["tiles"]})
        assert g.phase == "draw"  # continues with rinshan draw
        assert g.n_kan == 4


class TestExhaustive:
    def test_exhaustive_with_tenpai_settlement(self):
        # seat0 tenpai, others noten: +3000 / -1000 each
        hands = ["123m123p123s456m9m", D1, D2, D3B]
        g = make_game(hands, draws=["3z", "4z"], dora=["4z"], first_draw=False,
                      wall_left=2, config=RiichiConfig(nagashi_mangan=False))
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.round_result["type"] == "ryuukyoku:exhaustive"
        assert g.round_result["tenpai"] == [0]
        assert g.scores == [28000, 24000, 24000, 24000]

    def test_exhaustive_dealer_tenpai_renchan(self):
        hands = ["123m123p123s456m9m", D1, D2, D3B]
        g = make_game(hands, draws=["3z", "4z"], dora=["4z"], first_draw=False,
                      wall_left=2)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.honba == 1 and g.round_idx == 0 and g.dealer == 0

    def test_exhaustive_dealer_noten_dealer_changes(self):
        hands = ["3579m2468p2468s1z", "123m123p123s456m9m", D1, D2]
        g = make_game(hands, draws=["3z", "4z"], dora=["4z"], first_draw=False,
                      wall_left=2, config=RiichiConfig(nagashi_mangan=False))
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.round_idx == 1 and g.dealer == 1 and g.honba == 0
        assert g.scores[1] == 28000  # seat1 was the only tenpai

    def test_two_tenpai_split_1500(self):
        hands = ["123m123p123s456m9m", "111m234m678p678s9m", D1,
                 "77p88p234s567m99s5s"]
        g = make_game(hands, draws=["3z", "4z"], dora=["4z"], first_draw=False,
                      wall_left=2, config=RiichiConfig(nagashi_mangan=False))
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.round_result["tenpai"] == [0, 1]
        assert g.scores[0] == 26500 and g.scores[1] == 26500
        assert g.scores[2] == 23500 and g.scores[3] == 23500


class TestNagashiMangan:
    def test_nagashi_mangan_dealer(self):
        discards = [[{"tile": "1p", "tsumogiri": False, "riichi": False},
                     {"tile": "1z", "tsumogiri": False, "riichi": False},
                     {"tile": "9s", "tsumogiri": False, "riichi": False}],
                    [], [], []]
        hands = ["123m123p123s456m9m", D1, "112244556677z7z", D3B]
        g = make_game(hands, draws=["3z", "5p"], dora=["4z"], first_draw=False,
                      wall_left=2, discards=discards)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.round_result["nagashi"] == [0]
        # dealer nagashi mangan: 4000 all = +12000
        assert g.scores[0] == 37000

    def test_nagashi_requires_uncalled_discards(self):
        discards = [[{"tile": "1p", "tsumogiri": False, "riichi": False},
                     {"tile": "1z", "tsumogiri": False, "riichi": False,
                      "called": True}],
                    [], [], []]
        hands = ["123m123p123s456m9m", D1, "112244556677z7z", D3B]
        g = make_game(hands, draws=["3z", "5p"], dora=["4z"], first_draw=False,
                      wall_left=2, discards=discards)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.round_result["nagashi"] == []

    def test_nagashi_disabled_by_config(self):
        discards = [[{"tile": "1p", "tsumogiri": False, "riichi": False}], [], [], []]
        hands = ["123m123p123s456m9m", D1, D2, D3B]
        cfg = RiichiConfig(nagashi_mangan=False)
        g = make_game(hands, draws=["3z", "4z"], dora=["4z"], first_draw=False,
                      wall_left=2, discards=discards, config=cfg)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.round_result["nagashi"] == []
