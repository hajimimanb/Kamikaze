"""Additional engine edge cases (config switches, game-end rules, kan dora)."""
import pytest

from env.riichi_game import RiichiConfig

from conftest import make_game


D1 = "99p99s112233z4z5z6z"
D2 = "77p88p99s1122z33z4z"
D3B = "99p556677z9m5z7z7z6z"


class TestKuitan:
    def test_open_tanyao_ron_allowed_with_kuitan(self):
        melds = [[{"type": "pon", "tiles": ["2m", "2m", "2m"], "from": 1,
                   "open": True, "called": "2m"}], [], [], []]
        hands = ["234p234s456p4s", "99p99s112233z56z7z",
                 "77p88p1122z33z55z6z", "99p6677z112p9m8p4z4z"]
        g = make_game(hands, draws=["9m", "4s"], dora=["4z"], first_draw=False,
                      melds=melds)
        g.step({"type": "discard", "tile": g.drawn_tile})  # 9m
        g.step({"type": "discard", "tile": g.drawn_tile})  # 4s from seat1
        assert g.legal_actions(0)["ron"] is True
        r = g.step({"type": "ron"})
        assert "Tanyao" in r["round_end"]["winners"][0]["yaku"]

    def test_open_tanyao_ron_rejected_without_kuitan(self):
        melds = [[{"type": "pon", "tiles": ["2m", "2m", "2m"], "from": 1,
                   "open": True, "called": "2m"}], [], [], []]
        hands = ["234p234s456p4s", "99p99s112233z56z7z",
                 "77p88p1122z33z55z6z", "99p6677z112p9m8p4z4z"]
        cfg = RiichiConfig(kuitan=False)
        g = make_game(hands, draws=["9m", "4s"], dora=["4z"], first_draw=False,
                      melds=melds, config=cfg)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.legal_actions(0)["ron"] is False


class TestAgariYame:
    def test_s4_dealer_win_renchan_below_30000(self):
        # mjai.app parity: agari-yame needs return_score+ and top; a dealer
        # win at all-last below that renchans instead of ending the game.
        hands = ["123m123p123s456m1z", "11z99p99s567m567m9m",
                 "2288m6688p99s22z4z", "111333444m22p55s"]
        g = make_game(hands, draws=["3z", "1z"], dora=["7z"], first_draw=False,
                      round_idx=7)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        r = g.step({"type": "ron"})
        assert r["game_end"] is None
        assert g.phase == "draw"
        assert g.round_idx == 7 and g.honba == 1

    def test_s4_dealer_win_ends_game_with_30000_and_top(self):
        hands = ["123m123p123s456m1z", "11z99p99s567m567m9m",
                 "2288m6688p99s22z4z", "111333444m22p55s"]
        g = make_game(hands, draws=["3z", "1z"], dora=["7z"], first_draw=False,
                      round_idx=7, scores=[30000, 25000, 25000, 20000])
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        r = g.step({"type": "ron"})
        assert r["game_end"] is not None
        assert g.phase == "game_end"

    def test_s4_dealer_win_renchan_without_agari_yame(self):
        hands = ["123m123p123s456m1z", "11z99p99s567m567m9m",
                 "2288m6688p99s22z4z", "111333444m22p55s"]
        cfg = RiichiConfig(agari_yame=False)
        g = make_game(hands, draws=["3z", "1z"], dora=["7z"], first_draw=False,
                      round_idx=7, config=cfg)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "ron"})
        assert g.phase == "draw"
        assert g.round_idx == 7 and g.honba == 1


class TestRiichiStickPersistence:
    def test_sticks_stay_after_ryuukyoku(self):
        hands = ["123m123p123s456m9m", D1, D2, D3B]
        g = make_game(hands, draws=["8m", "8m"], dora=["4z"], first_draw=False,
                      wall_left=2, kyoutaku=2,
                      config=RiichiConfig(nagashi_mangan=False))
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.round_result["type"] == "ryuukyoku:exhaustive"
        # Tenhou (oracle-verified, 77/77 + 3000-game scan): riichi sticks
        # are RETURNED on any ryuukyoku; kyoutaku is zeroed.
        assert g.kyoutaku == 0  # sticks returned / pool cleared
        assert g.honba == 1


class TestKanDora:
    def test_dora_indicator_sequence(self):
        hands = ["1111m234567p99s4z", "99p99s112233z5z6z7z",
                 "77p88p1122z33z55z6z", "6677z112p9m9m8p4z5z7z"]
        g = make_game(hands, draws=["4z"], dora=["8p"], first_draw=False,
                      rinshan=["4z"])
        initial = g.dora_indicators[0]
        g.step({"type": "kan", "tiles": g.legal_actions(0)["kan"][0]["tiles"]})
        assert len(g.dora_indicators) == 2
        assert g.dora_indicators[0] == initial
        assert g.dora_indicators[1] == g.wall[127]  # first kan flips wall[127]


class TestChankanKokushi:
    def test_ankan_not_robbed_even_by_kokushi(self):
        # platform v1.7 / Tenhou: an ankan NEVER opens a chankan window,
        # not even for a kokushi hand
        hands = ["9999m234567p11s4z", "11m19p19s1234567z",
                 "77p88p1122z33z55z6z", "6677z112p8p4z5z7z9s9s"]
        g = make_game(hands, draws=["8m"], dora=["8p"], first_draw=False,
                      rinshan=["8m"])
        g.step({"type": "kan", "tiles": g.legal_actions(0)["kan"][0]["tiles"]})
        assert g.phase == "draw"   # no chankan window at all
        assert not g.chankan_queue
        assert g.legal_actions(1)["ron"] is False

    def test_ankan_not_robbed_by_non_kokushi(self):
        hands = ["2222m234567p99s4z", "456m456p456s13m99s",
                 "77p88p1122z33z55z6z", "6677z112p8p4z5z7z8s8s"]
        g = make_game(hands, draws=["8m"], dora=["8p"], first_draw=False,
                      rinshan=["8m"])
        g.step({"type": "kan", "tiles": g.legal_actions(0)["kan"][0]["tiles"]})
        assert g.phase == "draw"  # no chankan window for non-kokushi waits


class TestRiichiBoundary:
    def test_riichi_allowed_with_exactly_1000(self):
        hands = ["123m123p123s456m9m", D1, D2, D3B]
        g = make_game(hands, draws=["1p"], dora=["4z"], first_draw=False,
                      scores=[1000, 25000, 25000, 25000])
        assert g.legal_actions(0)["riichi"]
