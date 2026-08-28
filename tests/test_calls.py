"""Meld (pon/chi/kan) mechanics, claim priority, kuikae."""
import pytest

from env.riichi_game import RiichiGame
from riichi.tiles import kind_of

from conftest import DUMMY_HANDS, make_game


class TestPon:
    def test_pon_option_offered(self):
        hands = ["123m123p123s456m9m", "2468m99p1357s112z",
                 "99m2467m234p67s9p9p", "111333555m22p44s"]
        g = make_game(hands, draws=["9m", "3z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.phase == "claim"
        assert g.turn == 2
        la = g.legal_actions(2)
        assert len(la["pon"]) == 1

    def test_pon_executes(self):
        hands = ["123m123p123s456m9m", "2468m99p1357s112z",
                 "99m2467m234p67s9p9p", "111333555m22p44s"]
        g = make_game(hands, draws=["9m", "3z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        la = g.legal_actions(2)
        opt = la["pon"][0]["tiles"]
        n_hand = len(g.hands[2])
        g.step({"type": "pon", "tiles": opt})
        assert len(g.hands[2]) == n_hand - 2
        assert g.melds[2][0]["type"] == "pon"
        assert g.melds[2][0]["from"] == 0
        assert g.turn == 2
        assert g.phase == "draw"

    def test_pon_then_discard_continues_play(self):
        hands = ["123m123p123s456m9m", "2468m99p1357s112z",
                 "99m2467m234p67s9p9p", "111333555m22p44s"]
        g = make_game(hands, draws=["9m", "3z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "pon", "tiles": g.legal_actions(2)["pon"][0]["tiles"]})
        assert len(g.legal_actions(2)["discard"]) == 11
        g.step({"type": "discard", "tile": g.legal_actions(2)["discard"][0]})
        assert g.turn == 3

    def test_pon_requires_two_in_hand(self):
        hands = ["123m123p123s456m9m", "2468m99p1357s112z",
                 "9m2467m234p67s9p9p4s", "111333555m22p44s"]
        g = make_game(hands, draws=["9m", "3z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.legal_actions(2)["pon"] == []

    def test_discard_marked_called_after_pon(self):
        hands = ["123m123p123s456m9m", "2468m99p1357s112z",
                 "99m2467m234p67s9p9p", "111333555m22p44s"]
        g = make_game(hands, draws=["9m", "3z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "pon", "tiles": g.legal_actions(2)["pon"][0]["tiles"]})
        assert g.discards[0][-1]["called"] is True

class TestChi:
    def test_chi_only_from_left(self):
        # seat2 could chi 3m but is not the discarder left player
        hands = ["123m123p123s456m9m", "112233445566z7z",
                 "1122m3m9p9s1122z9s5z", "333555777p22s44m"]
        g = make_game(hands, draws=["3m", "3z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})  # 3m
        queue = [(x["seat"], x["kind"]) for x in g.claim_queue]
        assert (2, "chow") not in queue
        assert (1, "chow") not in queue

    def test_chi_options_listed_for_left(self):
        hands = ["123m123p123s456m9m", "112m9p9s1122z5z5z6z6z",
                 "2244m3m9p9s1122z9s5z", "333555777p22s88s"]
        g = make_game(hands, draws=["3m", "3z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})  # 3m
        assert g.turn == 1
        la = g.legal_actions(1)
        assert la["chow"], "chow options expected"
        for o in la["chow"]:
            kinds = sorted(t // 4 for t in o["tiles"])
            assert kinds == [0, 1, 2]  # 123m

    def test_chi_executes_and_uses_called_tile(self):
        hands = ["123m123p123s456m9m", "112m9p9s1122z5z5z6z6z",
                 "2244m3m9p9s1122z9s5z", "333555777p22s88s"]
        g = make_game(hands, draws=["3m", "3z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        opt = g.legal_actions(1)["chow"][0]["tiles"]
        g.step({"type": "chow", "tiles": opt})
        assert g.melds[1][0]["type"] == "chi"
        assert g.melds[1][0]["from"] == 0
        assert g.turn == 1

    def test_chi_removes_two_from_hand(self):
        hands = ["123m123p123s456m9m", "112m9p9s1122z5z5z6z6z",
                 "2244m3m9p9s1122z9s5z", "333555777p22s88s"]
        g = make_game(hands, draws=["3m", "3z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        n = len(g.hands[1])
        g.step({"type": "chow", "tiles": g.legal_actions(1)["chow"][0]["tiles"]})
        assert len(g.hands[1]) == n - 2

    def test_chi_not_for_honors(self):
        g = make_game(DUMMY_HANDS, draws=["3z", "3z", "3z"], dora=["7z"],
                      first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})  # 3z honor
        assert all(it["kind"] != "chow" for it in g.claim_queue)


class TestClaimPriority:
    def test_pon_beats_chi(self):
        hands = ["123m123p123s456m9m", "1233m9p9s112z33z44s",
                 "2244m6688p99s112z", "111333555p22s88s"]
        g = make_game(hands, draws=["3m", "3z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        q = [(x["seat"], x["kind"]) for x in g.claim_queue]
        assert q[0] == (1, "pon")
        assert q[1] == (1, "chow")

    def test_ron_beats_everything(self):
        hands = ["123m123p123s456m1z", "11z99p99s567m567m9m",
                 "2288m6688p99s22z4z", "111333444m22p55s"]
        g = make_game(hands, draws=["3z", "1z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})  # 3z, nobody claims
        g.step({"type": "discard", "tile": g.drawn_tile})  # 1z from seat1
        q = [(x["seat"], x["kind"]) for x in g.claim_queue]
        assert q == [(0, "ron")]

    def test_daiminkan_from_discard(self):
        hands = ["123m123p123s456m7s", "9m9m9m99p99s1222z3z4z",
                 "2266m6688p99s112z", "111333555p22s44m"]
        g = make_game(hands, draws=["9m", "3z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.claim_queue[0] == {"seat": 1, "kind": "daiminkan"}
        g.step({"type": "kan", "tiles": g.legal_actions(1)["kan"][0]["tiles"]})
        assert g.melds[1][0]["type"] == "kan"
        assert g.melds[1][0]["open"] is True
        assert g.n_kan == 1
        assert g.turn == 1

class TestPassFlow:
    def test_pass_advances_to_next_claimer(self):
        hands = ["123m123p123s456m9m", "2468m99p1357s112z",
                 "99m2467m234p67s9p9p", "111333555m22p44s"]
        g = make_game(hands, draws=["9m", "3z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "pass"})
        assert g.turn == 1  # next player draws (discarder+1)
        assert g.phase == "draw"

    def test_pass_marks_temp_furiten_for_declined_ron(self):
        hands = ["123m123p123s456m1z", "11z99p99s567m567m9m",
                 "2288m6688p99s22z4z", "111333444m22p55s"]
        g = make_game(hands, draws=["3z", "1z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})  # 3z
        g.step({"type": "discard", "tile": g.drawn_tile})  # 1z
        assert g.legal_actions(0)["ron"] is True
        g.step({"type": "pass"})
        assert g.temp_furiten[0] is True


class TestKuikae:
    def test_kuikae_forbidden_by_default(self):
        hands = ["123m123p123s456m9m", "1123m9p9s1222z3z44s",
                 "4466m6688p99s112z", "333555777p22s88s"]
        g = make_game(hands, draws=["1m", "3z", "3z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})  # 1m
        g.step({"type": "pass"})  # decline the pon first
        opts = g.legal_actions(1)["chow"]
        assert opts
        g.step({"type": "chow", "tiles": opts[0]["tiles"]})
        discards = g.legal_actions(1)["discard"]
        kinds = {t // 4 for t in discards}
        assert 0 not in kinds  # cannot discard 1m (kuikae: called kind)
        assert len(discards) == 9  # 11 tiles, two 1m forbidden

    def test_kuikae_allowed_with_config(self):
        from env.riichi_game import RiichiConfig
        hands = ["123m123p123s456m9m", "1123m9p9s1222z3z44s",
                 "4466m6688p99s112z", "333555777p22s88s"]
        cfg = RiichiConfig(kuikae=True)
        g = make_game(hands, draws=["1m", "3z", "3z"], dora=["7z"],
                      first_draw=False, config=cfg)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "pass"})  # decline the pon first
        g.step({"type": "chow", "tiles": g.legal_actions(1)["chow"][0]["tiles"]})
        discards = g.legal_actions(1)["discard"]
        kinds = {t // 4 for t in discards}
        assert 0 in kinds


class TestKan:
    def test_ankan_offered_with_quad(self):
        hands = ["1111m234567p99s4z", "2468m99p1357s112z",
                 "2244m6688p99s112z", "333555777m22p44s"]
        g = make_game(hands, draws=["3z", "3z", "3z"], dora=["7z"], first_draw=False)
        la = g.legal_actions(0)
        assert len(la["kan"]) == 1
        assert sorted(kind_of(t) for t in la["kan"][0]["tiles"]) == [0, 0, 0, 0]

    def test_ankan_executes_and_flips_dora(self):
        hands = ["1111m234567p99s4z", "2468m99p1357s112z",
                 "2244m6688p99s112z", "333555777m22p44s"]
        g = make_game(hands, draws=["3z", "3z", "3z"], dora=["7z"], first_draw=False)
        n_dora = len(g.dora_indicators)
        g.step({"type": "kan", "tiles": g.legal_actions(0)["kan"][0]["tiles"]})
        assert g.n_kan == 1
        assert len(g.dora_indicators) == n_dora + 1
        assert g.melds[0][0]["type"] == "kan"
        assert g.melds[0][0]["open"] is False
        assert g.turn == 0  # still the kan player
        assert g.phase == "draw"

    def test_ankan_draws_rinshan(self):
        hands = ["1111m234567p99s4z", "2468m99p1357s112z",
                 "2244m6688p99s112z", "333555777m22p44s"]
        g = make_game(hands, draws=["3z", "3z", "3z"], dora=["7z"], first_draw=False,
                      rinshan=["4z"])
        g.step({"type": "kan", "tiles": g.legal_actions(0)["kan"][0]["tiles"]})
        assert kind_of(g.drawn_tile) == 30  # 4z from the rinshan pool
        assert len(g.hands[0]) == 11

    def test_shouminkan_extends_pon(self):
        melds = [[{"type": "pon", "tiles": ["5m", "5m", "5m"], "from": 1,
                   "open": True, "called": "5m"}], [], [], []]
        hands = ["234m567p99s12z", "99p99s1357s3z4z5z6z6z",
                 "2288m6688p22s2z2z7z", "333555777p22s88s"]
        g = make_game(hands, draws=["5m", "3z", "3z"], dora=["7z"], first_draw=False,
                      melds=melds)
        la = g.legal_actions(0)
        assert la["kan"], "shouminkan option expected"
        g.step({"type": "kan", "tiles": la["kan"][0]["tiles"]})
        assert g.melds[0][0]["type"] == "shouminkan"
        assert len(g.melds[0][0]["tiles"]) == 4
        assert g.n_kan == 1

    def test_four_kans_blocks_fifth(self):
        hands = ["1111m234567p99s4z", "2468m99p1357s112z",
                 "2244m6688p99s112z", "333555777m22p44s"]
        g = make_game(hands, draws=["3z", "3z", "3z"], dora=["7z"], first_draw=False)
        g.n_kan = 4
        la = g.legal_actions(0)
        assert la["kan"] == []