"""Win detection and scoring: yaku, fu/han, payments, honba, riichi sticks."""
import pytest

from env.riichi_game import RiichiConfig

from conftest import make_game


D1 = "99p99s112233z4z5z6z"
D2 = "77p88p99s1122z33z4z"
D3B = "99p556677z9m5z7z7z6z"


def play_discards(g, n):
    for _ in range(n):
        g.step({"type": "discard", "tile": g.drawn_tile})


class TestTsumoWins:
    def test_tsumo_sanshoku(self):
        hands = ["123m123p123s456m9m", D1, D2, D3B]
        g = make_game(hands, draws=["9m"], dora=["4z"], first_draw=False)
        assert g.legal_actions(0)["tsumo"] is True
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert w["yaku"] == ["Tsumo", "Sanshoku"] or "Sanshoku" in w["yaku"]
        assert w["han"] == 3
        assert w["fu"] == 30

    def test_tsumo_dealer_payment_split(self):
        # dealer tsumo 3 han 30 fu: 2000 all
        hands = ["123m123p123s456m9m", D1, D2, D3B]
        g = make_game(hands, draws=["9m"], dora=["4z"], first_draw=False)
        g.step({"type": "tsumo"})
        assert g.scores == [31000, 23000, 23000, 23000]

    def test_tsumo_non_dealer_payment_split(self):
        # seat1 tsumo 3 han 30 fu: dealer 2000, others 1000
        hands = ["111m234m567p789s9m", "123m123p123s456m9m",
                 "99p112233z4z5z6z7z8p", "77p88p1122z33z44z6z"]
        g = make_game(hands, draws=["5z", "9m"], dora=["4z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})  # 5z
        assert g.turn == 1
        r = g.step({"type": "tsumo"})
        assert r["round_end"]["winners"][0]["seat"] == 1
        assert g.scores[1] == 25000 + 4000
        assert g.scores[0] == 25000 - 2000
        assert g.scores[2] == 25000 - 1000
        assert g.scores[3] == 25000 - 1000

    def test_tsumo_round_advances_without_renchan(self):
        hands = ["111m234m567p789s9m", "123m123p123s456m9m",
                 "99p112233z4z5z6z7z8p", "77p88p1122z33z44z6z"]
        g = make_game(hands, draws=["5z", "9m"], dora=["4z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "tsumo"})
        assert g.round_idx == 1
        assert g.dealer == 1
        assert g.honba == 0


class TestRonWins:
    
    def test_ron_honba_and_riichi_sticks(self):
        hands = ["123m123p123s456m1z", "11z99p99s567m567m9m",
                 "2288m6688p99s22z4z", "111333444m22p55s"]
        g = make_game(hands, draws=["3z", "1z"], dora=["7z"], first_draw=False,
                      honba=2, kyoutaku=3)
        g.step({"type": "discard", "tile": g.drawn_tile})  # 5z
        g.step({"type": "discard", "tile": g.drawn_tile})  # 1z
        r = g.step({"type": "ron"})
        w = r["round_end"]["winners"][0]
        # sanshoku 2 han 40 fu dealer ron -> 3900 + 600 honba + 3000 sticks
        assert w["han"] == 2
        assert g.scores[0] == 25000 + 3900 + 600 + 3000
        assert g.scores[1] == 25000 - 3900 - 600
        # dealer won -> renchan
        assert g.round_idx == 0 and g.honba == 3 and g.dealer == 0
        assert g.kyoutaku == 0

    def test_ron_renchan_honba_increments(self):
        hands = ["123m123p123s456m1z", "11z99p99s567m567m9m",
                 "2288m6688p99s22z4z", "111333444m22p55s"]
        g = make_game(hands, draws=["3z", "1z"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "ron"})
        assert g.round_idx == 0 and g.honba == 1 and g.dealer == 0


class TestKiriage:
    def test_kiriage_mangan_on(self):
        # riichi + iipeiko + tanyao + pinfu = 4 han 30 fu -> 8000 (kiriage)
        hands = ["234m234m234p34s55s", "99p112233z4z5z6z7z8p",
                 "77p88p1122z33z44z5z", "5577z112p9m7z8p9s9s6z"]
        cfg = RiichiConfig(kiriage_mangan=True)
        g = make_game(hands, draws=["6z", "5s"], dora=["4z"], first_draw=False,
                      riichi=[True, False, False, False], config=cfg)
        g._riichi_draws[0] = 1  # ippatsu expired
        g.step({"type": "discard", "tile": g.drawn_tile})  # 5z tsumogiri
        g.step({"type": "discard", "tile": g.drawn_tile})  # 5s from seat1
        assert g.legal_actions(0)["ron"] is True
        r = g.step({"type": "ron"})
        w = r["round_end"]["winners"][0]
        assert w["han"] == 4
        assert w["fu"] == 30
        assert w["level"] == "kiriage mangan"
        assert g.scores[0] == 25000 + 12000  # dealer kiriage mangan

    def test_kiriage_mangan_off(self):
        hands = ["234m234m234p34s55s", "99p112233z4z5z6z7z8p",
                 "77p88p1122z33z44z5z", "5577z112p9m7z8p9s9s6z"]
        cfg = RiichiConfig(kiriage_mangan=False)
        g = make_game(hands, draws=["6z", "5s"], dora=["4z"], first_draw=False,
                      riichi=[True, False, False, False], config=cfg)
        g._riichi_draws[0] = 1
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        r = g.step({"type": "ron"})
        w = r["round_end"]["winners"][0]
        assert w["han"] == 4
        assert g.scores[0] == 25000 + 11600  # dealer 4han30fu without kiriage


class TestYakuman:
    def test_kokushi_13_wait_single_by_default(self):
        # platform v1.6: no double yakuman -> 13-sided kokushi = single
        hands = ["19m19p19s1234567z", "567m567p234s22p88p",
                 "456m679p456s88p88s", "678m234p678s55m66p"]
        g = make_game(hands, draws=["7z"], dora=["4z"], first_draw=False)
        assert g.legal_actions(0)["tsumo"] is True
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert any("Kokushi" in y for y in w["yaku"])
        assert w["han"] == 13  # single yakuman

    def test_kokushi_13_wait_double_with_config(self):
        cfg = RiichiConfig(double_yakuman=True)
        hands = ["19m19p19s1234567z", "567m567p234s22p88p",
                 "456m679p456s88p88s", "678m234p678s55m66p"]
        g = make_game(hands, draws=["7z"], dora=["4z"], first_draw=False,
                      config=cfg)
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert w["han"] == 26

    def test_suuankou(self):
        hands = ["111m222m333m99s99m", "567m567p234s22p88p",
                 "456m679p456s88p88s", "678m234p678s55m66p"]
        g = make_game(hands, draws=["9s"], dora=["4z"], first_draw=False)
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert "Suuankou" in w["yaku"]
        assert w["han"] == 13

    def test_daisangen(self):
        hands = ["555z666z777z123m9m", "567m567p234s22p88p",
                 "456m679p456s88p88s", "678m234p678s55m66p"]
        g = make_game(hands, draws=["9m"], dora=["4z"], first_draw=False)
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert "Daisangen" in w["yaku"]
        assert w["han"] == 13  # single yakuman in this ruleset

    def test_tenhou(self):
        hands = ["123m123p123s456m9m", D1, D2, D3B]
        g = make_game(hands, draws=["9m"], dora=["4z"])  # first_draw=True
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert "Tenhou" in w["yaku"]
        assert g.scores[0] == 25000 + 48000

    def test_chiihou(self):
        hands = ["111m234m567p789s9m", "123m123p123s456m9m",
                 "99p112233z4z5z6z7z8p", "77p88p1122z33z44z6z"]
        g = make_game(hands, draws=["5z", "9m"], dora=["4z"])  # first_draw=True
        g.step({"type": "discard", "tile": g.drawn_tile})  # 5z, no calls
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert "Chiihou" in w["yaku"]
        assert g.scores[1] == 25000 + 32000


class TestDoubleRon:
    def test_atamahane_first_ronner_wins(self):
        # seat2 (turn-order closest to seat1) and seat0 both wait 9m
        hands = ["123m123p123s456m9m", "99p99s5566z112p8p9m",
                 "333z234m567p789s9m", "99p4466z7z7z7z8p1p2p9s"]
        cfg = RiichiConfig(atamahane=True, double_ron=False)
        g = make_game(hands, draws=["6m", "9m"], dora=["4z"], first_draw=False,
                      config=cfg)
        g.step({"type": "discard", "tile": g.drawn_tile})  # 6m
        g.step({"type": "discard", "tile": g.drawn_tile})  # 9m from seat1
        q = [(x["seat"], x["kind"]) for x in g.claim_queue]
        # engine queries ronners in turn order; atamahane = first declaration wins
        assert q == [(2, "ron"), (0, "ron")]
        r = g.step({"type": "ron"})  # seat2 declares first -> seat0 skipped
        assert [x["seat"] for x in r["round_end"]["winners"]] == [2]
        assert g.scores[2] > 25000
        assert g.scores[0] == 25000

    def test_double_ron_both_win(self):
        hands = ["123m123p123s456m9m", "99p99s5566z112p8p9m",
                 "333z234m567p789s9m", "99p4466z7z7z7z8p1p2p9s"]
        cfg = RiichiConfig(atamahane=False, double_ron=True)
        g = make_game(hands, draws=["6m", "9m"], dora=["4z"], first_draw=False,
                      config=cfg)
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        q = [(x["seat"], x["kind"]) for x in g.claim_queue]
        assert q == [(2, "ron"), (0, "ron")]
        g.step({"type": "ron"})  # seat2
        g.step({"type": "ron"})  # seat0
        winners = [x["seat"] for x in g.round_result["winners"]]
        assert winners == [2, 0]
        assert g.scores[1] < 20000  # paid twice


class TestChankan:
    def test_chankan_on_shouminkan(self):
        melds = [[{"type": "pon", "tiles": ["5m", "5m", "5m"], "from": 1,
                   "open": True, "called": "5m"}], [], [], []]
        hands = ["234m789p99s12z", "99p99s1357s3z4z5z6z6z",
                 "234m345p345s34m55z", "333555777p22s88s"]
        g = make_game(hands, draws=["5m"], dora=["4z"], first_draw=False,
                      melds=melds)
        la = g.legal_actions(0)
        assert la["kan"], "shouminkan expected"
        g.step({"type": "kan", "tiles": la["kan"][0]["tiles"]})
        assert g.phase == "chankan"
        assert g.turn == 2
        assert g.legal_actions(2)["ron"] is True
        r = g.step({"type": "ron"})
        w = r["round_end"]["winners"][0]
        assert "Chankan" in w["yaku"]
        assert "Sanshoku" in w["yaku"]

    def test_chankan_pass_then_dora_flip_and_rinshan(self):
        melds = [[{"type": "pon", "tiles": ["5m", "5m", "5m"], "from": 1,
                   "open": True, "called": "5m"}], [], [], []]
        hands = ["234m789p99s12z", "99p99s1357s3z4z5z6z6z",
                 "234m345p345s34m55z", "333555777p22s88s"]
        g = make_game(hands, draws=["5m"], dora=["4z"], first_draw=False,
                      melds=melds, rinshan=["4z"])
        n_dora = len(g.dora_indicators)
        g.step({"type": "kan", "tiles": g.legal_actions(0)["kan"][0]["tiles"]})
        g.step({"type": "pass"})  # decline chankan
        # mjai.app parity: kakan reveals its dora at the next discard,
        # not at the kan itself
        assert len(g.dora_indicators) == n_dora
        assert g.turn == 0
        assert g.phase == "draw"
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert len(g.dora_indicators) == n_dora + 1


class TestPlatformV16:
    """平台口径 v1.6（rules_spec §7）：无切上、单倍役满、双响上家取、三家和流局。"""

    def test_default_no_kiriage_4han30fu(self):
        # riichi + iipeiko + tanyao + pinfu = 4 han 30 fu; platform default
        # kiriage_mangan=False -> dealer ron 11600 (not mangan 12000)
        hands = ["234m234m234p34s55s", "99p112233z4z5z6z7z8p",
                 "77p88p1122z33z44z5z", "5577z112p9m7z8p9s9s6z"]
        g = make_game(hands, draws=["6z", "5s"], dora=["4z"], first_draw=False,
                      riichi=[True, False, False, False])
        g._riichi_draws[0] = 1  # ippatsu expired
        g.step({"type": "discard", "tile": g.drawn_tile})  # 5z tsumogiri
        g.step({"type": "discard", "tile": g.drawn_tile})  # 5s from seat1
        r = g.step({"type": "ron"})
        w = r["round_end"]["winners"][0]
        assert w["han"] == 4 and w["fu"] == 30
        assert w["level"] != "kiriage mangan"
        assert g.scores[0] == 25000 + 11600  # dealer 4han30fu without kiriage

    def test_suuankou_tanki_single_yakuman_dealer_ron_48000(self):
        # platform v1.6: no double yakuman -> suuankou tanki = 13 han single;
        # dealer ron = 48000
        hands = ["111m222s333p999m4z", "99p99s5566z112p8p8p",
                 "234m567p789s22p88p", "678m234p678s55m66p"]
        g = make_game(hands, draws=["1z", "4z"], dora=["4z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat0: 1z
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat1: 4z
        r = g.step({"type": "ron"})
        w = r["round_end"]["winners"][0]
        assert any("Suuankou" in y for y in w["yaku"])
        assert w["han"] == 13
        assert g.scores[0] == 25000 + 48000
        assert g.scores[1] == 25000 - 48000

    def test_double_ron_sticks_and_honba_to_closest_winner(self):
        # seats 2 and 0 both ron 9m from seat1; seat2 (closest in turn order
        # from the discarder) takes ALL honba and the riichi stick.
        # seat2: 1han40fu=1300 + honba 300 + stick 1000 = 2600
        # seat0 (dealer): 2han40fu=3900 (no honba/stick)
        hands = ["123m123p123s456m9m", "99p99s5566z112p8p9m",
                 "333z234m567p789s9m", "99p4466z7z7z7z8p1p2p9s"]
        g = make_game(hands, draws=["6m", "9m"], dora=["4z"], first_draw=False,
                      honba=1, kyoutaku=1)
        g.step({"type": "discard", "tile": g.drawn_tile})  # 6m
        g.step({"type": "discard", "tile": g.drawn_tile})  # 9m from seat1
        q = [(x["seat"], x["kind"]) for x in g.claim_queue]
        assert q == [(2, "ron"), (0, "ron")]
        g.step({"type": "ron"})  # seat2 (closest)
        r = g.step({"type": "ron"})  # seat0
        assert [x["seat"] for x in r["round_end"]["winners"]] == [2, 0]
        assert g.scores[2] == 25000 + 2600
        assert g.scores[0] == 25000 + 3900
        assert g.scores[1] == 25000 - 5500
        assert g.scores[3] == 25000

    def test_triple_ron_abortive_draw(self):
        # three ronners on one discard -> ron3 abortive draw: no settlement,
        # renchan, riichi sticks stay in the pool
        hands = ["123m123p123s456m9m", "99p99s5566z112p8p2z",
                 "333z234m567p789s9m", "444z234m567p789s9m"]
        g = make_game(hands, draws=["6m", "9m"], dora=["4z"], first_draw=False,
                      kyoutaku=1)
        g.step({"type": "discard", "tile": g.drawn_tile})  # 6m
        g.step({"type": "discard", "tile": g.drawn_tile})  # 9m from seat1
        q = [(x["seat"], x["kind"]) for x in g.claim_queue]
        assert q == [(2, "ron"), (3, "ron"), (0, "ron")]
        g.step({"type": "ron"})  # seat2
        g.step({"type": "ron"})  # seat3
        r = g.step({"type": "ron"})  # seat0 -> 3rd ron -> ron3
        assert r["round_end"]["type"] == "ryuukyoku:ron3"
        assert g.scores == [25000, 25000, 25000, 25000]  # no settlement
        assert g.kyoutaku == 0                            # Tenhou: sticks returned
        assert g.round_idx == 0 and g.honba == 1          # renchan
        assert g.dealer == 0


class TestLastTileWins:
    def test_haitei_tsumo(self):
        hands = ["123m123p123s456m9m", "99p99s5566z112p8p9m",
                 "77p88p1122z33z44z5z", "99p4466z7z7z7z8p1p2p9m"]
        g = make_game(hands, draws=["9m"], dora=["7z"], first_draw=False,
                      wall_left=1)
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert "Haitei" in w["yaku"]

    def test_houtei_ron(self):
        hands = ["123m123p123s456m9m", "123m123p123s456m3z",
                 "77p88p1122z44z5z6z7z", "99p4466z7z7z7z8p1p2p9m"]
        g = make_game(hands, draws=["3z"], dora=["6z"], first_draw=False,
                      wall_left=1)
        g.step({"type": "discard", "tile": g.drawn_tile})  # last tile discarded
        assert g.legal_actions(1)["ron"] is True
        r = g.step({"type": "ron"})
        w = r["round_end"]["winners"][0]
        assert "Houtei" in w["yaku"]


class TestRinshanAndDora:
    def test_rinshan_kaihou(self):
        hands = ["1111m234p567p99s4z", "99p99s112233z5z6z7z",
                 "77p88p1122z33z55z6z", "6677z112p9m9m8p4z5z7z"]
        g = make_game(hands, draws=["4z"], dora=["8p"], first_draw=False,
                      rinshan=["4z"])
        g.step({"type": "kan", "tiles": g.legal_actions(0)["kan"][0]["tiles"]})
        assert g.legal_actions(0)["tsumo"] is True
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert "Rinshan" in w["yaku"]

    def test_aka_dora_counted(self):
        hands = ["123m123p123s456m0s", "99p99s112233z4z5z6z",
                 "77p88p99s1122z33z4z", "99p556677z9m5z7z7z6z"]
        g = make_game(hands, draws=["5s"], dora=["4s"], first_draw=False)
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert "AkaDora" in w["yaku"]
        assert "Dora" in w["yaku"]
        # sanshoku 2 + tsumo 1 + dora 2 + aka 1 = 6 han
        assert w["han"] == 6

    def test_ura_dora_for_riichi_winner(self):
        hands = ["123m123p123s456m0s", "99p99s112233z4z5z6z",
                 "77p88p99s1122z33z4z", "99p556677z9m5z7z7z6z"]
        g = make_game(hands, draws=["5s"], dora=["4z"], ura=["4s"],
                      first_draw=False, riichi=[True, False, False, False])
        g._riichi_draws[0] = 2  # tsumo after riichi, no ippatsu
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert "UraDora" in w["yaku"]
        assert "Riichi" in w["yaku"]
        # riichi 1 + sanshoku 2 + tsumo 1 + ura 2 + aka 1 = 7 han
        assert w["han"] == 7
