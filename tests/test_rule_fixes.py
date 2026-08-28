# -*- coding: utf-8 -*-
"""User-requested rule fixes (2026-08): unit tests.

1. Same-go-around (temp) furiten for shape-completing tiles even without
   yaku; cleared by the player's own next draw.
2. Chankan: kakan robbable with any yaku; ankan robbable by NOBODY
   (platform v1.7 / Tenhou).
3. West extension: after South-4 a negative score ends the game (tobi);
   otherwise the game continues into West only while every score is below
   return_score; west_extension now defaults to True.
4. explain.py: shanten / ukeire / river safety (genbutsu, suji, kabe,
   early) / yaku expectation spot-checks.
"""
from env.riichi_game import RiichiConfig
from riichi.explain import calc_shanten, calc_ukeire, tile_safety, yaku_expectation
from riichi.tiles import parse_mpsz

from conftest import make_game


class TestShapeFuriten:
    """同巡振听：形状可和(无役) 见逃也置位，直到自己下一次摸牌解除。"""

    def test_no_yaku_shape_pass_sets_temp_furiten(self):
        # seat 1 (S wind): waits 2z (seat-wind yaku) / 4z (no yaku).
        # seat 2 discards 4z: shape-completing but no yaku -> ron not even
        # offered, only the pon. Passing it must still set temp furiten.
        hands = ["667788m334455p9p",     # seat 0 discards its draws
                 "123m456p789s2244z",    # subject
                 "111m999m222p888p1s",   # draws+discards 4z
                 "777888999s55s66s"]     # draws+discards 2z
        g = make_game(hands, draws=["3s", "1z", "4z", "2z", "5z", "6z"],
                      dora=["4z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat0: 3s
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat1: 1z
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat2: 4z
        # no ron offer (no yaku), but a pon offer exists
        q = [(x["seat"], x["kind"]) for x in g.claim_queue]
        assert q == [(1, "pon")]
        assert g.legal_actions(1)["ron"] is False
        g.step({"type": "pass"})
        assert g.temp_furiten[1] is True, "declining a shape-completing tile must set temp furiten"
        # seat3 discards 2z: a *legal* ron normally (seat-wind triplet), but
        # temp furiten blocks it
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat3: 2z
        assert g.legal_actions(1)["ron"] is False
        assert g.temp_furiten[1] is True
        g.step({"type": "pass"})          # decline the pon on 2z as well
        # seat0 discards 5z, then seat1 draws 6z -> temp furiten cleared
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat0: 5z
        assert g.temp_furiten[1] is False, "own draw must clear temp furiten"

    def test_call_lifts_temp_furiten_immediately(self):
        # rules spec 2: a successful call before the next draw clears
        # same-go-around furiten at once ("通过副露提前解除")
        hands = ["667788m334455p9p",
                 "123m456p789s2244z",
                 "111m999m222p888p1s",
                 "777888999s55s66s"]
        g = make_game(hands, draws=["3s", "1z", "4z", "2z"], dora=["4z"],
                      first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})  # 3s
        g.step({"type": "discard", "tile": g.drawn_tile})  # 1z
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat2: 4z
        assert g.turn == 1 and g.phase == "claim"
        # mark seat1 as temp-furiten (declined a shape-completing tile
        # earlier in the same go-around)
        g.temp_furiten[1] = True
        g.step({"type": "pon", "tiles": g.legal_actions(1)["pon"][0]["tiles"]})
        assert g.temp_furiten[1] is False, "a successful call must lift temp furiten"
        # progress: seat1 discards a hand tile
        g.step({"type": "discard", "tile": g.legal_actions(1)["discard"][0]})


class TestChankanRules:
    """抢杠：加杠任意役可抢；暗杠任何人不可抢（v1.7 天凤口径）。"""

    def test_kakan_robbed_with_any_yaku(self):
        # seat0 shouminkans its 5m pon; seat1 robs it with plain tanyao
        melds = [[{"type": "pon", "tiles": ["5m", "5m", "5m"], "from": 1,
                   "open": True, "called": "5m"}], [], [], []]
        hands = ["234m789p99s12z",
                 "123p456p789p34m22m",   # waits 5m via 34m; tanyao only
                 "111222333p44s55s",
                 "667788m555p99s34z"]
        g = make_game(hands, draws=["5m"], dora=["4z"], first_draw=False,
                      melds=melds)
        la = g.legal_actions(0)
        assert la["kan"], "shouminkan expected"
        g.step({"type": "kan", "tiles": la["kan"][0]["tiles"]})
        assert g.phase == "chankan"
        assert g.turn == 1
        assert g.legal_actions(1)["ron"] is True
        r = g.step({"type": "ron"})
        w = r["round_end"]["winners"][0]
        assert "Chankan" in w["yaku"]
        # an ordinary (non-kokushi) hand robbed the kakan
        assert not any("Kokushi" in y for y in w["yaku"])

    def test_riichi_allowed_after_ankan(self):
        # an ankan preserves menzen, so riichi must be legal (regression for
        # _riichi_options returning [] whenever any meld existed).
        # seat0: concealed 123m456p789s + ankan 2222z + drawn 1s -> tanki riichi.
        from riichi.tiles import parse_mpsz
        meld = {"type": "kan", "tiles": parse_mpsz("2222z"),
                "from": -1, "open": False}
        g = make_game(hands=["123m456p11s78s", "123m456m789m11p22p",
                             "123p456p789p11s22s", "234s456s789s11z33z"],
                      draws=[], dora=["1m"], melds=[[meld], [], [], []],
                      first_draw=False, config=RiichiConfig(kyushu_kyuhai=True))
        assert g.phase == "draw" and g.turn == 0
        la = g.legal_actions(0)
        assert la["riichi"], "riichi after ankan must be offered"

    def test_ankan_cannot_be_robbed_even_by_kokushi(self):
        # platform v1.7 / Tenhou: an ankan NEVER opens a chankan window.
        # seat1 is kokushi single-wait on 1m, seat2 waits 1m with a normal
        # yaku (East triplet); neither may rob the ankan.
        hands = ["1111m234p567p99s5z",
                 "9m1p9p1s9s1z2z2z3z4z5z6z7z",
                 "23m111z456p789p45s",
                 "667788m334455p9p"]
        g = make_game(hands, draws=["6z"], dora=["4z"], first_draw=False)
        la = g.legal_actions(0)
        assert la["kan"], "ankan expected"
        g.step({"type": "ankan", "tiles": la["kan"][0]["tiles"]})
        assert g.phase == "draw"      # no chankan window at all
        assert not g.chankan_queue
        assert g.legal_actions(1)["ron"] is False
        assert g.legal_actions(2)["ron"] is False

    def test_temp_furiten_player_cannot_rob_kakan(self):
        # seat1 waits 5m (Ittsu) but is same-go-around furiten; the chankan
        # window must skip them entirely
        melds = [[{"type": "pon", "tiles": ["5m", "5m", "5m"], "from": 1,
                   "open": True, "called": "5m"}], [], [], []]
        hands = ["234m789p99s12z",
                 "123p456p789p34m22m",
                 "111222333p44s55s",
                 "667788m555p99s34z"]
        g = make_game(hands, draws=["5m"], dora=["4z"], first_draw=False,
                      melds=melds)
        g.temp_furiten[1] = True
        g.step({"type": "kan", "tiles": g.legal_actions(0)["kan"][0]["tiles"]})
        assert g.phase == "draw"          # no chankan offer
        assert not g.chankan_queue

    def test_riichi_pass_furiten_player_cannot_rob_kakan(self):
        melds = [[{"type": "pon", "tiles": ["5m", "5m", "5m"], "from": 1,
                   "open": True, "called": "5m"}], [], [], []]
        hands = ["234m789p99s12z",
                 "123p456p789p34m22m",
                 "111222333p44s55s",
                 "667788m555p99s34z"]
        g = make_game(hands, draws=["5m"], dora=["4z"], first_draw=False,
                      melds=melds, riichi=[False, True, False, False])
        g.riichi_furiten[1] = True        # declined a legal ron after riichi
        g.step({"type": "kan", "tiles": g.legal_actions(0)["kan"][0]["tiles"]})
        assert g.phase == "draw"
        assert not g.chankan_queue

    def test_ankan_robbing_switch_is_reserved_noop(self):
        # v1.7: the chankan_kokushi flag is reserved; even with it True the
        # ankan still opens no chankan window
        cfg = RiichiConfig(chankan_kokushi=True)
        hands = ["1111m234p567p99s5z",
                 "9m1p9p1s9s1z2z2z3z4z5z6z7z",
                 "23m111z456p789p45s",
                 "667788m334455p9p"]
        g = make_game(hands, draws=["6z"], dora=["4z"], first_draw=False,
                      config=cfg)
        g.step({"type": "ankan", "tiles": g.legal_actions(0)["kan"][0]["tiles"]})
        assert g.phase == "draw"  # no chankan window at all


class TestWestExtension:
    """西入：负分立即终局；否则仅在全员 < return_score 时继续。"""

    HANDS = ["123m123p123s456m1z", "22p88p99s1122z4z5z7z",
             "2288m6688p99s22z4z", "111333444m33s55s"]
    # dealer = seat 3 (South-4): seat3 draws 3z, seat0 draws 9m, seat1 draws
    # 1z and discards it, seat0 rons (non-dealer win -> round advances).

    def test_all_below_30000_continues_to_west(self):
        g = make_game(self.HANDS, draws=["3z", "9m", "1z"], dora=["7z"],
                      first_draw=False, round_idx=7, dealer=3)
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat3: 3z
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat0: 9m
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat1: 1z
        r = g.step({"type": "ron"})                        # seat0 wins, not dealer
        assert r["game_end"] is None
        assert g.phase == "draw"
        assert g.round_idx == 8       # West-1
        assert g.dealer == 0
        assert max(g.scores) < 30000 and min(g.scores) >= 0

    def test_negative_score_ends_game_tobi(self):
        g = make_game(self.HANDS, draws=["3z", "9m", "1z"], dora=["7z"],
                      first_draw=False, round_idx=7, dealer=3,
                      scores=[25000, 1000, 25000, 25000])
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        r = g.step({"type": "ron"})
        assert g.scores[1] < 0
        assert r["game_end"] is not None
        assert g.phase == "game_end"

    def test_reaching_30000_ends_game(self):
        g = make_game(self.HANDS, draws=["3z", "9m", "1z"], dora=["7z"],
                      first_draw=False, round_idx=7, dealer=3,
                      scores=[29000, 25000, 25000, 21000])
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        r = g.step({"type": "ron"})
        assert max(g.scores) >= 30000
        assert r["game_end"] is not None
        assert g.phase == "game_end"

    def test_same_condition_after_each_west_round(self):
        # West-1 with all scores low: another non-renchan round -> West-2
        g = make_game(self.HANDS, draws=["3z", "9m", "1z"], dora=["7z"],
                      first_draw=False, round_idx=8, dealer=1,
                      scores=[20000, 20000, 20000, 20000])
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        r = g.step({"type": "ron"})
        assert r["game_end"] is None
        assert g.round_idx == 9
        # and with someone at 30000 the same round ends instead
        g2 = make_game(self.HANDS, draws=["3z", "9m", "1z"], dora=["7z"],
                       first_draw=False, round_idx=8, dealer=1,
                       scores=[32000, 20000, 20000, 20000])
        g2.step({"type": "discard", "tile": g2.drawn_tile})
        g2.step({"type": "discard", "tile": g2.drawn_tile})
        g2.step({"type": "discard", "tile": g2.drawn_tile})
        r2 = g2.step({"type": "ron"})
        assert r2["game_end"] is not None

    def test_west_extension_default_on(self):
        assert RiichiConfig().west_extension is True


class TestExplainFactors:
    """explain.py 因子 spot-check (§4)。"""

    def test_shanten_values(self):
        assert calc_shanten(parse_mpsz("123m456p789s1122z")) == 0    # tenpai
        assert calc_shanten(parse_mpsz("123m456p789s11122z")) == -1  # complete

    def test_ukeire_counts_and_visible_deduction(self):
        # tanki 9m: 4 copies - 1 held = 3; hide two more -> 1
        hand = parse_mpsz("123m123p123s456m9m")
        u = calc_ukeire(hand)
        assert u["kinds"] == [8]
        assert u["count"] == 3
        visible = [33, 34]  # two more 9m copies visible
        u2 = calc_ukeire(hand, visible136=visible)
        assert u2["count"] == 1

    def test_safety_markers_genbutsu_suji_kabe_early(self):
        discards = [
            [{"tile": t, "tsumogiri": False, "riichi": False}
             for t in parse_mpsz("4p2s9m1z")],      # seat0 river
            [{"tile": t, "tsumogiri": False, "riichi": False}
             for t in parse_mpsz("5p6p7p8p9p3m")],  # seat1 river
            [], [],                                 # seats 2/3 empty
        ]
        # all four 3p visible -> kabe at 3p protects 2p and 4p
        visible = parse_mpsz("3p3p3p3p")
        out = tile_safety(discards, visible)
        s0 = out[0]
        assert 12 in s0["genbutsu"]     # 4p (kind 12) discarded -> genbutsu
        assert 9 in s0["suji"]          # 4p suji: 1p
        assert 15 in s0["suji"]         # 4p suji: 7p
        assert 10 in s0["kabe"]         # kabe 3p -> 2p protected
        assert 12 in s0["kabe"]         # kabe 3p -> 4p protected
        assert 12 in s0["early"]        # 4p was an early discard
        assert 12 not in s0["danger"]
        assert 9 not in s0["danger"]
        assert 10 not in s0["danger"]
        s1 = out[1]
        assert 14 in s1["genbutsu"]     # 6p discarded
        # honors are never suji/kabe protected -> still dangerous
        assert 33 in s1["danger"]

    def test_yaku_expectation_factors(self):
        # East triplet + dora (2z indicator -> 3z dora) + menzen bonus
        hand = parse_mpsz("111z234m567p789s3z")
        r = yaku_expectation(hand, dora_indicators=parse_mpsz("2z"),
                             seat=1, round_idx=0)
        assert "yakuhai_27" in r["yaku"]
        assert r["dora"] == 1
        assert r["closed"] is True
        assert "menzen" in r["yaku"]
        assert r["han_est"] >= 3       # yakuhai + dora + menzen
        # aka: red five in hand counts
        r3 = yaku_expectation(parse_mpsz("123m456m0p789s112z"),
                              dora_indicators=[])
        assert r3["aka"] == 1


class TestWindPairAndTripletScoring:
    """连风/役牌 作雀头只计符不计番；连风刻子计 2 番（reviewer 对齐用）。

    对齐依据（mjai.app Rust 参考实现）:
    - docs/mjai.app-main/src/algo/agari.rs: 役牌(门风/场风/三元) 仅从
      all_kotsu_and_kantsu() 计番，雀头永不计番（小三元/小四喜仅用雀头判型）。
    - 同文件符计算: 雀头为场风 +2符、门风 +2符 → 连风雀头 = 4符（天凤规则，
      源码注释直接引用 tenhou.net/man/#RULE "連風牌は4符"）。
    - mahjong 库 fu.py 已实现 VALUED_PAIR(2符)/DOUBLE_VALUED_PAIR(4符)。
    """

    def test_double_wind_pair_gives_4_fu_not_han(self):
        # East round, seat0 = East dealer. Closed tsumo: 555m(暗刻4符) +
        # 78s(两面0符) + 11z 连风雀头. fu = 20+2+4+4 = 30 -> 30
        # (Tenhou-exact: 78s waiting 6s is ryanmen 0 fu).
        # 雀头不给番: yaku 只有 MenzenTsumo。
        hands = ["555m123m456p78s11z", "99p99s112233m889p",
                 "223344p223344s9s", "6688p6677s33445z"]
        g = make_game(hands, draws=["6s"], dora=["4z"], first_draw=False)
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert w["fu"] == 30
        assert "SeatWindEast" not in w["yaku"]
        assert "RoundWindEast" not in w["yaku"]
        assert "Tsumo" in w["yaku"]

    def test_single_valued_pair_gives_2_fu(self):
        # East round, seat1 = South. 22z = 门风(S)雀头 -> 2符.
        # fu = 20+2+4+2+2 = 30 (no rounding up).
        hands = ["99p99s112233m889p", "555m123m456p78s22z",
                 "223344p223344s9s", "6688p6677s33445z"]
        g = make_game(hands, draws=["9m", "6s"], dora=["4z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat0: 9m
        r = g.step({"type": "tsumo"})                      # seat1 tsumo 6s
        w = r["round_end"]["winners"][0]
        assert w["fu"] == 30
        assert "SeatWindSouth" not in w["yaku"]

    def test_double_wind_triplet_gives_2_han_closed(self):
        # East round, seat0 = East dealer, closed 111z -> 2 han 连风
        # dora indicator C(7z) -> dora White; hand has none, so the wind
        # triplet contributes exactly its 2 yaku han (1 seat + 1 round)
        hands = ["111z234m567p89s22p", "99p99s112233m889p",
                 "223344p223344s9s", "6688p6677s33445z"]
        g = make_game(hands, draws=["9m", "7s"], dora=["7z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat0: 9m
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat1: 7s
        r = g.step({"type": "ron"})
        w = r["round_end"]["winners"][0]
        assert "SeatWindEast" in w["yaku"]
        assert "RoundWindEast" in w["yaku"]
        assert w["han"] == 2

    def test_double_wind_triplet_gives_2_han_open_pon(self):
        # East round, seat0 = East dealer, open pon 111z -> still 2 han
        melds = [[{"type": "pon", "tiles": ["1z", "1z", "1z"], "from": 1,
                   "open": True, "called": "1z"}], [], [], []]
        hands = ["234m567p89s22p", "99p99s112233m889p",
                 "223344p223344s9s", "6688p6677s33445z"]
        g = make_game(hands, draws=["9m", "7s"], dora=["4z"], first_draw=False,
                      melds=melds)
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat0: 9m
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat1: 7s
        r = g.step({"type": "ron"})
        w = r["round_end"]["winners"][0]
        assert "SeatWindEast" in w["yaku"]
        assert "RoundWindEast" in w["yaku"]
        assert w["han"] >= 2


class TestYakumanSettlement:
    """役满结算按 docs/rules_spec.md §5.1/§5.3：庄家自摸每家付 2a。

    n>=13 -> a=8000（役满）；双重役满 a=16000。庄家自摸：每家 2a。
    """

    def test_dealer_tsumo_single_yakuman_each_pays_16000(self):
        # 大三元 (han 13): a=8000, each payer 2a=16000 -> winner +48000
        hands = ["555z666z777z234m9m", "567m567p234s22p88p",
                 "456m679p456s88p88s", "678m234p678s55m66p"]
        g = make_game(hands, draws=["9m"], dora=["4z"], first_draw=False)
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert w["han"] == 13
        assert g.scores == [73000, 9000, 9000, 9000]

    def test_ko_tsumo_single_yakuman_8000_16000_split(self):
        # 子家自摸役满: 闲家 a=8000, 庄家 2a=16000 -> winner +32000
        hands = ["456m679p456s88p88s", "555z666z777z234m9m",
                 "567m567p234s22p88p", "678m234p678s55m66p"]
        g = make_game(hands, draws=["3s", "9m"], dora=["4z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})  # seat0: 3s
        r = g.step({"type": "tsumo"})                      # seat1 tsumo
        w = r["round_end"]["winners"][0]
        assert w["han"] == 13
        assert g.scores == [9000, 57000, 17000, 17000]

    def test_dealer_tsumo_kokushi13_single_by_default(self):
        # platform v1.6: no double yakuman -> 国士13面 = single, each pays 16000
        hands = ["19m19p19s1234567z", "567m567p234s22p88p",
                 "456m679p456s88p88s", "678m234p678s55m66p"]
        g = make_game(hands, draws=["7z"], dora=["4z"], first_draw=False)
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert w["han"] == 13
        assert g.scores == [73000, 9000, 9000, 9000]

    def test_dealer_tsumo_double_yakuman_with_config(self):
        # explicit double_yakuman=True: han 26, each payer 32000 -> tobi
        cfg = RiichiConfig(double_yakuman=True)
        hands = ["19m19p19s1234567z", "567m567p234s22p88p",
                 "456m679p456s88p88s", "678m234p678s55m66p"]
        g = make_game(hands, draws=["7z"], dora=["4z"], first_draw=False,
                      config=cfg)
        r = g.step({"type": "tsumo"})
        w = r["round_end"]["winners"][0]
        assert w["han"] == 26
        assert g.scores == [121000, -7000, -7000, -7000]
        assert g.phase == "game_end"  # tobi: negative scores end the game
