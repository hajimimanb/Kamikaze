"""Explainability helpers (schema §4): shanten, ukeire, safety, expectation."""
import pytest

from riichi import explain as ex
from riichi.tiles import parse_mpsz


class TestShanten:
    def test_complete_hand_minus_one(self):
        h = parse_mpsz("123m123p123s456m8m")
        assert ex.calc_shanten(h) == 0

    def test_complete_hand(self):
        h = parse_mpsz("123m123p123s456m99m")
        assert ex.calc_shanten(h) == -1

    def test_far_from_tenpai(self):
        h = parse_mpsz("135m246p357s1122z")
        assert ex.calc_shanten(h) >= 3

    def test_chiitoitsu_shanten(self):
        h = parse_mpsz("1122334455667z")
        assert ex.calc_shanten(h) == 0

    def test_kokushi_13_wait(self):
        h = parse_mpsz("19m19p19s1234567z")
        assert ex.calc_shanten(h) == 0

    def test_open_meld_ignored_in_shanten(self):
        h = parse_mpsz("123m123p123s4m")
        melds = [{"type": "pon", "tiles": parse_mpsz("555z"), "open": True}]
        assert ex.calc_shanten(h, melds) == 0


class TestUkeire:
    def test_ukeire_lists_improving_tiles(self):
        h = parse_mpsz("123m123p123s456m8m")
        u = ex.calc_ukeire(h)
        assert 7 in u["kinds"]  # 8m
        assert u["count"] >= 1

    def test_ukeire_deducts_visible_tiles(self):
        h = parse_mpsz("123m123p123s456m8m")
        visible = parse_mpsz("8m8m")
        u = ex.calc_ukeire(h, visible136=visible)
        assert u["per_kind"][7] == 1

    def test_ukeire_zero_for_complete_hand(self):
        h = parse_mpsz("123m123p123s456m99m")
        u = ex.calc_ukeire(h)
        assert u["count"] == 0

    def test_tenpai_ukeire_are_winning_tiles(self):
        h = parse_mpsz("123m123p123s456m9m")
        u = ex.calc_ukeire(h)
        assert u["kinds"] == [8]
        assert u["count"] == 3  # one 9m is already in hand

    def test_ukeire_respects_four_copy_limit(self):
        h = parse_mpsz("123m123p123s456m9m")
        visible = parse_mpsz("9m9m9m9m")
        u = ex.calc_ukeire(h, visible136=visible)
        assert u["count"] == 0

class TestSafety:
    def test_genbutsu(self):
        discards = [[{"tile": t} for t in parse_mpsz("1m5p9s")], [], [], []]
        s = ex.tile_safety(discards)
        assert 0 in s[0]["genbutsu"]
        assert 13 in s[0]["genbutsu"]
        assert 26 in s[0]["genbutsu"]

    def test_suji_pairs(self):
        discards = [[{"tile": parse_mpsz("5m")[0]}], [], [], []]
        s = ex.tile_safety(discards)
        assert 1 in s[0]["suji"]
        assert 7 in s[0]["suji"]

    def test_suji_from_outer_discard(self):
        discards = [[{"tile": parse_mpsz("1m")[0]}], [], [], []]
        s = ex.tile_safety(discards)
        assert 3 in s[0]["suji"]

    def test_kabe_blocks_ryanmen(self):
        visible = parse_mpsz("5m5m5m5m")
        discards = [[], [], [], []]
        s = ex.tile_safety(discards, visible136=visible)
        assert 4 in s[0]["kabe"]
        assert 5 in s[0]["kabe"]

    def test_danger_contains_unmarked_tiles(self):
        discards = [[{"tile": parse_mpsz("1m")[0]}], [], [], []]
        s = ex.tile_safety(discards)
        assert 27 in s[0]["danger"]

    def test_safety_per_opponent(self):
        discards = [
            [{"tile": parse_mpsz("1m")[0]}],
            [{"tile": parse_mpsz("5p")[0]}],
            [{"tile": parse_mpsz("9s")[0]}],
            [],
        ]
        s = ex.tile_safety(discards)
        assert s[0]["genbutsu"] == [0]
        assert s[1]["genbutsu"] == [13]
        assert s[2]["genbutsu"] == [26]


class TestDoraCount:
    def test_dora_from_indicator(self):
        h = parse_mpsz("123m123p123s456m9m")
        d = ex.count_dora(h, dora_indicators=parse_mpsz("8m"))
        assert d["dora"] == 1

    def test_dora_cycle_9_to_1(self):
        h = parse_mpsz("123m123p123s456m9m")
        d = ex.count_dora(h, dora_indicators=parse_mpsz("9m"))
        assert d["dora"] == 1

    def test_aka_count(self):
        h = parse_mpsz("123m123p123s456m0s")
        d = ex.count_dora(h)
        assert d["aka"] == 1

    def test_dora_counts_melds(self):
        h = parse_mpsz("123m123p123s456m9s")
        melds = [{"type": "pon", "tiles": parse_mpsz("9m9m9m"), "open": True}]
        d = ex.count_dora(h, melds=melds, dora_indicators=parse_mpsz("8m"))
        assert d["dora"] == 3


class TestYakuExpectation:
    def test_sanshoku_detected(self):
        h = parse_mpsz("123m123p123s456m9m")
        y = ex.yaku_expectation(h)
        assert "sanshoku" in y["yaku"]

    def test_ittsu_detected(self):
        h = parse_mpsz("123m456m789m99s9s")
        y = ex.yaku_expectation(h)
        assert "ittsu" in y["yaku"]

    def test_tanyao_detected(self):
        h = parse_mpsz("234m234p234s456m5m")
        y = ex.yaku_expectation(h)
        assert "tanyao" in y["yaku"]

    def test_yakuhai_detected(self):
        h = parse_mpsz("123m123p123s555z9m")
        y = ex.yaku_expectation(h, seat=0, round_idx=0)
        assert any(k.startswith("yakuhai") for k in y["yaku"])

    def test_honitsu_detected(self):
        h = parse_mpsz("123456789m11222z")
        y = ex.yaku_expectation(h)
        assert "honitsu" in y["yaku"]

    def test_closed_flag_and_bonus(self):
        h = parse_mpsz("123m123p123s456m9m")
        y = ex.yaku_expectation(h)
        assert y["closed"] is True
        assert "menzen" in y["yaku"]
        melds = [{"type": "pon", "tiles": parse_mpsz("555z"), "open": True}]
        y2 = ex.yaku_expectation(h, melds=melds)
        assert y2["closed"] is False

    def test_dora_added_to_expectation(self):
        h = parse_mpsz("123m123p123s456m9m")
        y = ex.yaku_expectation(h, dora_indicators=parse_mpsz("8m"))
        assert y["dora"] == 1

    def test_expectation_dict_keys(self):
        h = parse_mpsz("123m123p123s456m9m")
        y = ex.yaku_expectation(h)
        for k in ("yaku", "han_est", "dora", "aka", "closed", "shanten"):
            assert k in y

    def test_explain_decision_shape(self):
        import conftest
        g = conftest.make_game(conftest.DUMMY_HANDS, ["1p"], ["7z"],
                               first_draw=False)
        exp = ex.explain_decision(g, 0)
        for k in ("shanten", "ukeire", "safety", "yaku_expectation"):
            assert k in exp