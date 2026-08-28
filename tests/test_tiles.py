"""Tile utility tests (encoding contract, schema §1)."""
import pytest

from riichi.tiles import (
    RED_FIVE_KINDS,
    RED_FIVE_MAN,
    RED_FIVE_PIN,
    RED_FIVE_SOU,
    RED_FIVES,
    copy_of,
    count_red_fives,
    counts34,
    is_dragon,
    is_honor,
    is_red_five,
    is_suited,
    is_terminal,
    is_terminal_or_honor,
    is_wind,
    kind_of,
    kind_str,
    next_dora_kind,
    parse_mpsz,
    rank_of,
    sort_tiles136,
    suit_of,
    tile136,
    tiles136_from_counts,
    tiles136_to_str,
)


class TestKindTile136:
    def test_kind_of_man(self):
        assert kind_of(0) == 0
        assert kind_of(3) == 0
        assert kind_of(16) == 4

    def test_kind_of_pin(self):
        assert kind_of(36) == 9
        assert kind_of(52) == 13

    def test_kind_of_sou(self):
        assert kind_of(72) == 18
        assert kind_of(88) == 22

    def test_kind_of_honors(self):
        assert kind_of(108) == 27
        assert kind_of(135) == 33

    def test_copy_of(self):
        assert copy_of(16) == 0
        assert copy_of(17) == 1
        assert copy_of(135) == 3

    def test_tile136_roundtrip(self):
        for kind in range(34):
            for copy in range(4):
                t = tile136(kind, copy)
                assert kind_of(t) == kind
                assert copy_of(t) == copy
                assert t == kind * 4 + copy


class TestRedFives:
    def test_fixed_ids(self):
        assert RED_FIVE_MAN == 16
        assert RED_FIVE_PIN == 52
        assert RED_FIVE_SOU == 88

    def test_red_kinds_are_the_5s(self):
        assert RED_FIVE_KINDS == (4, 13, 22)

    def test_is_red_five(self):
        assert is_red_five(16)
        assert is_red_five(52)
        assert is_red_five(88)
        assert not is_red_five(17)
        assert not is_red_five(0)

    def test_count_red_fives(self):
        assert count_red_fives([16, 52, 88]) == 3
        assert count_red_fives([16, 17, 52, 53, 88, 89]) == 3
        assert count_red_fives([]) == 0

    def test_red_fives_set(self):
        assert RED_FIVES == frozenset((16, 52, 88))


class TestCounts34:
    def test_counts(self):
        c = counts34([0, 1, 2, 3])
        assert c[0] == 4
        assert sum(c) == 4

    def test_counts_mixed(self):
        c = counts34([0, 4, 8, 16, 52, 88, 108])
        assert c[0] == 1 and c[1] == 1 and c[2] == 1
        assert c[4] == 1 and c[13] == 1 and c[22] == 1
        assert c[27] == 1

    def test_tiles136_from_counts_roundtrip(self):
        tiles = [0, 4, 8, 16, 52, 88]
        assert tiles136_from_counts(counts34(tiles)) == [0, 4, 8, 16, 52, 88]

    def test_tiles136_from_counts_uses_lowest_copies(self):
        c = [0] * 34
        c[0] = 3
        assert tiles136_from_counts(c) == [0, 1, 2]


class TestPredicates:
    def test_is_suited(self):
        assert is_suited(0) and is_suited(26)
        assert not is_suited(27)

    def test_is_honor(self):
        assert is_honor(27) and is_honor(33)
        assert not is_honor(26)

    def test_is_wind(self):
        assert is_wind(27) and is_wind(30)
        assert not is_wind(31)

    def test_is_dragon(self):
        assert is_dragon(31) and is_dragon(33)
        assert not is_dragon(30)

    def test_is_terminal(self):
        assert is_terminal(0) and is_terminal(8)
        assert is_terminal(9) and is_terminal(17)
        assert is_terminal(18) and is_terminal(26)
        assert not is_terminal(4)
        assert not is_terminal(27)

    def test_is_terminal_or_honor(self):
        assert is_terminal_or_honor(0)
        assert is_terminal_or_honor(33)
        assert not is_terminal_or_honor(5)

    def test_suit_of(self):
        assert suit_of(0) == 0 and suit_of(8) == 0
        assert suit_of(9) == 1 and suit_of(17) == 1
        assert suit_of(18) == 2 and suit_of(26) == 2
        assert suit_of(27) == 3 and suit_of(33) == 3

    def test_rank_of(self):
        assert rank_of(0) == 1 and rank_of(8) == 9
        assert rank_of(27) == 1 and rank_of(33) == 7

    def test_kind_str(self):
        assert kind_str(0) == "1m"
        assert kind_str(13) == "5p"
        assert kind_str(26) == "9s"
        assert kind_str(27) == "1z"
        assert kind_str(33) == "7z"


class TestNextDora:
    def test_suits_cycle(self):
        assert next_dora_kind(0) == 1
        assert next_dora_kind(7) == 8
        assert next_dora_kind(8) == 0
        assert next_dora_kind(9) == 10
        assert next_dora_kind(17) == 9
        assert next_dora_kind(18) == 19
        assert next_dora_kind(26) == 18

    def test_winds_cycle(self):
        assert next_dora_kind(27) == 28
        assert next_dora_kind(28) == 29
        assert next_dora_kind(29) == 30
        assert next_dora_kind(30) == 27

    def test_dragons_cycle(self):
        assert next_dora_kind(31) == 32
        assert next_dora_kind(32) == 33
        assert next_dora_kind(33) == 31


class TestParseMpsz:
    def test_basic(self):
        assert parse_mpsz("123m") == [0, 4, 8]

    def test_pin_sou(self):
        assert parse_mpsz("9p") == [68]
        assert parse_mpsz("9s") == [104]

    def test_honors(self):
        assert parse_mpsz("1234567z") == [108, 112, 116, 120, 124, 128, 132]

    def test_red_five_zero(self):
        assert parse_mpsz("0m") == [16]
        assert parse_mpsz("0p") == [52]
        assert parse_mpsz("0s") == [88]

    def test_red_five_r(self):
        assert parse_mpsz("rm") == [16]

    def test_plain_5_never_red(self):
        assert parse_mpsz("5m") == [17]
        assert parse_mpsz("55m") == [17, 18]

    def test_mixed_red_and_plain(self):
        assert parse_mpsz("055m") == [16, 17, 18]

    def test_full_hand(self):
        tiles = parse_mpsz("123m456p789s11z")
        assert len(tiles) == 11

    def test_invalid_suit_letter(self):
        with pytest.raises(ValueError):
            parse_mpsz("123x")

    def test_missing_suit_letter(self):
        with pytest.raises(ValueError):
            parse_mpsz("123")

    def test_red_in_honors_raises(self):
        with pytest.raises(ValueError):
            parse_mpsz("0z")

    def test_honor_rank_bounds(self):
        with pytest.raises(ValueError):
            parse_mpsz("8z")


class TestToStr:
    def test_roundtrip(self):
        s = "123m0p56s11z"
        assert tiles136_to_str(parse_mpsz(s)) == s

    def test_red_without_marker(self):
        assert tiles136_to_str([16], aka_marker=False) == "5m"

    def test_sorts_output(self):
        assert tiles136_to_str([8, 0, 4]) == "123m"

    def test_sort_tiles136(self):
        assert sort_tiles136([135, 0, 16]) == [0, 16, 135]
