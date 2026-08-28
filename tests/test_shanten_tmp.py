import sys
sys.path.insert(0, "C:/agentwork/src")
sys.path.insert(0, "C:/agentwork")

from mahjong.tile import TilesConverter
from tenhou.shanten_tmp import shanten34, tenpai_discards, is_complete

H = TilesConverter.one_line_string_to_34_array  # "123m456p..." -> counts34


def test_complete_hand():
    assert shanten34(H("123456789m123p11z")) == -1   # 14 tiles
    assert is_complete(H("123456789m123p11z")) is True


def test_tenpai_hand():
    # 3 sets + ryanmen + pair (13 tiles)
    assert shanten34(H("123m456p789s12m77z")) == 0


def test_tanki_tenpai():
    # 4 sets + lone honor (13 tiles)
    assert shanten34(H("123m456p789s111z5z")) == 0


def test_shanten_junk():
    # 124578m1369p258s : 4 taatsu + 5 lone, best partition covers 9 tiles
    assert shanten34(H("124578m1369p258s")) == 4


def test_chiitoi():
    assert shanten34(H("1122334455667m")) == 0   # 6 pairs + 1 = tenpai
    assert shanten34(H("112233445566m7s")) == 0  # tenpai on 7s
    # 6 pairs + 2 singles is also tenpai (draw 7s or 8p)
    assert shanten34(H("1122334455m667s8p")) == 0
    # 123,123,678s + 44m/88p pairs: shanten 0 (discard 5m -> tenpai)
    assert shanten34(H("112233445m678s88p")) == 0
    # 3 sets + two ryanmen, no pair: one draw reaches tenpai
    assert shanten34(H("123m456p78s111z45s")) == 1


def test_kokushi():
    assert shanten34(H("19m19p19s1234567z")) == 0
    assert shanten34(H("19m19p19s123456z5m")) == 1


def test_tenpai_discards_riichi():
    # 14 tiles: 123456789m + 99m + 77z + 5p ->
    # only discarding 5p keeps tenpai (123456789m sets + 99m/77z pairs)
    hand = TilesConverter.string_to_136_array(man="12345678999", pin="5", honors="77")
    out = tenpai_discards(hand)
    assert sorted(out) == [52]  # the red 5p (only 5p copy present)


def test_with_open_melds():
    # 2 open melds implied + closed 123m 55z 66z (8 tiles):
    # tenpai on 5z/6z -> shanten 0
    assert shanten34(H("123m55z66z")) == 0
    # 1 set + 4 lone honors with 2 open melds implied: 8 - 2*3 - 0 - 0 = 2
    assert shanten34(H("123m5z6z7z1s")) == 2
