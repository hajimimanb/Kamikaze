"""Wall construction, dealing and turn-flow tests."""
import random

import pytest

from env.riichi_game import RiichiConfig, RiichiGame
from riichi.tiles import count_red_fives, parse_mpsz, tiles136_to_str

from conftest import DUMMY_HANDS, make_game, make_wall


class TestWall:
    def test_wall_is_full_permutation(self):
        g = RiichiGame(seed=7)
        assert sorted(g.wall) == list(range(136))

    def test_same_seed_same_wall(self):
        g1 = RiichiGame(seed=123)
        g2 = RiichiGame(seed=123)
        assert g1.wall == g2.wall
        assert g1.hands == g2.hands

    def test_different_seeds_differ(self):
        g1 = RiichiGame(seed=1)
        g2 = RiichiGame(seed=2)
        assert g1.wall != g2.wall

    def test_three_red_fives_in_wall(self):
        g = RiichiGame(seed=5)
        assert count_red_fives(g.wall) == 3

    def test_wall_left_starts_70_after_first_draw_69(self):
        g = RiichiGame(seed=5)
        assert g.wall_left == 69  # dealer already drew the 14th tile

    def test_initial_dora_is_wall_126(self):
        g = RiichiGame(seed=5)
        assert g.dora_indicators == [g.wall[126]]

    def test_dead_wall_layout(self):
        wall = make_wall(draws=[0], dora=[100], ura=[101], rinshan=[102, 103, 104, 105])
        assert wall[126] == 100
        assert wall[131] == 101
        # rinshan drawn back-to-front: wall[125] first
        assert wall[125] == 102


class TestDeal:
    def test_each_player_gets_13(self):
        g = RiichiGame(seed=5)
        assert [len(h) for h in g.hands] == [14, 13, 13, 13]  # dealer drew

    def test_dealer_draws_14th(self):
        g = RiichiGame(seed=5)
        assert len(g.hands[0]) == 14
        assert g.turn == 0
        assert g.phase == "draw"

    def test_no_overlap_between_hands_and_draw(self):
        g = RiichiGame(seed=5)
        dealt = [t for h in g.hands for t in h]
        assert len(set(dealt)) == 53  # 52 dealt + 1 drawn, all distinct

    def test_first_events_are_start_kyoku_and_tsumo(self):
        g = RiichiGame(seed=5)
        assert g.events[0]["type"] == "start_kyoku"
        assert g.events[1]["type"] == "tsumo"
        assert g.events[1]["seat"] == 0
        assert g.events[1]["actor"] == 0
        assert g.events[1]["pai"]

    def test_dealer_starts_at_seat_0(self):
        g = RiichiGame(seed=5)
        assert g.dealer == 0
        assert g.round_idx == 0


class TestTurnFlow:
    def test_turn_order_east_south_west_north(self):
        g = make_game(
            DUMMY_HANDS,
            draws=["3z", "3z", "3z", "3z"], dora=["7z"], first_draw=False)
        seq = []
        for _ in range(4):
            seq.append(g.turn)
            t = g.drawn_tile
            g.step({"type": "discard", "tile": t})
        assert seq == [0, 1, 2, 3]

    def test_discard_removes_tile_from_hand(self):
        g = make_game(
            DUMMY_HANDS,
            draws=["3z", "3z", "3z", "3z"], dora=["7z"], first_draw=False)
        n_before = len(g.hands[0])
        tile = g.legal_actions(0)["discard"][0]
        g.step({"type": "discard", "tile": tile})
        assert len(g.hands[0]) == n_before - 1
        assert tile not in g.hands[0]

    def test_discard_recorded_in_river(self):
        g = make_game(
            DUMMY_HANDS,
            draws=["3z", "3z", "3z", "3z"], dora=["7z"], first_draw=False)
        tile = g.legal_actions(0)["discard"][0]
        g.step({"type": "discard", "tile": tile})
        assert g.discards[0][-1]["tile"] == tile

    def test_tsumogiri_flag(self):
        g = make_game(
            DUMMY_HANDS,
            draws=["3z", "3z", "3z", "3z"], dora=["7z"], first_draw=False)
        drawn = g.drawn_tile
        g.step({"type": "discard", "tile": drawn})
        assert g.discards[0][-1]["tsumogiri"] is True

    def test_hand_cut_not_tsumogiri(self):
        g = make_game(
            DUMMY_HANDS,
            draws=["3z", "3z", "3z", "3z"], dora=["7z"], first_draw=False)
        drawn = g.drawn_tile
        others = [t for t in g.hands[0] if t != drawn]
        g.step({"type": "discard", "tile": others[0]})
        assert g.discards[0][-1]["tsumogiri"] is False

    def test_illegal_discard_raises(self):
        g = make_game(
            DUMMY_HANDS,
            draws=["3z", "3z", "3z", "3z"], dora=["7z"], first_draw=False)
        with pytest.raises(ValueError):
            g.step({"type": "discard", "tile": 999})

    def test_illegal_action_from_wrong_seat_raises(self):
        g = make_game(
            DUMMY_HANDS,
            draws=["3z", "3z", "3z", "3z"], dora=["7z"], first_draw=False)
        with pytest.raises(ValueError):
            g.step({"type": "discard", "tile": g.hands[1][0]})

    def test_wall_left_decrements_per_draw(self):
        g = make_game(
            DUMMY_HANDS,
            draws=["3z", "3z", "3z", "3z"], dora=["7z"], first_draw=False)
        w0 = g.wall_left
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.wall_left == w0 - 2

    def test_last_event_discard(self):
        g = make_game(
            DUMMY_HANDS,
            draws=["3z", "3z", "3z", "3z"], dora=["7z"], first_draw=False)
        tile = g.legal_actions(0)["discard"][0]
        g.step({"type": "discard", "tile": tile})
        assert g.last_event["type"] == "discard"
        assert g.last_event["seat"] == 0
        assert g.last_event["tile"] == tile

    def test_unknown_action_raises(self):
        g = make_game(
            DUMMY_HANDS,
            draws=["3z"], dora=["7z"], first_draw=False)
        with pytest.raises(ValueError):
            g.step({"type": "explode"})
