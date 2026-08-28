"""Observation schema (§2) compliance and decision-record sampling (§3)."""
import pytest

from env.riichi_game import RiichiGame

from conftest import make_game


D1 = "99p99s112233z4z5z6z"
D2 = "77p88p99s1122z33z4z"
D3 = "99p556677z112p9m5z"
D3B = "99p556677z9m5z7z7z6z"


SCHEMA_KEYS = [
    "game_id", "source", "seat", "round", "honba", "riichi_sticks",
    "wall_left", "dora_indicators", "scores", "oya", "hand", "melds",
    "discards", "n_kan", "riichi_declared", "last_event", "legal_actions",
    "label",
]


class TestSchemaCompliance:
    def test_all_schema_keys_present(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3],
                      draws=["1p"], dora=["4z"], first_draw=False)
        obs = g.state.get_observation(0)
        for k in SCHEMA_KEYS:
            assert k in obs, "missing key %s" % k

    def test_field_types(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3],
                      draws=["1p"], dora=["4z"], first_draw=False)
        obs = g.state.get_observation(0)
        assert isinstance(obs["game_id"], str)
        assert obs["source"] == "engine"
        assert isinstance(obs["seat"], int)
        assert isinstance(obs["round"], int)
        assert isinstance(obs["honba"], int)
        assert isinstance(obs["riichi_sticks"], int)
        assert isinstance(obs["wall_left"], int)
        assert isinstance(obs["dora_indicators"], list)
        assert isinstance(obs["scores"], list) and len(obs["scores"]) == 4
        assert isinstance(obs["hand"], list)
        assert isinstance(obs["melds"], list) and len(obs["melds"]) == 4
        assert isinstance(obs["discards"], list) and len(obs["discards"]) == 4
        assert isinstance(obs["n_kan"], int)
        assert isinstance(obs["riichi_declared"], list)
        assert isinstance(obs["last_event"], dict)
        assert isinstance(obs["legal_actions"], dict)

    def test_hand_contains_14_tiles_after_draw(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3],
                      draws=["1p"], dora=["4z"], first_draw=False)
        obs = g.state.get_observation(0)
        assert len(obs["hand"]) == 14
        assert g.drawn_tile in obs["hand"]

    def test_wall_left_and_dora(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3],
                      draws=["1p"], dora=["4z"], first_draw=False)
        obs = g.state.get_observation(0)
        assert obs["wall_left"] == 69
        assert obs["dora_indicators"] == g.dora_indicators

    def test_discards_record_format(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3B],
                      draws=["1p"], dora=["4z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        obs = g.state.get_observation(0)
        d = obs["discards"][0][-1]
        assert set(d.keys()) == {"tile", "tsumogiri", "riichi"}
        assert d["tsumogiri"] is True
        assert d["riichi"] is False

    def test_legal_actions_keys(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3],
                      draws=["1p"], dora=["4z"], first_draw=False)
        la = g.state.get_observation(0)["legal_actions"]
        for k in ("discard", "riichi", "chow", "pon", "kan", "ron", "tsumo"):
            assert k in la

    def test_observation_for_other_seat_has_empty_legal(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3],
                      draws=["1p"], dora=["4z"], first_draw=False)
        obs = g.state.get_observation(1)
        assert obs["legal_actions"]["discard"] == []
        assert obs["legal_actions"]["ron"] is False

    def test_observation_seat_field_is_decision_maker(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3B],
                      draws=["1p", "4z"], dora=["4z"], first_draw=False)
        g.step({"type": "discard", "tile": g.drawn_tile})
        assert g.turn == 1
        assert g.state.get_observation()["seat"] == 1

    def test_round_index_minus_one_in_west_rounds(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3],
                      draws=["1p"], dora=["4z"], first_draw=False, round_idx=8)
        assert g.state.get_observation(0)["round"] == -1


class TestDecisionRecords:
    def test_records_collected_with_labels(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3],
                      draws=["1p"], dora=["4z"], first_draw=False)
        g.record_decisions = True
        tile = g.legal_actions(0)["discard"][0]
        g.step({"type": "discard", "tile": tile})
        assert len(g.records) == 1
        rec = g.records[0]
        assert rec["label"] == {"type": "discard", "tile": tile}
        assert rec["seat"] == 0
        assert "legal_actions" in rec

    def test_pass_not_recorded(self):
        hands = ["123m123p123s456m1z", "11z99p99s567m567m9m",
                 "2288m6688p99s22z4z", "111333444m22p55s"]
        g = make_game(hands, draws=["3z", "1z"], dora=["4z"], first_draw=False)
        g.record_decisions = True
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "pass"})
        assert len(g.records) == 2  # two discards, no pass

    def test_ron_record_label(self):
        hands = ["123m123p123s456m1z", "11z99p99s567m567m9m",
                 "2288m6688p99s22z4z", "111333444m22p55s"]
        g = make_game(hands, draws=["3z", "1z"], dora=["4z"], first_draw=False)
        g.record_decisions = True
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "discard", "tile": g.drawn_tile})
        g.step({"type": "ron"})
        assert g.records[-1]["label"] == {"type": "ron"}

    def test_riichi_record_label(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3],
                      draws=["1p"], dora=["4z"], first_draw=False)
        g.record_decisions = True
        tile = next(t for t in g.legal_actions(0)["riichi"] if t == g.drawn_tile)
        g.step({"type": "riichi", "tile": tile})
        assert g.records[-1]["label"] == {"type": "riichi", "tile": tile}


class TestExplainBlock:
    def test_explain_included_on_request(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3],
                      draws=["1p"], dora=["4z"], first_draw=False)
        obs = g.state.get_observation(0, include_explain=True)
        assert "explain" in obs
        exp = obs["explain"]
        for k in ("shanten", "ukeire", "safety", "yaku_expectation"):
            assert k in exp, "missing explain key %s" % k

    def test_explain_absent_by_default(self):
        g = make_game(["123m123p123s456m9m", D1, D2, D3],
                      draws=["1p"], dora=["4z"], first_draw=False)
        assert "explain" not in g.state.get_observation(0)
