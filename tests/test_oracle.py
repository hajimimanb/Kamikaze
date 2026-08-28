# -*- coding: utf-8 -*-
"""Independent oracle cross-validation against mjai.app (Rust reference).

Every round of full hanchan games played by the native mjai arena
(mjai.mlibriichi, the Rust mjai.app implementation) is replayed with the
local RiichiGame engine: same hands, same wall, same decisions. The round
outcome (winner, losers, score deltas, tenpai settlement) must match the
oracle exactly, which validates yaku/han/fu/score computation, furiten,
kan dora timing and ryuukyoku settlement independently of the mahjong lib.

The oracle logs are cached under fixtures/oracle/ (deterministic arena +
deterministic agents). Set ORACLE_LIVE=1 to regenerate them against the
real arena.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle_util import get_oracle_log, split_rounds, replay_round  # noqa: E402

SEEDS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]


def _round_kind(revents):
    horas = [e for e in revents if e["type"] == "hora"]
    if horas:
        return "ron" if horas[0]["target"] != horas[0]["actor"] else "tsumo"
    calls = sum(1 for e in revents if e["type"] in ("chi", "pon", "daiminkan", "kakan", "ankan"))
    reaches = sum(1 for e in revents if e["type"] == "reach")
    return "draw_c%d_r%d" % (calls, reaches)


def _collect_cases():
    cases = []
    for seed in SEEDS:
        events = get_oracle_log(seed)
        for i, revents in enumerate(split_rounds(events)):
            cases.append(pytest.param(seed, i, id="seed%02d_r%02d_%s" % (seed[0], i, _round_kind(revents))))
    return cases


CASES = _collect_cases()


def test_oracle_coverage_meets_target():
    """The captain's target: at least 30 oracle-channel rule tests."""
    assert len(CASES) >= 30, "expected >=30 oracle round tests, got %d" % len(CASES)


@pytest.mark.parametrize("seed,round_idx", CASES)
def test_oracle_round_matches_native_engine(seed, round_idx):
    events = get_oracle_log(seed)
    revents = split_rounds(events)[round_idx]
    r = replay_round(revents)
    assert r["ok"], "oracle mismatch: %s (ours=%s oracle=%s, result=%s)" % (
        r.get("error"), r["our_deltas"], r["oracle_deltas"], r.get("our_result"))
