# -*- coding: utf-8 -*-
"""Unit tests for the feature encoder + mode switch (captain supplement)."""
import numpy as np
import pytest

from model.features import (
    VALID_MODES, action_availability, build_features, channel_layout,
    channel_names, channel_scale_max, discard_mask, feature_channels,
    group_summary, indicator_to_dora_kind, label_kind, save_channel_layout,
    score_bucket_col, tile_to_kind,
)


def base_obs():
    return {
        "game_id": "t", "source": "test", "seat": 1, "round": 3, "honba": 1,
        "riichi_sticks": 1, "wall_left": 40, "dora_indicators": [0],
        "scores": [25000, 30000, 20000, 25000], "oya": 0,
        "hand": [16, 17, 18, 19, 0, 4, 36, 40, 72, 108, 109, 110, 120],
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "n_kan": 0, "riichi_declared": [False] * 4,
        "last_event": {"type": "draw", "seat": 1, "tile": 120},
        "legal_actions": {"discard": [16, 17, 0, 4, 36, 40, 72, 108, 109, 110, 120],
                          "riichi": [], "chow": [], "pon": [], "kan": [],
                          "ron": False, "tsumo": False},
        "label": {"type": "discard", "tile": 120},
    }


def test_mode_validation():
    with pytest.raises(ValueError):
        build_features(base_obs(), mode="nope")
    with pytest.raises(ValueError):
        channel_names("bogus")


def test_channel_counts():
    c_simple = feature_channels("simple")
    c_full = feature_channels("full")
    assert c_simple == 17, c_simple   # hand5 + dora2 + basic10
    assert c_full == 126, c_full      # Suphx-style expanded baseline
    assert c_simple < c_full
    assert len(channel_names("simple")) == c_simple
    assert len(channel_names("full")) == c_full


def test_simple_excludes_field_groups():
    gs = group_summary("simple")
    assert not {"discards", "melds", "state"} & set(gs), gs
    gs_full = group_summary("full")
    assert "discards" in gs_full and "melds" in gs_full and "state" in gs_full
    # simple mode must be invariant to discards/melds (curriculum stage 1)
    obs = base_obs()
    a = build_features(obs, mode="simple")
    obs2 = dict(obs)
    obs2["discards"] = [[{"tile": 50, "tsumogiri": True, "riichi": False}], [], [], []]
    obs2["melds"] = [[{"type": "pon", "tiles": [20, 21, 23], "from": 1, "red": False}],
                     [], [], []]
    obs2["riichi_declared"] = [True, False, False, False]
    obs2["n_kan"] = 2
    b = build_features(obs2, mode="simple")
    assert np.array_equal(a, b)
    # ... but full mode sees the difference
    c = build_features(obs, mode="full")
    d = build_features(obs2, mode="full")
    assert not np.array_equal(c, d)


def test_hand_channels():
    obs = base_obs()
    # tile136 ids: 16-19 = 5m x4 (16 is red), 0,1 = 1m x2, 36 = 9p
    obs["hand"] = [16, 17, 18, 19, 0, 1, 36]
    x = build_features(obs, mode="simple")[:, :, 0]
    names = channel_names("simple")
    idx = {n: i for i, n in enumerate(names)}
    assert x[idx["hand_count4"], 4] == 1.0   # 4 copies of 5m
    assert x[idx["hand_count2"], 0] == 1.0   # 2 copies of 1m
    assert x[idx["hand_count1"], 0] == 1.0   # >=1 copy of 1m
    assert x[idx["hand_red5"], 4] == 1.0     # red 5m in hand
    assert x[idx["hand_red5"], 13] == 0.0


def test_dora_channels():
    obs = base_obs()
    # tile136 ids: 0,0 = 1m x2 -> 2m dora x2; 120 = N wind -> E wind dora
    obs["dora_indicators"] = [0, 0, 120]
    x = build_features(obs, mode="simple")[:, :, 0]
    names = channel_names("simple")
    idx = {n: i for i, n in enumerate(names)}
    assert x[idx["dora_indicators"], 0] == 2.0
    assert x[idx["dora_indicators"], 30] == 1.0
    assert x[idx["dora_kinds"], 1] == 2.0    # 2m is dora (2 indicators)
    assert x[idx["dora_kinds"], 27] == 1.0   # E wind is dora
    # indicator mapping table spot checks
    assert indicator_to_dora_kind(8) == 0    # 9m -> 1m
    assert indicator_to_dora_kind(17) == 9   # 9p -> 1p
    assert indicator_to_dora_kind(26) == 18  # 9s -> 1s
    assert indicator_to_dora_kind(30) == 27  # N -> E
    assert indicator_to_dora_kind(33) == 31  # chun -> haku


def test_basic_channels():
    obs = base_obs()  # seat=1 oya=0 round=3 -> seat wind = W(29), preval E(27)
    x = build_features(obs, mode="simple")[:, :, 0]
    names = channel_names("simple")
    idx = {n: i for i, n in enumerate(names)}
    assert x[idx["seat_wind"], 28] == 1.0   # seat1, oya0 -> South (kind 28)
    assert x[idx["prevalent_wind"], 27] == 1.0
    assert x[idx["is_oya"], 0] == 0.0
    assert x[idx["round"], 0] == pytest.approx(3 / 7)
    assert x[idx["honba"], 0] == pytest.approx(0.25)
    assert x[idx["wall_left"], 0] == pytest.approx(40 / 70)
    assert x[idx["own_score_bucket"], score_bucket_col(30000)] == 1.0
    assert x[idx["rank"], 0] == pytest.approx(0.0)  # 30000 is top score
    assert x[idx["last_draw"], 120 // 4] == 1.0


def test_discard_mask_and_availability():
    obs = base_obs()
    mask = discard_mask(obs)
    assert mask[16 // 4] and mask[120 // 4]
    assert not mask[33]  # no dragon in hand
    obs2 = dict(obs)
    obs2["legal_actions"] = None
    assert discard_mask(obs2).all()  # inference fallback
    av = action_availability(obs)
    assert av["discard"] is True and av["riichi"] is False


def test_label_kind():
    assert label_kind({"label": {"type": "discard", "tile": 120}}) == 30
    assert label_kind({"label": {"type": "riichi", "tile": 16}}) == 4
    assert label_kind({"label": {"type": "chow", "tiles": [0, 4, 8]}}) == 0
    with pytest.raises(KeyError):
        label_kind({"label": {"type": "ron"}})


def test_tile_helpers():
    assert tile_to_kind(16) == 4
    assert tile_to_kind(135) == 33
    assert score_bucket_col(0) == 0
    assert score_bucket_col(45000) == 12
    assert score_bucket_col(999999) == 33


def test_channel_layout_json(tmp_path):
    """Registry exposes names + semantics + quant scale; JSON dump works."""
    layout = channel_layout("full")
    assert len(layout) == 126
    assert layout[0]["name"] == "hand_count1" and layout[0]["group"] == "hand"
    assert all(set(e) >= {"index", "group", "name", "description", "quant_max"}
               for e in layout)
    # simple channels are the first 17 of full (shard slicing invariant)
    names_full = channel_names("full")
    names_simple = channel_names("simple")
    assert names_full[:17] == names_simple
    # quant scale: only the two count dora channels exceed 1
    scale = channel_scale_max(names_full)
    assert (scale > 1).sum() == 2
    out = tmp_path / "channel_layout.json"
    save_channel_layout(str(out))
    doc = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert doc["modes"]["full"]["n_channels"] == 126
    assert doc["modes"]["simple"]["n_channels"] == 17

