# -*- coding: utf-8 -*-
"""评测工具链自测 (eval-tooling-developer, 任务 t9 自测要求).

覆盖: 类别谓词/分类器、推理契约与基线、场况生成与验证、通道B 抽样去重与
对照/差异表、专家报表、离线指标正确性、留出集过滤。
不依赖天凤大数据 (用合成 records), 也不依赖引擎具体版本 (场况用已生成样例)。
"""
import gzip
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from riichi.tiles import parse_mpsz, tile136
from riichi.explain import calc_shanten

from eval.categories import (
    CATEGORIES,
    classify_point,
    light_explain,
    obs_digest,
    primary_category,
)
from eval.inference import (
    heuristic_baseline_inference,
    random_baseline_inference,
    sanity_check_decision,
)
from eval.similarity_eval import (
    exact_action_equal,
    iter_records,
    load_holdout_ids,
    sample_points,
)
from eval.offline_metrics import compute_metrics
from eval import expert_report, scenario_gen

SCHEMA_KEYS = ["game_id", "source", "seat", "round", "honba", "riichi_sticks", "wall_left",
               "dora_indicators", "scores", "oya", "hand", "melds", "discards", "n_kan",
               "riichi_declared", "last_event", "legal_actions", "label"]
LA_KEYS = ["discard", "riichi", "chow", "pon", "kan", "ron", "tsumo"]


def synth_obs(hand_mpsz, seat=0, label_tile=None, riichi=None, wall_left=60, scores=None,
              dora=None, discards=None, legal_extra=None, game_id="synth-1", last_draw=True):
    hand = parse_mpsz(hand_mpsz)
    if label_tile is None:
        label_tile = hand[0]
    la = {"discard": list(hand), "riichi": [], "chow": [], "pon": [],
          "kan": [], "ron": False, "tsumo": False}
    la.update(legal_extra or {})
    drawn = None
    if last_draw and len(hand) == 14:
        drawn = hand[-1]
    return {
        "game_id": game_id, "source": "synth", "seat": seat, "round": 0, "honba": 0,
        "riichi_sticks": 0, "wall_left": wall_left,
        "dora_indicators": dora or [parse_mpsz("1z")[0]],
        "scores": scores or [25000] * 4, "oya": 0, "hand": hand,
        "melds": [[], [], [], []],
        "discards": discards or [[], [], [], []],
        "n_kan": 0, "riichi_declared": riichi or [False] * 4,
        "last_event": {"type": "draw", "seat": seat, "tile": drawn} if drawn is not None
        else {"type": "debug", "seat": -1, "tile": -1},
        "legal_actions": la,
        "label": {"type": "discard", "tile": label_tile},
    }


def write_records(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 类别谓词

class TestCategories:
    def test_defense_predicate(self):
        obs = synth_obs("334455m7788p1z2z3z6s", riichi=[False, True, False, False],
                        discards=[[], [{"tile": parse_mpsz("1z")[0], "tsumogiri": False, "riichi": True}],
                                  [], []])
        assert 1 in classify_point(obs)

    def test_attack_predicate(self):
        obs = synth_obs("123m456m79p2379s1z6s",
                        discards=[[], [{"tile": parse_mpsz("8s")[0], "tsumogiri": False, "riichi": False}] * 4, [], []])
        assert 2 in classify_point(obs)

    def test_riichi_predicate(self):
        obs = synth_obs("123m456m789p22388s", legal_extra={"riichi": [parse_mpsz("2s")[0]]})
        assert 3 in classify_point(obs)

    def test_call_predicate(self):
        obs = synth_obs("112233m456p12s12z", legal_extra={"chow": [{"tiles": parse_mpsz("123s")}]})
        assert 4 in classify_point(obs)

    def test_big_hand_predicate(self):
        obs = synth_obs("19m19p19s123456z7s7z")
        assert 5 in classify_point(obs)
        obs2 = synth_obs("2233445566m789p9m")
        assert 5 in classify_point(obs2)

    def test_score_predicate(self):
        obs = synth_obs("334455m7788p1z2z3z6s", scores=[42000, 20000, 18000, 12000])
        assert 6 in classify_point(obs)

    def test_bust_predicate(self):
        obs = synth_obs("234m567m99p2356s1z4p", scores=[1200, 24000, 22000, 20000])
        assert 7 in classify_point(obs)

    def test_wall_predicate(self):
        obs = synth_obs("123m456m789p2388s3s", wall_left=15)
        assert 8 in classify_point(obs)

    def test_dora_predicate(self):
        obs = synth_obs("3334455m7788p99s6s", dora=[parse_mpsz("2m")[0]])  # 指示2m->宝牌3m x3
        assert 9 in classify_point(obs)

    def test_aka_predicate(self):
        obs = synth_obs("11234567m0569p2z6m")  # 含赤5p
        assert 10 in classify_point(obs)

    def test_primary_category_order(self):
        obs = synth_obs("234m567m99p2356s1z4p", scores=[1200, 24000, 22000, 20000])
        assert primary_category(obs) == 6  # 点数状况先于濒临飞人

    def test_obs_digest(self):
        obs = synth_obs("334455m7788p1z2z3z6s")
        d = obs_digest(obs)
        assert d["hand"] == "334455m7788p6s123z" or "6s" in d["hand"]
        assert d["wall_left"] == 60


# ------------------------------------------------------------------ 推理契约

class TestInference:
    def test_heuristic_respects_legal_mask(self):
        obs = synth_obs("123m456m789p22388s")
        d = heuristic_baseline_inference(obs, rng=__import__("random").Random(0))
        ok, why = sanity_check_decision(obs, d)
        assert ok, why
        assert set(d) >= {"action", "top_k", "factors", "rationale"}
        assert d["rationale"]

    def test_heuristic_deterministic(self):
        obs = synth_obs("123m456m789p22388s")
        r1 = heuristic_baseline_inference(obs, rng=__import__("random").Random(7))["action"]
        r2 = heuristic_baseline_inference(obs, rng=__import__("random").Random(7))["action"]
        assert r1 == r2

    def test_random_baseline_uniform(self):
        obs = synth_obs("123m456m789p22388s")
        d = random_baseline_inference(obs, rng=__import__("random").Random(1))
        ok, _ = sanity_check_decision(obs, d)
        assert ok
        n = len(obs["legal_actions"]["discard"])
        assert d["probs_discard"] and abs(sum(d["probs_discard"].values()) - 1.0) < 1e-6
        assert abs(d["probs_discard"][obs["hand"][0]] - 1.0 / n) < 1e-9

    def test_exact_action_equal(self):
        assert exact_action_equal({"type": "discard", "tile": 5}, {"type": "discard", "tile": 5})
        assert not exact_action_equal({"type": "discard", "tile": 5}, {"type": "riichi", "tile": 5})
        assert exact_action_equal({"type": "pon", "tiles": [1, 2, 3]}, {"type": "pon", "tiles": [3, 2, 1]})


# -------------------------------------------------------------- 场况生成 (样例)

class TestScenarioGen:
    def test_committed_scenarios_valid(self):
        import glob
        files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "..", "eval", "scenarios", "cat*.json")))
        assert len(files) >= 20
        by_cat = {}
        for f in files:
            sc = json.load(open(f, encoding="utf-8"))
            obs = sc["obs"]
            assert sc["predicate_verified"]
            assert sc["category"] in CATEGORIES
            for k in SCHEMA_KEYS:
                assert k in obs, (f, k)
            assert set(LA_KEYS) <= set(obs["legal_actions"].keys())
            assert sc["category"] in classify_point(obs)
            assert set(sc["scoring"]) == {"action_quality", "explain_quality", "notes"}
            by_cat[sc["category"]] = by_cat.get(sc["category"], 0) + 1
        assert len(by_cat) == 10
        assert all(v >= 1 for v in by_cat.values())

    def test_live_generation_smoke(self, tmp_path):
        import random
        out = str(tmp_path)
        for cat in (1, 3, 4, 9):
            sc, obs, expl = scenario_gen.generate_scenario(cat, random.Random(100 + cat))
            assert sc["category"] == cat
            assert cat in classify_point(obs, light_explain(obs, with_yaku=True))

    def test_fallback_fixtures(self, tmp_path):
        out = str(tmp_path / "scen")
        idx = scenario_gen.fallback_from_fixtures(out)
        assert len(idx["scenarios"]) >= 20


# ------------------------------------------------------------------ 通道B 抽样

class TestSimilarity:
    def _synth_corpus(self, tmp_path, n_games=8, points_per_game=6):
        recs = []
        hands = ["123m456m789p22388s", "334455m7788p1z2z3z6s", "234m567m99p2356s1z4p"]
        for g in range(n_games):
            gid = "synth-game-%02d" % g
            for p in range(points_per_game):
                h = hands[p % len(hands)]
                obs = synth_obs(h, seat=p % 4, game_id=gid, wall_left=60 - p * 5)
                recs.append(obs)
        path = os.path.join(str(tmp_path), "records.jsonl")
        write_records(path, recs)
        return path

    def test_sample_dedup_and_quota(self, tmp_path):
        path = self._synth_corpus(tmp_path)
        # 数据充足时 (n=16 <= 8局x2) 每局上限生效
        picked, s1, s2 = sample_points([path], n=16, seed=1, per_game_cap=2, per_player_cap=1)
        assert len(picked) == 16
        per_game = {}
        for p in picked:
            per_game[p["game_id"]] = per_game.get(p["game_id"], 0) + 1
        assert max(per_game.values()) <= 2
        # 请求超出上限时放宽每局数, 但每(局,座位)去重不变
        picked2, s1, s2 = sample_points([path], n=24, seed=1, per_game_cap=2, per_player_cap=1)
        assert len(picked2) == 24
        players = {}
        for p in picked2:
            key = (p["game_id"], p["seat"])
            assert key not in players
            players[key] = True

    def test_holdout_filter(self, tmp_path):
        path = self._synth_corpus(tmp_path, n_games=4, points_per_game=2)
        holdout = os.path.join(str(tmp_path), "holdout.txt")
        with open(holdout, "w", encoding="utf-8") as f:
            f.write("synth-game-00\nsynth-game-02\n")
        ids = load_holdout_ids(holdout)
        got = [r["game_id"] for r in iter_records([path], ids)]
        assert set(got) <= {"synth-game-00", "synth-game-02"}
        assert len(got) == 4


# ------------------------------------------------------------------ 离线指标

class TestOfflineMetrics:
    def _corpus(self, tmp_path, n=24):
        recs = []
        hands = ["123m456m789p22388s", "334455m7788p1z2z3z6s"]
        for i in range(n):
            gid = "m-game-%02d" % (i // 4)
            h = hands[i % 2]
            r = synth_obs(h, seat=i % 4, game_id=gid)
            recs.append(r)
        path = os.path.join(str(tmp_path), "m.jsonl")
        write_records(path, recs)
        return path

    def test_copy_human_perfect(self, tmp_path):
        path = self._corpus(tmp_path)
        m = compute_metrics([path], inference_fn=lambda obs: {
            "action": dict(obs["label"]), "top_k": [{"action": dict(obs["label"]), "p": 1.0, "factors": {}}],
            "factors": {}, "rationale": "copy", "probs_discard": {}}, cap=100, classify=False)
        assert m["consistency"]["action_exact"] == 1.0
        assert m["consistency"]["discard_kind_top1"] == 1.0
        assert m["n_illegal"] == 0

    def test_random_baseline_metrics(self, tmp_path):
        path = self._corpus(tmp_path)
        m = compute_metrics([path], inference_fn=random_baseline_inference, cap=100, classify=False)
        assert m["baselines"]["random_top1_expected"] > 0
        assert 0 <= m["consistency"]["discard_kind_top1"] <= 1
        assert m["entropy"]["mean"] > 0

    def test_illegal_decision_counted(self, tmp_path):
        path = self._corpus(tmp_path, n=4)
        m = compute_metrics([path], inference_fn=lambda obs: {
            "action": {"type": "discard", "tile": 999}, "top_k": [], "factors": {},
            "rationale": "x", "probs_discard": {}}, cap=100, classify=False)
        assert m["n_illegal"] == 4


# ------------------------------------------------------------------ 专家报表

class TestExpertReport:
    def test_scenario_sheet(self, tmp_path):
        scen_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "scenarios")
        out = str(tmp_path)
        rows = expert_report.scenario_sheet(scen_dir, out)
        assert len(rows) >= 20
        assert os.path.exists(os.path.join(out, "scenarios_sheet.md"))
        assert os.path.exists(os.path.join(out, "scenarios_sheet.csv"))
        import csv as _csv
        with open(os.path.join(out, "scenarios_sheet.csv"), encoding="utf-8") as f:
            rd = list(_csv.reader(f))
        assert rd[0][-3:] == ["action_quality(1-5)", "explain_quality(1-5)", "notes"]
        assert len(rd) == len(rows) + 1

    def test_diff_sheet(self, tmp_path):
        sim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "outputs", "similarity")
        if not os.path.exists(os.path.join(sim_dir, "diff_annotations.csv")):
            pytest.skip("similarity outputs not generated")
        out = str(tmp_path)
        body = expert_report.diff_sheet(sim_dir, out)
        assert len(body) >= 1
        assert os.path.exists(os.path.join(out, "similarity_diff_sheet.md"))


# ------------------------------------------------------------------ 端到端 (小数据)

class TestEndToEnd:
    def test_similarity_run_on_synth(self, tmp_path):
        from eval.similarity_eval import run_similarity
        recs = []
        hands = ["123m456m789p22388s", "334455m7788p1z2z3z6s", "234m567m99p2356s1z4p",
                 "19m19p19s123456z7s7z"]
        for g in range(10):
            gid = "e2e-game-%02d" % g
            for p in range(8):
                obs = synth_obs(hands[p % 4], seat=p % 4, game_id=gid)
                obs["scores"] = [42000, 25000, 20000, 9000] if p % 3 == 0 else [25000] * 4
                recs.append(obs)
        path = os.path.join(str(tmp_path), "e2e.jsonl")
        write_records(path, recs)
        out = str(tmp_path / "sim")
        summary = run_similarity([path], n=40, out_dir=out, seed=3)
        assert summary["n_sampled"] == 40
        assert os.path.exists(os.path.join(out, "compare.csv"))
        assert os.path.exists(os.path.join(out, "diff_annotations.csv"))
        assert os.path.exists(os.path.join(out, "summary.md"))
        import csv as _csv
        with open(os.path.join(out, "compare.csv"), encoding="utf-8") as f:
            rd = list(_csv.reader(f))
        assert len(rd) == 41  # header + 40
        with open(os.path.join(out, "diff_annotations.csv"), encoding="utf-8") as f:
            dd = list(_csv.reader(f))
        assert dd[0][0].startswith("# 判定值")
        n_diff = len(dd) - 2
        assert n_diff == summary["expert_diff"]["n_diff"]
