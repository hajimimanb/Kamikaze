# -*- coding: utf-8 -*-
"""L0-L4 分层测评工具链自测 (eval-tooling-developer, t9 扩展)。

覆盖: runner 统一接口、baselines 对手池、sanity_smoke(L0)、offline_metrics(L1
基线区间/过拟合标记)、ev_eval(L2 状态重建+rollout)、league_eval(L3)、run_eval
汇总。L2/L3 用极小规模 (秒级), 全量参数见各模块 CLI。
"""
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from riichi.tiles import parse_mpsz

from eval.runner import build_runner, ModelRunner
from eval.baselines import BASELINES, get_baseline
from eval.inference import sanity_check_decision
from eval.sanity_smoke import build_smoke_cases, check_factor_consistency, run_smoke
from eval.offline_metrics import compute_metrics
from eval.league_eval import play_game, run_pool, elo_from_rank
from eval.run_eval import write_html

RECORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "data", "processed", "tenhou", "records-20260801.jsonl.gz")
HOLDOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "data", "processed", "tenhou", "splits", "eval_holdout_games.txt")


def synth_obs(hand_mpsz, seat=0, label_tile=None, riichi=None, wall_left=None, scores=None,
              dora=None, discards=None, game_id="layers-1", legal_extra=None):
    hand = parse_mpsz(hand_mpsz)
    la = {"discard": list(hand), "riichi": [], "chow": [], "pon": [],
          "kan": [], "ron": False, "tsumo": False}
    la.update(legal_extra or {})
    if wall_left is None:
        wall_left = 65
    return {
        "game_id": game_id, "source": "synth", "seat": seat, "round": 0, "honba": 0,
        "riichi_sticks": 0, "wall_left": wall_left,
        "dora_indicators": dora or [parse_mpsz("1z")[0]],
        "scores": scores or [25000] * 4, "oya": 0, "hand": hand,
        "melds": [[], [], [], []],
        "discards": discards or [[], [], [], []],
        "n_kan": 0, "riichi_declared": riichi or [False] * 4,
        "last_event": {"type": "draw", "seat": seat, "tile": hand[-1]},
        "legal_actions": la,
        "label": {"type": "discard", "tile": label_tile or hand[0]},
    }


class TestRunner:
    def test_all_runners_legal_and_batched(self):
        obs_list = [synth_obs(h, seat=i % 4, game_id="r-%d" % i)
                    for i, h in enumerate(["123m456m789p22388s", "334455m7788p123z6s",
                                           "234567m99p2356s1z", "11234567m0569p2z6m"])]
        for name in ("random", "heuristic", "tsumogiri", "shanten_greedy"):
            r = build_runner(name, seed=0)
            ds = r.predict(obs_list)
            assert len(ds) == len(obs_list)
            for o, d in zip(obs_list, ds):
                ok, why = sanity_check_decision(o, d)
                assert ok, (name, why, d["action"])

    def test_baseline_registry(self):
        assert set(BASELINES) == {"random", "tsumogiri", "shanten_greedy"}
        for name in BASELINES:
            assert isinstance(get_baseline(name), ModelRunner)

    def test_tsumogiri_discards_drawn(self):
        obs = synth_obs("123m456m789p22388s")
        r = build_runner("tsumogiri")
        d = r.predict([obs])[0]
        assert d["action"]["tile"] == obs["last_event"]["tile"]

    def test_shanten_greedy_never_worsens(self):
        from riichi.explain import calc_shanten
        obs = synth_obs("123m456m789p2388s3s")
        r = build_runner("shanten_greedy", seed=0)
        d = r.predict([obs])[0]
        hand = obs["hand"]
        sh0 = calc_shanten(hand)
        sh_after = calc_shanten([t for t in hand if t != d["action"]["tile"]])
        assert sh_after <= sh0


class TestSanitySmoke:
    def test_cases_count(self):
        cases = build_smoke_cases(0)
        assert len(cases) >= 20

    def test_smoke_pass_on_heuristic(self):
        s = run_smoke(build_runner("heuristic", seed=0), seed=0)
        assert s["pass"] and s["n_illegal"] == 0

    def test_smoke_legal_only_on_all_runners(self):
        for name in ("random", "tsumogiri", "shanten_greedy"):
            s = run_smoke(build_runner(name, seed=0), seed=0)
            assert s["pass"], name

    def test_factor_consistency_catches_bad(self):
        obs = synth_obs("123m456m789p22388s")
        good = build_runner("heuristic", seed=0).predict([obs])[0]
        ok, probs = check_factor_consistency(good, obs)
        assert ok, probs
        # 破坏: factors.shanten 与规则重算不符
        bad = dict(good)
        bad["factors"] = dict(good["factors"])
        bad["factors"]["shanten"] = 99
        ok2, probs2 = check_factor_consistency(bad, obs)
        assert not ok2
        # 破坏: rationale 引用的数字不在 factors
        bad3 = dict(good)
        bad3["rationale"] = "切牌: 向听777, 进张888张"
        ok3, _ = check_factor_consistency(bad3, obs)
        assert not ok3


class TestOfflineL1:
    def test_overfit_flag_on_perfect_copy(self, tmp_path):
        path = os.path.join(str(tmp_path), "l1.jsonl")
        NL = chr(10)
        with open(path, "w", encoding="utf-8") as f:
            for i in range(12):
                o = synth_obs("123m456m789p22388s", game_id="o-%d" % i)
                f.write(json.dumps(o, ensure_ascii=False) + NL)
        m = compute_metrics([path], inference_fn=lambda obs: {
            "action": dict(obs["label"]), "top_k": [{"action": dict(obs["label"]), "p": 1.0, "factors": {}}],
            "factors": {}, "rationale": "copy", "probs_discard": {}},
            cap=100, classify=False)
        assert m["human_human_baseline"]["verdict"] == "above_baseline"
        assert m["overfit_warning"]

    def test_verdict_below(self, tmp_path):
        path = os.path.join(str(tmp_path), "l1b.jsonl")
        NL = chr(10)
        with open(path, "w", encoding="utf-8") as f:
            for i in range(12):
                o = synth_obs("123m456m789p22388s", game_id="ob-%d" % i)
                f.write(json.dumps(o, ensure_ascii=False) + NL)
        from eval.inference import random_baseline_inference
        m = compute_metrics([path], inference_fn=lambda obs: random_baseline_inference(obs, rng=random.Random(1)),
                            cap=100, classify=False)
        assert m["human_human_baseline"]["verdict"] == "below_baseline"


class TestEvEvalL2:
    def test_reconstruct_and_rollout(self):
        if not os.path.exists(RECORDS):
            pytest.skip("holdout records not available")
        from eval.ev_eval import reconstruct_game, rollout_from
        from eval.similarity_eval import iter_records
        from eval.runner import build_runner
        # records-202608* 本身即留出窗口 (>=2026-08-01 日期规则);
        # splits 清单在 data-engineer 重组期间可能滞后, 不依赖它
        for r in iter_records([RECORDS]):
            lab = r.get("label") or {}
            if lab.get("type") in ("discard", "riichi") and (r.get("legal_actions") or {}).get("discard"):
                g = reconstruct_game(r, 12345)
                assert g.phase == "draw" and g.turn == r["seat"]
                la = g.legal_actions(g.turn)
                assert len(la["discard"]) == 14
                delta, early = rollout_from(g, r["seat"], {"type": "discard", "tile": la["discard"][0]},
                                            build_runner("random", seed=0))
                assert isinstance(delta, int)
                break

    def test_tiny_ev_run(self):
        if not os.path.exists(RECORDS):
            pytest.skip("holdout records not available")
        from eval.ev_eval import run_ev_eval
        from eval.runner import build_runner
        s = run_ev_eval([RECORDS], n=2, candidates=2, rollouts=2, seed=5,
                        rollouter=build_runner("shanten_greedy", seed=0),
                        out=None, holdout_file=None)
        assert s["n_evaluated"] >= 1
        assert 0.0 <= s["ev_best_agrees_human"] <= 1.0


class TestLeagueL3:
    def test_play_game_terminates(self):
        agents = [build_runner("random", seed=i) for i in range(4)]
        res = play_game(agents, game_seed=99)
        assert 1 <= res["rank"] <= 4
        assert res["steps"] > 0

    def test_run_pool_metrics(self):
        runner = build_runner("shanten_greedy", seed=0)
        p = run_pool(runner, "random", 2, base_seed=7)
        assert p["games"] == 2
        assert p["anchored_elo"] > 0
        assert 0 <= p["win_rate"]
        assert len(p["anchored_elo_ci95"]) == 2

    def test_elo_formula(self):
        assert abs(elo_from_rank(2.5) - 1500) < 1e-6
        assert elo_from_rank(1.0) > 2000
        assert elo_from_rank(4.0) < 1000


class TestHumanBaseline:
    def _corpus(self, tmp_path):
        import json as _json
        recs = []
        hands = ["123m456m789p22388s", "334455m7788p123z6s"]
        NL = chr(10)
        for g in range(10):
            for s in range(2):
                hand = parse_mpsz(hands[s])
                o = synth_obs(hands[s], seat=s, game_id="hb-%02d" % g,
                              label_tile=hand[0] if s == 0 else hand[1])
                recs.append(o)
        p = os.path.join(str(tmp_path), "hb.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(_json.dumps(r, ensure_ascii=False) + NL)
        return p

    def test_measure_hand_pairs(self, tmp_path):
        from eval.human_baseline import measure_human_consistency
        path = self._corpus(tmp_path)
        s = measure_human_consistency([path], max_scan=1000, out=None)
        hand = s["levels"]["hand"]
        assert hand["n_pairs"] >= 1  # 同手牌不同局 -> 至少 1 对
        assert 0.0 <= hand["agreement"] <= 1.0

    def test_load_measured_prefers_hand(self, tmp_path):
        import json as _json
        from eval.human_baseline import load_measured_baseline
        p = os.path.join(str(tmp_path), "mb.json")
        _json.dump({"scanned": 100, "levels": {
            "hand": {"n_pairs": 200, "agreement": 0.79, "agreement_ci95": [0.76, 0.81]},
            "medium": {"n_pairs": 5000, "agreement": 0.17, "agreement_ci95": [0.16, 0.18]},
        }}, open(p, "w", encoding="utf-8"))
        m = load_measured_baseline(p)
        assert m["top1_range"] == [0.76, 0.81]
        assert "hand" in m["source"]


class TestRunEval:
    def test_report_html(self, tmp_path):
        report = {"generated_at": "t", "runner": "x", "quick": True, "overall": "PASS",
                  "levels": {"L0": {"status": "OK", "elapsed_s": 0.1,
                                    "summary": {"n_cases": 27, "pass": True}}}}
        p = os.path.join(str(tmp_path), "report.html")
        write_html(report, p)
        html = open(p, encoding="utf-8").read()
        assert "L0" in html and "PASS" in html
