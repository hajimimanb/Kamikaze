# -*- coding: utf-8 -*-
"""Validate a Kamikaze experiment config against the 7-block contract.

Pure-standard-library implementation (no ``jsonschema`` dependency) so it runs
in any environment, including a machine without torch/mahjong installed.

Usage::

    python configs/config_validate.py [config.json]

Exit code 0 = PASS, 1 = FAIL (CI-friendly). Catches the class of mistakes that
feedback1 section 27 calls out: seed stored as string, negative games, unknown
reward mode, opponent-pool probabilities that do not sum to 1, etc.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REQUIRED_TOP = [
    "environment", "model", "training", "reward",
    "opponent_pool", "evaluation", "runtime",
]

_ENV_BOOL_FIELDS = [
    "aka", "kuitan", "kuikae", "kiriage_mangan", "atamahane",
    "double_ron", "nagashi_mangan", "kyushu_kyuhai",
    "four_winds_ryuukyoku", "four_kans_ryuukyoku", "chankan_kokushi",
    "agari_yame", "west_extension", "double_yakuman",
]

_MODEL_ALLOWED_HEADS = {"riichi", "chow", "pon", "kan", "ron", "tsumo", "kyushu"}
_OPTIMIZERS = {"adamw", "adam", "sgd"}
_DENSE_REWARDS = {"phi_diff"}
_TERMINAL_REWARDS = {"settlement_pt"}
_SETTLEMENT_RULES = {"tenhou", "majsoul", "mleague"}
_POOL_STRATEGIES = {"self_hist_sl", "self_only", "self_hist"}
_PRIMARY_METRICS = {"mean_rank"}
_SECONDARY_METRICS = {"rank1_rate", "placement_score", "pt_per_100"}
_CI_METHODS = {"bootstrap"}


def _is_bool(v) -> bool:
    return isinstance(v, bool)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(config: dict) -> list:
    """Return a list of human-readable error strings (empty => valid)."""
    errs: list = []

    for key in REQUIRED_TOP:
        if key not in config:
            errs.append(f"missing top-level block: {key!r}")
    if errs:  # cannot proceed without full blocks
        return errs

    _env(config.get("environment"), errs)
    _model(config.get("model"), errs)
    _training(config.get("training"), errs)
    _reward(config.get("reward"), errs)
    _opponent_pool(config.get("opponent_pool"), errs)
    _evaluation(config.get("evaluation"), errs)
    _runtime(config.get("runtime"), errs)
    return errs


def _env(env, errs):
    for f in _ENV_BOOL_FIELDS:
        if f in env and not _is_bool(env[f]):
            errs.append(f"environment.{f} must be boolean, got {env[f]!r}")
    for f in ("start_score", "return_score"):
        if f in env and (not _is_int(env[f]) or env[f] < 0):
            errs.append(f"environment.{f} must be a non-negative int, got {env[f]!r}")


def _model(m, errs):
    if m.get("feature_mode") not in (None, "simple", "full"):
        errs.append(f"model.feature_mode must be 'simple'|'full', got {m.get('feature_mode')!r}")
    for f in ("channels", "n_blocks"):
        if f in m and (not _is_int(m[f]) or m[f] < (0 if f == "n_blocks" else 1)):
            errs.append(f"model.{f} must be a positive int, got {m[f]!r}")
    heads = m.get("binary_heads")
    if heads is not None:
        if not isinstance(heads, list) or not heads:
            errs.append(f"model.binary_heads must be a non-empty list, got {heads!r}")
        else:
            bad = [h for h in heads if h not in _MODEL_ALLOWED_HEADS]
            if bad:
                errs.append(f"model.binary_heads has unknown heads {bad}")
    for f in ("include_value", "use_event_attn"):
        if f in m and not _is_bool(m[f]):
            errs.append(f"model.{f} must be boolean, got {m[f]!r}")


def _training(t, errs):
    for f in ("games_per_epoch", "epochs", "eval_every", "eval_games",
              "save_every", "pool_k", "t5_games", "seed"):
        if f in t and (not _is_int(t[f]) or t[f] < (0 if f == "seed" else 1)):
            errs.append(f"training.{f} must be a non-negative int, got {t[f]!r}")
    if t.get("optimizer") not in _OPTIMIZERS:
        errs.append(f"training.optimizer must be one of {sorted(_OPTIMIZERS)}, got {t.get('optimizer')!r}")
    for f in ("lr", "lr_trunk_ratio", "clip", "clip_grad", "temperature"):
        if f in t and (not _is_num(t[f]) or t[f] <= 0):
            errs.append(f"training.{f} must be > 0, got {t[f]!r}")
    for f in ("gamma", "lam"):
        if f in t and (not _is_num(t[f]) or not (0 <= t[f] <= 1)):
            errs.append(f"training.{f} must be in [0,1], got {t[f]!r}")
    for f in ("entropy_coef", "vf_coef"):
        if f in t and (not _is_num(t[f]) or t[f] < 0):
            errs.append(f"training.{f} must be >= 0, got {t[f]!r}")


def _reward(r, errs):
    if r.get("dense") not in _DENSE_REWARDS:
        errs.append(f"reward.dense must be one of {sorted(_DENSE_REWARDS)}, got {r.get('dense')!r}")
    if r.get("terminal") not in _TERMINAL_REWARDS:
        errs.append(f"reward.terminal must be one of {sorted(_TERMINAL_REWARDS)}, got {r.get('terminal')!r}")
    if r.get("settlement_rule") not in _SETTLEMENT_RULES:
        errs.append(f"reward.settlement_rule must be one of {sorted(_SETTLEMENT_RULES)}, got {r.get('settlement_rule')!r}")
    for f in ("agari_bonus", "deal_in_penalty"):
        if f in r and not _is_bool(r[f]):
            errs.append(f"reward.{f} must be boolean, got {r[f]!r}")
    phi = r.get("phi")
    if phi is not None:
        for f in ("feat_dim", "hidden", "num_layers"):
            if f in phi and (not _is_int(phi[f]) or phi[f] < 1):
                errs.append(f"reward.phi.{f} must be a positive int, got {phi[f]!r}")


def _opponent_pool(op, errs):
    if op.get("strategy") not in _POOL_STRATEGIES:
        errs.append(f"opponent_pool.strategy must be one of {sorted(_POOL_STRATEGIES)}, got {op.get('strategy')!r}")
    for f in ("p_self", "p_past", "p_sl"):
        if f in op and (not _is_num(op[f]) or not (0 <= op[f] <= 1)):
            errs.append(f"opponent_pool.{f} must be in [0,1], got {op[f]!r}")
    for f in ("k", "past_cache_max"):
        if f in op and (not _is_int(op[f]) or op[f] < 1):
            errs.append(f"opponent_pool.{f} must be a positive int, got {op[f]!r}")
    s = op.get("p_self", 0) + op.get("p_past", 0) + op.get("p_sl", 0)
    if not math.isclose(s, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        errs.append(f"opponent_pool probabilities must sum to 1, got {s}")


def _evaluation(ev, errs):
    if ev.get("primary_metric") not in _PRIMARY_METRICS:
        errs.append(f"evaluation.primary_metric must be one of {sorted(_PRIMARY_METRICS)}, got {ev.get('primary_metric')!r}")
    sec = ev.get("secondary_metrics")
    if sec is not None:
        bad = [x for x in sec if x not in _SECONDARY_METRICS]
        if bad:
            errs.append(f"evaluation.secondary_metrics has unknown metrics {bad}")
    if ev.get("ci_method") not in _CI_METHODS:
        errs.append(f"evaluation.ci_method must be one of {sorted(_CI_METHODS)}, got {ev.get('ci_method')!r}")
    if "ci_level" in ev and (not _is_num(ev["ci_level"]) or not (0 < ev["ci_level"] < 1)):
        errs.append(f"evaluation.ci_level must be in (0,1), got {ev['ci_level']!r}")
    if "n_bootstrap" in ev and (not _is_int(ev["n_bootstrap"]) or ev["n_bootstrap"] < 100):
        errs.append(f"evaluation.n_bootstrap must be an int >= 100, got {ev['n_bootstrap']!r}")
    for stage in ("pilot", "full"):
        s = ev.get(stage)
        if s is not None:
            for f in ("seeds", "games_per_seed"):
                if f in s and (not _is_int(s[f]) or s[f] < 1):
                    errs.append(f"evaluation.{stage}.{f} must be a positive int, got {s[f]!r}")


def _runtime(p, errs):
    if not isinstance(p, dict):
        errs.append(f"runtime must be an object, got {type(p).__name__}")
        return
    for k, v in p.items():
        if not isinstance(v, str):
            errs.append(f"runtime.{k} must be a string, got {v!r}")


def main(argv):
    path = Path(argv[0]) if argv else Path(__file__).resolve().parent / "rl_v1.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"FAIL: config not found: {path}")
        return 1
    except json.JSONDecodeError as e:
        print(f"FAIL: invalid JSON in {path}: {e}")
        return 1

    if not isinstance(config, dict):
        print("FAIL: config root must be a JSON object")
        return 1

    errs = validate(config)
    if errs:
        print(f"FAIL: {len(errs)} validation error(s) in {path}")
        for e in errs:
            print(f"  - {e}")
        return 1

    print(f"PASS: {path} (7 blocks, {config.get('_meta', {}).get('name', '?')})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))