# -*- coding: utf-8 -*-
"""D0 — Regression / Sanity Suite（feedback2 §八 / §三十一）。

在改 D1–D10 的过程中防止已有系统被破坏。逐项自检，任何一项 FAIL 即非零退出（CI 友好）。

分级：
  - 纯 Python 层（无 GPU/无 torch/mahjong 也能跑）：paths 解析、config 校验、py_compile
    语法检查、eval CLI ``--help`` 分类诊断。
  - torch / mahjong 依赖层：import 主要模块、加载 RL checkpoint、建局、观测、合法动作、
    random rollout、现有测试套件。依赖缺失时标 **SKIP**（给出原因），不判 FAIL。

用法：:

    python tools/d0_regression.py

退出码：0 = 无 FAIL；1 = 有 FAIL（SKIP 不影响）。
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_MAHJONG = importlib.util.find_spec("mahjong") is not None

MAJOR_MODULES = [
    "agent.policy",
    "env.riichi_game",
    "model.model",
    "model.features",
    "model.net",
    "model.rl_reward",
    "riichi.tiles",
    "utils.paths",
]

CHECKPOINT = REPO / "models" / "kamikaze_rl_v1_fp16.pt"


# --------------------------------------------------------------------------- helpers

def _dep_missing() -> str:
    msgs = []
    if not HAS_TORCH:
        msgs.append("torch")
    if not HAS_MAHJONG:
        msgs.append("mahjong")
    return ", ".join(msgs)


def _import_modules():
    """Import all major modules; SKIP on torch/mahjong missing, FAIL on other errors."""
    imported = []
    for name in MAJOR_MODULES:
        __import__(name)
        imported.append(name)
    return "ok", "imported " + ", ".join(imported)


def _load_checkpoint():
    from agent.policy import RiichiPolicy
    p = RiichiPolicy(str(CHECKPOINT), seed=0, use_event_attn=False)
    n_params = p.model.num_parameters()
    return "ok", "loaded %s (params=%d)" % (CHECKPOINT.name, n_params)


def _create_game():
    from env.riichi_game import RiichiGame, RiichiConfig
    g = RiichiGame(RiichiConfig(), seed=0)
    return "ok", "phase=%s turn=%d" % (g.phase, g.turn)


def _create_observation():
    from env.riichi_game import RiichiGame, RiichiConfig
    g = RiichiGame(RiichiConfig(), seed=0)
    obs = g.state.get_observation()
    if not isinstance(obs, dict):
        raise AssertionError("observation is not a dict")
    if "legal_actions" not in obs:
        raise AssertionError("observation missing legal_actions")
    return "ok", "observation keys=%d (has legal_actions)" % len(obs)


def _legal_actions():
    from env.riichi_game import RiichiGame, RiichiConfig
    g = RiichiGame(RiichiConfig(), seed=0)
    la = g.state.legal_actions()
    expect = {"discard", "riichi", "chow", "pon", "kan", "ron", "tsumo", "kyushu"}
    missing = expect - set(la.keys())
    if missing:
        raise AssertionError("legal_actions missing keys: %s" % sorted(missing))
    if not la.get("discard"):
        raise AssertionError("no legal discard on first decision")
    return "ok", "legal_actions keys=%s (discard=%d)" % (sorted(la.keys()), len(la["discard"]))


def _random_action(la, rng):
    if la.get("ron"):
        return {"type": "ron"}
    if la.get("tsumo"):
        return {"type": "tsumo"}
    if la.get("kyushu"):
        return {"type": "kyushu"}
    for h in ("kan", "pon", "chow"):
        if la.get(h):
            return {"type": h, "tiles": list(la[h][0]["tiles"])}
    if la.get("riichi"):
        return {"type": "riichi", "tile": la["riichi"][0]}
    if la.get("discard"):
        return {"type": "discard", "tile": rng.choice(la["discard"])}
    return {"type": "pass"}


def _random_rollout():
    import random
    from env.riichi_game import RiichiGame, RiichiConfig
    rng = random.Random(12345)
    g = RiichiGame(RiichiConfig(), seed=12345)
    steps = 0
    while g.phase != "game_end" and steps < 10000:
        la = g.state.legal_actions()
        g.step(_random_action(la, rng))
        steps += 1
    if g.phase != "game_end":
        raise AssertionError("game did not finish in 10000 steps")
    if len(g.scores) != 4:
        raise AssertionError("scores length != 4")
    return "ok", "finished in %d steps, scores=%s, events=%d" % (
        steps, list(map(int, g.scores)), len(g.events))


def _paths():
    from utils.paths import repo_root, data_dir, checkpoint_dir, log_dir, model_dir
    assert repo_root().is_dir(), "repo_root not a dir"
    assert str(model_dir()) == str(checkpoint_dir().parent / "models"), "model_dir mismatch"
    _ = (data_dir(), checkpoint_dir(), log_dir(), model_dir())
    return "ok", "repo_root=%s (model_dir ok, checkpoint exists=%s)" % (
        repo_root().name, CHECKPOINT.exists())


def _py_compile():
    bad = []
    n = 0
    for root in ("src", "tools"):
        for p in (REPO / root).rglob("*.py"):
            n += 1
            try:
                compile(p.read_text(encoding="utf-8"), str(p), "exec")
            except Exception as e:
                bad.append("%s: %s" % (p, e))
    if bad:
        return "fail", "; ".join(bad[:5])
    return "ok", "compiled %d files under src/ + tools/" % n


def _config_validate():
    cfg = REPO / "configs" / "config_validate.py"
    for name in ("rl_v1.json", "rl_v1_event_attn.json", "rl_v1_reward_full.json"):
        r = subprocess.run([sys.executable, str(cfg), str(REPO / "configs" / name)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return "fail", "%s: %s" % (name, (r.stdout + r.stderr).strip()[-400:])
    return "ok", "3 configs PASS config_validate.py"


def _cli_help():
    """Run --help for eval tools. The tool imports torch/mahjong at module top-level,
    so on a machine without them the CLI cannot even reach argparse -> SKIP. On a full
    env: FAIL only on a path regression ('agentwork'), PASS on clean exit 0."""
    if _dep_missing():
        return "skip", "eval tools need torch+mahjong to run --help (path check deferred)"
    for tool in ("eval_vs_sl.py", "eval_sliding.py"):
        r = subprocess.run([sys.executable, str(REPO / "tools" / tool), "--help"],
                           capture_output=True, text=True)
        combined = (r.stdout + r.stderr).lower()
        if "agentwork" in combined:
            return "fail", "%s --help surfaced hardcoded 'agentwork'" % tool
        if r.returncode != 0:
            return "fail", "%s --help rc=%d: %s" % (tool, r.returncode, combined[-300:])
    return "ok", "eval_vs_sl.py + eval_sliding.py --help exit 0"


def _test_suite():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                       cwd=str(REPO), capture_output=True, text=True)
    combined = r.stdout + r.stderr
    low = combined.lower()
    if r.returncode == 0:
        return "ok", "pytest -q passed"
    if "agentwork" in low:
        return "fail", "pytest surfaced hardcoded 'agentwork'"
    if ("no module named 'torch'" in low and "no module named 'mahjong'" in low
            and "error" in low):
        return "skip", "pytest requires torch+mahjong (missing locally)"
    if "no module named" in low and ("torch" in low or "mahjong" in low):
        return "skip", "pytest blocked by missing torch/mahjong; tail: %s" % combined[-300:]
    return "fail", "pytest rc=%d; tail: %s" % (r.returncode, combined[-600:])


# --------------------------------------------------------------------------- registry

def _needs(env_only: bool = False):
    """Return (skip_reason or None). env_only=True -> requires torch+mahjong."""
    if env_only and _dep_missing():
        return "missing: " + _dep_missing()
    return None


CHECKS = [
    ("paths",             False, _paths),
    ("config_validate",   False, _config_validate),
    ("py_compile",        False, _py_compile),
    ("cli_help",          False, _cli_help),
    ("import_modules",    True,  _import_modules),
    ("load_checkpoint",   True,  _load_checkpoint),
    ("create_game",       True,  _create_game),
    ("create_observation", True, _create_observation),
    ("legal_actions",     True,  _legal_actions),
    ("random_rollout",    True,  _random_rollout),
    ("test_suite",        True,  _test_suite),
]


def main() -> int:
    results = []
    for name, env_only, fn in CHECKS:
        reason = _needs(env_only)
        if reason:
            results.append((name, "SKIP", reason))
            print("[SKIP] %-18s %s" % (name, reason))
            continue
        try:
            status, detail = fn()
        except Exception as e:
            status = "fail"
            detail = "%s: %s" % (type(e).__name__, e)
        status = {"ok": "PASS", "skip": "SKIP", "fail": "FAIL"}.get(status, status.upper())
        results.append((name, status, detail))
        print("[%s] %-18s %s" % (status.ljust(4), name, detail))

    fails = [r for r in results if r[1] == "FAIL"]
    skips = [r for r in results if r[1] == "SKIP"]
    passes = [r for r in results if r[1] == "PASS"]
    print("\n=== D0 SUMMARY ===")
    print("PASS=%d SKIP=%d FAIL=%d (total=%d)"
          % (len(passes), len(skips), len(fails), len(results)))
    if skips:
        print("skipped (missing deps): %s" % ", ".join(r[0] for r in skips))
    if fails:
        print("FAILED: %s" % ", ".join(r[0] for r in fails))
        return 1
    print("D0 PASS" if not fails else "D0 FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())