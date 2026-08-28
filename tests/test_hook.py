# -*- coding: utf-8 -*-
"""Model hook integration tests (task t16): checkpoint -> ModelDecision."""
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parents[1])
SRC = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from model.synthetic import make_record, write_records  # noqa: E402


def _train_tiny_ckpt(tmp_path, mode="full"):
    records = str(tmp_path / "synth.jsonl")
    write_records(records, n=1500, seed=17)
    cmd = [sys.executable, str(Path(SRC) / "model" / "train_sl.py"),
           "--records", records, "--mode", mode, "--out", str(tmp_path / "ckpt"),
           "--log-dir", str(tmp_path / "logs"), "--batch-size", "64",
           "--max-steps", "20", "--blocks", "2", "--channels", "16",
           "--epochs", "1", "--device", "cpu"]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = SRC
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd=SRC, timeout=600)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    stage = 1 if mode == "simple" else 2
    return str(tmp_path / "ckpt" / f"stage{stage}_{mode}" / "last.pt")


def _check_decisions(ckpt, n=20):
    from eval.inference import make_inference_hook, sanity_check_decision
    from eval.runner import build_runner
    hook = make_inference_hook(model_path=ckpt, device="cpu")
    runner = build_runner("hook:" + ckpt)
    rng = random.Random(3)
    obs_list = [make_record(rng, i) for i in range(n)]
    decisions = runner.predict(obs_list)
    assert len(decisions) == n
    for obs, d in zip(obs_list, decisions):
        ok, reason = sanity_check_decision(obs, d)
        assert ok, reason
        assert set(d) >= {"action", "top_k", "factors", "rationale", "probs_discard"}
        if d["action"]["type"] == "discard":
            # rationale numbers must match factors (protocol §3)
            assert str(d["factors"]["shanten"]) in d["rationale"]
            assert str(d["factors"]["ukeire"]) in d["rationale"]
            assert sum(d["probs_discard"].values()) == pytest.approx(1.0, abs=1e-3)
            assert d["action"]["tile"] in obs["legal_actions"]["discard"]
    # single-point hook agrees with batch runner (保序)
    for obs, d in zip(obs_list, decisions):
        d2 = hook(obs)
        assert d2["action"] == d["action"]


def test_hook_full_mode(tmp_path):
    _check_decisions(_train_tiny_ckpt(tmp_path, mode="full"))


def test_hook_simple_mode(tmp_path):
    """stage-1 checkpoint (no binary heads) must load and stay legal."""
    _check_decisions(_train_tiny_ckpt(tmp_path, mode="simple"))
