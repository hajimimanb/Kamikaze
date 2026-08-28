# -*- coding: utf-8 -*-
"""Smoke test: run the t5 training loop end-to-end on synthetic records."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from model.synthetic import write_records

SRC = str(Path(__file__).resolve().parents[1] / "src")


def run_cli(records, mode, tmp_path, extra=()):
    cmd = [sys.executable, str(Path(SRC) / "model" / "train_sl.py"),
           "--records", records, "--mode", mode, "--out", str(tmp_path / "ckpt"),
           "--log-dir", str(tmp_path / "logs"), "--epochs", "1",
           "--max-steps", "120", "--eval-every", "60", "--blocks", "2",
           "--channels", "32", "--device", "cpu", *extra]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = SRC
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=SRC, timeout=600)


@pytest.mark.parametrize("mode", ["simple", "full"])
def test_train_smoke(mode, tmp_path):
    records = str(tmp_path / "synth.jsonl")
    write_records(records, n=400, seed=1)
    r = run_cli(records, mode, tmp_path)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = tmp_path / "ckpt" / f"stage{1 if mode == 'simple' else 2}_{mode}"
    assert (out / "last.pt").exists()
    assert (out / "config.json").exists()
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["mode"] == mode
    log = tmp_path / "logs" / f"train_sl_stage{1 if mode == 'simple' else 2}_{mode}.jsonl"
    assert log.exists()
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert lines and "discard_top1" in lines[-1]
    # tsumogiri accuracy is the stage-1 baseline metric; synthetic data is
    # 85% tsumogiri so a trained (or even random-masked) model stays > 0
    assert lines[-1]["tsumogiri_acc"] is not None
    assert 0.0 <= lines[-1]["tsumogiri_acc"] <= 1.0
    # stdout reports curriculum-relevant metric keys
    joined = r.stdout + r.stderr
    assert "top1" in joined and "channels=" in joined
