# -*- coding: utf-8 -*-
"""Tests for offline tensorization (preprocess.py) + shard-based training."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from model.features import BINARY_HEADS, build_features, channel_names, feature_channels, discard_mask
from model.preprocess import NO_LABEL, channel_scales, fnv1a64, load_shard
from model.synthetic import write_records

SRC = str(Path(__file__).resolve().parents[1] / "src")


def run_preprocess(records, out, tmp_path, workers=1, shard_size=1000, chunk=500, limit=0):
    cmd = [sys.executable, str(Path(SRC) / "model" / "preprocess.py"),
           "--records", records, "--out", out,
           "--shard-size", str(shard_size), "--chunk", str(chunk),
           "--workers", str(workers)]
    if limit:
        cmd += ["--limit", str(limit)]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = SRC
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd=SRC, timeout=1200)
    return r


def _expected_arrays(records):
    """Recompute expected shard fields in-process (single worker => ordered)."""
    from model.preprocess import encode_batch
    return encode_batch(records)


def test_preprocess_roundtrip_exact(tmp_path):
    records_file = str(tmp_path / "synth.jsonl")
    write_records(records_file, n=2500, seed=3)
    with open(records_file, encoding="utf-8") as fh:
        records = [json.loads(x) for x in fh]

    out = str(tmp_path / "shards")
    r = run_preprocess(records_file, out, tmp_path, workers=1, shard_size=1000)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]

    shard_paths = sorted(Path(out).glob("shard-*.npz.zst"))
    assert len(shard_paths) == 3  # 2500 / 1000 -> 2 full + 1 partial
    manifest = json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_samples"] == 2500 and manifest["n_shards"] == 3

    expected = _expected_arrays(records)
    got = {"feat": [], "dmask": [], "discard": [], "drawn": [], "gid": []}
    for h in BINARY_HEADS:
        got[f"{h}_y"] = []
        got[f"{h}_cand"] = []
    scales = []
    for p in shard_paths:
        d = load_shard(p)
        for k in got:
            got[k].append(d[k])
        scales.append(d["scale"])
    for k in got:
        got[k] = np.concatenate(got[k], axis=0)
    for k in got:
        assert np.array_equal(got[k], expected[k]), k
    # game-id hashes for partition audit
    assert got["gid"].dtype == np.uint64
    assert got["gid"][0] == fnv1a64(records[0]["game_id"])
    assert (got["gid"] != 0).all()
    # per-channel scale metadata consistent + saved npy matches
    # (stored scale = max/255; decode x = feat * scale)
    saved_scale = np.load(Path(out) / "channel_scale.npy")
    assert all(np.array_equal(s, saved_scale) for s in scales)
    assert np.array_equal(saved_scale, channel_scales(channel_names("full")) / 255.0)
    assert len(saved_scale) == feature_channels("full") == 126
    assert "samples/s" in (r.stdout + r.stderr)


def test_preprocess_multiprocess_totals(tmp_path):
    records_file = str(tmp_path / "synth.jsonl")
    write_records(records_file, n=6000, seed=5)
    out = str(tmp_path / "shards")
    r = run_preprocess(records_file, out, tmp_path, workers=3, shard_size=2048)
    assert r.returncode == 0, r.stderr[-2000:]
    shard_paths = sorted(Path(out).glob("shard-*.npz.zst"))
    total = sum(load_shard(p)["feat"].shape[0] for p in shard_paths)
    assert total == 6000
    # every sample has >=1 legal discard kind in its mask
    for p in shard_paths:
        dmask = load_shard(p)["dmask"]
        assert (dmask.sum(axis=1) >= 1).all()
        feat = load_shard(p)["feat"]
        assert feat.min() >= 0 and feat.max() <= 255
        disc = load_shard(p)["discard"]
        assert ((disc == NO_LABEL) | (disc <= 33)).all()


def test_decode_matches_live_encoding(tmp_path):
    """Decoded shard features == live build_features within quantization tol."""
    records_file = str(tmp_path / "synth.jsonl")
    write_records(records_file, n=900, seed=11)
    with open(records_file, encoding="utf-8") as fh:
        records = [json.loads(x) for x in fh]
    out = str(tmp_path / "shards")
    r = run_preprocess(records_file, out, tmp_path, workers=1, shard_size=1000)
    assert r.returncode == 0, r.stderr[-2000:]
    d = load_shard(sorted(Path(out).glob("shard-*.npz.zst"))[0])
    scale = d["scale"]
    for i in (0, 17, 500, 899):
        live = build_features(records[i], mode="full")[:, :, 0]
        decoded = d["feat"][i].astype(np.float32) * scale[:, None]
        assert np.abs(decoded - live).max() <= 0.02, i
        mask = discard_mask(records[i])
        assert np.array_equal(d["dmask"][i].astype(bool), mask)
        # simple mode = first 17 channels of full (curriculum stage 1)
        live_s = build_features(records[i], mode="simple")[:, :, 0]
        decoded_s = d["feat"][i, :17].astype(np.float32) * scale[:17, None]
        assert np.abs(decoded_s - live_s).max() <= 0.02


def test_train_on_shards_smoke(tmp_path):
    records_file = str(tmp_path / "synth.jsonl")
    write_records(records_file, n=4000, seed=13)
    out = str(tmp_path / "shards")
    r = run_preprocess(records_file, out, tmp_path, workers=2, shard_size=1500)
    assert r.returncode == 0, r.stderr[-2000:]
    cmd = [sys.executable, str(Path(SRC) / "model" / "train_sl.py"),
           "--shards", out, "--mode", "simple", "--out", str(tmp_path / "ckpt"),
           "--log-dir", str(tmp_path / "logs"), "--batch-size", "64",
           "--max-steps", "30", "--eval-every", "1200", "--blocks", "2",
           "--channels", "32", "--epochs", "1", "--device", "cpu"]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = SRC
    r2 = subprocess.run(cmd, capture_output=True, text=True, env=env,
                        cwd=SRC, timeout=900)
    assert r2.returncode == 0, r2.stdout[-2000:] + r2.stderr[-2000:]
    assert (tmp_path / "ckpt" / "stage1_simple" / "last.pt").exists()
    log = tmp_path / "logs" / "train_sl_stage1_simple.jsonl"
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert lines and "samples_per_sec" in lines[-1]
    assert lines[-1]["samples_per_sec"] > 0
    assert "tsumogiri_acc" in lines[-1]
