"""Split manifest integrity + record partition consistency (reviewer t7)."""
import gzip
import glob
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from utils.paths import processed_dir

from tenhou.stats import assign_split
from tenhou.verify_splits import load_lists


def test_manifests_disjoint_and_self_consistent():
    parts = load_lists()
    assert parts, "no partitions in splits.json"
    for a in parts:
        for b in parts:
            if a < b:
                assert not (parts[a] & parts[b]), (a, b)
    for part, games in parts.items():
        for g in list(games)[:2000]:
            assert assign_split(g) == part, (g, part)


def test_every_record_game_id_in_its_partition():
    shards = glob.glob(str(processed_dir() / "tenhou" / "records-*.jsonl.gz"))
    if not shards:
        import pytest
        pytest.skip("no record shards on disk yet")
    parts = load_lists()
    n = 0
    for path in shards[:3]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gid = rec.get("game_id")
                assert gid, "record without game_id"
                part = assign_split(str(gid))
                assert gid in parts[part], (gid, part)
                n += 1
    assert n > 1000
