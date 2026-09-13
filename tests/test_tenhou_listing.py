import gzip
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from utils.paths import raw_dir, tmp_dir

from tenhou.listing import parse_scc_bytes, phoenix_rows


def _load_gz(path):
    return open(path, "rb").read()


def test_parse_scc_2026_hour0():
    data = _load_gz(str(tmp_dir() / "scc_20260824" / "scc2026082400.html.gz"))
    rows = parse_scc_bytes(data)
    assert rows, "hour 00 should have rows"
    lobbies = set(r["lobby"] for r in rows)
    assert "00a9" in lobbies
    assert "00b9" in lobbies  # sanma south
    assert "00e1" in lobbies  # tonpu fast
    for r in rows:
        assert r["logid"].startswith("2026082400gm-")
        assert r["time"] is not None


def test_phoenix_filter():
    data = _load_gz(str(tmp_dir() / "scc_20260824" / "scc2026082400.html.gz"))
    rows = phoenix_rows(parse_scc_bytes(data))
    assert rows
    assert all(r["lobby"] == "00a9" for r in rows)


def test_full_day_count():
    import glob
    files = sorted(glob.glob(str(raw_dir() / "listings" / "scc_20260824" / "*.gz")))
    assert len(files) == 24
    total = 0
    for f in files:
        total += len(phoenix_rows(parse_scc_bytes(open(f, "rb").read())))
    assert total == 430


def test_zip_fallback_extraction():
    # synthetic scraw zip: one daily scc member for 20090615
    import io
    import zipfile
    import gzip as gz
    from tenhou.listing import _extract_scc_from_zip, parse_scc_bytes, phoenix_rows

    html = ('00:10 | 20 | 四鳳南喰赤－ | <a href="http://tenhou.net/0/?'
            'log=2009061500gm-00a9-0000-deadbeef">牌譜</a> | a(+50) b(0) '
            'c(-20) d(-30)<br>\n').encode("utf-8")
    member = gz.compress(html)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("2009/scc20090615.html.gz", member)
    data = buf.getvalue()
    out = _extract_scc_from_zip(data, "20090615", None)
    assert out is not None
    rows = phoenix_rows(parse_scc_bytes(out))
    assert len(rows) == 1
    assert rows[0]["logid"] == "2009061500gm-00a9-0000-deadbeef"
    assert rows[0]["lobby"] == "00a9"


def test_daily_scheme_january():
    # older dates only have daily files (scc{yyyymmdd}.html.gz)
    import glob
    files = glob.glob(str(raw_dir() / "listings" / "scc_20260110" / "*.gz"))
    if not files:
        import pytest
        pytest.skip("January daily listing not cached yet")
    rows = phoenix_rows(parse_scc_bytes(open(files[0], "rb").read()))
    assert rows
    assert all(r["logid"].startswith("20260110") for r in rows)
    assert all(r["lobby"] == "00a9" for r in rows)
