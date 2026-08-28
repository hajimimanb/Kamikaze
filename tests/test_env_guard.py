# -*- coding: utf-8 -*-
"""Version-guard tests (task t16)."""
import importlib.metadata

import pytest

from riichi.version_guard import assert_versions


def test_guard_passes_current_env():
    assert importlib.metadata.version("mahjong") == "2.0.0"
    assert_versions()  # no raise


def test_guard_raises_on_mismatch(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda pkg: "1.2.1" if pkg == "mahjong" else "2.0.0")
    with pytest.raises(RuntimeError, match="mahjong==1.2.1"):
        assert_versions()
