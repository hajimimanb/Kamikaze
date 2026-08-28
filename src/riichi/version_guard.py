# -*- coding: utf-8 -*-
"""Startup dependency version assertions (requirements-lock.txt, task t16).

mahjong==2.0.0 is REQUIRED: engine (env/riichi_game.py) and explain.py use the
2.0 API (HandCalculator/Shanten/Agari). pip may silently DOWNGRADE mahjong
when installing mjai (mjai 0.2.1 metadata declares mahjong~=1.2.0, though mjai
works fine with 2.0.0 - verified 2026-08-25, full test suite green).

Importing riichi.tiles triggers this guard, so every ML/engine entrypoint
(features -> preprocess/train/eval) asserts at import time. A mismatch raises
immediately with remediation instructions.
"""
from __future__ import annotations

import importlib.metadata as _md

_REQUIRED = {"mahjong": "2.0.0"}


def assert_versions() -> None:
    problems = []
    for pkg, want in sorted(_REQUIRED.items()):
        try:
            got = _md.version(pkg)
        except _md.PackageNotFoundError:
            problems.append(pkg + ": not installed")
            continue
        if got != want:
            problems.append(pkg + "==" + got + " (required " + want + ")")
    if problems:
        raise RuntimeError(
            "[version-guard] dependency mismatch: " + "; ".join(problems) + ". "
            "Fix: C:/agentwork/.venv/Scripts/python.exe -m pip install mahjong==2.0.0 "
            "(mjai 0.2.1 metadata may downgrade it on install; mjai still works "
            "with 2.0.0 - see requirements.txt note)")
