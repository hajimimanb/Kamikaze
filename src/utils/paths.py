# -*- coding: utf-8 -*-
"""Repository-root-aware path helpers.

All machine-specific absolute paths (historically hardcoded as ``C:/agentwork``)
must be resolved through this module so the project works on any machine.

Import bootstrap (put ``src`` on ``sys.path`` once per entrypoint)::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from utils.paths import repo_root, data_dir
"""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Absolute path to the repository root (the directory that contains ``src/``).

    This file lives at ``src/utils/paths.py``:
    ``parents[0]=utils`` -> ``parents[1]=src`` -> ``parents[2]=root``.
    """
    return Path(__file__).resolve().parents[2]


def src_dir() -> Path:
    return repo_root() / "src"


def data_dir() -> Path:
    return repo_root() / "data"


def checkpoints_dir() -> Path:
    return repo_root() / "checkpoints"


def logs_dir() -> Path:
    return repo_root() / "logs"


def tools_dir() -> Path:
    return repo_root() / "tools"


def docs_dir() -> Path:
    return repo_root() / "docs"


def raw_dir() -> Path:
    return data_dir() / "raw"


def processed_dir() -> Path:
    return data_dir() / "processed"


def tmp_dir() -> Path:
    return data_dir() / "tmp"


def venv_dir() -> Path:
    return repo_root() / ".venv"


def model_dir() -> Path:
    """模型权重目录（models/），含已随仓库分发的 kamikaze_rl_v1_fp16.pt。"""
    return repo_root() / "models"


# feedback2 §九 建议统一命名 model_dir / checkpoint_dir / log_dir / data_dir。
# 下方单数形式为规范名；checkpoints_dir / logs_dir 保留为历史兼容别名，
# 不破坏 tools/ 与 src/ 中既有调用点。
def checkpoint_dir() -> Path:
    return checkpoints_dir()


def log_dir() -> Path:
    return logs_dir()