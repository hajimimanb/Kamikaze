# -*- coding: utf-8 -*-
"""N：重启前归档/清空 RL 训练日志（metrics/games/analysis/train/奖励相关）。

用法：python tools/archive_rl_logs.py
- 将 logs/rl_train_metrics.jsonl、rl_train_games.jsonl、rl_train_epoch_analysis.jsonl、
  rl_train.txt、rl_gate_monitor.jsonl、reward_pred.txt、calibrate_rl_result.json、
  rl_phi_monitor.json/txt 移入 logs/archive_<时间戳>/
- 保留 logs/past_ckpts.txt（历史对手池环形索引，重启后继续累计）
- 面板（rl_train_dashboard）会在新日志出现后自动续读
"""
import os, shutil, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
from utils.paths import logs_dir

LOGS = str(logs_dir())
FILES = [
    "rl_train_metrics.jsonl",
    "rl_train_games.jsonl",
    "rl_train_epoch_analysis.jsonl",
    "rl_train.txt",
    "rl_gate_monitor.jsonl",
    "reward_pred.txt",
    "calibrate_rl_result.json",
    "rl_phi_monitor.json",
    "rl_phi_monitor.txt",
]


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(LOGS, "archive_" + ts)
    os.makedirs(dst, exist_ok=True)
    moved = 0
    for f in FILES:
        src = os.path.join(LOGS, f)
        if os.path.exists(src):
            shutil.move(src, os.path.join(dst, f))
            moved += 1
    print("ARCHIVED %d files -> %s" % (moved, dst))


if __name__ == "__main__":
    main()
