# -*- coding: utf-8 -*-
"""重启 RL 自对弈训练（detached，PID 写入 logs/rl_train.pid）。

用法：python tools/start_rl_train.py
日志：logs/rl_train.txt（追加）；PID：logs/rl_train.pid
"""
import os, subprocess, sys

CWD = "C:/agentwork"
ARGS = [
    "-u", "src/model/train_rl_vec.py",
    "--vec", "8", "--games", "20", "--epochs", "500",   # 20局/epoch × 500 = 10000 局（牌运平滑）
    "--event-attn", "--seed", "0",
    "--ckpt", "checkpoints/sl/rl/rl_v1_resume_ep51.pt",   # 从保存的继续点（epoch 51）继续
    "--out", "checkpoints/sl/rl/rl_v1.pt",
    "--phi-ckpt", "checkpoints/sl/rl/reward_predictor.pt",
    "--metrics-path", "logs/rl_train_metrics.jsonl",
    "--games-path", "logs/rl_train_games.jsonl",
    "--eval-every", "30", "--eval-games", "20", "--save-every", "50",
    "--inner-epochs", "1", "--pool-size", "32",
    "--clip", "0.15", "--entropy-coef", "0.06", "--clip-grad", "0.5",
    "--win-bonus", "3.0", "--rare-w-cap", "4.0",
    "--kl-stop", "0.3", "--bias-snap", "1", "--t5-every", "0",
    "--lr-schedule", "fixed",
    "--reset-heads", "riichi",   # 根因修复（事件注入放大）+ 保守 lr：慢但有效
]

if __name__ == "__main__":
    os.chdir(CWD)
    log = open("logs/rl_train.txt", "ab")
    p = subprocess.Popen([sys.executable] + ARGS, stdout=log,
                         stderr=subprocess.STDOUT, cwd=CWD,
                         creationflags=0x00000008 | 0x00000200)
    with open("logs/rl_train.pid", "w") as f:
        f.write(str(p.pid))
    print("SPAWNED PID", p.pid)
