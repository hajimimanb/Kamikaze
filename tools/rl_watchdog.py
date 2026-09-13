# -*- coding: utf-8 -*-
"""训练进程 watchdog：检测训练死亡 → 自动归档日志并重启（从最新 rl_v1.pt 继续）。

用法：python tools/rl_watchdog.py
- 每 30s 检查 logs/rl_train.pid 进程存活 + rl_train.txt 最近 4 分钟有写入
- 死亡且未 DONE → 重启 start_rl_train.py，写 logs/rl_watchdog.log
"""
import os, subprocess, sys, time, datetime
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
from utils.paths import repo_root, logs_dir

CWD = str(repo_root())
PID_FILE = str(logs_dir() / "rl_train.pid")
TRAINLOG = str(logs_dir() / "rl_train.txt")
WLOG = str(logs_dir() / "rl_watchdog.log")


def log(msg):
    line = "%s %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    try:
        with open(WLOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def alive():
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
    except Exception:
        return False
    r = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid],
                       capture_output=True, text=True, timeout=10)
    return "python.exe" in r.stdout


def train_updating():
    try:
        age = time.time() - os.path.getmtime(TRAINLOG)
        return age < 300
    except Exception:
        return False


def done():
    try:
        return "RL_V1 DONE" in open(TRAINLOG, encoding="utf-8", errors="ignore").read()
    except Exception:
        return False


if __name__ == "__main__":
    log("watchdog started")
    down_since = None
    while True:
        time.sleep(30)
        try:
            if done():
                log("training DONE - watchdog exiting")
                break
            ok = alive() and train_updating()
            if ok:
                down_since = None
                continue
            if down_since is None:
                down_since = time.time()
                log("training appears down (alive=%s updating=%s)" % (alive(), train_updating()))
            if time.time() - down_since > 90:
                log("confirmed down >90s -> restarting")
                try:
                    subprocess.run([sys.executable, "tools/archive_rl_logs.py"],
                                   cwd=CWD, timeout=60)
                except Exception as e:
                    log("archive err: %s" % e)
                p = subprocess.Popen([sys.executable, "tools/start_rl_train.py"],
                                     cwd=CWD, creationflags=0x00000008 | 0x00000200)
                log("restarted (spawn pid=%d)" % p.pid)
                down_since = None
                time.sleep(10)
        except Exception as e:
            log("watchdog err: %s" % e)
