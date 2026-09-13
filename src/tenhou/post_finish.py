"""Watchdog: wait for the re-extraction batch to finish, then run the
post-processing chain (mjai re-export incl. masked files, shards manifest,
stats report) and write a completion marker."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import docs_dir, processed_dir, src_dir

PY = sys.executable
LOG = str(processed_dir() / "tenhou/logs/reextract.log")
OUT_LOG = str(processed_dir() / "tenhou/logs/post_finish.log")
DONE_MARK = str(processed_dir() / "tenhou/POST_FINISH_DONE")


def log(msg: str) -> None:
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_dir())
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([PY] + args, env=env, capture_output=True, text=True)
    log("run %s -> rc=%d" % (args, r.returncode))
    if r.stdout.strip():
        log(r.stdout.strip()[-500:])
    if r.returncode != 0 and r.stderr.strip():
        log("STDERR: " + r.stderr.strip()[-800:])


def main() -> int:
    log("watchdog started, waiting for reextract.log to show 2x BATCH DONE")
    while True:
        try:
            text = open(LOG, encoding="utf-8").read()
        except OSError:
            text = ""
        if text.count("BATCH DONE") >= 2:
            break
        time.sleep(60)
    log("re-extraction finished; starting post-processing")
    run(["src/tenhou/reexport_mjai.py"])
    run(["src/tenhou/manifest.py"])
    run(["-m", "tenhou.stats", "--out",
         str(docs_dir() / "data_report.md")])
    with open(DONE_MARK, "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("ALL POST-PROCESSING DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
