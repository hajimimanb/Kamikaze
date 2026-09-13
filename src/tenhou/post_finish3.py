"""Watchdog v3: wait for ALL batches (my redoA/redoB + the captain's fast
bands) to finish, then produce the final data_report.md + manifests + mjai
re-export and write POST_FINISH_DONE.

Completion conditions:
- redoA.log and redoB.log contain "BATCH DONE";
- every batch_fast_*.log in the logs dir contains "BATCH DONE";
- no running process has "tenhou.batch" in its command line.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import docs_dir, processed_dir, src_dir

PY = sys.executable
LOG_DIR = str(processed_dir() / "tenhou/logs")
OUT_LOG = os.path.join(LOG_DIR, "post_finish3.log")
DONE_MARK = str(processed_dir() / "tenhou/POST_FINISH_DONE")


def log(msg: str) -> None:
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\\n")


def any_batch_running() -> bool:
    import subprocess as sp
    try:
        out = sp.run(["wmic", "process", "where", "name='python.exe'",
                      "get", "CommandLine"], capture_output=True, text=True,
                     timeout=20).stdout or ""
    except Exception:
        return True
    return "tenhou.batch" in out


def all_done() -> bool:
    for name in ("redoA.log", "redoB.log", "redoC.log"):
        try:
            text = open(os.path.join(LOG_DIR, name), encoding="utf-8").read()
        except OSError:
            text = ""
        if "BATCH DONE" not in text:
            return False
    # every batch sweep log (captain's fast bands or later sweeps) must be
    # finished; note: a killed sweep leaves a log without BATCH DONE and
    # blocks finalization until the captain re-runs it (safer than firing
    # prematurely on an incomplete corpus)
    for name in sorted(os.listdir(LOG_DIR)):
        if name.startswith("batch") and name.endswith(".log"):
            try:
                text = open(os.path.join(LOG_DIR, name),
                            encoding="utf-8").read()
            except OSError:
                text = ""
            if "BATCH DONE" not in text:
                return False
    # no half-written shards (atomic writes leave .tmp only mid-run)
    import glob
    if glob.glob(str(processed_dir() / "tenhou" / "records-*.tmp")):
        return False
    if any_batch_running():
        return False
    return True


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
    log("watchdog v3 started; waiting for all batches to complete")
    while not all_done():
        time.sleep(120)
    log("all batches finished; final post-processing")
    run(["src/tenhou/reexport_mjai.py"])
    run(["src/tenhou/manifest.py"])
    run(["-m", "tenhou.stats", "--out",
         str(docs_dir() / "data_report.md")])
    with open(DONE_MARK, "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("FINAL POST-PROCESSING DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
