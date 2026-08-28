"""Detached background launcher for long jobs (Windows).

Usage: python bg.py --name NAME --out LOGFILE -- python ... args
Spawns the command fully detached and returns immediately.
"""
import subprocess
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    name, logfile = None, None
    while args and args[0].startswith("--"):
        if args[0] == "--name":
            name, args = args[1], args[2:]
        elif args[0] == "--out":
            logfile, args = args[1], args[2:]
        else:
            raise SystemExit("unknown option " + args[0])
    if not args:
        raise SystemExit("no command given")
    if logfile:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
    f = open(logfile, "ab", buffering=0) if logfile else None
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    import os as _os
    env = dict(_os.environ)
    env["PYTHONPATH"] = "C:/agentwork/src"
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.Popen(args, stdout=f, stderr=f, stdin=subprocess.DEVNULL,
                         creationflags=flags, close_fds=True, env=env)
    print("started pid", p.pid, "->", logfile)
