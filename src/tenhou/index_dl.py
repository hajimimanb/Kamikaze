"""Fetch tenhou.net/sc/raw/list.cgi (+ ?old) and store the file index manifest.

URLs (docs/observation_schema.md section 6):
- https://tenhou.net/sc/raw/list.cgi       -> recent ~7 days, no year prefix
- https://tenhou.net/sc/raw/list.cgi?old   -> history, paths carry {yyyy}/ prefix
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from typing import Dict, List

ENTRY_RE = re.compile(r"\{file:'([^']+)',size:(\d+)\}")

URLS = {
    "recent": "https://tenhou.net/sc/raw/list.cgi",
    "old": "https://tenhou.net/sc/raw/list.cgi?old",
}

DEFAULT_OUT = "C:/agentwork/data/processed/tenhou/index_manifest.json"


def _fetch(url: str, timeout: int = 30, retries: int = 3) -> str:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError("failed to fetch %s: %s" % (url, last))


def parse_listing(text: str) -> List[Dict]:
    out = []
    for m in ENTRY_RE.finditer(text):
        out.append({"file": m.group(1), "size": int(m.group(2))})
    return out


def build_manifest() -> Dict:
    manifest = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "recent": [],
        "old": [],
        "scc_files": [],
    }
    for kind, url in URLS.items():
        entries = parse_listing(_fetch(url))
        manifest[kind] = entries
    seen = set()
    scc = []
    for kind in ("recent", "old"):
        for e in manifest[kind]:
            name = e["file"].rsplit("/", 1)[-1]
            if re.match(r"^scc\d{8}(?:\d{2})?\.html\.gz$", name):
                key = e["file"]
                if key not in seen:
                    seen.add(key)
                    scc.append(e)
    scc.sort(key=lambda e: e["file"])
    manifest["scc_files"] = scc
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--urls", nargs="*", choices=list(URLS) + ["all"], default=["all"])
    args = ap.parse_args(argv)

    manifest = build_manifest()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print("saved manifest: %s" % args.out)
    print("recent files: %d, old files: %d, scc files: %d"
          % (len(manifest["recent"]), len(manifest["old"]), len(manifest["scc_files"])))
    if manifest["scc_files"]:
        first = manifest["scc_files"][0]["file"]
        last = manifest["scc_files"][-1]["file"]
        print("scc range: %s .. %s" % (first, last))
    return 0


if __name__ == "__main__":
    sys.exit(main())
