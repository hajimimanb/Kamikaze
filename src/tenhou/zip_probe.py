"""Background prober: poll tenhou scraw yearly zips; log when available."""
import time, urllib.request, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import processed_dir
ZIPS = ['scraw2025.zip', 'scraw2024.zip', 'scraw2023.zip', 'scraw2022.zip', 'scraw2021.zip', 'scraw2020.zip', 'scraw2019.zip', 'scraw2018.zip', 'scraw2017.zip', 'scraw2016.zip', 'scraw2015.zip', 'scraw2014.zip', 'scraw2013.zip', 'scraw2012.zip', 'scraw2011.zip', 'scraw2010.zip', 'scraw2009.zip']
BASE = 'https://tenhou.net/sc/raw/'
LOG = str(processed_dir() / "tenhou/logs/zip_probe.log")
state = str(processed_dir() / "tenhou/logs/zip_probe_state.json")

def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + msg
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)

last = {}
if os.path.exists(state):
    last = json.load(open(state, encoding='utf-8'))
while True:
    for z in ZIPS:
        url = BASE + z
        try:
            req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://tenhou.net/sc/raw/'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                code, size = resp.status, resp.headers.get('Content-Length', '?')
        except Exception as e:
            code = getattr(e, 'code', None)
            size = '?'
        prev = last.get(z)
        cur = (code, str(size))
        if cur != prev:
            log('zip %s -> %s %s' % (z, code, size))
            last[z] = cur
            json.dump(last, open(state, 'w', encoding='utf-8'))
    time.sleep(1800)