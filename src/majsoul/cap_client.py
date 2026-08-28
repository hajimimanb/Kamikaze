# -*- coding: utf-8 -*-
"""CAP token client for amae-koromo data mirrors.

Flow (verified against the official cap-widget source):
  POST {CAP_ENDPOINT}/challenge
  -> {challenge:{c,s,d}, token, instrumentation?, format?}
  For i in 1..c: salt=prng(token+i, s), target=prng(token+i+"d", d)
     nonce = smallest n>=0 with sha256(salt+n).hexdigest().startswith(target)
  (format 2 instead carries a "challenges" list of {protocol,payload}.)
  If instrumentation present: base64 -> deflate-raw -> script, execute inside a
  sandbox iframe via Playwright (system Edge) and capture
  postMessage {type:"cap:instr", result} as the instr payload.
  POST {CAP_ENDPOINT}/redeem {token, solutions, instr?}
  -> {token, expires}  (expires = ISO date string; cached locally)

IMPORTANT: akcap.pikapika.me rejects plain urllib with HTTP 403 (TLS
fingerprint), so challenge AND redeem requests are made from inside a real
Edge browser context via Playwright. The PoW itself is solved in Python with
multiprocessing (much faster than in-browser JS). The instrumentation script
runs in a sandbox iframe in the same page.

Constraints (per captain):
  - headless=True ALWAYS (never pop windows on the user's desktop).
  - page.evaluate(js) is called with a SINGLE argument; all dynamic data is
    embedded into the JS string via json.dumps.
  - Fresh browser + context per attempt, random viewport, random UA from a
    list, --disable-blink-features=AutomationControlled, navigator.webdriver
    masked via init script.
  - Max 8 attempts, sleep 2s between attempts.
  - Token cached to data/tmp/cap_token.json {token, expires}.
"""
import base64
import concurrent.futures
import hashlib
import io
import json
import os
import random
import sys
import time
import zlib

CAP_ENDPOINT = "https://akcap.pikapika.me/14f343ec68"
TOKEN_CACHE = "C:/agentwork/data/tmp/cap_token.json"
TOKEN_CACHE_LEGACY = "C:/agentwork/data/tmp/majsoul_cap_token.json"
ATTEMPT_LOG = "C:/agentwork/data/tmp/cap_attempts.jsonl"

UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
]

MAX_ATTEMPTS = int(os.environ.get("CAP_MAX_ATTEMPTS", "8"))
POW_WORKERS = int(os.environ.get("CAP_POW_WORKERS", "8"))
INSTR_TIMEOUT_MS = 20000


def _log_attempt(rec):
    line = json.dumps(rec, ensure_ascii=False, default=str)
    os.makedirs(os.path.dirname(ATTEMPT_LOG), exist_ok=True)
    with io.open(ATTEMPT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print("[cap] attempt:", line, flush=True)


def fnv1a(s):
    h = 2166136261
    for b in s.encode("utf-8"):
        h ^= b
        h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) & 0xFFFFFFFF
    return h


def prng(seed, length):
    """fnv1a-seeded xorshift32 -> lowercase hex string of given length."""
    state = fnv1a(seed)
    out = []
    while len(out) * 8 < length:
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        out.append("%08x" % state)
    return "".join(out)[:length]


def _solve_pow_one(salt, target):
    """Return smallest nonce whose sha256(salt+nonce) hex starts with target."""
    nonce = 0
    while True:
        h = hashlib.sha256(("%s%d" % (salt, nonce)).encode("utf-8")).hexdigest()
        if h.startswith(target):
            return nonce
        nonce += 1


def _pow_worker(args):
    salt, target = args
    try:
        return _solve_pow_one(salt, target)
    except Exception as e:  # pragma: no cover
        return {"error": "%s: %s" % (type(e).__name__, e)}


def solve_pow_challenges(pairs, workers=POW_WORKERS):
    """Solve [(salt,target), ...] in parallel processes; return nonce list."""
    results = [None] * len(pairs)
    if len(pairs) <= 2:
        for i, (salt, target) in enumerate(pairs):
            results[i] = _solve_pow_one(salt, target)
        return results
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(pairs))) as ex:
            for i, res in zip(range(len(pairs)), ex.map(_pow_worker, pairs)):
                if isinstance(res, dict) and "error" in res:
                    raise RuntimeError("worker error: %s" % res["error"])
                results[i] = res
        return results
    except Exception as e:
        print("[cap] process pool failed (%s), falling back to serial" % e, flush=True)
        for i, (salt, target) in enumerate(pairs):
            if results[i] is None:
                results[i] = _solve_pow_one(salt, target)
        return results


def _solve_rsw(payload):
    """RSW: y = x^(2^t) mod N."""
    n = int(payload["N"], 16)
    x = int(payload["x"], 16)
    t = int(payload["t"])
    return {"y": format(pow(x, pow(2, t), n), "x")}


def solve_challenges(resp, token):
    """Return solutions list for a challenge response (format 1 or 2)."""
    if resp.get("format") == 2 and isinstance(resp.get("challenges"), list):
        solutions = []
        for ch in resp["challenges"]:
            proto = ch.get("protocol")
            pl = ch.get("payload") or {}
            if proto == "sha256-pow":
                solutions.append({"nonce": _solve_pow_one(pl["salt"], pl["target"])})
            elif proto == "rsw":
                solutions.append(_solve_rsw(pl))
            elif proto == "instrumentation":
                solutions.append({"timeout": True})
            else:
                raise RuntimeError("unknown protocol %s" % proto)
        return solutions
    ch = resp.get("challenge") or {}
    c, s, d = ch.get("c", 0), ch.get("s", 0), ch.get("d", 0)
    pairs = [(prng("%s%d" % (token, i), s), prng("%s%dd" % (token, i), d)) for i in range(1, c + 1)]
    return list(solve_pow_challenges(pairs))


# --------------------------------------------------------------------------
# Browser-side JS (challenge/redeem fetched in-page to pass TLS fingerprint).
# Every page.evaluate() call takes a SINGLE argument: the full JS source with
# all dynamic data embedded via json.dumps.
# --------------------------------------------------------------------------

def _challenge_js(endpoint):
    return (
        "(async () => {"
        "  const endpoint = %s;"
        "  const r = await fetch(endpoint + '/challenge', { method: 'POST', headers: {"
        "    'Origin': 'https://amae-koromo.sapk.ch', 'Referer': 'https://amae-koromo.sapk.ch/' } });"
        "  const text = await r.text();"
        "  let json = null; try { json = JSON.parse(text); } catch (e) {}"
        "  return { status: r.status, json: json, text: text.slice(0, 200) };"
        "})()" % json.dumps(endpoint)
    )


def _redeem_js(endpoint, body):
    return (
        "(async () => {"
        "  const endpoint = %s;"
        "  const body = %s;"
        "  const r = await fetch(endpoint + '/redeem', { method: 'POST',"
        "    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });"
        "  const text = await r.text();"
        "  let json = null; try { json = JSON.parse(text); } catch (e) {}"
        "  return { status: r.status, json: json, text: text.slice(0, 300) };"
        "})()" % (json.dumps(endpoint), json.dumps(body))
    )


def _instr_js(script_text, timeout_ms):
    return (
        "(async () => {"
        "  const scriptText = %s;"
        "  const timeoutMs = %d;"
        "  return await new Promise((resolve) => {"
        "    const timeout = setTimeout(() => { cleanup(); resolve({__timeout: true}); }, timeoutMs);"
        "    const iframe = document.createElement('iframe');"
        "    iframe.setAttribute('sandbox', 'allow-scripts');"
        "    iframe.setAttribute('aria-hidden', 'true');"
        "    iframe.style.cssText = 'position:absolute;width:2px;height:2px;top:-9999px;left:-9999px;border:none;opacity:0;pointer-events:none;';"
        "    let resolved = false;"
        "    function cleanup() {"
        "      if (resolved) return;"
        "      resolved = true;"
        "      clearTimeout(timeout);"
        "      window.removeEventListener('message', handler);"
        "      if (iframe.parentNode) iframe.parentNode.removeChild(iframe);"
        "    }"
        "    function handler(ev) {"
        "      if (!iframe.contentWindow || ev.source !== iframe.contentWindow) return;"
        "      const d = ev.data;"
        "      if (!d || typeof d !== 'object') return;"
        "      if (d.type === 'cap:instr') {"
        "        cleanup();"
        "        if (d.blocked) resolve({__blocked: true, blockReason: d.blockReason || 'automated_browser'});"
        "        else if (d.result) resolve({result: d.result});"
        "        else resolve({__timeout: true});"
        "      } else if (d.type === 'cap:error') {"
        "        cleanup();"
        "        resolve({__timeout: true});"
        "      }"
        "    }"
        "    window.addEventListener('message', handler);"
        "    const open = '<!DOCTYPE html><html><head><meta charset=\"UTF-8\"></head><body><script>';"
        "    iframe.srcdoc = open + scriptText + '\\n</scr' + 'ipt></body></html>';"
        "    document.body.appendChild(iframe);"
        "  });"
        "})()" % (json.dumps(script_text), timeout_ms)
    )


def run_instrumentation(page, instr_b64, timeout_ms=INSTR_TIMEOUT_MS):
    """Decompress + execute the instrumentation script in a sandbox iframe.

    Returns dict with one of:
      {"result": <json>} / {"__blocked": True, "blockReason": ...} / {"__timeout": True}
    """
    raw = base64.b64decode(instr_b64)
    script_text = zlib.decompress(raw, -15).decode("utf-8")
    return page.evaluate(_instr_js(script_text, timeout_ms))


def _browser_attempt(ua):
    """One full CAP attempt inside a fresh headless Edge browser.
    Returns (token, expires, rec). Always headless=True."""
    from playwright.sync_api import sync_playwright
    rec = {}
    viewport = {"width": random.randint(1024, 1680), "height": random.randint(768, 1050)}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True,
                                    args=["--disable-blink-features=AutomationControlled"])
        try:
            ctx = browser.new_context(user_agent=ua, viewport=viewport,
                                      locale=random.choice(["zh-CN", "en-US", "ja-JP"]),
                                      timezone_id="Asia/Shanghai")
            ctx.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
            )
            page = ctx.new_page()
            page.goto("about:blank")

            t0 = time.time()
            ch = page.evaluate(_challenge_js(CAP_ENDPOINT))
            rec["challenge_status"] = ch.get("status")
            resp = ch.get("json") or {}
            if ch.get("status") != 200 or "token" not in resp:
                raise RuntimeError("challenge failed: http %s %s" % (ch.get("status"), ch.get("text")))
            token = resp["token"]
            rec["has_instrumentation"] = bool(resp.get("instrumentation"))

            solutions = solve_challenges(resp, token)
            rec["pow_solved"] = len(solutions)
            rec["pow_seconds"] = round(time.time() - t0, 1)

            instr = None
            if resp.get("instrumentation"):
                r = run_instrumentation(page, resp["instrumentation"])
                if "__blocked" in r:
                    raise RuntimeError("instrumentation blocked: %s" % r.get("blockReason"))
                if "__error" in r:
                    raise RuntimeError("instrumentation error: %s" % r["__error"])
                if "__timeout" in r:
                    rec["instr_timeout"] = True
                else:
                    instr = r.get("result")
                    rec["instr_ok"] = True

            body = {"token": token, "solutions": solutions}
            if instr:
                body["instr"] = instr
            elif resp.get("instrumentation"):
                body["instr_timeout"] = True
            rd = page.evaluate(_redeem_js(CAP_ENDPOINT, body))
            rec["redeem_status"] = rd.get("status")
            rj = rd.get("json") or {}
            if rd.get("status") == 200 and rj.get("success") and rj.get("token"):
                return rj["token"], _parse_expiry(rj.get("expires")), rec
            raise RuntimeError("redeem failed: http %s %s" % (rd.get("status"), rd.get("text")))
        finally:
            try:
                browser.close()
            except Exception:
                pass


def acquire_token(max_attempts=MAX_ATTEMPTS):
    """Run the full CAP flow with retries (headless only, 2s between attempts).
    Returns (token, expires_epoch_s) or raises."""
    reasons = []
    for attempt in range(1, max_attempts + 1):
        rec = {"ts": time.time(), "attempt": attempt, "headless": True}
        ua = random.choice(UA_LIST)
        try:
            token, exp, extra = _browser_attempt(ua)
            rec.update(extra)
            rec["ok"] = True
            rec["token_prefix"] = token.split(":")[0] if ":" in token else token[:12]
            _log_attempt(rec)
            _save_cache(token, exp)
            return token, exp
        except Exception as e:
            rec["fail"] = "%s: %s" % (type(e).__name__, str(e)[:250])
            _log_attempt(rec)
            reasons.append(rec["fail"])
            if attempt < max_attempts:
                time.sleep(2)
    raise RuntimeError("CAP acquisition failed after %d attempts. Reasons: %s"
                       % (max_attempts, json.dumps(reasons, ensure_ascii=False)[:800]))


def _parse_expiry(exp):
    """expires may be ISO string or epoch ms."""
    if exp is None:
        return time.time() + 600
    if isinstance(exp, (int, float)):
        v = float(exp)
        return v / 1000.0 if v > 1e12 else v
    try:
        from datetime import datetime, timezone
        s = str(exp).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return time.time() + 600


def _save_cache(token, exp_epoch):
    os.makedirs(os.path.dirname(TOKEN_CACHE), exist_ok=True)
    with io.open(TOKEN_CACHE, "w", encoding="utf-8") as f:
        json.dump({"token": token, "expires": exp_epoch, "obtained_at": time.time()}, f)


def cache_age():
    """Seconds since the cached token was obtained (None if no cache)."""
    for path in (TOKEN_CACHE, TOKEN_CACHE_LEGACY):
        if not os.path.exists(path):
            continue
        try:
            with io.open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return time.time() - float(d.get("obtained_at", 0))
        except Exception:
            pass
    return None


def load_cached_token(margin_seconds=120):
    for path in (TOKEN_CACHE, TOKEN_CACHE_LEGACY):
        if not os.path.exists(path):
            continue
        try:
            with io.open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if time.time() < float(d.get("expires", 0)) - margin_seconds:
                return d["token"]
        except Exception:
            pass
    return None


def get_token(force=False):
    """Return a valid CAP token (cached if possible)."""
    if not force:
        cached = load_cached_token()
        if cached:
            return cached
    token, _exp = acquire_token()
    return token


if __name__ == "__main__":
    print(json.dumps({"token": get_token()}, ensure_ascii=False))
