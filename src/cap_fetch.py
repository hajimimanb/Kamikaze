"""Solve CAP challenge from akcap.pikapika.me and fetch Majsoul data via amae-koromo mirrors.

Uses Playwright with the system Edge (browser fingerprint) to:
  1. POST {cap_endpoint}/challenge
  2. solve sha256-pow (JS fallback) / rsw challenges
  3. POST {cap_endpoint}/redeem -> token
  4. call data mirror API with Bearer token
"""
import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from utils.paths import tmp_dir

CAP_ENDPOINT = "https://akcap.pikapika.me/14f343ec68/"
DATA_MIRRORS = [
    "https://5-data.amae-koromo.com/",
    "https://1.data.amae-koromo.com/",
    "https://4.data.amae-koromo.com/",
]

JS_SOLVER = r"""
(async () => {
async function capSolve(endpoint, mirrors, apiPath) {
  const chResp = await fetch(endpoint + 'challenge', { method: 'POST', headers: { 'Origin': 'https://amae-koromo.sapk.ch', 'Referer': 'https://amae-koromo.sapk.ch/' } });
  if (!chResp.ok) return { error: "challenge http " + chResp.status };
  let resp;
  try { resp = await chResp.json(); } catch (e) { return { error: "challenge parse: " + e }; }
  if (resp.error) return { error: "challenge error: " + resp.error };

  let challenges = [];
  if (resp.format === 2 && Array.isArray(resp.challenges)) {
    challenges = resp.challenges;
  } else {
    const { challenge, token } = resp;
    function fnv1a(str) {
      let hash = 2166136261;
      for (let i = 0; i < str.length; i++) {
        hash ^= str.charCodeAt(i);
        hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
      }
      return hash >>> 0;
    }
    function prng(seed, length) {
      let state = fnv1a(seed);
      let result = "";
      function next() {
        state ^= state << 13;
        state ^= state >>> 17;
        state ^= state << 5;
        return state >>> 0;
      }
      while (result.length < length) {
        const rnd = next();
        result += rnd.toString(16).padStart(8, "0");
      }
      return result.slice(0, length);
    }
    let ii = 0;
    for (let i = 0; i < challenge.c; i++) {
      ii++;
      challenges.push({
        protocol: "sha256-pow",
        payload: { salt: prng(`${token}${ii}`, challenge.s), target: prng(`${token}${ii}d`, challenge.d) },
      });
    }
  }

  function sha256Sync(bytes) {
    const K = [0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
    const H = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    const l = bytes.length;
    const withLen = l + 9;
    const total = Math.ceil(withLen / 64) * 64;
    const msg = new Uint8Array(total);
    msg.set(bytes);
    msg[l] = 0x80;
    const bitLenHi = Math.floor(l / 0x20000000);
    const bitLenLo = (l << 3) >>> 0;
    const dv = new DataView(msg.buffer);
    dv.setUint32(total - 8, bitLenHi, false);
    dv.setUint32(total - 4, bitLenLo, false);
    const w = new Uint32Array(64);
    for (let off = 0; off < total; off += 64) {
      for (let i = 0; i < 16; i++) w[i] = dv.getUint32(off + i * 4, false);
      for (let i = 16; i < 64; i++) {
        const s0 = ((w[i-15] >>> 7) | (w[i-15] << 25)) ^ ((w[i-15] >>> 18) | (w[i-15] << 14)) ^ (w[i-15] >>> 3);
        const s1 = ((w[i-2] >>> 17) | (w[i-2] << 15)) ^ ((w[i-2] >>> 19) | (w[i-2] << 13)) ^ (w[i-2] >>> 10);
        w[i] = (w[i-16] + s0 + w[i-7] + s1) >>> 0;
      }
      let a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
      for (let i = 0; i < 64; i++) {
        const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
        const ch = (e & f) ^ (~e & g);
        const t1 = (h + S1 + ch + K[i] + w[i]) >>> 0;
        const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
        const maj = (a & b) ^ (a & c) ^ (b & c);
        const t2 = (S0 + maj) >>> 0;
        h=g; g=f; f=e; e=(d+t1)>>>0; d=c; c=b; b=a; a=(t1+t2)>>>0;
      }
      H[0]=(H[0]+a)>>>0; H[1]=(H[1]+b)>>>0; H[2]=(H[2]+c)>>>0; H[3]=(H[3]+d)>>>0;
      H[4]=(H[4]+e)>>>0; H[5]=(H[5]+f)>>>0; H[6]=(H[6]+g)>>>0; H[7]=(H[7]+h)>>>0;
    }
    const out = new Uint8Array(32);
    const odv = new DataView(out.buffer);
    for (let i = 0; i < 8; i++) odv.setUint32(i * 4, H[i], false);
    return out;
  }

  const encoder = new TextEncoder();
  async function solvePow(salt, target) {
    let nonce = 0;
    const batchSize = 20000;
    const targetBits = target.length * 4;
    const fullBytes = Math.floor(targetBits / 8);
    const remainingBits = targetBits % 8;
    const paddedTarget = target.length % 2 === 0 ? target : target + "0";
    const targetBytesLength = paddedTarget.length / 2;
    const targetBytes = new Uint8Array(targetBytesLength);
    for (let k = 0; k < targetBytesLength; k++) {
      targetBytes[k] = parseInt(paddedTarget.substring(k * 2, k * 2 + 2), 16);
    }
    const partialMask = remainingBits > 0 ? (0xff << (8 - remainingBits)) & 0xff : 0;
    while (true) {
      for (let i = 0; i < batchSize; i++) {
        const inputString = salt + nonce;
        const hashBytes = (typeof crypto !== "undefined" && crypto.subtle) ? new Uint8Array(await crypto.subtle.digest("SHA-256", encoder.encode(inputString))) : sha256Sync(encoder.encode(inputString));

        let matches = true;
        for (let k = 0; k < fullBytes; k++) {
          if (hashBytes[k] !== targetBytes[k]) { matches = false; break; }
        }
        if (matches && remainingBits > 0) {
          if ((hashBytes[fullBytes] & partialMask) !== (targetBytes[fullBytes] & partialMask)) matches = false;
        }
        if (matches) return { nonce: Number(nonce) };
        nonce++;
      }
      await new Promise(r => setTimeout(r, 0));
    }
  }
  async function solveRsw(N, x, t) {
    let y = BigInt("0x" + x);
    const n = BigInt("0x" + N);
    const tt = t | 0;
    for (let i = 0; i < tt; i += 200000) {
      const end = Math.min(tt, i + 200000);
      for (let j = i; j < end; j++) y = (y * y) % n;
      await new Promise(r => setTimeout(r, 0));
    }
    return { y: y.toString(16) };
  }

  async function runInstrumentation(instrBytes) {
    if (!instrBytes) return null;
    const bin = atob(instrBytes);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    const scriptText = new TextDecoder().decode(await new Response(
      new Blob([arr]).stream().pipeThrough(new DecompressionStream("deflate-raw"))
    ).arrayBuffer());
    window.__capInstrScript = scriptText;
    return new Promise((resolve) => {
      var timeout = setTimeout(() => { cleanup(); resolve({ __timeout: true }); }, 20000);
      var iframe = document.createElement("iframe");
      iframe.setAttribute("sandbox", "allow-scripts");
      iframe.style.cssText = "position:absolute;width:1px;height:1px;top:-9999px;left:-9999px;border:none;opacity:0;pointer-events:none;";
      var resolved = false;
      function cleanup() {
        if (resolved) return;
        resolved = true;
        clearTimeout(timeout);
        window.removeEventListener("message", handler);
        if (iframe.parentNode) iframe.parentNode.removeChild(iframe);
      }
      function handler(ev) {
        if (!iframe.contentWindow || ev.source !== iframe.contentWindow) return;
        var d = ev.data;
        if (!d || typeof d !== "object") return;
        if (d.type === "cap:instr") {
          cleanup();
          if (d.blocked) resolve({ __blocked: true, blockReason: d.blockReason || "automated_browser" });
          else if (d.result) resolve(d.result);
          else resolve({ __timeout: true });
        } else if (d.type === "cap:error") {
          cleanup();
          resolve({ __timeout: true });
        }
      }
      window.addEventListener("message", handler);
      iframe.srcdoc = '<!DOCTYPE html><html><head><meta charset="UTF-8"></head><body><script>' + scriptText + '\n</scr' + 'ipt></body></html>';
      document.body.appendChild(iframe);
    });
  }

  const isFormat1 = resp.format !== 2;
  const solutions = [];
  for (const ch of challenges) {
    if (ch.protocol === "sha256-pow") {
      const s = await solvePow(ch.payload.salt, ch.payload.target);
      solutions.push(isFormat1 ? s.nonce : s);
    } else if (ch.protocol === "rsw") {
      solutions.push(await solveRsw(ch.payload.N, ch.payload.x, ch.payload.t));
    } else if (ch.protocol === "instrumentation") {
      solutions.push({ timeout: true });
    } else {
      return { error: "unknown protocol " + ch.protocol };
    }
  }
  let instrOut = null;
  if (resp.instrumentation) {
    instrOut = await runInstrumentation(resp.instrumentation);
    if (instrOut && instrOut.__blocked) {
      return { error: "instrumentation blocked: " + instrOut.blockReason, instrScriptHead: window.__capInstrScript ? window.__capInstrScript.slice(0, 60000) : null };
    }
  }
  const redeemResp = await fetch(endpoint + "redeem", {
    method: "POST",
    body: JSON.stringify({ token: resp.token, solutions, ...(instrOut && !instrOut.__timeout ? { instr: instrOut } : {}) }),
    headers: { "Content-Type": "application/json" },
  });
  let rj;
  try { rj = await redeemResp.json(); } catch (e) { return { error: "redeem parse: " + e }; }
  if (!rj.success) return { error: "redeem failed: " + JSON.stringify(rj).slice(0, 300), format: isFormat1, sampleSol: solutions[0], instrOut: instrOut ? JSON.stringify(instrOut).slice(0, 200) : null, includedInstr: !!(instrOut && !instrOut.__timeout) };
  const capToken = rj.token;

  for (const m of mirrors) {
    try {
      const apiResp = await fetch(m + apiPath, { headers: { authorization: "Bearer " + capToken } });
      if (apiResp.ok) {
        const body = await apiResp.text();
        return { ok: true, mirror: m, token: capToken, body };
      }
    } catch (e) {}
  }
  return { error: "all mirrors failed", token: capToken };
}
return await capSolve(__ENDPOINT__, __MIRRORS__, __APIPATH__);
})().catch(e => ({ error: String(e && e.message || e) }));
"""


async def main():
    api_path = sys.argv[1] if len(sys.argv) > 1 else "api/v2/pl4/games/0/0?limit=1"
    js = (JS_SOLVER.replace("__ENDPOINT__", json.dumps(CAP_ENDPOINT)).replace("__MIRRORS__", json.dumps(DATA_MIRRORS)).replace("__APIPATH__", json.dumps(api_path)))
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            channel="msedge",
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        await browser.new_context.__self__ if False else None
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
        )
        await ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
        page = await ctx.new_page()
        await page.goto("about:blank")
        result = await page.evaluate(js)
        with open(str(tmp_dir() / "last_cap_result.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False))
        print(json.dumps(result, ensure_ascii=False)[:20000])
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())