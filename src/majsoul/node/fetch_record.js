// fetch_record.js - fetch one Majsoul game record via official WS gateway (guest attempt)
// Usage: node fetch_record.js <uuid> [gateway_host]
const protobuf = require("protobufjs");
const fs = require("fs");
const path = require("path");
const REPO = path.join(__dirname, "..", "..", "..");

const LIQI_PATH = path.join(REPO, "data", "tmp", "liqi.json");
const VERSION_JSON = JSON.parse(fs.readFileSync(path.join(REPO, "data", "tmp", "mjs_version.json"), "utf8")
  .replace(/^\uFEFF/, ""));
const GATEWAYS = [
  "route-2.maj-soul.com",
  "route-3.maj-soul.com",
  "route-4.maj-soul.com",
  "route-5.maj-soul.com",
  "route-6.maj-soul.com",
];

function encodeVarint(buf, value) {
  let v = BigInt(value);
  while (true) {
    let b = Number(v & 0x7fn);
    v >>= 7n;
    if (v !== 0n) b |= 0x80;
    buf.push(b);
    if (v === 0n) break;
  }
}

function encodeWrapper(name, data) {
  const buf = [];
  buf.push(0x0a);
  encodeVarint(buf, Buffer.byteLength(name));
  for (const b of Buffer.from(name, "utf8")) buf.push(b);
  if (data && data.length) {
    buf.push(0x12);
    encodeVarint(buf, data.length);
    for (const b of data) buf.push(b);
  }
  return Buffer.from(buf);
}

function decodeWrapper(buf) {
  // returns {name, data}
  let pos = 0;
  let name = "";
  let data = Buffer.alloc(0);
  while (pos < buf.length) {
    const tag = buf[pos++];
    const field = tag >> 3;
    const wt = tag & 7;
    let v = 0n, shift = 0n, b;
    if (wt === 0) {
      do { b = buf[pos++]; v |= BigInt(b & 0x7f) << shift; shift += 7n; } while (b & 0x80);
    } else if (wt === 2) {
      do { b = buf[pos++]; v |= BigInt(b & 0x7f) << shift; shift += 7n; } while (b & 0x80);
      const len = Number(v);
      if (field === 1) name = buf.slice(pos, pos + len).toString("utf8");
      else if (field === 2) data = buf.slice(pos, pos + len);
      pos += len;
    } else if (wt === 5) pos += 4;
    else if (wt === 1) pos += 8;
  }
  return { name, data };
}

function connect(gatewayHost) {
  const url = "wss://" + gatewayHost + "/gateway";
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url, {
      headers: { Origin: "https://game.maj-soul.com" },
    });
    const timer = setTimeout(() => { try { ws.close(); } catch (e) {} reject(new Error("ws connect timeout")); }, 15000);
    ws.onopen = () => { clearTimeout(timer); resolve(ws); };
    ws.onerror = (e) => { clearTimeout(timer); reject(new Error("ws error: " + (e.message || "unknown"))); };
  });
}

function rpc(ws, pb, name, reqType, payload, timeoutMs) {
  return new Promise((resolve, reject) => {
    const idx = (Math.floor(Math.random() * 60000) + 1) & 0xffff;
    const reqMsg = reqType.encode(payload).finish();
    const wrapped = encodeWrapper(name, reqMsg);
    const packet = Buffer.concat([Buffer.from([0x02, idx & 0xff, (idx >> 8) & 0xff]), wrapped]);
    const timer = setTimeout(() => { ws.removeEventListener("message", onMsg); reject(new Error("rpc timeout")); }, timeoutMs || 30000);
    async function onMsg(ev) {
      const raw = ev.data instanceof Blob ? await ev.data.arrayBuffer() : ev.data;
      const data = Buffer.from(raw);
      if (data.length < 3) return;
      if (data[0] === 3) {
        const ridx = data[1] | (data[2] << 8);
        if (ridx !== idx) return;
        clearTimeout(timer);
        ws.removeEventListener("message", onMsg);
        try {
          const w = decodeWrapper(data.slice(3));
          resolve(w);
        } catch (e) { reject(e); }
      } else if (data[0] === 2) {
        // request from server: ignore
      }
    }
    ws.addEventListener("message", onMsg);
    ws.send(packet);
  });
}

async function main() {
  const uuid = process.argv[2];
  const forceGateway = process.argv[3];
  if (!uuid) { console.error("usage: node fetch_record.js <uuid> [gateway]"); process.exit(2); }
  const root = protobuf.Root.fromJSON(JSON.parse(fs.readFileSync(LIQI_PATH, "utf8")));
  const lq = root.lookupType("lq.ReqGameRecord");
  const resType = root.lookupType("lq.ResGameRecord");
  const wrapperType = root.lookupType("lq.Wrapper");
  const detailType = root.lookupType("lq.GameDetailRecords");

  const versionStr = "web-" + VERSION_JSON.version.replace(/\.w$/, "");
  const gateways = forceGateway ? [forceGateway] : GATEWAYS;
  let lastErr = null;
  for (const gw of gateways) {
    try {
      const ws = await connect(gw);
      // heartbeat like the reference (Lobby.heatbeat) - optional
      const w = await rpc(ws, root, ".lq.Lobby.fetchGameRecord", lq,
        { game_uuid: uuid, client_version_string: versionStr }, 30000);
      ws.close();
      const res = resType.decode(w.data);
      const resJson = resType.toObject(res, { longs: String, enums: Number, defaults: true });
      if (res.error && res.error.code) {
        console.error(JSON.stringify({ gateway: gw, error: resJson.error }));
        lastErr = resJson.error;
        continue;
      }
      if (!res.data || !res.data.length) {
        console.error(JSON.stringify({ gateway: gw, error: "no data in response", head: resJson.head }));
        lastErr = "no data";
        continue;
      }
      // decode data: Wrapper -> GameDetailRecords
      let recs;
      let inner = decodeWrapper(res.data);
      if (inner.name === "" && inner.data.length) {
        // data may be raw GameDetailRecords without wrapper
        try {
          const det = detailType.decode(inner.data);
          recs = detailType.toObject(det, { longs: String, enums: Number, defaults: true }).records;
        } catch (e) {
          recs = null;
        }
        if (!recs) {
          try {
            const det = detailType.decode(res.data);
            recs = detailType.toObject(det, { longs: String, enums: Number, defaults: true }).records;
          } catch (e2) { recs = null; }
        }
        if (!recs) {
          // data might be Wrapper-wrapped
          inner = decodeWrapper(res.data);
          const det = detailType.decode(inner.data);
          recs = detailType.toObject(det, { longs: String, enums: Number, defaults: true }).records;
        }
      } else {
        const det = detailType.decode(inner.data);
        recs = detailType.toObject(det, { longs: String, enums: Number, defaults: true }).records;
      }
      const log = (recs || []).map((rec) => {
        const innerName = rec.name;
        // rec.data is a Buffer (bytes) -> decode with the named type if known, else raw
        let dataJson;
        try {
          const t = root.lookupType("lq." + innerName);
          const m = t.decode(rec.data);
          dataJson = t.toObject(m, { longs: String, enums: Number, defaults: true });
        } catch (e) {
          dataJson = null;
        }
        const seat = dataJson && dataJson.seat !== undefined ? dataJson.seat : 0;
        return [seat, { name: innerName, data: dataJson }];
      });
      console.log(JSON.stringify({ head: resJson.head, log }));
      return;
    } catch (e) {
      lastErr = String(e && e.message || e);
      console.error(JSON.stringify({ gateway: gw, error: lastErr }));
    }
  }
  console.error(JSON.stringify({ fatal: lastErr }));
  process.exit(1);
}

main().catch((e) => { console.error(String(e)); process.exit(1); });
