// probe_rpcs.js - test public RPCs as guest
const protobuf = require("protobufjs");
const fs = require("fs");
const { execSync } = require("child_process");

const LIQI_PATH = "C:/agentwork/data/tmp/liqi.json";
const VERSION_JSON = JSON.parse(fs.readFileSync("C:/agentwork/data/tmp/mjs_version.json", "utf8"));
const GATEWAYS = ["route-2.maj-soul.com", "route-4.maj-soul.com", "route-5.maj-soul.com", "route-6.maj-soul.com"];

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
  let pos = 0, name = "", data = Buffer.alloc(0);
  while (pos < buf.length) {
    const tag = buf[pos++];
    const field = tag >> 3, wt = tag & 7;
    let v = 0n, shift = 0n, b;
    if (wt === 0) { do { b = buf[pos++]; v |= BigInt(b & 0x7f) << shift; shift += 7n; } while (b & 0x80); }
    else if (wt === 2) {
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
  return new Promise((resolve, reject) => {
    const ws = new WebSocket("wss://" + gatewayHost + "/gateway", { headers: { Origin: "https://game.maj-soul.com" } });
    const timer = setTimeout(() => { try { ws.close(); } catch (e) {} reject(new Error("connect timeout")); }, 15000);
    ws.onopen = () => { clearTimeout(timer); resolve(ws); };
    ws.onerror = () => { clearTimeout(timer); reject(new Error("ws error")); };
  });
}
function rpc(ws, name, dataBuf) {
  return new Promise((resolve, reject) => {
    const idx = (Math.floor(Math.random() * 60000) + 1) & 0xffff;
    const packet = Buffer.concat([Buffer.from([0x02, idx & 0xff, (idx >> 8) & 0xff]), encodeWrapper(name, dataBuf)]);
    const timer = setTimeout(() => reject(new Error("timeout")), 20000);
    async function onMsg(ev) {
      const raw = ev.data instanceof Blob ? await ev.data.arrayBuffer() : ev.data;
      const data = Buffer.from(raw);
      if (data.length < 3 || data[0] !== 3) return;
      const ridx = data[1] | (data[2] << 8);
      if (ridx !== idx) return;
      clearTimeout(timer);
      ws.removeEventListener("message", onMsg);
      resolve(data.slice(3));
    }
    ws.addEventListener("message", onMsg);
    ws.send(packet);
  });
}

async function main() {
  const root = protobuf.Root.fromJSON(JSON.parse(fs.readFileSync(LIQI_PATH, "utf8")));
  const tests = [
    [".lq.Lobby.fetchGameRecordList", "lq.ReqGameRecordList", { start: 0, count: 10, type: 5 }],
    [".lq.Lobby.fetchGameLiveList", "lq.ReqGameLiveList", { filter_id: 0 }],
    [".lq.Lobby.heatbeat", "lq.ReqCommon", { no_operation_counter: 0 }],
  ];
  for (const [method, typeName, payload] of tests) {
    let type = null;
    try { type = root.lookupType(typeName); } catch (e) { type = null; }
    for (const gw of GATEWAYS) {
      try {
        const ws = await connect(gw);
        const data = type ? type.encode(payload).finish() : Buffer.alloc(0);
        const resp = await rpc(ws, method, data);
        ws.close();
        let respJson = null;
        try {
          const t = root.lookupType("lq." + method.split(".").pop().replace(/^fetch/, "Res").replace(/^heatbeat/, "Res"));
          respJson = JSON.stringify(t.toObject(t.decode(resp), { longs: String }));
        } catch (e) {
          respJson = "len=" + resp.length + " hex=" + resp.slice(0, 60).toString("hex");
        }
        console.log(method, gw, "->", respJson.slice(0, 400));
        break;
      } catch (e) {
        console.log(method, gw, "ERR", String(e).slice(0, 120));
      }
    }
  }
}
main().catch((e) => { console.error(String(e)); process.exit(1); });
