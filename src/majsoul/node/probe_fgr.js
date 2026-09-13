// probe_fgr.js - dump raw fetchGameRecord response bytes
const protobuf = require("protobufjs");
const fs = require("fs");
const path = require("path");
const REPO = path.join(__dirname, "..", "..", "..");
const LIQI_PATH = path.join(REPO, "data", "tmp", "liqi.json");
const VERSION_JSON = JSON.parse(fs.readFileSync(path.join(REPO, "data", "tmp", "mjs_version.json"), "utf8"));
const GATEWAYS = ["route-2.maj-soul.com", "route-4.maj-soul.com"];

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
    const timer = setTimeout(() => reject(new Error("timeout")), 30000);
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
  const uuid = process.argv[2] || "260805-49733dc4-762e-4155-9a6b-71dd9d337370";
  const root = protobuf.Root.fromJSON(JSON.parse(fs.readFileSync(LIQI_PATH, "utf8")));
  const versionStr = "web-" + VERSION_JSON.version.replace(/\.w$/, "");
  for (const gw of GATEWAYS) {
    try {
      const ws = await connect(gw);
      try {
        const hb = root.lookupType("lq.ReqHeatBeat").encode({ no_operation_counter: 0 }).finish();
        await rpc(ws, ".lq.Lobby.heatbeat", hb);
      } catch (e) {}
      const fl = root.lookupType("lq.ReqFastLogin").encode({ client_version_string: versionStr }).finish();
      const flResp = await rpc(ws, ".lq.Lobby.fastLogin", fl);
      console.log(gw, "fastLogin raw:", flResp.toString("hex").slice(0, 120));
      // fetchGameRecord for several uuids
      for (const u of [uuid, "999ds5F4N7u", "260806-00000000-0000-0000-0000-000000000000"]) {
        const rq = root.lookupType("lq.ReqGameRecord").encode({ game_uuid: u, client_version_string: versionStr }).finish();
        const resp = await rpc(ws, ".lq.Lobby.fetchGameRecord", rq);
        console.log(gw, "fgr", u.slice(0, 20), "raw:", resp.toString("hex").slice(0, 200), "len", resp.length);
      }
      // fetchGameRecordListV2 as guest
      const v2 = root.lookupType("lq.ReqGameRecordListV2").encode({ modes: [16], begin_time: Math.floor(Date.now()/1000) - 86400, end_time: Math.floor(Date.now()/1000) }).finish();
      const v2resp = await rpc(ws, ".lq.Lobby.fetchGameRecordListV2", v2);
      console.log(gw, "listV2 raw:", v2resp.toString("hex").slice(0, 200), "len", v2resp.length);
      ws.close();
    } catch (e) {
      console.log(gw, "ERR", String(e).slice(0, 120));
    }
  }
}
main().catch((e) => { console.error(String(e)); process.exit(1); });
