// probe_handshake.js - exact client handshake then fetchGameRecord
const protobuf = require("protobufjs");
const fs = require("fs");
const path = require("path");
const REPO = path.join(__dirname, "..", "..", "..");
const LIQI_PATH = path.join(REPO, "data", "tmp", "liqi.json");
const VERSION_JSON = JSON.parse(fs.readFileSync(path.join(REPO, "data", "tmp", "mjs_version.json"), "utf8"));

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
  const gateways = ["route-2.maj-soul.com", "route-4.maj-soul.com", "route-5.maj-soul.com"];
  for (const gw of gateways) {
    try {
      const ws = await connect(gw);
      // 1. requestConnection (like the real client)
      const rcType = root.lookupType("lq.ReqRequestConnection");
      const rc = rcType.encode({ type: 1, route_id: gw.replace(/\..*$/, ""),
                                 timestamp: Math.floor(Date.now()), platform: "Web" }).finish();
      const rcResp = await rpc(ws, ".lq.Route.requestConnection", rc);
      console.log(gw, "requestConnection:", rcResp.toString("hex").slice(0, 100));
      // 2. Route.heartbeat
      const hb = root.lookupType("lq.ReqHeatBeat").encode({ delay: 0, no_operation_counter: 0, platform: 11, network_quality: 0 }).finish();
      await rpc(ws, ".lq.Route.heartbeat", hb);
      console.log(gw, "heartbeat ok");
      // 3. fastLogin
      const fl = root.lookupType("lq.ReqFastLogin").encode({ client_version_string: versionStr }).finish();
      const flResp = await rpc(ws, ".lq.Lobby.fastLogin", fl);
      console.log(gw, "fastLogin:", flResp.toString("hex").slice(0, 100));
      // 4. fetchGameRecord
      const rq = root.lookupType("lq.ReqGameRecord").encode({ game_uuid: uuid, client_version_string: versionStr }).finish();
      const resp = await rpc(ws, ".lq.Lobby.fetchGameRecord", rq);
      console.log(gw, "fetchGameRecord:", resp.toString("hex").slice(0, 160), "len", resp.length);
      ws.close();
    } catch (e) {
      console.log(gw, "ERR", String(e).slice(0, 150));
    }
  }
}
main().catch((e) => { console.error(String(e)); process.exit(1); });
