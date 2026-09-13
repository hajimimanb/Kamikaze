// fetch_record2.js - fastLogin (guest) then fetchGameRecord
const protobuf = require("protobufjs");
const fs = require("fs");
const path = require("path");
const REPO = path.join(__dirname, "..", "..", "..");
const LIQI_PATH = path.join(REPO, "data", "tmp", "liqi.json");
const VERSION_JSON = JSON.parse(fs.readFileSync(path.join(REPO, "data", "tmp", "mjs_version.json"), "utf8"));
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
  const uuid = process.argv[2];
  const root = protobuf.Root.fromJSON(JSON.parse(fs.readFileSync(LIQI_PATH, "utf8")));
  const versionStr = "web-" + VERSION_JSON.version.replace(/\.w$/, "");
  for (const gw of GATEWAYS) {
    try {
      const ws = await connect(gw);
      // heatbeat (typo name preserved in protocol)
      try {
        const hbType = root.lookupType("lq.ReqHeatBeat");
        const hb = hbType.encode({ no_operation_counter: 0 }).finish();
        await rpc(ws, ".lq.Lobby.heatbeat", hb);
      } catch (e) { /* ignore */ }
      // fastLogin
      const flType = root.lookupType("lq.ReqFastLogin");
      const flResp = await rpc(ws, ".lq.Lobby.fastLogin", flType.encode({ client_version_string: versionStr }).finish());
      const flRes = root.lookupType("lq.ResFastLogin").decode(flResp);
      const flJson = root.lookupType("lq.ResFastLogin").toObject(flRes, { longs: String });
      console.log("fastLogin:", JSON.stringify(flJson).slice(0, 300));
      // fetchGameRecord
      const rqType = root.lookupType("lq.ReqGameRecord");
      const resp = await rpc(ws, ".lq.Lobby.fetchGameRecord",
        rqType.encode({ game_uuid: uuid, client_version_string: versionStr }).finish());
      const resType = root.lookupType("lq.ResGameRecord");
      const res = resType.decode(resp);
      const resJson = resType.toObject(res, { longs: String, defaults: true });
      ws.close();
      console.log("fetchGameRecord: head=", JSON.stringify(resJson.head || null).slice(0, 200));
      console.log("error=", JSON.stringify(resJson.error || null));
      if (res.data && res.data.length) {
        fs.writeFileSync(path.join(REPO, "data", "tmp", "record_data.bin"), res.data);
        console.log("data bytes:", res.data.length, "-> saved record_data.bin");
        // decode
        const wrapper = decodeWrapper(res.data);
        const detType = root.lookupType("lq.GameDetailRecords");
        let recs = null;
        try { recs = detType.toObject(detType.decode(wrapper.data), { longs: String, defaults: true }).records; }
        catch (e) { try { recs = detType.toObject(detType.decode(res.data), { longs: String, defaults: true }).records; } catch (e2) {} }
        console.log("records:", recs ? recs.length : null);
        if (recs) {
          const log = recs.map((rec) => {
            let dataJson = null;
            try {
              const t = root.lookupType("lq." + rec.name);
              const m = t.decode(rec.data);
              dataJson = t.toObject(m, { longs: String, enums: Number, defaults: true });
            } catch (e) {}
            return [dataJson && dataJson.seat !== undefined ? dataJson.seat : 0, { name: rec.name, data: dataJson }];
          });
          fs.writeFileSync(path.join(REPO, "data", "tmp", "fetched_record.json"), JSON.stringify({ head: resJson.head, log }));
          console.log("SAVED fetched_record.json");
        }
      } else if (res.data_url) {
        console.log("data_url:", res.data_url);
      }
      return;
    } catch (e) {
      console.log(gw, "ERR", String(e).slice(0, 150));
    }
  }
}
main().catch((e) => { console.error(String(e)); process.exit(1); });
