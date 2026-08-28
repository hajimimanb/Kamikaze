# -*- coding: utf-8 -*-
"""RL 训练实时监控面板：真实进度 + 数据曲线 + 异常报出。
数据源: logs/rl_train_metrics.jsonl（每 epoch）+ logs/rl_train.txt（200 局粒度进度）
输出: checkpoints/rl_train_panel.html（5s 刷新）
异常机制: ratio 偏离 / NaN / gate 锁死 / clip 率 / value 尺度 / wr 回退 / 进度停滞 / 进程死亡
"""
import datetime, json, os, re, subprocess, time

METRICS = "C:/agentwork/logs/rl_train_metrics.jsonl"
TRAINLOG = "C:/agentwork/logs/rl_train.txt"
GAMES_PATH = "C:/agentwork/logs/rl_train_games.jsonl"
OUT = "C:/agentwork/checkpoints/rl_train_panel.html"
PHI_MON = "C:/agentwork/logs/rl_phi_monitor.json"

STYLE = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5"><title>RL 训练实时监控</title>
<style>
 body{background:#0f172a;color:#e2e8f0;font-family:Consolas,'Microsoft YaHei',monospace;margin:16px}
 h1{font-size:19px;color:#f8fafc} .sub{color:#94a3b8;font-size:12px;margin-bottom:14px}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:14px}
 .card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:10px 14px}
 .card .k{color:#94a3b8;font-size:11px} .card .v{font-size:17px;font-weight:700;margin-top:2px}
 .panel{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;margin-bottom:14px}
 .panel h2{font-size:14px;color:#f8fafc;margin:0 0 8px}
 .chart{width:100%;height:auto}
 table{width:100%;border-collapse:collapse;font-size:11px}
 th{color:#94a3b8;text-align:left;padding:3px 6px;border-bottom:1px solid #334155}
 td{padding:3px 6px;border-bottom:1px solid #1e293b}
 .alert{background:#7f1d1d;border:1px solid #f87171;border-radius:8px;padding:10px 14px;margin-bottom:10px}
 .warn{background:#78350f;border:1px solid #fbbf24;border-radius:8px;padding:10px 14px;margin-bottom:10px}
 .bad{color:#f87171} .ok{color:#4ade80}
</style></head><body>"""


def read_metrics():
    pts = []
    if os.path.exists(METRICS):
        for ln in open(METRICS, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    pts.append(json.loads(ln))
                except Exception:
                    pass
    return pts


def read_progress():
    """训练进度行 [rl] epoch X game Y/Z avg_p=.. avg_v=.. H=.. ratio=.. clip=.."""
    last = None
    if os.path.exists(TRAINLOG):
        for ln in open(TRAINLOG, encoding="utf-8", errors="ignore"):
            m = re.search(r"\[rl\] epoch (\d+) game (\d+)/(\d+) avg_p=([\d.-]+) avg_v=([\d.-]+) H=([\d.]+) ratio=([\d.]+) clip=([\d.]+)", ln)
            if m:
                last = {"epoch": int(m.group(1)), "game": int(m.group(2)), "games": int(m.group(3)),
                        "lp": float(m.group(4)), "lv": float(m.group(5)), "H": float(m.group(6)),
                        "ratio": float(m.group(7)), "clip": float(m.group(8))}
    return last


def read_phi_monitor():
    """读取 Φ 漂移监控日志（P0-B compliance）：返回 dict 或 None。"""
    if not os.path.exists(PHI_MON):
        return None
    try:
        return json.load(open(PHI_MON, encoding="utf-8"))
    except Exception:
        return None


def check_alerts(pts, last, prog, done=False):
    alerts = []
    if pts:
        m = pts[-1]
        if m.get("ratio") is not None and m["ratio"] > 0 and not (0.8 <= m["ratio"] <= 1.2):
            alerts.append(("bad", "ratio 偏离 [0.8,1.2]: %.4f（采样/重放 logp 不一致风险）" % m["ratio"]))
        for k in ("lp", "lv", "H"):
            v = m.get(k)
            if v is not None and (v != v or v in (float("inf"), float("-inf"))):
                alerts.append(("bad", "%s = NaN/Inf（训练数值异常）" % k))
        if m.get("clip_rate") is not None and m["clip_rate"] > 0.10:
            alerts.append(("warn", "PPO clip 率 %.1f%% > 10%%（更新步长过大）" % (100 * m["clip_rate"])))
        g = m.get("gate_mean")
        if g is not None and g < 0.02 and len(pts) >= 2:
            alerts.append(("warn", "gate 均值 %.4f < 0.02（事件旁支疑似锁死，检查分层 lr）" % g))
        vs = m.get("value_std")
        if vs is not None and (vs < 0.1 or vs > 100):
            alerts.append(("warn", "value_std %.2f 异常（尺度失配，检查 Φ 奖励/adv 归一化）" % vs))
        if len(pts) >= 2:
            w0, w1 = pts[-2].get("wr_vs_sl"), m.get("wr_vs_sl")
            if w0 is not None and w1 is not None and w1 < w0 - 0.05:
                alerts.append(("warn", "wr_vs_sl 回退 %.3f → %.3f（策略退化信号）" % (w0, w1)))
    else:
        if time.time() - os.path.getmtime(TRAINLOG) > 60 if os.path.exists(TRAINLOG) else True:
            alerts.append(("warn", "尚无 metrics（训练未启动或未完成首个 epoch）"))
    # 进度停滞 / 进程死亡
    # P0② 死亡阈值自适应：基于最近两个 metrics 的 elapsed 间隔（epoch 时长），无 metrics 用 300s
    if os.path.exists(TRAINLOG):
        age = time.time() - os.path.getmtime(TRAINLOG)
        thr = 300
        if len(pts) >= 2:
            d_el = pts[-1].get("elapsed_s", 0) - pts[-2].get("elapsed_s", 0)
            if d_el > 0:
                thr = max(300, int(d_el * 1.5))
        if age > thr and not done:
            alerts.append(("bad", "训练日志 %ds 无更新（>自适应阈值 %ds，进程可能死亡/卡死）" % (int(age), thr)))
    return alerts


def svg_series(series, w=760, h=150, colors=None):
    colors = colors or ["#60a5fa", "#f472b6", "#4ade80", "#fbbf24", "#a78bfa", "#f87171"]
    allpts = [p for _, pts in series for p in pts if p is not None and p[1] is not None]
    if len(allpts) < 2:
        return '<div class="empty" style="color:#64748b;padding:10px;text-align:center">数据积累中…</div>'
    xs = [p[0] for p in allpts]; ys = [p[1] for p in allpts]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    if x1 == x0: x1 = x0 + 1
    if y1 == y0: y1 = y0 + 1
    pad = 8
    px = lambda x: pad + (x - x0) / (x1 - x0) * (w - 2 * pad)
    py = lambda y: h - pad - (y - y0) / (y1 - y0) * (h - 2 * pad)
    gl = []
    for i in range(5):
        gx = x0 + (x1 - x0) * i / 4
        gl.append('<line x1="%.1f" y1="0" x2="%.1f" y2="%d" stroke="#1e293b"/>'
                  '<text x="%.1f" y="%d" fill="#64748b" font-size="9" text-anchor="middle">%.0f</text>'
                  % (px(gx), px(gx), h, px(gx), h - 2, gx))
        gy = y0 + (y1 - y0) * i / 4
        gl.append('<line x1="0" y1="%.1f" x2="%d" y2="%.1f" stroke="#1e293b"/>'
                  '<text x="4" y="%.1f" fill="#64748b" font-size="9">%.2f</text>'
                  % (py(gy), w, py(gy), py(gy) - 2, gy))
    gl.append('<text x="%d" y="%d" fill="#475569" font-size="9" text-anchor="end">x: 局/epoch</text>'
              '<text x="6" y="12" fill="#475569" font-size="9">y: 值</text>' % (w - 4, h - 4))
    polylines, leg = "", ""
    for i, (name, pts) in enumerate(series):
        good = [p for p in pts if p is not None and p[1] is not None]
        if not good: continue
        c = colors[i % len(colors)]
        p = " ".join("%.1f,%.1f" % (px(x), py(y)) for x, y in good)
        polylines += '<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>' % (p, c)
        leg += '<span style="margin-right:10px;font-size:11px;color:#94a3b8"><font color="%s">■</font> %s</span>' % (c, name)
    return ('<svg viewBox="0 0 %d %d" class="chart">%s%s</svg><div style="margin-top:2px">%s</div>'
            % (w, h, "".join(gl), polylines, leg))


from _games_block import games_block


def sys_stats():
    """GPU/CPU 占用（每 5s 查询）。"""
    gpu = "—"
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if o.stdout.strip():
            u, mu, mt = [x.strip() for x in o.stdout.strip().split(",")]
            gpu = "%s%% · %s/%sMB" % (u, mu, mt)
    except Exception:
        pass
    cpu = "—"
    try:
        o = subprocess.run(["wmic", "cpu", "get", "loadpercentage"],
                           capture_output=True, text=True, timeout=5)
        for ln in o.stdout.splitlines():
            if ln.strip().isdigit():
                cpu = ln.strip() + "%"
                break
    except Exception:
        pass
    return gpu, cpu


def _fmt_hms(sec):
    if sec is None or sec < 0:
        return "—"
    h, rem = int(sec) // 3600, int(sec) % 3600
    return "%dh %02dm" % (h, rem // 60)


_RES_CACHE = {"t": 0.0, "data": None}
_NCPU = None


def _ncpu():
    global _NCPU
    if _NCPU is None:
        try:
            _NCPU = int(subprocess.run(["wmic", "cpu", "get", "NumberOfLogicalProcessors", "/value"],
                                       capture_output=True, text=True, timeout=10).stdout.split("=")[-1].strip().split("\r")[0])
        except Exception:
            _NCPU = 8
    return max(_NCPU, 1)


def _res_snapshot():
    """采集 3 进程 + 系统资源（CPU/GPU/内存），缓存 15s。用 psutil 可靠取进程资源。"""
    import time as _t
    import psutil as _psutil
    now = _t.time()
    if _RES_CACHE["data"] is not None and now - _RES_CACHE["t"] < 15:
        return _RES_CACHE["data"]
    d = {"train": {"cpu": 0.0, "mem": 0, "gpu_mem": 0},
         "eval": {"cpu": 0.0, "mem": 0, "gpu_mem": 0},
         "panel": {"cpu": 0.0, "mem": 0, "gpu_mem": 0}}
    try:
        # psutil：按 cmdline 匹配三进程（含 shim+system 树），可靠取 CPU%/内存
        roles = [("train", "train_rl_vec"), ("eval", "eval_sliding"), ("panel", "rl_panel_server")]
        _pids_by_role = {}
        procs = []
        try:
            for pr in _psutil.process_iter(["pid", "cmdline", "memory_info"]):
                try:
                    cl = pr.info.get("cmdline") or []
                    cmd = " ".join(cl)
                    mi = pr.info.get("memory_info")
                    rss = mi.rss if mi else 0
                    # 复用缓存的 Process 实例（cpu_percent 需同实例两次采样）
                    _pr = _RES_CACHE.get("ps_%d" % pr.pid) or _psutil.Process(pr.pid)
                    _RES_CACHE["ps_%d" % pr.pid] = _pr
                    procs.append((pr.pid, cmd, rss, _pr))
                except Exception:
                    pass
        except Exception:
            pass
        for name, pat in roles:
            pids = [pid for pid, cmd, _rss, _p in procs if pat in cmd]
            _pids_by_role[name] = pids
            d[name]["mem"] = sum(rss for pid, cmd, rss, _p in procs if pid in pids)
            d[name]["cpu"] = 0.0
            for pid, cmd, _rss, pr in procs:
                if pid in pids:
                    try:
                        d[name]["cpu"] += pr.cpu_percent(interval=None) / _ncpu()
                    except Exception:
                        pass
        # GPU（compute-apps 按 PID 分显存）
        try:
            g = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            for ln in g.splitlines():
                parts = ln.split(",")
                if len(parts) >= 2:
                    try:
                        pid = int(parts[0].strip())
                        mem = int(parts[1].strip())
                    except Exception:
                        continue
                    for name in ("train", "eval", "panel"):
                        if pid in _pids_by_role.get(name, []):
                            d[name]["gpu_mem"] = mem
        except Exception:
            pass
        # 系统
        sys_cpu = 0.0
        try:
            c = subprocess.run(["wmic", "cpu", "get", "LoadPercentage", "/value"],
                               capture_output=True, text=True, timeout=10).stdout
            for ln in c.splitlines():
                if ln.strip().startswith("LoadPercentage="):
                    sys_cpu = float(ln.strip().split("=")[1])
        except Exception:
            pass
        d["sys_cpu"] = sys_cpu
        d["total_mem_kb"] = 0
        d["free_mem_kb"] = 0
        try:
            o = subprocess.run(["wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/value"],
                               capture_output=True, text=True, timeout=10).stdout
            for ln in o.splitlines():
                if ln.strip().startswith("TotalVisibleMemorySize="):
                    d["total_mem_kb"] = int(ln.strip().split("=")[1])
                elif ln.strip().startswith("FreePhysicalMemory="):
                    d["free_mem_kb"] = int(ln.strip().split("=")[1])
        except Exception:
            pass
        d["gpu_total"] = 0
        d["gpu_used"] = 0
        d["gpu_util"] = 0
        try:
            gg = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            for ln in gg.splitlines():
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) >= 3:
                    d["gpu_util"] = float(parts[0])
                    d["gpu_used"] = int(parts[1])
                    d["gpu_total"] = int(parts[2])
        except Exception:
            pass
        _RES_CACHE["data"] = d
        _RES_CACHE["t"] = now
    except Exception:
        pass
    return d


def _resource_html():
    """页面最上方：资源占用（3 进程 CPU/GPU/内存 分段进度条）。"""
    d = _res_snapshot()
    ncpu = _ncpu()
    tb = 1024.0 * 1024.0
    # CPU 三段 + 其他 + 剩余（按整机 100%）
    t = d.get("train", {}).get("cpu", 0) or 0
    e_ = d.get("eval", {}).get("cpu", 0) or 0
    p = d.get("panel", {}).get("cpu", 0) or 0
    sysc = d.get("sys_cpu", 0) or 0
    other = max(0.0, sysc - t - e_ - p)
    rest = max(0.0, 100.0 - sysc)
    REST_C = "#334155"   # 剩余/可用段：浅灰蓝，与深色背景 #1e293b 区分
    def bar(segs):
        # 归一化：段宽按总和缩放，防挤压
        tot_w = sum(max(0.0, w) for w, _c, _t in segs)
        tot_w = max(tot_w, 1e-9)
        parts = "".join('<div style="width:%s%%;background:%s" title="%s"></div>'
                        % (max(0.4, min(100, w / tot_w * 100.0)), c, tt)
                        for w, c, tt in segs)
        return ('<div style="display:flex;height:16px;border-radius:4px;overflow:hidden;'
                'background:#1e293b;margin-top:4px">%s</div>' % parts)
    # CPU 条
    cpu_bar = bar([(t, "#4ade80", "训练 %.0f%%" % t),
                   (e_, "#60a5fa", "评估 %.0f%%" % e_),
                   (p, "#fbbf24", "面板 %.0f%%" % p),
                   (other, "#94a3b8", "其他 %.0f%%" % other),
                   (rest, REST_C, "剩余 %.0f%%" % rest)])
    # GPU 条（compute-apps 不可用时：训练=整卡已用近似，评估/面板=0）
    gt = d.get("gpu_total", 0) or 1
    _gu = d.get("gpu_used", 0) or 0
    gm_train = d.get("train", {}).get("gpu_mem", 0) or 0
    gm_eval = d.get("eval", {}).get("gpu_mem", 0) or 0
    gm_panel = d.get("panel", {}).get("gpu_mem", 0) or 0
    if gm_train == 0 and gm_eval == 0 and gm_panel == 0:
        gm_train = _gu   # 驱动不报每进程显存：训练占整卡（评估 CPU/面板无 GPU）
    gm = [("train", gm_train), ("eval", gm_eval), ("panel", gm_panel)]
    gm_other = max(0, _gu - sum(x[1] for x in gm))
    gm_rest = max(0, gt - _gu)
    gpu_bar = bar([(x[1] / gt * 100, c, "%s %.0fMB" % (n, x[1]))
                   for n, x, c in [("训练", gm[0], "#4ade80"), ("评估", gm[1], "#60a5fa"), ("面板", gm[2], "#fbbf24")]] +
                  [(gm_other / gt * 100, "#94a3b8", "其他 %.0fMB" % gm_other),
                   (gm_rest / gt * 100, REST_C, "剩余 %.0fMB" % gm_rest)])
    # 内存条（单位统一为 KB：psutil rss=bytes → /1024；wmic 总内存=KB）
    tot = d.get("total_mem_kb", 0) or 1
    free = d.get("free_mem_kb", 0) or 0
    used = tot - free
    _m3_kb = [d.get("train", {}).get("mem", 0) / 1024.0,
              d.get("eval", {}).get("mem", 0) / 1024.0,
              d.get("panel", {}).get("mem", 0) / 1024.0]
    mem_3 = sum(_m3_kb)
    mem_other = max(0, used - mem_3)
    mem_rest = max(0, tot - used)
    mem_bar = bar([(_m3_kb[0] / tot * 100, "#4ade80", "训练 %.1fGB" % (_m3_kb[0] / (1024.0 * 1024.0))),
                   (_m3_kb[1] / tot * 100, "#60a5fa", "评估 %.1fGB" % (_m3_kb[1] / (1024.0 * 1024.0))),
                   (_m3_kb[2] / tot * 100, "#fbbf24", "面板 %.1fGB" % (_m3_kb[2] / (1024.0 * 1024.0))),
                   (mem_other / tot * 100, "#94a3b8", "其他 %.1fGB" % (mem_other / (1024.0 * 1024.0))),
                   (mem_rest / tot * 100, REST_C, "剩余 %.1fGB" % (mem_rest / (1024.0 * 1024.0)))])
    def legend(items):
        return '<div style="margin-top:2px;font-size:11px;color:#94a3b8">%s</div>' % "".join(
            '<span style="margin-right:10px"><font color="%s">■</font> %s</span>' % (c, t) for c, t in items)
    gpu_txt = "GPU 利用率 %d%% · 显存 %d/%d MB" % (d.get("gpu_util", 0), d.get("gpu_used", 0), d.get("gpu_total", 0))
    return ('<div class="panel"><h2>🖥️ 资源占用（训练 / 评估 / 面板 三进程）</h2>'
            '<div style="font-size:12px;color:#64748b">CPU（%d 核，整机负载 %.0f%%）</div>%s%s'
            '<div style="margin-top:8px;font-size:12px;color:#64748b">GPU %s</div>%s%s'
            '<div style="margin-top:8px;font-size:12px;color:#64748b">内存（总计 %.1f GB，已用 %.1f GB）</div>%s%s'
            '</div>' % (ncpu, sysc, cpu_bar,
                        legend([("#4ade80", "训练"), ("#60a5fa", "评估"), ("#fbbf24", "面板"), ("#94a3b8", "其他"), ("#1e293b", "剩余")]),
                        gpu_txt, gpu_bar,
                        legend([("#4ade80", "训练"), ("#60a5fa", "评估"), ("#fbbf24", "面板"), ("#94a3b8", "其他"), (REST_C, "剩余")]),
                        tot / tb, used / tb, mem_bar,
                        legend([("#4ade80", "训练"), ("#60a5fa", "评估"), ("#fbbf24", "面板"), ("#94a3b8", "其他"), (REST_C, "剩余")])))


def _sliding_eval_html():
    """页面最上方：并行模型评估系统（滑动平均可视化 + 最近几局对弈记录）。
    数据：logs/vs_sl_sliding.jsonl（聚合点）+ vs_sl_sliding_games.jsonl（每局）"""
    import json as _json
    rows = []
    if os.path.exists("C:/agentwork/logs/vs_sl_sliding.jsonl"):
        for _ln in open("C:/agentwork/logs/vs_sl_sliding.jsonl", encoding="utf-8", errors="ignore"):
            _ln = _ln.strip()
            if _ln:
                try:
                    rows.append(_json.loads(_ln))
                except Exception:
                    pass
    gs = []
    if os.path.exists("C:/agentwork/logs/vs_sl_sliding_games.jsonl"):
        for _ln in open("C:/agentwork/logs/vs_sl_sliding_games.jsonl", encoding="utf-8", errors="ignore"):
            _ln = _ln.strip()
            if _ln:
                try:
                    gs.append(_json.loads(_ln))
                except Exception:
                    pass
    # 实时对局进度（评估进行中）
    prog = {}
    try:
        if os.path.exists("C:/agentwork/logs/vs_sl_eval_progress.json"):
            prog = _json.loads(open("C:/agentwork/logs/vs_sl_eval_progress.json", encoding="utf-8").read())
    except Exception:
        pass
    prog_html = ""
    if prog:
        _sc = prog.get("scores", [])
        while len(_sc) < 4:
            _sc.append(0)
        seat = prog.get("seat", 0)
        if isinstance(seat, int):
            seat = seat + 1
        _rr = prog.get("results", [])
        rr_txt = "".join('<span style="margin-right:8px;color:#94a3b8">%s</span>' % (
            ("%s %s家和%s %d番%s" % (r.get("t", ""), str(r.get("和", "") + 1) if isinstance(r.get("和"), int) else r.get("和"),
                                        "自摸" if r.get("tsumo") else ("铳" + str((r.get("铳") or 0) + 1) if isinstance(r.get("铳"), int) else "荣和"),
                                        r.get("han", 0), "")) if "流局" not in r
                          else ("%s 流局" % r.get("t", ""))) for r in _rr[-6:])
        prog_html = ('<div style="margin-top:6px;padding:8px;border:1px solid #334155;border-radius:6px;font-size:12px">'
                     '<b>⏳ 评估进行中</b> · epoch %s · 第 <b>%s/%s</b> 局 · 座位%s · <b>%s</b> · %s<br/>'
                     '各家打点：<span style="color:#4ade80">%s</span> <span style="color:#60a5fa">%s</span> '
                     '<span style="color:#fbbf24">%s</span> <span style="color:#f87171">%s</span><br/>'
                     '本半庄和铳记录：%s</div>'
                     % (prog.get("epoch", "?"), prog.get("game", 0), prog.get("total", 100), seat,
                        prog.get("round", "?"),
                        "✅ 终局（顺位%s · pt%+d）" % (prog.get("rank", "?"), prog.get("spt", 0))
                        if prog.get("phase") == "done" else "🔄 对局中",
                        _sc[0], _sc[1], _sc[2], _sc[3], rr_txt))
    if not rows:
        return ('<div class="panel"><h2>🎛️ 并行模型评估（滑动 vs-SL · 独立进程不占训练 GPU）</h2>'
                '<div class="empty" style="color:#64748b;padding:10px">评估数据积累中…（每 30 分钟 16 局，CPU 推理）</div></div>')
    xs = list(range(len(rows)))   # 顺序编号（避免多轮次 games_done 乱序）
    # 用户需求：曲线改为最近 20 个评估点的滑动平均（平滑噪声，看趋势）
    def _ma(key, fbkey=None, w=20):
        vals = [r.get(key, r.get(fbkey, 0.0) if fbkey else 0.0) for r in rows]
        out = []
        for i in range(len(vals)):
            win = vals[max(0, i - w + 1):i + 1]
            if len(win) >= 2:
                out.append((xs[i], sum(win) / len(win)))
        return out
    s1 = [("pt加权胜率(20点均)", _ma("wr_pt"))]
    s2 = [("每百局pt(20点均)", _ma("avg_pt_per100"))]
    s3 = [("rank胜率(20点均)", _ma("rank_wr", "wr_pt"))]
    s4 = [("平均顺位(20点均)", _ma("avg_rank"))]
    _early = ''
    if len(rows) < 2:
        _lr = rows[-1]
        _early = ('<div class="sub" style="color:#94a3b8">评估点积累中（当前 %d 点：pt加权胜率 %.3f · 每百局pt %+.1f · rank胜率 %.3f · 平均顺位 %.2f）</div>'
                  % (len(rows), _lr.get("wr_pt", 0), _lr.get("avg_pt_per100", 0),
                     _lr.get("rank_wr", 0), _lr.get("avg_rank", 0)))
    last = rows[-1]
    _rd = last.get("rank_dist", [0, 0, 0, 0])
    while len(_rd) < 4:
        _rd.append(0)
    cards = ""
    cards += ('<div class="card"><div class="k">最近评估（ep%d · %d局）</div><div class="v">pt加权胜率 %.3f</div>'
              '<div class="k">均势</div><div class="v" style="font-size:13px">0.500</div></div>'
              % (last.get("epoch", 0), last.get("n", 0), last.get("wr_pt", 0)))
    cards += ('<div class="card"><div class="k">平均每百局pt</div><div class="v" style="color:%s">%+.1f</div>'
              '<div class="k">平均pt ±std</div><div class="v" style="font-size:13px">%+.2f ± %.1f</div></div>'
              % ("#4ade80" if last.get("avg_pt_per100", 0) > 0 else "#f87171",
                 last.get("avg_pt_per100", 0), last.get("avg_pt", 0), last.get("pt_std", 0)))
    cards += ('<div class="card"><div class="k">顺位分布（1/2/3/4位）</div>'
              '<div class="v" style="font-size:13px">%d / %d / %d / %d</div>'
              '<div class="k">rank胜率</div><div class="v" style="font-size:14px">%.3f（均势0.375）</div></div>'
              % (_rd[0], _rd[1], _rd[2], _rd[3], last.get("rank_wr", 0)))
    cards += ('<div class="card"><div class="k">胜负局（胜/负）</div><div class="v" style="font-size:13px">%d / %d</div>'
              '<div class="k">单局最大胜/负</div><div class="v" style="font-size:13px">%+d / %+d</div></div>'
              % (last.get("wins", 0), last.get("losses", 0), last.get("max_win", 0), last.get("max_loss", 0)))
    rows_html = ""
    for g in gs[-10:][::-1]:
        col = "#4ade80" if g.get("win") else "#f87171"
        seat = g.get("seat", "")
        if isinstance(seat, int):
            seat = seat + 1
        rows_html += ('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td style="color:%s">%+d</td><td>%s</td></tr>'
                      % (g.get("t", "")[11:19], seat, g.get("rank", ""), g.get("score", ""),
                         col, g.get("spt", 0), "✅胜" if g.get("win") else "❌负"))
    games_tbl = ('<table style="width:100%%;font-size:12px"><tr><th>时间</th><th>座位</th><th>顺位</th><th>分数</th><th>pt</th><th>结果</th></tr>%s</table>'
                 % rows_html) if gs else '<div class="empty" style="color:#64748b;padding:6px">尚无对局记录</div>'
    _early = _early if len(rows) < 2 else ''
    # 单次准确数据表：每个评估点一行（100 局统计）
    hrows = ""
    for r in rows[-10:][::-1]:
        _rdx = r.get("rank_dist", [0, 0, 0, 0])
        while len(_rdx) < 4:
            _rdx.append(0)
        hrows += ('<tr><td>%s</td><td>ep%d</td><td>%d</td><td>%.3f</td><td style="color:%s">%+.1f</td>'
                  '<td>%+.2f±%.1f</td><td>%.3f</td><td>%.2f</td><td>%d/%d/%d/%d</td></tr>'
                  % (r.get("t", "")[5:16], r.get("epoch", 0), r.get("n", 0), r.get("wr_pt", 0),
                     "#4ade80" if r.get("avg_pt_per100", 0) > 0 else "#f87171",
                     r.get("avg_pt_per100", 0), r.get("avg_pt", 0), r.get("pt_std", 0),
                     r.get("rank_wr", 0), r.get("avg_rank", 0),
                     _rdx[0], _rdx[1], _rdx[2], _rdx[3]))
    hist_tbl = ('<table style="width:100%%;font-size:11px"><tr><th>时间</th><th>ep</th><th>局数</th><th>pt加权胜率</th><th>每百局pt</th>'
                '<th>平均pt±std</th><th>rank胜率</th><th>均顺位</th><th>顺位分布</th></tr>%s</table>' % hrows)
    return ('<div class="panel"><h2>🎛️ 并行模型评估（滑动 vs-SL · 独立进程不占训练 GPU）</h2>'
            '%s'
            '%s'
            '<div class="cards">%s</div>'
            '<div style="margin-top:8px">%s</div>'
            '<div style="margin-top:8px">%s</div>'
            '<div style="margin-top:8px">%s</div>'
            '<div style="margin-top:8px"><b style="font-size:12px;color:#94a3b8">每次评估详细数据（100 局/次，最近 %d 次）</b></div>%s'
            '<div style="margin-top:8px"><b style="font-size:12px;color:#94a3b8">最近 %d 局对弈记录</b></div>%s'
            '</div>' % (_early, prog_html, cards, svg_series(s1, colors=["#4ade80"]),
                        svg_series(s2, colors=["#60a5fa"]),
                        svg_series(s3 + s4, colors=["#fbbf24", "#a78bfa"]),
                        min(len(rows), 10), hist_tbl, min(len(gs), 10), games_tbl))


def _vs_modules_html():
    """用户需求：本 epoch 详细分析下方三个模块——
    1) 对 SL 胜率详细分析（rank/pt加权/每百局pt） 2) 对历史版本胜率详细分析 3) 进步分析"""
    import json as _json
    games = []
    if os.path.exists(GAMES_PATH):
        for _ln in open(GAMES_PATH, encoding="utf-8", errors="ignore"):
            _ln = _ln.strip()
            if _ln:
                try:
                    games.append(_json.loads(_ln))
                except Exception:
                    pass
    n = len(games)
    pts = read_metrics()
    html = ""
    # ---------- 模块1：对 SL 胜率详细分析 ----------
    sl_evals = [p for p in pts if p.get("wr_vs_sl") is not None]
    if sl_evals:
        xs = [p.get("games_done", p.get("epoch", i)) for i, p in enumerate(sl_evals)]
        series = [("rank胜率", [(xs[i], p["wr_vs_sl"]) for i, p in enumerate(sl_evals)])]
        if sl_evals[-1].get("wr_pt_vs_sl") is not None:
            series.append(("pt加权胜率", [(xs[i], p.get("wr_pt_vs_sl")) for i, p in enumerate(sl_evals)]))
        if sl_evals[-1].get("avg_pt_per100_vs_sl") is not None:
            series.append(("每百局pt", [(xs[i], p.get("avg_pt_per100_vs_sl")) for i, p in enumerate(sl_evals)]))
        last = sl_evals[-1]
        _wr = last.get("wr_vs_sl") or 0
        _wrpt = last.get("wr_pt_vs_sl")
        _p100 = last.get("avg_pt_per100_vs_sl")
        _avgpt = last.get("avg_pt_vs_sl")
        _win = last.get("wr_pt_window")
        _ee = last.get("eval_every", 30) or 30
        _eg = last.get("eval_games", 20) or 20
        cards = ""
        cards += ('<div class="card"><div class="k">最近评估 ep%d</div><div class="v">rank胜率 %.3f</div>'
                  '<div class="k">目标 0.66 · 均势 0.375</div><div class="v" style="font-size:13px">窗口 %d 局</div></div>'
                  % (last.get("epoch", 0), _wr, _win or _eg))
        if _wrpt is not None:
            cards += '<div class="card"><div class="k">pt加权胜率</div><div class="v">%.3f</div><div class="k">均势</div><div class="v" style="font-size:13px">0.500</div></div>' % _wrpt
        if _p100 is not None:
            cards += ('<div class="card"><div class="k">平均每百局pt</div><div class="v" style="color:%s">%+.1f</div>'
                      '<div class="k">平均pt</div><div class="v" style="font-size:13px">%+.2f</div></div>'
                      % ("#4ade80" if _p100 > 0 else "#f87171", _p100, _avgpt or 0))
        html += ('<div class="panel"><h2>🤖 模型对 SL 胜率详细分析（每 %d epoch · %d 局）</h2>'
                 '<div class="cards">%s</div><div style="margin-top:8px">%s</div></div>'
                 % (_ee, _eg, cards, svg_series(series, colors=["#4ade80", "#fbbf24", "#60a5fa"])))
    else:
        html += ('<div class="panel"><h2>🤖 模型对 SL 胜率详细分析</h2>'
                 '<div class="empty" style="color:#64748b;padding:10px">尚无评估点（每 30 epoch 一次，首次约 600 局）</div></div>')
    # ---------- 模块2：对历史版本胜率 ----------
    past = [g for g in games if g.get("opp") == "past"]
    if past:
        n_p = len(past)
        d = [0, 0, 0, 0]
        for g in past:
            r = g.get("rank", 2)
            if 1 <= r <= 4:
                d[r - 1] += 1
        wr = (d[0] + 0.5 * d[1]) / max(n_p, 1)
        spts = [g.get("settlement", 0) for g in past]
        pospt = sum(x for x in spts if x > 0)
        negpt = sum(-x for x in spts if x < 0)
        wr_pt = pospt / max(pospt + negpt, 1e-9)
        avg_s = sum(spts) / n_p
        def slide_past(k):
            w = 30
            out = []
            for i in range(0, n_p, 5):
                win = past[max(0, i - w):i + 5]
                if len(win) >= 5:
                    out.append((i + len(win), float(sum(g.get(k, 0) for g in win) / len(win))))
            return out
        cards = ('<div class="card"><div class="k">历史版本对局</div><div class="v">%d 局</div>'
                 '<div class="k">rank胜率</div><div class="v" style="font-size:14px">%.1f%%</div></div>' % (n_p, 100 * wr))
        cards += ('<div class="card"><div class="k">pt加权胜率</div><div class="v">%.3f</div>'
                  '<div class="k">平均结算</div><div class="v" style="font-size:14px">%+.1f</div></div>' % (wr_pt, avg_s))
        cards += ('<div class="card"><div class="k">顺位分布</div><div class="v" style="font-size:12px">1位%d · 2位%d · 3位%d · 4位%d</div>'
                  '<div class="k">spt&gt;0</div><div class="v" style="font-size:14px">%.0f%%</div></div>'
                  % (d[0], d[1], d[2], d[3], 100.0 * sum(1 for x in spts if x > 0) / max(n_p, 1)))
        html += ('<div class="panel"><h2>🕰️ 对历史版本胜率详细分析（虚构自对弈）</h2>'
                 '<div class="cards">%s</div><div style="margin-top:8px">%s</div></div>'
                 % (cards, svg_series([("结算滑均", slide_past("settlement")), ("rank滑均", slide_past("rank"))],
                                      colors=["#4ade80", "#f472b6"])))
    else:
        html += ('<div class="panel"><h2>🕰️ 对历史版本胜率详细分析（虚构自对弈）</h2>'
                 '<div class="empty" style="color:#64748b;padding:10px">尚无历史版本对局（首个存档后出现，约 200 局）</div></div>')
    # ---------- 模块3：进步分析 ----------
    if n >= 20:
        wsize = 100
        wins = []
        for i in range(0, n, 20):
            win = games[max(0, i - wsize):i + 20]
            if len(win) >= 20:
                wr_ = (sum(1 for g in win if g.get("rank") == 1) + 0.5 * sum(1 for g in win if g.get("rank") == 2)) / len(win)
                avg_s = sum(g.get("settlement", 0) for g in win) / len(win)
                avg_r = sum(g.get("rank", 2) for g in win) / len(win)
                wins.append((i + len(win), wr_, avg_s, avg_r))
        if len(wins) >= 2:
            first, last_ = wins[0], wins[-1]
            d_s = last_[2] - first[2]
            d_wr = last_[1] - first[1]
            d_r = first[3] - last_[3]
            def _trend(d, good_pos=True):
                if abs(d) < 0.005:
                    return '<span style="color:#94a3b8">持平</span>'
                good = d > 0 if good_pos else d < 0
                col = "#4ade80" if good else "#f87171"
                return '<span style="color:%s">%s%+.2f</span>' % (col, "▲" if d > 0 else "▼", d)
            cards = ('<div class="card"><div class="k">每百局pt增量</div><div class="v" style="color:%s">%+.1f</div>'
                     '<div class="k">首→尾窗口结算</div><div class="v" style="font-size:13px">%+.1f → %+.1f</div></div>'
                     % ("#4ade80" if d_s > 0 else "#f87171", d_s, first[2], last_[2]))
            cards += ('<div class="card"><div class="k">胜率进步</div><div class="v">%s</div>'
                      '<div class="k">顺位进步</div><div class="v" style="font-size:14px">%s</div></div>' % (_trend(d_wr), _trend(d_r, good_pos=False)))
            cards += ('<div class="card"><div class="k">最近窗口胜率</div><div class="v">%.1f%%</div>'
                      '<div class="k">最近顺位</div><div class="v" style="font-size:14px">%.2f</div></div>' % (100 * last_[1], last_[3]))
            html += ('<div class="panel"><h2>📈 进步分析（每 100 局窗口滑动 · 共 %d 局）</h2>'
                     '<div class="cards">%s</div><div style="margin-top:8px">%s</div><div style="margin-top:8px">%s</div></div>'
                     % (n, cards,
                        svg_series([("胜率(窗口)", [(x, v) for x, v, _, _ in wins])], colors=["#4ade80"]),
                        svg_series([("结算pt(窗口)", [(x, s) for x, _, s, _ in wins])], colors=["#60a5fa"])))
        else:
            html += '<div class="panel"><h2>📈 进步分析</h2><div class="empty" style="color:#64748b;padding:10px">样本不足（需 ≥40 局）</div></div>'
    else:
        html += '<div class="panel"><h2>📈 进步分析</h2><div class="empty" style="color:#64748b;padding:10px">样本不足（需 ≥20 局）</div></div>'
    return html


def render():
    pts = read_metrics()
    # 未更新行判定（32 局池更新点之间全 0）：不显示数值、不画曲线点
    _UPD_KEYS = ("lp", "lv", "H", "ratio", "clip_rate", "kl", "lp_disc", "lp_bin")
    def _no_upd(p):
        return bool(p.get("lp", 0) == 0 and p.get("ratio", 0) == 0)
    prog = read_progress()
    _logtxt = ""
    if os.path.exists(TRAINLOG):
        _logtxt = open(TRAINLOG, encoding="utf-8", errors="ignore").read()
    _done = "RL_V1 DONE" in _logtxt
    alerts = check_alerts(pts, None, prog, done=_done)
    now_s = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # P0④ 状态判定：训练进程 PID 存在（_done 在 check_alerts 前已算）
    import subprocess
    _alive = False
    try:
        _out = subprocess.run(["wmic", "process", "where", "name='python.exe' and CommandLine like '%train_rl%'",
                               "get", "ProcessId"], capture_output=True, text=True, timeout=10)
        _alive = bool(_out.stdout and len(_out.stdout.split()) > 1)
    except Exception:
        pass
    running = _alive and not _done
    cur = pts[-1] if pts else {}
    _gpu, _cpu = sys_stats()
    if cur:
        _elapsed = cur.get("elapsed_s", 0)
    elif os.path.exists(TRAINLOG):
        _elapsed = max(time.time() - os.path.getmtime(TRAINLOG), 0)
    else:
        _elapsed = 0
    _gpp = 0
    try:
        for _ln in open(TRAINLOG, encoding="utf-8", errors="ignore"):
            _m = re.search(r"game (\d+)/(\d+)", _ln)
            if _m:
                _gpp = int(_m.group(2))
                break
    except Exception:
        pass
    _total_games = 0
    try:
        _o = subprocess.run(["wmic", "process", "where", "name='python.exe' and CommandLine like '%train_rl%'",
                             "get", "CommandLine"], capture_output=True, text=True, timeout=5)
        _mm = re.search(r"--epochs (\d+)", _o.stdout)
        if _mm:
            _total_games = _gpp * int(_mm.group(1))
    except Exception:
        pass
    _done_games = cur.get("games_done", 0) if cur else 0
    _eta = None
    if _elapsed > 0 and _done_games > 0 and _total_games > _done_games:
        _rate = _done_games / _elapsed
        _eta = (_total_games - _done_games) / _rate
    _cards_sys = '<div class="card"><div class="k">GPU 利用率/显存</div><div class="v" style="font-size:13px">%s</div><div class="k">CPU 负载</div><div class="v" style="font-size:13px">%s</div></div>' % (_gpu, _cpu)
    _cards_sys += '<div class="card"><div class="k">已用时间</div><div class="v" style="font-size:13px">%s</div><div class="k">预估剩余</div><div class="v" style="font-size:13px">%s</div></div>' % (_fmt_hms(_elapsed), _fmt_hms(_eta))

    # 卡片
    cards = ""
    cards += '<div class="card"><div class="k">训练状态</div><div class="v">%s</div><div class="k" style="margin-top:4px">%s</div></div>' % (
        "🔄 运行中" if running else ("✅ 完成" if pts and "DONE" in (open(TRAINLOG, encoding="utf-8", errors="ignore").read() if os.path.exists(TRAINLOG) else "") else "⏳ 等待"),
        ("epoch %d · %d 局完成" % (cur.get("epoch", 0), cur.get("games_done", 0))) if cur else "尚未开始")
    _upd = bool(cur and not _no_upd(cur))
    cards += '<div class="card"><div class="k">loss_p</div><div class="v">%s</div><div class="k">loss_v</div><div class="v" style="font-size:14px">%s</div></div>' % (
        ("%.4f" % cur.get("lp", 0)) if _upd else "—", ("%.2f" % cur.get("lv", 0)) if _upd else "—")
    cards += '<div class="card"><div class="k">熵 H</div><div class="v">%s</div><div class="k">ratio</div><div class="v" style="font-size:14px">%s</div></div>' % (
        ("%.3f" % cur.get("H", 0)) if _upd else "—",
        ("—" if (cur and (cur.get("ratio") is None or cur.get("ratio", 0) <= 0))
         else ("%.4f" % cur.get("ratio", 0))))
    cards += '<div class="card"><div class="k">clip 率</div><div class="v">%s</div><div class="k">gate 均值/std</div><div class="v" style="font-size:14px">%s</div></div>' % (
        ("%.1f%%" % (100 * cur.get("clip_rate", 0))) if _upd else "—",
        ("%.5f / %.5f" % (cur.get("gate_mean", 0), cur.get("gate_std", 0))) if cur.get("gate_mean") is not None else "n/a")
    _wr_sl = cur.get("wr_vs_sl") if cur else None
    cards += '<div class="card"><div class="k">vs-SL 胜率</div><div class="v">%s</div><div class="k">对手构成 自/史/SL</div><div class="v" style="font-size:14px">%d / %d / %d</div></div>' % (
        ("%.1f%%" % (100 * _wr_sl)) if _wr_sl is not None else "—",
        cur.get("opp_self", 0) if cur else 0, cur.get("opp_past", 0) if cur else 0,
        cur.get("opp_sl", 0) if cur else 0)
    cards += '<div class="card"><div class="k">吞吐</div><div class="v">%s 局/时</div><div class="k">value mean/std</div><div class="v" style="font-size:14px">%s</div></div>' % (
        ("%.0f" % cur.get("throughput_gph", 0)) if cur.get("throughput_gph") is not None else "—",
        ("%.2f / %.2f" % (cur.get("value_mean", 0), cur.get("value_std", 0)))
        if cur.get("value_mean") is not None else "—")

    # 告警
    alert_html = ""
    for sev, msg in alerts:
        cls = "alert" if sev == "bad" else "warn"
        icon = "🚨" if sev == "bad" else "⚠️"
        alert_html += '<div class="%s">%s %s</div>' % (cls, icon, msg)
    if not alerts:
        alert_html = '<div class="panel" style="border-color:#4ade80"><div class="ok">✅ 无异常</div></div>'
    # 页面最上方：资源占用（3 进程 CPU/GPU/内存）
    resource_html = _resource_html()
    # 页面最上方：并行模型评估系统（滑动平均 + 最近对弈记录）
    sliding_html = _sliding_eval_html()

    # 曲线（x=epoch 或 games_done）
    X = [p.get("games_done", p.get("epoch", i)) for i, p in enumerate(pts)]
    def col(k):
        out = []
        for i, p in enumerate(pts):
            v = p.get(k)
            if k in _UPD_KEYS and _no_upd(p):
                v = None   # 未更新行：曲线不记录（svg 自动跳过 None）
            out.append((X[i], v))
        return out
    charts = ""
    charts += '<div class="panel"><h2>📉 loss_p / loss_v</h2>%s</div>' % svg_series(
        [("loss_p", col("lp")), ("loss_v", col("lv"))], colors=["#60a5fa", "#f87171"])
    charts += '<div class="panel"><h2>🧠 熵 H / ratio / clip 率</h2>%s</div>' % svg_series(
        [("H", col("H")), ("ratio", col("ratio")), ("clip", col("clip_rate"))], colors=["#4ade80", "#fbbf24", "#a78bfa"])
    charts += '<div class="panel"><h2>🚪 gate 均值/方差（事件旁支开启状态）</h2>%s</div>' % svg_series(
        [("gate_mean", col("gate_mean")), ("gate_std", col("gate_std"))], colors=["#f472b6", "#60a5fa"])
    charts += '<div class="panel"><h2>📊 KL / reward_mean / 吞吐（更新幅度与奖励）</h2>%s</div>' % svg_series(
        [("KL", col("kl")), ("reward_mean", col("reward_mean")), ("吞吐局时", col("throughput_gph"))],
        colors=["#fbbf24", "#4ade80", "#a78bfa"])
    charts += '<div class="panel"><h2>🎯 head 分项 loss（discard vs binary，T2 加权效果）</h2>%s</div>' % svg_series(
        [("lp_disc", col("lp_disc")), ("lp_bin", col("lp_bin"))], colors=["#60a5fa", "#f87171"])
    charts += '<div class="panel"><h2>🏆 vs-SL 胜率（每 epoch 自对战 8 局）· 目标 0.66</h2>%s</div>' % svg_series(
        [("wr_vs_sl", col("wr_vs_sl"))], colors=["#4ade80"])

    # Φ 漂移监控（P0-B compliance：每 500 局天凤日志 MSE，>30% 恶化触发重训）
    pmon = read_phi_monitor()
    if pmon and pmon.get("history"):
        hist = pmon["history"]
        px = [h.get("games_done", i) for i, h in enumerate(hist)]
        pmse = [(px[i], h.get("mse")) for i, h in enumerate(hist) if h.get("mse") is not None]
        pcorr = [(px[i], h.get("corr")) for i, h in enumerate(hist) if h.get("corr") is not None]
        base = pmon.get("baseline_mse")
        bl = (" · 基线 MSE %.1f" % base) if base is not None else ""
        last_ev = hist[-1]
        phi_alert = ""
        if last_ev.get("drift_triggered"):
            phi_alert = '<div class="alert">🚨 Φ 漂移: MSE %.1f > 基线×%.2f（%s）%s</div>' % (
                last_ev.get("mse", 0), last_ev.get("drift_ratio", 1.3),
                ("已触发重训" if last_ev.get("retrain") else "待处理"), bl)
        elif last_ev.get("retrain"):
            phi_alert = '<div class="warn">⚠️ Φ 已重训（50%% 自对弈+50%% 人类，lr 1e-4）%s</div>' % bl
        charts += ('<div class="panel"><h2>🧪 Φ 漂移监控（天凤日志 MSE/相关性）· 最新 %.1f%s</h2>%s%s</div>'
                   % (last_ev.get("mse", 0) or 0, bl, phi_alert,
                      svg_series([("Φ MSE", pmse), ("Φ corr", pcorr)],
                                 colors=["#f87171", "#4ade80"])))
    elif pmon is None:
        charts += '<div class="panel"><h2>🧪 Φ 漂移监控</h2><div class="sub">尚未产生监控点（训练满 500 局后出现）</div></div>'

    # 详细表
    rows = ""
    for p in pts[-12:][::-1]:
        _f = lambda k, fmt: (fmt % p.get(k, 0)) if p.get(k) is not None else "—"
        _up = not _no_upd(p)
        rows += ('<tr><td>%d</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
                 '<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
                 % (p.get("epoch", 0), p.get("games_done", 0),
                    ("%.4f" % p.get("lp", 0)) if _up else "—",
                    ("%.2f" % p.get("lv", 0)) if _up else "—",
                    ("%.3f" % p.get("H", 0)) if _up else "—",
                    ("%.4f" % p.get("ratio", 0)) if _up else "—",
                    _f("kl", "%.3f") if _up else "—",
                    _f("gate_mean", "%.5f"), _f("wr_vs_sl", "%.3f"),
                    _f("value_std", "%.2f"), _f("reward_mean", "%.3f"),
                    _f("throughput_gph", "%.0f")))
    tbl = ('<div class="panel"><h2>📋 每 epoch 详细指标（最近 12）</h2>'
           '<table><tr><th>ep</th><th>局数</th><th>loss_p</th><th>loss_v</th><th>H</th><th>ratio</th>'
           '<th>KL</th><th>gate</th><th>wr_vs_sl</th><th>v_std</th><th>r_mean</th><th>局/时</th></tr>%s</table></div>' % rows)

    # 每局实时区块（games.jsonl）
    games_html = games_block()
    # 用户需求：本 epoch 详细分析下方——SL胜率/历史版本/进步分析 三个模块
    vs_html = _vs_modules_html()

    # 训练进度行
    prog_line = ""
    if prog:
        prog_line = ('<div class="panel"><h2>🔄 当前进度</h2><div class="sub">epoch %d · game %d/%d · '
                     'avg_p=%.4f avg_v=%.4f H=%.3f ratio=%.4f clip=%.3f</div></div>'
                     % (prog["epoch"], prog["game"], prog["games"], prog["lp"], prog["lv"],
                        prog["H"], prog["ratio"], prog["clip"]))

    html = STYLE + """
<h1>🎯 RL 训练实时监控（M1）</h1>
%s
<div class="sub">5s 刷新 · 更新时间 %s · 数据源 logs/rl_train_metrics.jsonl + logs/rl_train.txt</div>
%s
%s
%s
<div class="cards">%s</div>
%s
%s
%s
%s
%s
</body></html>""" % (resource_html, now_s, alert_html, sliding_html, _cards_sys, cards, prog_line, games_html, vs_html, charts, tbl)
    open(OUT, "w", encoding="utf-8").write(html)
    return len(pts), len(alerts)


if __name__ == "__main__":
    print("panel:", render())