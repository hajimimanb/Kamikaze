# -*- coding: utf-8 -*-
import json, os, sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
from utils.paths import logs_dir

GAMES = str(logs_dir() / "rl_train_games.jsonl")



def svg_lines(series, w=760, h=120, colors=None):
    """内联折线（避免与 rl_train_dashboard 循环 import）。"""
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
    gl = ""
    for i in range(5):
        gx = x0 + (x1 - x0) * i / 4
        gl += ('<line x1="%.1f" y1="0" x2="%.1f" y2="%d" stroke="#1e293b"/>'
               '<text x="%.1f" y="%d" fill="#64748b" font-size="9" text-anchor="middle">%.0f</text>'
               % (px(gx), px(gx), h, px(gx), h - 2, gx))
        gy = y0 + (y1 - y0) * i / 4
        gl += ('<line x1="0" y1="%.1f" x2="%d" y2="%.1f" stroke="#1e293b"/>'
               '<text x="4" y="%.1f" fill="#64748b" font-size="9">%.2f</text>'
               % (py(gy), w, py(gy), py(gy) - 2, gy))
    gl += '<text x="%d" y="%d" fill="#475569" font-size="9" text-anchor="end">x: 局</text>' % (w - 4, h - 4)
    polylines, leg = "", ""
    for i, (name, pts) in enumerate(series):
        good = [p for p in pts if p is not None and p[1] is not None]
        if not good: continue
        c = colors[i % len(colors)]
        p = " ".join("%.1f,%.1f" % (px(x), py(y)) for x, y in good)
        polylines += '<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>' % (p, c)
        leg += '<span style="margin-right:10px;font-size:11px;color:#94a3b8"><font color="%s">■</font> %s</span>' % (c, name)
    return ('<svg viewBox="0 0 %d %d" class="chart">%s%s</svg><div style="margin-top:2px">%s</div>'
            % (w, h, gl, polylines, leg))


def read_games(n=400):
    out = []
    if os.path.exists(GAMES):
        for ln in open(GAMES, encoding="utf-8", errors="ignore"):
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    return out[-n:]


def epoch_analytics(games):
    """按 epoch 聚合全套指标：和牌率/分布/被飞/打点/铳点/放铳/流局/自摸荣和/流听/了巡/顺位。"""
    by_ep = {}
    for g in games:
        ep = g.get("epoch", 0)
        s = by_ep.setdefault(ep, {"wins": 0, "deals": 0, "rounds": 0, "ranks": [],
                                  "tsumo": 0, "ron": 0, "ryuukyoku": 0, "tenpai": 0,
                                  "riichi": 0, "meld": 0, "silent": 0,
                                  "wp_s": 0.0, "wp_n": 0, "dp_s": 0.0, "dp_n": 0,
                                  "turns_s": 0, "turns_n": 0, "bust": 0})
        s["wins"] += g.get("wins", 0); s["deals"] += g.get("deals", 0)
        s["rounds"] += g.get("rounds", 1); s["ranks"].append(g.get("rank", 2))
        s["tsumo"] += g.get("tsumo_wins", 0); s["ron"] += g.get("ron_wins", 0)
        s["ryuukyoku"] += g.get("ryuukyoku", 0); s["tenpai"] += g.get("ryuukyoku_tenpai", 0)
        s["riichi"] += g.get("riichi_wins", 0); s["meld"] += g.get("meld_wins", 0)
        s["silent"] += g.get("silent_wins", 0)
        s["wp_s"] += g.get("win_pt_sum", 0); s["wp_n"] += g.get("win_pt_n", 0)
        s["dp_s"] += g.get("deal_pt_sum", 0); s["dp_n"] += g.get("deal_pt_n", 0)
        s["turns_s"] += g.get("win_turns_sum", 0); s["turns_n"] += g.get("win_turns_n", 0)
        s["bust"] += g.get("busted", 0)
        s["settle_s"] = s.get("settle_s", 0.0) + g.get("settlement", 0)
        for _h in ("tsumo", "ron", "chow", "pon", "kan", "riichi", "kyushu"):
            s[_h + "_opp"] = s.get(_h + "_opp", 0) + g.get(_h + "_opp", 0)
            s[_h + "_exec"] = s.get(_h + "_exec", 0) + g.get(_h + "_exec", 0)
        for _h in ("chow", "pon", "kan"):
            for _st in ("mf", "md"):
                s[_h + "_opp_" + _st] = (s.get(_h + "_opp_" + _st, 0)
                                         + g.get(_h + "_opp_" + _st, 0))
                s[_h + "_exec_" + _st] = (s.get(_h + "_exec_" + _st, 0)
                                          + g.get(_h + "_exec_" + _st, 0))
        s["ankan_opp"] = s.get("ankan_opp", 0) + g.get("ankan_opp", 0)
        s["ankan_exec"] = s.get("ankan_exec", 0) + g.get("ankan_exec", 0)
    out = []
    for ep in sorted(by_ep):
        s = by_ep[ep]
        n = len(s["ranks"])
        if not n:
            continue
        wins = s["wins"]; rds = max(s["rounds"], 1)
        out.append({"epoch": ep, "n": n,
                    "win_rate": 100.0 * wins / rds,
                    "deal_rate": 100.0 * s["deals"] / rds,
                    "ryuukyoku_rate": 100.0 * s["ryuukyoku"] / rds,
                    "bust_rate": 100.0 * s["bust"] / n,
                    "avg_win": (s["wp_s"] / s["wp_n"]) if s["wp_n"] else None,
                    "avg_deal": (s["dp_s"] / s["dp_n"]) if s["dp_n"] else None,
                    "avg_turns": (s["turns_s"] / s["turns_n"]) if s["turns_n"] else None,
                    "avg_rank": float(sum(s["ranks"]) / n),
                    "avg_settle": float(s.get("settle_s", 0.0) / n) if n else None,
                    "silent_rate": 100.0 * s["silent"] / wins if wins else None,
                    "riichi_rate": 100.0 * s["riichi"] / wins if wins else None,
                    "meld_rate": 100.0 * s["meld"] / wins if wins else None,
                    "tsumo_rate": 100.0 * s["tsumo"] / wins if wins else None,
                    "ron_rate": 100.0 * s["ron"] / wins if wins else None,
                    "tenpai_rate": (100.0 * s["tenpai"] / s["ryuukyoku"])
                                    if s["ryuukyoku"] else None,
                    "tsumo_exec_rate": (100.0 * s.get("tsumo_exec", 0) / s["tsumo_opp"])
                                       if s.get("tsumo_opp") else None,
                    "ron_exec_rate": (100.0 * s.get("ron_exec", 0) / s["ron_opp"])
                                     if s.get("ron_opp") else None,
                    "meld_exec_rate": (100.0 * (s.get("chow_exec", 0) + s.get("pon_exec", 0)
                                                + s.get("kan_exec", 0)) / max(s.get("chow_opp", 0)
                                                + s.get("pon_opp", 0) + s.get("kan_opp", 0), 1))
                                      if (s.get("chow_opp", 0) + s.get("pon_opp", 0)
                                          + s.get("kan_opp", 0)) else None,
                    "riichi_exec_rate": (100.0 * s.get("riichi_exec", 0) / s["riichi_opp"])
                                        if s.get("riichi_opp") else None,
                    "kyushu_exec_rate": (100.0 * s.get("kyushu_exec", 0) / s["kyushu_opp"])
                                        if s.get("kyushu_opp") else None,
                    "tsumo_exec": s.get("tsumo_exec", 0), "tsumo_opp": s.get("tsumo_opp", 0),
                    "ron_exec": s.get("ron_exec", 0), "ron_opp": s.get("ron_opp", 0),
                    "meld_exec": (s.get("chow_exec", 0) + s.get("pon_exec", 0)
                                  + s.get("kan_exec", 0)),
                    "meld_opp": (s.get("chow_opp", 0) + s.get("pon_opp", 0)
                                 + s.get("kan_opp", 0)),
                    "riichi_exec": s.get("riichi_exec", 0), "riichi_opp": s.get("riichi_opp", 0),
                    "kyushu_exec": s.get("kyushu_exec", 0), "kyushu_opp": s.get("kyushu_opp", 0),
                    "chow_exec_mf": s.get("chow_exec_mf", 0), "chow_opp_mf": s.get("chow_opp_mf", 0),
                    "chow_exec_md": s.get("chow_exec_md", 0), "chow_opp_md": s.get("chow_opp_md", 0),
                    "pon_exec_mf": s.get("pon_exec_mf", 0), "pon_opp_mf": s.get("pon_opp_mf", 0),
                    "pon_exec_md": s.get("pon_exec_md", 0), "pon_opp_md": s.get("pon_opp_md", 0),
                    "kan_exec_mf": s.get("kan_exec_mf", 0), "kan_opp_mf": s.get("kan_opp_mf", 0),
                    "kan_exec_md": s.get("kan_exec_md", 0), "kan_opp_md": s.get("kan_opp_md", 0),
                    "ankan_exec": s.get("ankan_exec", 0), "ankan_opp": s.get("ankan_opp", 0),
                    "chow_exec_mf_rate": (100.0 * s.get("chow_exec_mf", 0) / s["chow_opp_mf"])
                                         if s.get("chow_opp_mf") else None,
                    "chow_exec_md_rate": (100.0 * s.get("chow_exec_md", 0) / s["chow_opp_md"])
                                         if s.get("chow_opp_md") else None,
                    "pon_exec_mf_rate": (100.0 * s.get("pon_exec_mf", 0) / s["pon_opp_mf"])
                                        if s.get("pon_opp_mf") else None,
                    "pon_exec_md_rate": (100.0 * s.get("pon_exec_md", 0) / s["pon_opp_md"])
                                        if s.get("pon_opp_md") else None,
                    "kan_exec_mf_rate": (100.0 * s.get("kan_exec_mf", 0) / s["kan_opp_mf"])
                                        if s.get("kan_opp_mf") else None,
                    "kan_exec_md_rate": (100.0 * s.get("kan_exec_md", 0) / s["kan_opp_md"])
                                        if s.get("kan_opp_md") else None,
                    "ankan_exec_rate": (100.0 * s.get("ankan_exec", 0) / s["ankan_opp"])
                                       if s.get("ankan_opp") else None})
    return out


def epoch_progress(games):
    """按 epoch 聚合：和牌率/放铳率/平均顺位（体现强化学习的进步趋势）。"""
    by_ep = {}
    for g in games:
        ep = g.get("epoch", 0)
        s = by_ep.setdefault(ep, {"wins": 0, "deals": 0, "rounds": 0, "ranks": []})
        s["wins"] += g.get("wins", 0)
        s["deals"] += g.get("deals", 0)
        s["rounds"] += g.get("rounds", 1)
        s["ranks"].append(g.get("rank", 2))
    out = []
    for ep in sorted(by_ep):
        s = by_ep[ep]
        n = len(s["ranks"])
        if n == 0:
            continue
        out.append({"epoch": ep, "n": n,
                    "win_rate": 100.0 * s["wins"] / max(s["rounds"], 1),
                    "deal_rate": 100.0 * s["deals"] / max(s["rounds"], 1),
                    "avg_rank": float(sum(s["ranks"]) / n)})
    return out


def games_block():
    games = read_games()
    if not games:
        return '<div class="panel"><h2>🎮 每局结果（实时）</h2><div class="empty" style="color:#64748b;padding:10px">数据积累中…</div></div>'
    # 累计统计
    n = len(games)
    wr_all = sum(1 for g in games if g.get("rank") == 1) + 0.5 * sum(1 for g in games if g.get("rank") == 2)
    win_rate = wr_all / n
    avg_rank = sum(g.get("rank", 2) for g in games) / n
    win_pct = 100.0 * sum(g.get("wins", 0) for g in games) / max(sum(g.get("rounds", 1) for g in games), 1)
    deal_pct = 100.0 * sum(g.get("deals", 0) for g in games) / max(sum(g.get("rounds", 1) for g in games), 1)
    avg_settle = sum(g.get("settlement", 0) for g in games) / n
    # 滑动窗口（50 局）
    def slide(k):
        w = 50
        pts = []
        for i in range(0, n, 10):
            win = games[max(0, i - w):i + 10]
            if len(win) >= 10:
                pts.append((i + len(win), float(sum(g.get(k, 0) for g in win) / len(win))))
        return pts
    # 卡片
    cards = ""
    cards += '<div class="card"><div class="k">近 %d 局胜率</div><div class="v">%.1f%%</div><div class="k">平均顺位</div><div class="v" style="font-size:14px">%.2f</div></div>' % (n, 100 * win_rate, avg_rank)
    _g_rand = [g for g in games if g.get("opp") == "random"]
    _g_sl = [g for g in games if g.get("opp") == "sl"]
    _rk = lambda gg: (sum(1 for x in gg if x.get("rank") == 1) + 0.5 * sum(1 for x in gg if x.get("rank") == 2)) / max(len(gg), 1)
    cards += '<div class="card"><div class="k">对随机胜率(顺位)</div><div class="v">%s</div><div class="k">对 SL 胜率</div><div class="v" style="font-size:14px">%s</div></div>' % (
        ("%.1f%%" % (100 * _rk(_g_rand))) if _g_rand else "—",
        ("%.1f%% (n=%d)" % (100 * _rk(_g_sl), len(_g_sl))) if _g_sl else "— (epoch1+ 起)")
    cards += '<div class="card"><div class="k">和牌率（局/轮）</div><div class="v">%.1f%%</div><div class="k">放铳率</div><div class="v" style="font-size:14px">%.1f%%</div></div>' % (win_pct, deal_pct)
    cards += '<div class="card"><div class="k">平均结算</div><div class="v">%+.1f</div><div class="k">净和牌-放铳</div><div class="v" style="font-size:14px">%+d</div></div>' % (
        avg_settle, sum(g.get("wins", 0) for g in games) - sum(g.get("deals", 0) for g in games))
    # 曲线：结算滑动均值 + 和牌率滑动
    ch = svg_lines([("settlement 滑均(50局)", slide("settlement")),
                     ("和牌率滑均", slide("wins"))], colors=["#4ade80", "#60a5fa"])
    ch2 = svg_lines([("rank 滑均(50局)", slide("rank"))], colors=["#f472b6"])
    # 进步效果：本 epoch vs 上一 epoch（和牌率/放铳率/平均顺位）
    eps = epoch_progress(games)
    prog_html = ""
    if eps:
        cur_ep = eps[-1]
        prev_ep = eps[-2] if len(eps) >= 2 else None

        def _delta(cur, prev, lower_better, unit="pp"):
            if prev is None:
                return ""
            d = cur - prev
            if abs(d) < 0.005:
                return ' <span style="color:#94a3b8;font-size:11px">持平</span>'
            good = (d < 0) if lower_better else (d > 0)
            col = "#4ade80" if good else "#f87171"
            arrow = "▼" if d < 0 else "▲"
            return ' <span style="color:%s;font-size:11px">%s%+.1f%s</span>' % (col, arrow, d, unit)

        _pv = ("上一 epoch %d" % prev_ep["epoch"]) if prev_ep else "上一 epoch"
        prog_html = ('<div class="panel"><h2>📈 进步效果（本 epoch %d · 对比 %s · 样本 %d 局）</h2>'
                     '<div class="cards">' % (cur_ep["epoch"], _pv, cur_ep["n"]))
        prog_html += ('<div class="card"><div class="k">本 epoch 和牌率</div><div class="v">%.1f%%%s</div></div>'
                      % (cur_ep["win_rate"],
                         _delta(cur_ep["win_rate"], prev_ep["win_rate"] if prev_ep else None, False)))
        prog_html += ('<div class="card"><div class="k">本 epoch 放铳率</div><div class="v">%.1f%%%s</div></div>'
                      % (cur_ep["deal_rate"],
                         _delta(cur_ep["deal_rate"], prev_ep["deal_rate"] if prev_ep else None, True)))
        prog_html += ('<div class="card"><div class="k">本 epoch 平均顺位</div><div class="v">%.2f%s</div></div>'
                      % (cur_ep["avg_rank"],
                         _delta(cur_ep["avg_rank"], prev_ep["avg_rank"] if prev_ep else None, True, unit="")))
        prog_html += ('<div class="card"><div class="k">vs-SL 胜率</div><div class="v">—</div>'
                      '<div class="k">目标 0.66</div><div class="v" style="font-size:13px">评估点每 100 局</div></div>')
        prog_html += '</div>'
        if len(eps) >= 2:
            p_wr = [(e["epoch"], e["win_rate"]) for e in eps]
            p_dr = [(e["epoch"], e["deal_rate"]) for e in eps]
            p_rk = [(e["epoch"], e["avg_rank"]) for e in eps]
            prog_html += '<div style="margin-top:8px">%s</div>' % svg_lines(
                [("和牌率 %", p_wr), ("放铳率 %", p_dr)], colors=["#4ade80", "#f87171"])
            prog_html += '<div style="margin-top:8px">%s</div>' % svg_lines(
                [("平均顺位", p_rk)], colors=["#f472b6"])
        prog_html += '</div>'
    # 本 epoch 详细分析（全套指标 + 与上一 epoch 对比）
    eps = epoch_analytics(games)
    ana_html = ""
    if eps:
        cur = eps[-1]
        prev = eps[-2] if len(eps) >= 2 else None

        def _d(v1, v0, good_dir=1, fmt="%+.1f", unit=""):
            """Δ 标注：good_dir 1=升好 -1=降好 0=中性；颜色绿/红/灰。"""
            if v0 is None or v1 is None:
                return ""
            d = v1 - v0
            if abs(d) < 1e-9:
                return ' <span style="color:#94a3b8;font-size:11px">持平</span>'
            good = (d > 0) if good_dir == 1 else ((d < 0) if good_dir == -1 else None)
            col = "#4ade80" if good else ("#f87171" if good is False else "#94a3b8")
            arrow = "▲" if d > 0 else "▼"
            return ' <span style="color:%s;font-size:11px">%s%s%s</span>' % (col, arrow, fmt % d, unit)

        def _cell(label, v, fmt, pv, good_dir, unit=""):
            vs = (fmt % v) if v is not None else "—"
            return ('<div class="card"><div class="k">%s</div><div class="v">%s%s</div></div>'
                    % (label, vs, _d(v, pv, good_dir, unit=unit)))

        _pv = ("上一 epoch %d" % prev["epoch"]) if prev else "上一 epoch"
        ana_html = ('<div class="panel"><h2>📊 本 epoch 详细分析（epoch %d · %d 局 · 对比 %s）</h2><div class="cards">'
                    % (cur["epoch"], cur["n"], _pv))
        ana_html += _cell("和牌率", cur["win_rate"], "%.1f%%", prev["win_rate"] if prev else None, 1)
        ana_html += _cell("放铳率", cur["deal_rate"], "%.1f%%", prev["deal_rate"] if prev else None, -1)
        ana_html += _cell("流局率", cur["ryuukyoku_rate"], "%.1f%%", prev["ryuukyoku_rate"] if prev else None, 0)
        ana_html += _cell("被飞率", cur["bust_rate"], "%.1f%%", prev["bust_rate"] if prev else None, -1)
        ana_html += '</div><div class="cards">'
        ana_html += _cell("平均打点", cur["avg_win"], "%.0f", prev["avg_win"] if prev else None, 1)
        ana_html += _cell("平均铳点", cur["avg_deal"], "%.0f", prev["avg_deal"] if prev else None, -1)
        ana_html += _cell("和了巡数", cur["avg_turns"], "%.1f巡", prev["avg_turns"] if prev else None, 0)
        ana_html += _cell("平均顺位", cur["avg_rank"], "%.2f", prev["avg_rank"] if prev else None, -1)
        ana_html += '</div><div class="cards">'
        ana_html += _cell("默听和率", cur["silent_rate"], "%.1f%%", prev["silent_rate"] if prev else None, 0)
        ana_html += _cell("立直和率", cur["riichi_rate"], "%.1f%%", prev["riichi_rate"] if prev else None, 0)
        ana_html += _cell("副露和率", cur["meld_rate"], "%.1f%%", prev["meld_rate"] if prev else None, 0)
        ana_html += _cell("自摸和率", cur["tsumo_rate"], "%.1f%%", prev["tsumo_rate"] if prev else None, 0)
        ana_html += '</div><div class="cards">'
        ana_html += _cell("荣和率", cur["ron_rate"], "%.1f%%", prev["ron_rate"] if prev else None, 0)
        ana_html += _cell("流听率(罚符)", cur["tenpai_rate"], "%.1f%%", prev["tenpai_rate"] if prev else None, 0)
        def _cell_n(label, ex, opp, pv, good_dir):
            """执行率卡片：ex/opp (pct%)；opp=0 -> N/A；Δ 为执行率变化。"""
            if opp is None or opp <= 0:
                vs = "N/A"
                rate = None
            else:
                rate = 100.0 * ex / opp
                vs = "%d/%d (%.1f%%)" % (ex, opp, rate)
            return ('<div class="card"><div class="k">%s</div><div class="v">%s%s</div></div>'
                    % (label, vs, _d(rate, pv, good_dir, fmt="%+.1f", unit="pp")))
        ana_html += '</div><div class="cards">'
        ana_html += _cell_n("自摸执行率", cur["tsumo_exec"], cur["tsumo_opp"],
                            prev["tsumo_exec_rate"] if prev else None, 1)
        ana_html += _cell_n("荣和执行率", cur["ron_exec"], cur["ron_opp"],
                            prev["ron_exec_rate"] if prev else None, 1)
        ana_html += _cell_n("吃碰杠执行率", cur["meld_exec"], cur["meld_opp"],
                            prev["meld_exec_rate"] if prev else None, 0)
        ana_html += _cell_n("立直执行率", cur["riichi_exec"], cur["riichi_opp"],
                            prev["riichi_exec_rate"] if prev else None, 0)
        ana_html += '</div><div class="cards">'
        ana_html += _cell_n("九种九牌执行率", cur["kyushu_exec"], cur["kyushu_opp"],
                            prev["kyushu_exec_rate"] if prev else None, 0)
        ana_html += '</div><div class="cards">'
        ana_html += _cell_n("门清吃执行率", cur["chow_exec_mf"], cur["chow_opp_mf"],
                            prev["chow_exec_mf_rate"] if prev else None, 0)
        ana_html += _cell_n("副露吃执行率", cur["chow_exec_md"], cur["chow_opp_md"],
                            prev["chow_exec_md_rate"] if prev else None, 0)
        ana_html += _cell_n("门清碰执行率", cur["pon_exec_mf"], cur["pon_opp_mf"],
                            prev["pon_exec_mf_rate"] if prev else None, 0)
        ana_html += _cell_n("副露碰执行率", cur["pon_exec_md"], cur["pon_opp_md"],
                            prev["pon_exec_md_rate"] if prev else None, 0)
        ana_html += '</div><div class="cards">'
        ana_html += _cell_n("门清明杠执行率", cur["kan_exec_mf"], cur["kan_opp_mf"],
                            prev["kan_exec_mf_rate"] if prev else None, 0)
        ana_html += _cell_n("副露明杠执行率", cur["kan_exec_md"], cur["kan_opp_md"],
                            prev["kan_exec_md_rate"] if prev else None, 0)
        ana_html += _cell_n("暗杠执行率", cur["ankan_exec"], cur["ankan_opp"],
                            prev["ankan_exec_rate"] if prev else None, 0)
        ana_html += '</div>'
        # 滑动平均 helper（窗口 5 epoch，降噪）
        def _sm(pts, w=5):
            out = []
            for i in range(len(pts)):
                seg = pts[max(0, i - w + 1):i + 1]
                out.append((pts[i][0], float(sum(x[1] for x in seg)) / len(seg)))
            return out
        # 和点/铳点 + 期望值曲线（滑动平均）
        p_aw = _sm([(e["epoch"], e["avg_win"]) for e in eps if e.get("avg_win")])
        p_ad = _sm([(e["epoch"], e["avg_deal"]) for e in eps if e.get("avg_deal")])
        p_ev = _sm([(e["epoch"], (e["win_rate"] / 100.0) * e["avg_win"])
                    for e in eps if e.get("avg_win")])
        p_ed = _sm([(e["epoch"], (e["deal_rate"] / 100.0) * e["avg_deal"])
                    for e in eps if e.get("avg_deal")])
        ana_html += ('<div class="panel"><h2>🀄 和点/铳点（每 epoch 滑动平均·5）</h2><div style="margin-top:8px">%s</div></div>'
                     % svg_lines([("和点", p_aw), ("铳点", p_ad)],
                                 colors=["#4ade80", "#f87171"]))
        ana_html += ('<div class="panel"><h2>📊 期望值：和率×和点 / 铳率×铳点（滑动平均·5）</h2>'
                     '<div style="margin-top:8px">%s</div></div>'
                     % svg_lines([("和率×和点", p_ev), ("铳率×铳点", p_ed)],
                                 colors=["#60a5fa", "#fbbf24"]))
        # 🎯 决策头执行率分组（滑动平均，每组一张图）
        _groups = [
            ("立直", [("riichi_exec", "立直")]),
            ("吃（门清/副露）", [("chow_exec_mf", "门清吃"), ("chow_exec_md", "副露吃")]),
            ("碰（门清/副露）", [("pon_exec_mf", "门清碰"), ("pon_exec_md", "副露碰")]),
            ("杠（门清/副露/暗杠）", [("kan_exec_mf", "门清明杠"), ("kan_exec_md", "副露明杠"),
                                    ("ankan_exec", "暗杠")]),
            ("流局（九种九牌）", [("kyushu_exec", "流局")]),
            ("和牌（自摸/荣和）", [("tsumo_exec", "自摸"), ("ron_exec", "荣和")]),
        ]
        for _gname, _items in _groups:
            _series = []
            for _k, _name in _items:
                _opp_k = _k.replace("_exec", "_opp")
                _pts = []
                for e in eps:
                    _op = e.get(_opp_k, 0)
                    if _op and _op > 0:
                        _pts.append((e["epoch"], 100.0 * e.get(_k, 0) / _op))
                if _pts:
                    _series.append((_name, _sm(_pts)))
            if _series:
                ana_html += ('<div class="panel"><h2>🎯 %s执行率（每 epoch 滑动平均·5）</h2>'
                             '<div style="margin-top:8px">%s</div></div>'
                             % (_gname, svg_lines(_series)))
        # 💰 平均结算（滑动平均）
        p_settle = _sm([(e["epoch"], e["avg_settle"]) for e in eps
                        if e.get("avg_settle") is not None])
        if p_settle:
            ana_html += ('<div class="panel"><h2>💰 平均结算变化（每 epoch 滑动平均·5）</h2>'
                         '<div style="margin-top:8px">%s</div></div>'
                         % svg_lines([("平均结算", p_settle)], colors=["#60a5fa"]))
        ana_html += '</div>'
    # 滚动表（最近 12 局）
    rows = ""
    for g in games[-12:][::-1]:
        rows += '<tr><td>%d.%d</td><td>%d</td><td>%+d</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td><td>%dms</td></tr>' % (
            g.get("epoch", 0), g.get("game", 0), g.get("rank", 0), g.get("settlement", 0),
            g.get("wins", 0), g.get("deals", 0), g.get("melds", 0), g.get("rounds", 0),
            g.get("duration_ms", 0))
    tbl = ('<div class="panel"><h2>📋 最近 12 局结果（实时）</h2>'
           '<table><tr><th>局</th><th>顺位</th><th>结算</th><th>和牌</th><th>放铳</th>'
           '<th>副露</th><th>轮数</th><th>时长</th></tr>%s</table></div>' % rows)
    return ('<div class="cards">%s</div>%s%s'
            '<div class="panel"><h2>📈 结算/和牌率 滑动（50 局窗，每 10 局取样）</h2>%s</div>'
            '<div class="panel"><h2>🏆 平均顺位滑动</h2>%s</div>%s'
            % (cards, prog_html, ana_html, ch, ch2, tbl))