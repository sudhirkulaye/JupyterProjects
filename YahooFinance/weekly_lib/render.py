"""Renders the dashboard model as one self-contained HTML page (inline CSS, inline SVG charts, a little JS)."""
import html
import json
import math

import numpy as np
import pandas as pd

MINUS = "−"


# ================================================================ formatting helpers
def E(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return html.escape(str(x))


def ok(x):
    try:
        return x is not None and not pd.isna(x) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def f_pct(x, d=1, signed=True):
    if not ok(x):
        return "–"
    v = float(x) * 100
    if round(v, d) == 0:
        return f"{0:.{d}f}%"                      # no '+0.0%' or '-0.0%'
    s = f"{v:+.{d}f}%" if signed else f"{v:.{d}f}%"
    return s.replace("-", MINUS)


def f_num(x, d=0, signed=False):
    if not ok(x):
        return "–"
    s = f"{float(x):+,.{d}f}" if signed else f"{float(x):,.{d}f}"
    return s.replace("-", MINUS)


def f_inr(x, d=0):
    return "–" if not ok(x) else "₹" + f_num(x, d)


def f_cr(x):
    if not ok(x):
        return "–"
    return f"{x / 1e5:,.2f} L Cr" if x >= 1e5 else f"{x:,.0f} Cr"


def f_date(d, fmt="%d %b %Y"):
    return pd.Timestamp(d).strftime(fmt) if ok_date(d) else "–"


def ok_date(d):
    return d is not None and not pd.isna(d)


def sign_cls(x, eps=0.0005):
    if not ok(x):
        return ""
    return "pos" if x > eps else ("neg" if x < -eps else "")


def dv(x):
    return f"{float(x):.6g}" if ok(x) else ""


def td_pct(x, d=1, signed=True, extra="", tip=None):
    cls = " ".join(c for c in ("num", sign_cls(x) if signed else "", extra) if c)
    t = f' data-tip="{E(tip)}" tabindex="0"' if tip else ""
    return f'<td class="{cls}" data-v="{dv(x)}"{t}>{f_pct(x, d, signed)}</td>'


def td_stat(r, base, d=1):
    """Cell holding the average / lowest / highest-estimate versions of a figure; the table's toggle picks one."""
    vals = {"avg": r[base], "min": r[f"{base}_min"], "max": r[f"{base}_max"]}
    spans = "".join(f'<span class="s{k} {sign_cls(v)}">{f_pct(v, d)}</span>' for k, v in vals.items())
    data = " ".join(f'data-{k}="{dv(v)}"' for k, v in vals.items())
    return f'<td class="num" data-v="{dv(vals["avg"])}" {data}>{spans}</td>'


def td_num(x, d=0, extra="", text=None):
    return f'<td class="num {extra}" data-v="{dv(x)}">{text if text is not None else f_num(x, d)}</td>'


def td_txt(s, extra="", sort=None):
    sv = f' data-v="{E(sort)}"' if sort is not None else ""
    return f'<td class="{extra}"{sv}>{s}</td>'


def td_stock(tk, name, badges=""):
    nm = name if isinstance(name, str) and name != tk else ""
    return f'<td class="tk" data-v="{E(tk)}"><b>{E(tk)}</b>{badges}<span class="nm">{E(nm)}</span></td>'


def fno_badge(is_fno):
    return '<span class="badge" title="Futures &amp; options available">F&amp;O</span>' if is_fno else ""


ICONS = {"good": "▲", "bad": "▼", "neutral": "●", "warn": "◆"}


def chip(kind, label, tip=""):
    t = f' data-tip="{E(tip)}" tabindex="0"' if tip else ""
    return (f'<span class="chip {kind}"{t}><span class="ic" aria-hidden="true">{ICONS[kind]}</span>'
            f'{E(label)}</span>')


def table(head, rows, tid="", cls="", attrs=""):
    ths = []
    for h in head:
        label, align, tip = (h + ("", ""))[:3] if isinstance(h, tuple) else (h, "", "")
        t = f' title="{E(tip)}"' if tip else ""
        ths.append(f'<th class="{align}"{t} scope="col">{E(label)}</th>')
    tid_attr = f' id="{tid}"' if tid else ""
    return (f'<div class="tw"><table class="data {cls}"{tid_attr} {attrs}><thead><tr>{"".join(ths)}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def tile(label, value, sub="", tip=""):
    t = f' data-tip="{E(tip)}" tabindex="0"' if tip else ""
    return f'<div class="tile"{t}><div class="tl">{E(label)}</div><div class="tv">{value}</div><div class="ts">{sub}</div></div>'


def jsonable(v):
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


# ================================================================ SVG charts
def nice_ticks(lo, hi, n=3):
    if hi <= lo:
        hi = lo + (abs(lo) * 0.1 or 1)
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    start, end = math.floor(lo / step) * step, math.ceil(hi / step) * step
    return [start + i * step for i in range(int(round((end - start) / step)) + 1)]


def tick_label(v, kind, span):
    if kind == "pct":
        return f"{v * 100:.0f}%".replace("-", MINUS)
    if kind == "ic":
        return f"{v:.2f}".replace("-", MINUS)
    if abs(v) >= 10000:
        return f"{v / 1000:.0f}K"
    d = 0 if span >= 20 else (1 if span >= 2 else 2)
    return f"{v:,.{d}f}".replace("-", MINUS)


def _segments(xs, ys):
    seg, out = [], []
    for x, y in zip(xs, ys):
        if y is None:
            if seg:
                out.append(seg)
            seg = []
        else:
            seg.append((x, y))
    if seg:
        out.append(seg)
    return out


def line_chart(dates, series, band=None, kind="price", w=340, h=150, title="", zero=False):
    """series: [(name, values, slot)], band: (low, high, slot, name). Crosshair tooltip via JS."""
    L, R, T, B = 46, 12, 10, 22
    pw, ph = w - L - R, h - T - B
    ts = [pd.Timestamp(d).value for d in dates]
    t0, span_t = ts[0], max(ts[-1] - ts[0], 1)
    X = [L + pw * (t - t0) / span_t for t in ts]
    vals = [float(v) for _, arr, _ in series for v in arr if ok(v)]
    if band:
        vals += [float(v) for v in list(band[0]) + list(band[1]) if ok(v)]
    if zero:
        vals.append(0.0)
    if not vals:
        return '<p class="muted small">No data</p>'
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06 or (abs(hi) * 0.05) or 1
    ticks = nice_ticks(lo - pad, hi + pad, 4 if h >= 140 else 3)
    y0, y1 = ticks[0], ticks[-1]

    def Y(v):
        return T + ph * (1 - (v - y0) / (y1 - y0))

    out = [f'<svg class="lc" viewBox="0 0 {w} {h}" role="img" aria-label="{E(title)}">']
    for tv in ticks:
        yy = Y(tv)
        out.append(f'<line class="grid" x1="{L}" x2="{w - R}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        out.append(f'<text class="ax" x="{L - 6}" y="{yy + 3.5:.1f}" text-anchor="end">{tick_label(tv, kind, y1 - y0)}</text>')
    if zero and y0 < 0 < y1:
        out.append(f'<line class="base" x1="{L}" x2="{w - R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
    n = len(dates)
    for i in sorted({0, n // 3, (2 * n) // 3, n - 1}):
        anchor = "start" if i == 0 else ("end" if i == n - 1 else "middle")
        out.append(f'<text class="ax" x="{X[i]:.1f}" y="{h - 6}" text-anchor="{anchor}">{pd.Timestamp(dates[i]):%b %y}</text>')
    if band:
        lows = [float(v) if ok(v) else None for v in band[0]]
        highs = [float(v) if ok(v) else None for v in band[1]]
        both = [(x, Y(lo_), Y(hi_)) if lo_ is not None and hi_ is not None else None
                for x, lo_, hi_ in zip(X, lows, highs)]
        seg = []
        for item in both + [None]:
            if item is None:
                if len(seg) > 1:
                    pts = [f"{x:.1f},{yh:.1f}" for x, _, yh in seg] + [f"{x:.1f},{yl:.1f}" for x, yl, _ in reversed(seg)]
                    out.append(f'<polygon class="band s{band[2]}" points="{" ".join(pts)}"/>')
                seg = []
            else:
                seg.append(item)
    data_series = []
    for name, arr, slot in series:
        ys = [Y(float(v)) if ok(v) else None for v in arr]
        for seg in _segments(X, ys):
            d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in seg)
            out.append(f'<path class="ln s{slot}" d="{d}"/>')
        last = max((i for i, y in enumerate(ys) if y is not None), default=None)
        if last is not None:
            out.append(f'<circle class="dot s{slot}" r="4" cx="{X[last]:.1f}" cy="{ys[last]:.1f}"/>')
        data_series.append({"n": name, "slot": slot, "v": [float(v) if ok(v) else None for v in arr], "y": ys})
    out.append(f'<line class="xh" x1="0" x2="0" y1="{T}" y2="{T + ph}" visibility="hidden"/>')
    out.append(f'<rect class="hit" x="{L}" y="{T}" width="{pw}" height="{ph}"/>')
    payload = {"x": X, "d": [pd.Timestamp(d).strftime("%d %b %Y") for d in dates], "k": kind, "s": data_series}
    if band:
        payload["b"] = {"n": band[3], "lo": [float(v) if ok(v) else None for v in band[0]],
                        "hi": [float(v) if ok(v) else None for v in band[1]]}
    out.append("</svg>")
    out[0] = out[0].replace('role="img"', f'role="img" data-c="{E(json.dumps(jsonable(payload), separators=(",", ":")))}"')
    return "".join(out)


def legend(items):
    parts = []
    for kind, slot, label in items:
        parts.append(f'<span><i class="{kind} s{slot}" aria-hidden="true"></i>{E(label)}</span>')
    return f'<div class="legend">{"".join(parts)}</div>'


def ic_chart(hist, w=860, h=210):
    L, R, T, B = 46, 12, 12, 24
    pw, ph = w - L - R, h - T - B
    ic = hist["ic"].astype(float)
    roll = ic.rolling(12, min_periods=4).mean()
    ticks = nice_ticks(min(ic.min(), -0.05), max(ic.max(), 0.15), 4)
    y0, y1 = ticks[0], ticks[-1]

    def Y(v):
        return T + ph * (1 - (v - y0) / (y1 - y0))

    n = len(ic)
    band_w = pw / n
    bw = max(2.0, min(24.0, band_w - 2))
    out = [f'<svg class="bc" viewBox="0 0 {w} {h}" role="img" aria-label="Weekly IC of the composite score">']
    for tv in ticks:
        out.append(f'<line class="grid" x1="{L}" x2="{w - R}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}"/>')
        out.append(f'<text class="ax" x="{L - 6}" y="{Y(tv) + 3.5:.1f}" text-anchor="end">{tick_label(tv, "ic", 1)}</text>')
    base = Y(0)
    for i, (d, v) in enumerate(ic.items()):
        cx = L + band_w * (i + 0.5)
        x0, x1 = cx - bw / 2, cx + bw / 2
        yv = Y(v)
        r = min(4, bw / 2, abs(base - yv))
        if v >= 0:
            p = (f"M{x0:.1f},{base:.1f} L{x0:.1f},{yv + r:.1f} Q{x0:.1f},{yv:.1f} {x0 + r:.1f},{yv:.1f} "
                 f"L{x1 - r:.1f},{yv:.1f} Q{x1:.1f},{yv:.1f} {x1:.1f},{yv + r:.1f} L{x1:.1f},{base:.1f} Z")
        else:
            p = (f"M{x0:.1f},{base:.1f} L{x0:.1f},{yv - r:.1f} Q{x0:.1f},{yv:.1f} {x0 + r:.1f},{yv:.1f} "
                 f"L{x1 - r:.1f},{yv:.1f} Q{x1:.1f},{yv:.1f} {x1:.1f},{yv - r:.1f} L{x1:.1f},{base:.1f} Z")
        out.append(f'<path class="bar" d="{p}"/>')
        tip = (f"IC {f_num(v, 3, True)}\nWeek of {pd.Timestamp(d):%d %b %Y} · 12-week average "
               f"{f_num(roll.iloc[i], 3, True)} · {int(hist['n'].iloc[i])} stocks")
        out.append(f'<rect class="hitbar" x="{L + band_w * i:.1f}" y="{T}" width="{band_w:.1f}" height="{ph}" '
                   f'data-tip="{E(tip)}"/>')
    out.append(f'<line class="base" x1="{L}" x2="{w - R}" y1="{base:.1f}" y2="{base:.1f}"/>')
    pts = [(L + band_w * (i + 0.5), Y(v)) for i, v in enumerate(roll) if ok(v)]
    if pts:
        out.append('<path class="ln s2" d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
    for i in sorted({0, n // 2, n - 1}):
        anchor = "start" if i == 0 else ("end" if i == n - 1 else "middle")
        out.append(f'<text class="ax" x="{L + band_w * (i + 0.5):.1f}" y="{h - 6}" text-anchor="{anchor}">'
                   f'{pd.Timestamp(ic.index[i]):%d %b}</text>')
    out.append("</svg>")
    return "".join(out)


# ================================================================ page sections
def status_chip(kind_label_tip):
    kind, label, tip = kind_label_tip
    return chip(kind, label, tip)


def est_status(r):
    if not ok(r.eps4):
        return ("neutral", "No data", "No 4-week EPS revision available")
    tip = f"Consensus EPS {f_pct(r.eps4)} over 4 weeks, {f_pct(r.eps13)} over 13 weeks"
    if r.eps4 > 0.01:
        return ("good", "Rising", tip)
    if r.eps4 < -0.01:
        return ("bad", "Falling", tip)
    return ("neutral", "Flat", tip)


def tp_status(r):
    if not ok(r.tp4):
        return ("neutral", "No data", "No 4-week TP revision available")
    tip = f"Consensus TP {f_pct(r.tp4)} over 4 weeks, {f_pct(r.tp13)} over 13 weeks"
    if r.tp4 > 0.01:
        return ("good", "Raised", tip)
    if r.tp4 < -0.01:
        return ("bad", "Cut", tip)
    return ("neutral", "Unchanged", tip)


def broker_status(r):
    if not ok(r.b_n) or r.b_n == 0:
        return ("neutral", "None", "No MOSL / AMBIT / PL / AXIS view in the last 9 months")
    counts = f"{int(r.b_n_pos)} positive, {int(r.b_n_neu)} neutral, {int(r.b_n_neg)} negative"
    tip = f"{counts}; stance {r.b_stance:+.2f} vs each broker's usual rating; avg broker TP upside {f_pct(r.b_tp_upside)}"
    label = f"{int(r.b_n_pos)}/{int(r.b_n_neu)}/{int(r.b_n_neg)}"
    if r.b_stance > 0.3:
        return ("good", label, tip)
    if r.b_stance < -0.3:
        return ("bad", label, tip)
    return ("neutral", label, tip)


def rs_status(r):
    if not ok(r.rel_3M):
        return ("neutral", "No data", "")
    tip = f"3-month return {f_pct(r.ret_3M)} vs NIFTY: {f_pct(r.rel_3M)}"
    if r.rel_3M > 0.05:
        return ("good", "Ahead", tip)
    if r.rel_3M < -0.05:
        return ("bad", "Behind", tip)
    return ("neutral", "In line", tip)


def cushion_status(r):
    if not ok(r.upside):
        return ("neutral", "No TP", "")
    tip = f"Consensus TP {f_inr(r.tp)} vs price {f_inr(r.price)}; lowest TP {f_inr(r.tp_min)} ({f_pct(r.upside_min)})"
    if r.upside > 0.15:
        return ("good", f_pct(r.upside, 0), tip)
    if r.upside < 0:
        return ("bad", f_pct(r.upside, 0), tip)
    return ("neutral", f_pct(r.upside, 0), tip)


def flags(r, min_analysts):
    out = []
    if ok(r.eps4) and ok(r.ret_1M):
        if r.eps4 < -0.01 and r.ret_1M > 0.03:
            out.append(chip("bad", "EPS cut, price up", f"EPS {f_pct(r.eps4)} (4 wks) while price {f_pct(r.ret_1M)} (1M)"))
        if r.eps4 > 0.01 and r.ret_1M < -0.03:
            out.append(chip("good", "EPS up, price down", f"EPS {f_pct(r.eps4)} (4 wks) while price {f_pct(r.ret_1M)} (1M)"))
    if ok(r.b_down_30d) and r.b_down_30d > 0:
        out.append(chip("bad", "Broker downgrade", "A broker cut its rating in the last 30 days"))
    if ok(r.b_up_30d) and r.b_up_30d > 0:
        out.append(chip("good", "Broker upgrade", "A broker raised its rating in the last 30 days"))
    if ok(r.b_tpcut_30d) and r.b_tpcut_30d > 0:
        out.append(chip("bad", "Broker TP cut", "A broker cut its target by 10%+ in the last 30 days"))
    if ok(r.count_chg13) and r.count_chg13 <= -2:
        out.append(chip("warn", "Coverage down", f"{int(r.count_chg13)} analysts vs 13 weeks ago"))
    if ok(r.upside) and r.upside < 0:
        out.append(chip("warn", "Above avg TP", f"Price is {f_pct(-r.upside)} above the consensus TP"))
    if ok(r.age) and r.age >= 13:
        out.append(chip("neutral", f"Stale {int(r.age)} wks", "Yahoo consensus has not changed for this many weeks"))
    if not ok(r["count"]) or r["count"] < min_analysts:
        out.append(chip("neutral", "Thin coverage", f"{f_num(r['count'])} analysts on Yahoo (score needs {min_analysts}+)"))
    return " ".join(out)


def drivers(r, short=False):
    comps = {"Estimate upgrades": r.z_eps, "Target raised": r.z_tp, "Price pullback": r.z_rev,
             "12-month uptrend": r.z_mom12}
    if short:
        comps = {"Estimate cuts": -r.z_eps if ok(r.z_eps) else np.nan,
                 "Target cut": -r.z_tp if ok(r.z_tp) else np.nan,
                 "Price run-up": -r.z_rev if ok(r.z_rev) else np.nan,
                 "12-month downtrend": -r.z_mom12 if ok(r.z_mom12) else np.nan}
    ranked = sorted(((v, k) for k, v in comps.items() if ok(v) and v > 0.25), reverse=True)
    return " + ".join(k for _, k in ranked[:2]) or "Mixed"


def score_cell(r):
    if not ok(r.pct):
        return '<td class="num muted" data-v="">–</td>'
    tip = (f"Composite percentile {r.pct:.0f} (rank {int(r['rank'])})\nEPS z {f_num(r.z_eps, 2, True)} · "
           f"TP z {f_num(r.z_tp, 2, True)} · reversal z {f_num(r.z_rev, 2, True)} · "
           f"12-1M momentum z {f_num(r.z_mom12, 2, True)}")
    return f'<td class="num" data-v="{dv(r.pct)}" data-tip="{E(tip)}" tabindex="0"><span class="meter"><i style="width:{r.pct:.0f}%"></i></span>{r.pct:.0f}</td>'


def age_cell(r):
    if not ok(r.age):
        return '<td class="num muted" data-v="">–</td>'
    cls = "muted" if r.age < 8 else ""
    return f'<td class="num {cls}" data-v="{dv(r.age)}">{int(r.age)}w</td>'


# ---------------------------------------------------------------- header and checks
def sec_header(M):
    nav = [("checks", "Data checks"), ("market", "Market"), ("focus", "Focus Five"), ("shortlist", "Shortlisted"),
           ("long", "Long ideas"), ("short", "Short ideas"), ("divergence", "Divergence"), ("emerging", "Emerging"),
           ("scoreboard", "Scoreboard"), ("method", "Method")]
    links = "".join(f'<a href="#{k}">{E(v)}</a>' for k, v in nav)
    return (f'<header class="top"><div class="wrap hdr"><div><h1>Weekly stock dashboard</h1>'
            f'<p class="sub">Consensus as of {f_date(M["as_of"])} · prices to {f_date(M["px_date"])} · '
            f'built {f_date(M["today"])}</p></div>'
            f'<button id="theme" class="btn" type="button" aria-label="Switch light or dark theme">Theme</button></div>'
            f'<nav class="wrap" aria-label="Sections">{links}</nav></header>')


def sec_checks(M):
    icon = {"ok": ("good", "OK"), "info": ("neutral", "Note"), "warn": ("warn", "Check")}
    items = []
    for level, text in M["checks"]:
        kind, label = icon.get(level, ("neutral", "Note"))
        items.append(f'<li>{chip(kind, label)} <span>{E(text)}</span></li>')
    n_warn = sum(1 for lvl, _ in M["checks"] if lvl == "warn")
    summary = f"{len(M['checks'])} items, {n_warn} to check" if n_warn else f"{len(M['checks'])} items, all fine"
    return (f'<section class="card" id="checks"><details><summary><h2>Data checks</h2><span class="muted">{summary}</span>'
            f'</summary><ul class="checks">{"".join(items)}</ul></details></section>')


# ---------------------------------------------------------------- market
SECTOR_DEFS = {
    "breadth": "Share of the industry's scored stocks whose consensus EPS rose more than 0.5% in 4 weeks, minus the "
               "share whose EPS fell more than 0.5%",
    "eps4": "Median 4-week change in consensus average EPS across the industry's scored stocks",
    "eps13": "Median 13-week change in consensus average EPS",
    "tp4": "Median 4-week change in consensus average target price",
    "rel1m": "Median of (stock 1-month return minus NIFTY 1-month return)",
    "rel3m": "Median of (stock 3-month return minus NIFTY 3-month return)",
}


def sec_market(M):
    C = M["cfg"]
    ir = M["idx_rets"]
    nifty = ir.loc[C.MAIN_INDEX]
    tiles = [
        tile(f"{C.MAIN_INDEX} 1 week", f_pct(nifty["1W"]), f"2 weeks {f_pct(nifty['2W'])}"),
        tile(f"{C.MAIN_INDEX} 1 month", f_pct(nifty["1M"]), f"3 months {f_pct(nifty['3M'])}"),
        tile(f"{C.MAIN_INDEX} 6 months", f_pct(nifty["6M"]), f"12 months {f_pct(nifty['12M'])}"),
        tile(f"{C.MAIN_INDEX} year to date", f_pct(nifty["YTD"]), f"since {pd.Timestamp(M['px_date']).year - 1}-12-31"),
    ]
    head = [("Index", ""), *[(w, "num") for w in C.RETURN_WINDOWS]]
    rows = []
    for name in C.INDICES:
        if name in ir.index:
            rows.append("<tr>" + td_txt(f"<b>{E(name)}</b>", sort=name) + "".join(td_pct(ir.at[name, w]) for w in C.RETURN_WINDOWS) + "</tr>")
    idx_table = table(head, rows, cls="compact")

    ih = M["idx_hist"]
    idx_chart = ""
    if len(ih) > 5:
        series = [(name, ih[name].tolist(), i + 1) for i, name in enumerate(ih.columns)]
        weekly = ih.resample("W-FRI").last().iloc[::-1]
        idx_chart = (f'<h3>Index performance over the last 12 months</h3><p class="note">Cumulative return since '
                     f'{f_date(ih.index[0])}, from index_valuation.</p>'
                     + legend([("lk", i + 1, name) for i, name in enumerate(ih.columns)])
                     + line_chart(list(ih.index), series, kind="pct", w=860, h=200, zero=True,
                                  title="Index cumulative returns, last 12 months")
                     + details_table("Weekly data", ["Week", *ih.columns],
                                     [[f_date(d), *[f_pct(r[c]) for c in ih.columns]] for d, r in weekly.iterrows()]))

    ind, sub = M["ind_tbl"], M["sub_tbl"]
    caps = {"breadth": 0.5, "eps4": 0.05, "eps13": 0.10, "tp4": 0.05, "rel1m": 0.10, "rel3m": 0.15}
    labels = {"breadth": "Net EPS upgrades", "eps4": "EPS 4W", "eps13": "EPS 13W", "tp4": "TP 4W",
              "rel1m": "1M vs NIFTY", "rel3m": "3M vs NIFTY"}
    head = [("Industry (AMBIT)", "", "AMBIT Industry from the Tracker sheet of AnalystVeiwInsight.xlsx"),
            ("Stocks", "num", "Scored stocks in this industry (5+ analysts and a price)"),
            *[(labels[k], "num", SECTOR_DEFS[k]) for k in caps]]
    rows = []
    for name, r in ind.iterrows():
        cells = [td_txt(f"<b>{E(name)}</b>", sort=name), td_num(r["n"])]
        for k, cap in caps.items():
            cells.append(heat_cell(r[k], cap, f"{name}: median {labels[k]} {f_pct(r[k])}" if k != "breadth"
                                   else f"{name}: {f_pct(r[k], 0)} net EPS upgrades ({int(r['n'])} stocks)"))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    heat = table(head, rows, cls="heatmap")
    drill = []
    for name in ind.index:
        if name not in sub.index.get_level_values(0):
            continue
        s = sub.loc[name].sort_values("breadth", ascending=False)
        srows = ["<tr>" + td_txt(E(si), sort=si) + td_num(r["n"]) +
                 "".join(heat_cell(r[k], cap, f"{si}: {labels[k]} {f_pct(r[k])}") for k, cap in caps.items()) + "</tr>"
                 for si, r in s.iterrows()]
        drill.append(f'<details class="drill"><summary>{E(name)} <span class="muted">({len(s)} sub-industries)</span></summary>'
                     + table([("Sub-industry", ""), ("Stocks", "num"), *[(labels[k], "num", SECTOR_DEFS[k]) for k in caps]],
                             srows, cls="heatmap")
                     + "</details>")
    scale = ('<div class="scale" aria-hidden="true"><span>cut / lagging</span><i class="sw hn"></i><i class="sw mid"></i>'
             '<i class="sw hp"></i><span>raised / leading</span></div>')

    defs = "".join(f"<li><b>{E(labels[k])}</b>: {E(SECTOR_DEFS[k])}.</li>" for k in caps)
    return (f'<section class="card" id="market"><h2>Market</h2><p class="lede">Index returns from index_valuation, '
            f'then estimate changes by AMBIT industry.</p><div class="tiles">{"".join(tiles)}</div>'
            f'<h3>Index returns</h3>{idx_table}{idx_chart}'
            f'<h3>Industries: estimate revisions and relative price</h3><p class="note">Each scored stock (5+ analysts on '
            f'Yahoo and a price) is placed in its AMBIT Industry and Sub-Industry by ticker, from the Tracker sheet of '
            f'AnalystVeiwInsight.xlsx. Every stock counts equally: the figures are medians, not market-cap weighted. '
            f'Stocks not in that sheet appear as Unclassified. Sorted by net EPS upgrades.</p>'
            f'<details class="drill"><summary>What the columns mean</summary><ul class="defs">{defs}</ul></details>'
            f'{scale}{heat}<div class="drills">{"".join(drill)}</div></section>')


def heat_cell(x, cap, tip):
    if not ok(x):
        return '<td class="num heat" data-v="">–</td>'
    h = min(abs(x) / cap, 1.0) * 0.6
    pol = "hp" if x > 0 else ("hn" if x < 0 else "")
    return (f'<td class="num heat {pol}" style="--h:{h:.3f}" data-v="{dv(x)}" data-tip="{E(tip)}" tabindex="0">'
            f'{f_pct(x, 1)}</td>')


def details_table(summary, head, rows):
    body = "".join("<tr>" + "".join(f"<td>{E(c)}</td>" for c in row) + "</tr>" for row in rows)
    ths = "".join(f"<th>{E(h)}</th>" for h in head)
    return (f'<details class="tv"><summary>{E(summary)}</summary><div class="tw"><table class="data compact">'
            f'<thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table></div></details>')


# ---------------------------------------------------------------- focus five
TWRR_COLS = ["MTD", "3M", "6M", "1Y", "YTD", "2Y", "3Y", "5Y"]


def sec_focus(M):
    C, m, hold = M["cfg"], M["m"], M["holdings"]
    pt = M["port_twrr"]
    it = M["idx_twrr"].set_index("name")
    rows = []
    port = pt.iloc[0] if len(pt) else None
    if port is not None:
        cells = []
        for c in TWRR_COLS:                         # green = ahead of the main index, red = behind (whatever the sign)
            pv = _f(port[c])
            nv = _f(it.at[C.MAIN_INDEX, c]) if C.MAIN_INDEX in it.index else np.nan
            diff = pv - nv
            cls = "pos" if ok(diff) and diff > 0 else ("neg" if ok(diff) and diff < 0 else "")
            word = "ahead of" if ok(diff) and diff > 0 else "behind"
            tip = f"Focus Five {f_pct(pv)} vs {C.MAIN_INDEX} {f_pct(nv)}\n{f_pct(abs(diff)) if ok(diff) else ''} {word} {C.MAIN_INDEX}"
            cells.append(f'<td class="num {cls}" data-v="{dv(pv)}" data-tip="{E(tip)}" tabindex="0"><b>{f_pct(pv)}</b></td>')
        rows.append("<tr>" + td_txt("<b>Focus Five</b>", sort="0") + "".join(cells) + "</tr>")
    for name in C.INDICES:
        if name in it.index:
            rows.append("<tr>" + td_txt(E(name), sort=name) + "".join(td_pct(_f(it.at[name, c])) for c in TWRR_COLS) + "</tr>")
    if port is not None and C.MAIN_INDEX in it.index:
        rows.append('<tr class="sum">' + td_txt(f"<b>Focus Five vs {E(C.MAIN_INDEX)}</b>", sort="z")
                    + "".join(td_pct(_f(port[c]) - _f(it.at[C.MAIN_INDEX, c])) for c in TWRR_COLS) + "</tr>")
    twrr = table([("Time-weighted return", ""), *[(c, "num") for c in TWRR_COLS]], rows, cls="compact")

    total = float(hold["portfolio_value"].iloc[0]) if len(hold) else np.nan
    pnl = float(hold["net_profit"].astype(float).sum()) if len(hold) else np.nan
    cost = float(hold["total_cost"].astype(float).sum()) if len(hold) else np.nan
    tiles = [tile("Portfolio value", f_inr(total), f"{len(M['focus'])} stocks + cash"),
             tile("Unrealised P&L", f_inr(pnl), f"{f_pct(pnl / cost) if cost else ''} on cost")]
    if port is not None and C.MAIN_INDEX in it.index:
        tiles.append(tile("1-year TWRR", f_pct(_f(port["1Y"])), f"{C.MAIN_INDEX} {f_pct(_f(it.at[C.MAIN_INDEX, '1Y']))}"))
        tiles.append(tile("Year to date", f_pct(_f(port["YTD"])), f"{C.MAIN_INDEX} {f_pct(_f(it.at[C.MAIN_INDEX, 'YTD']))}"))

    # holdings and price table
    prow = []
    for h in hold.itertuples(index=False):
        tk = h.ticker
        weight = _f(h.weight)
        if tk in C.CASH_TICKERS:
            prow.append("<tr>" + td_stock(tk, "Cash") + td_pct(weight, 1, False) + "<td></td>" * (len(C.RETURN_WINDOWS) + 2) + "</tr>")
            continue
        r = m.loc[tk]
        cells = [td_stock(tk, r["name"], fno_badge(r["fno"])), td_pct(weight, 1, False),
                 td_pct(_f(h.net_profit) / _f(h.total_cost) if _f(h.total_cost) else np.nan)]
        for w in C.RETURN_WINDOWS:
            rel, ab = r["rel_" + w], r["ret_" + w]
            cells.append(f'<td class="num" data-v="{dv(rel)}" data-abs="{dv(ab)}">'
                         f'<span class="rel {sign_cls(rel)}">{f_pct(rel)}</span><span class="abs {sign_cls(ab)}">{f_pct(ab)}</span></td>')
        cells.append(td_pct(r["from_high"]))
        prow.append("<tr>" + "".join(cells) + "</tr>")
    phead = [("Stock", ""), ("Weight", "num"), ("P&L on cost", "num"), *[(w, "num") for w in C.RETURN_WINDOWS],
             ("From 52W high", "num")]
    price_tbl = table(phead, prow, tid="ff-prices", cls="rets")

    # estimates and brokers table
    erow = []
    bl = M["bv"]["latest"]
    for tk in M["focus"]:
        r = m.loc[tk]
        cells = [td_stock(tk, r["name"]), score_cell(r),
                 td_stat(r, "eps1"), td_stat(r, "eps4"), td_stat(r, "eps13"),
                 td_stat(r, "tp4"), td_stat(r, "tp13"), td_stat(r, "upside", 0),
                 td_num(r["count"], text=f'{f_num(r["count"])} <span class="muted">({f_num(r["count_chg13"], 0, True)})</span>'),
                 td_pct(r["disp"], 0, False), age_cell(r)]
        cells += [broker_cell(bl, tk, b, r["price"]) for b in C.BROKERS]
        cells.append(td_txt(flags(r, C.MIN_ANALYSTS), "flags"))
        erow.append("<tr>" + "".join(cells) + "</tr>")
    ehead = [("Stock", ""), ("Score", "num", "Composite percentile among scored stocks (100 = best)"),
             ("EPS 1W", "num", "Change in consensus EPS over 1 week (same fiscal year, average of the 2 years shown)"),
             ("EPS 4W", "num", "Change in consensus EPS over 4 weeks"), ("EPS 13W", "num", "Change in consensus EPS over 13 weeks"),
             ("TP 4W", "num", "Change in consensus target price over 4 weeks"),
             ("TP 13W", "num", "Change in consensus target price over 13 weeks"),
             ("TP upside", "num", "Consensus target price vs today's price"),
             ("Analysts (13W chg)", "num"), ("TP spread", "num", "(highest TP - lowest TP) / average TP"),
             ("Age", "num", "Weeks since the Yahoo consensus last changed"),
             *[(b, "") for b in C.BROKERS], ("Flags", "")]
    est_tbl = table(ehead, erow, tid="ff-est", cls="est st")
    stat_ctrl = ('<div class="seg" role="group" aria-label="Which analyst estimate">'
                 + "".join(f'<button type="button" data-stat-for="ff-est" data-stat="{k}" aria-pressed="{str(k == "avg").lower()}">'
                           f'{lbl}</button>' for k, lbl in (("min", "Lowest"), ("avg", "Average"), ("max", "Highest")))
                 + "</div>")

    cards = [focus_card(M, tk) for tk in M["focus"]]
    return (f'<section class="card" id="focus"><h2>Focus Five</h2><p class="lede">Holdings of portfolio '
            f'{E(hold["description"].iloc[0]) if len(hold) else ""} from the database, with returns vs {E(C.MAIN_INDEX)} and the '
            f'latest consensus and broker views.</p><div class="tiles">{"".join(tiles)}</div>'
            f'<h3>Portfolio vs indices (TWRR from the database)</h3><p class="note">Focus Five row: green means ahead of '
            f'{E(C.MAIN_INDEX)} over that period and red means behind, even when the return itself is negative or positive.</p>{twrr}'
            f'<div class="h3row"><h3>Holdings: price returns</h3><button class="btn" type="button" data-toggle-abs="ff-prices" '
            f'aria-pressed="false">Showing returns vs {E(C.MAIN_INDEX)}</button></div>{price_tbl}'
            f'<div class="h3row"><h3>Holdings: estimates and broker views</h3>{stat_ctrl}</div>'
            f'<p class="note">Revisions compare this week\'s Yahoo consensus with the same fiscal year 1, 4 or 13 weeks ago, '
            f'averaged over the two fiscal years Yahoo shows. Switch between the lowest, average and highest analyst '
            f'estimate. Lowest and highest can jump when a single analyst joins or drops out. Broker columns are the latest '
            f'verified AnalystsReco report (hover for the date and previous view).</p>{est_tbl}'
            f'<h3>Price vs consensus target, and consensus EPS</h3>'
            f'{legend([("lk", 1, "Price"), ("lk", 2, "Consensus TP (avg)"), ("lb", 2, "TP range (min to max)")])}'
            f'<div class="cards">{"".join(cards)}</div></section>')


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def broker_cell(latest, tk, broker, price):
    try:
        r = latest.loc[(tk, broker)]
    except KeyError:
        return '<td class="muted" data-v="">–</td>'
    if isinstance(r, pd.DataFrame):
        r = r.iloc[-1]
    score = r["score"]
    dot = "good" if ok(score) and score > 0 else ("bad" if ok(score) and score < 0 else "neutral")
    arrow = ""
    if ok(r["reco_chg"]) and r["reco_chg"] != 0:
        kind, label, sym = ("good", "upgraded", ICONS["good"]) if r["reco_chg"] > 0 else ("bad", "downgraded", ICONS["bad"])
        arrow = f'<span class="arr {kind}" role="img" aria-label="{label}">{sym}</span>'
    tpchg = f' ({f_pct(r["tp_chg"], 0)})' if ok(r["tp_chg"]) and abs(r["tp_chg"]) >= 0.005 else ""
    prev = (f'; previous {r["prev_Reco"]} {f_inr(r["prev_Target"])} on {f_date(r["prev_Date"])}'
            if ok_date(r["prev_Date"]) else "")
    tip = (f'{broker} {r["Reco"]} {f_inr(r["Target"])}\n{f_date(r["Date"])}, upside {f_pct(r["Target"] / price - 1 if ok(price) else np.nan)}'
           f'{prev}')
    return (f'<td data-v="{dv(score)}" data-tip="{E(tip)}" tabindex="0"><span class="rd {dot}" aria-hidden="true"></span>'
            f'{E(r["Reco"])}{arrow}<span class="sub">{f_inr(r["Target"])}{tpchg}</span></td>')


def focus_card(M, tk):
    C, m, ch = M["cfg"], M["m"], M["charts"][tk]
    r = m.loc[tk]
    dates = ch["dates"]
    price_chart = line_chart(dates, [("Price", ch["price"], 1), ("Consensus TP (avg)", ch["tp"], 2)],
                             band=(ch["tp_min"], ch["tp_max"], 2, "TP range") if ch["tp"] else None,
                             kind="price", title=f"{tk} price vs consensus target")
    eps_series = [(f"{lab} EPS", vals, i + 1) for i, (lab, vals) in enumerate(ch["eps"])]
    eps_chart = (line_chart(dates, eps_series, kind="eps", h=120, title=f"{tk} consensus EPS")
                 if eps_series else '<p class="muted small">No Yahoo EPS estimates</p>')
    eps_leg = legend([("lk", i + 1, f"{lab} EPS (avg)") for i, (lab, _) in enumerate(ch["eps"])]) if ch["eps"] else ""
    rows = []
    for i, d in enumerate(dates):
        eps_vals = [f_num(vals[i], 2) for _, vals in ch["eps"]]
        rows.append([f_date(d), f_num(ch["price"][i], 1), f_num(ch["tp"][i] if ch["tp"] else np.nan),
                     f_num(ch["tp_min"][i] if ch["tp_min"] else np.nan), f_num(ch["tp_max"][i] if ch["tp_max"] else np.nan),
                     *eps_vals])
    head = ["Week", "Price", "TP avg", "TP min", "TP max", *[f"{lab} EPS" for lab, _ in ch["eps"]]]
    score = f'score {r["pct"]:.0f}' if ok(r["pct"]) else "not scored"
    return (f'<article class="fc"><header><b>{E(tk)}</b> <span class="muted">{E(r["name"])}</span>'
            f'<span class="fc-meta">{score} · TP upside {f_pct(r["upside"], 0)} · {f_num(r["count"])} analysts</span></header>'
            f'{price_chart}<div class="fc-sub">Consensus EPS</div>{eps_leg}{eps_chart}'
            f'{details_table("Data table", head, rows[::-1])}</article>')


# ---------------------------------------------------------------- shortlisted
def sec_shortlist(M):
    C, m = M["cfg"], M["m"]
    focus = set(M["focus"])
    rows = []
    for tk, lists in sorted(M["shortlist"].items(), key=lambda kv: -(m.at[kv[0], "pct"] if kv[0] in m.index and ok(m.at[kv[0], "pct"]) else -1)):
        if tk in focus or tk not in m.index:
            continue
        r = m.loc[tk]
        d1 = r["pct"] - r["pct_1w"] if ok(r["pct"]) and ok(r["pct_1w"]) else np.nan
        cells = [td_stock(tk, r["name"], fno_badge(r["fno"])), td_txt(E(", ".join(lists))), td_txt(E(r["industry"])),
                 td_txt(status_chip(est_status(r))), td_txt(status_chip(tp_status(r))),
                 td_txt(status_chip(broker_status(r))), td_txt(status_chip(rs_status(r))),
                 td_txt(status_chip(cushion_status(r))), score_cell(r),
                 td_num(d1, text=f_num(d1, 0, True) if ok(d1) else "–", extra=sign_cls(d1 / 100 if ok(d1) else np.nan)),
                 td_pct(r["rel_1M"]), age_cell(r), td_txt(flags(r, C.MIN_ANALYSTS), "flags")]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = [("Stock", ""), ("Lists", ""), ("Industry", ""), ("Estimates", "", "4-week consensus EPS revision"),
            ("Target", "", "4-week consensus TP revision"), ("Brokers", "", "Positive / neutral / negative among MOSL, AMBIT, PL, AXIS"),
            ("Vs NIFTY 3M", ""), ("TP upside", ""), ("Score", "num"), ("Score chg 1W", "num"),
            ("1M vs NIFTY", "num"), ("Age", "num"), ("Flags", "")]
    overlap = sorted(set(M["shortlist"]) & focus)
    note = f'<p class="note">{len(overlap)} shortlisted stocks are Focus Five holdings and are shown above: {E(", ".join(overlap))}.</p>' if overlap else ""
    return (f'<section class="card" id="shortlist"><h2>Shortlisted</h2><p class="lede">Lists 002 to 005 from '
            f'watchlists.properties, read fresh on every run: edit the tickers there and the next run follows. A health '
            f'check per stock: green is supportive, red needs a look. Hover a chip for the numbers.</p>'
            f'{note}{table(head, rows, cls="health")}</section>')


# ---------------------------------------------------------------- ideas
def idea_rows(df, short=False, limit=None):
    rows = []
    for i, (tk, r) in enumerate(df.iterrows(), start=1):
        hidden = " hidden" if limit and i > limit else ""
        cells = [td_num(i), td_stock(tk, r["name"], fno_badge(r["fno"])), td_txt(E(r["industry"]), "ind"),
                 td_num(r["mcap"], text=f_cr(r["mcap"])), score_cell(r), td_pct(r["eps4"]),
                 td_pct(r["tp4"]), td_pct(r["mom"]), td_pct(r["mom12"], 0), td_pct(r["rel_1M"]), td_pct(r["upside"], 0),
                 td_num(r["count"]),
                 td_txt(broker_mix(r), sort=r["b_stance"] if ok(r["b_stance"]) else ""), age_cell(r),
                 td_txt(E(drivers(r, short)), "drv")]
        rows.append(f'<tr data-fno="{1 if r["fno"] else 0}" data-rank="{i}"{hidden}>' + "".join(cells) + "</tr>")
    return rows


IDEA_HEAD = [("#", "num"), ("Stock", ""), ("Industry", ""), ("Mkt cap", "num"), ("Score", "num"),
             ("EPS 4W", "num"), ("TP 4W", "num"), ("Price 4W", "num"),
             ("12-1M", "num", "Price return from 12 months ago to 1 month ago (12-1 month momentum)"),
             ("1M vs NIFTY", "num"), ("TP upside", "num"),
             ("Analysts", "num"), ("Brokers", "", "Positive / neutral / negative ratings among MOSL, AMBIT, PL, AXIS"),
             ("Age", "num"), ("Main drivers", "")]


def broker_mix(r):
    if not ok(r["b_n"]) or r["b_n"] == 0:
        return '<span class="muted">none</span>'
    return f'{int(r["b_n_pos"])}/{int(r["b_n_neu"])}/{int(r["b_n_neg"])}'


def n_scored(M):
    return int(M["m"]["eligible"].sum())


def sec_long(M):
    C = M["cfg"]
    df = M["lists"]["long"]
    rows = idea_rows(df, limit=C.TOP_N)
    return (f'<section class="card" id="long"><h2>Long ideas, 1-month horizon</h2><p class="lede">Ranked every week '
            f'across all {n_scored(M)} scored stocks in your Yahoo consensus list, whether or not they are on a watchlist, '
            f'so the list changes as estimates and prices move. Highest scores: consensus EPS and TP raised over 4 weeks '
            f'while the price has not run up in the last month, in a stock that rose over the year before. In the '
            f'back-test the top fifth beat the average of these stocks by '
            f'{f_pct(M["stats"].get("q5"))} per 4 weeks.</p>'
            f'<div class="filters"><label><input type="checkbox" data-fno-filter="long-tbl"> F&amp;O stocks only</label>'
            f'<span class="muted">Top {C.TOP_N} of {len(df)} ranked</span></div>'
            f'{table(IDEA_HEAD, rows, tid="long-tbl", attrs=f"data-limit={C.TOP_N}")}</section>')


def sec_short(M):
    df = M["lists"]["short"]
    rows = idea_rows(df, short=True)
    return (f'<section class="card" id="short"><h2>Short ideas, 1-month horizon</h2><p class="lede">Lowest scores among '
            f'the F&amp;O stocks in the same ranking (an overnight short in India needs futures or options; F&amp;O status '
            f'is the is_fno flag in stock_universe). In the back-test the bottom fifth trailed the average by '
            f'{f_pct(M["stats"].get("q1"))} per 4 weeks.</p>{table(IDEA_HEAD, rows)}</section>')


def sec_divergence(M):
    L = M["lists"]
    cut = idea_rows(L["div_short"], short=True, limit=10)
    raised = idea_rows(L["div_long"], limit=10)
    return (f'<section class="card" id="divergence"><h2>Divergence: estimates and price moving apart</h2>'
            f'<p class="lede">Scored stocks where the 4-week change in consensus EPS and the 4-week price move disagree. '
            f'Ranked by the size of the gap.</p>'
            f'<h3>Estimates cut, price up</h3><p class="note">Consensus EPS down more than 1% in 4 weeks while the price rose '
            f'more than 3%.</p>'
            f'<div class="filters"><label><input type="checkbox" data-fno-filter="div-cut"> F&amp;O stocks only</label></div>'
            f'{table(IDEA_HEAD, cut, tid="div-cut", attrs="data-limit=10")}'
            f'<h3>Estimates raised, price down</h3><p class="note">Consensus EPS up more than 1% in 4 weeks while the price '
            f'fell more than 3%.</p>'
            f'<div class="filters"><label><input type="checkbox" data-fno-filter="div-raised"> F&amp;O stocks only</label></div>'
            f'{table(IDEA_HEAD, raised, tid="div-raised", attrs="data-limit=10")}</section>')


# ---------------------------------------------------------------- emerging
def sec_emerging(M):
    L, m = M["lists"], M["m"]

    def mini(df, head, fn):
        rows = ["<tr>" + "".join(fn(tk, r)) + "</tr>" for tk, r in df.iterrows()]
        return table(head, rows, cls="compact") if rows else '<p class="muted">None this week.</p>'

    top = mini(L["new_top"], [("Stock", ""), ("Score", "num"), ("4 wks ago", "num"), ("EPS 4W", "num")],
               lambda tk, r: [td_stock(tk, r["name"], fno_badge(r["fno"])), score_cell(r),
                              td_num(r["pct_4w"], text=f_num(r["pct_4w"])), td_pct(r["eps4"])])
    bottom = mini(L["new_bottom"], [("Stock", ""), ("Score", "num"), ("4 wks ago", "num"), ("EPS 4W", "num")],
                  lambda tk, r: [td_stock(tk, r["name"], fno_badge(r["fno"])), score_cell(r),
                                 td_num(r["pct_4w"], text=f_num(r["pct_4w"])), td_pct(r["eps4"])])
    cov = mini(L["coverage"], [("Stock", ""), ("Analysts", "num"), ("13W chg", "num"), ("EPS 13W", "num")],
               lambda tk, r: [td_stock(tk, r["name"]), td_num(r["count"]), td_num(r["count_chg13"], text=f_num(r["count_chg13"], 0, True)),
                              td_pct(r["eps13"])])
    per = mini(L["persistent"], [("Stock", ""), ("Net up weeks", "num", "Upgrade weeks minus downgrade weeks, last 13"),
                                 ("EPS 13W", "num"), ("Score", "num")],
               lambda tk, r: [td_stock(tk, r["name"]), td_num(r["persist"], text=f_num(r["persist"], 0, True)),
                              td_pct(r["eps13"]), score_cell(r)])
    wk = M["bv"]["week"].sort_values(["Date", "Ticker"], ascending=[False, True])
    rows = []
    for r in wk.itertuples(index=False):
        chg = ""
        if ok(r.reco_chg) and r.reco_chg != 0:
            chg = chip("good" if r.reco_chg > 0 else "bad", "Upgrade" if r.reco_chg > 0 else "Downgrade")
        elif ok(r.tp_chg) and r.tp_chg >= 0.10:
            chg = chip("good", "TP raised 10%+")
        elif ok(r.tp_chg) and r.tp_chg <= -0.10:
            chg = chip("bad", "TP cut 10%+")
        reco_txt = f"{E(r.prev_Reco)} → {E(r.Reco)}" if isinstance(r.prev_Reco, str) else E(r.Reco)
        name = m.at[r.Ticker, "name"] if r.Ticker in m.index else ""
        rows.append("<tr>" + td_txt(f_date(r.Date, "%d %b"), sort=f_date(r.Date, "%Y%m%d")) + td_stock(r.Ticker, name)
                    + td_txt(E(r.Broker)) + td_txt(reco_txt) + td_num(r.Target, text=f_inr(r.Target))
                    + td_pct(r.tp_chg, 0) + td_pct(r.eps2_chg) + td_txt(chg) + "</tr>")
    actions = table([("Date", ""), ("Stock", ""), ("Broker", ""), ("Rating", ""), ("Target", "num"),
                     ("TP chg", "num"), ("EPS Y2 chg", "num", "Only when both reports use the same fiscal years"),
                     ("Change", "")], rows) if rows else '<p class="muted">No new broker reports this week.</p>'
    return (f'<section class="card" id="emerging"><h2>Emerging</h2><p class="lede">Stocks starting to move, before they '
            f'reach the long or short lists. The score is a percentile among the {n_scored(M)} scored stocks: 100 is the '
            f'best, 80 or more is the top 20%, 20 or less the bottom 20%. F&amp;O stocks carry a badge (is_fno in '
            f'stock_universe).</p>'
            f'<div class="grid2"><div><h3>Moved into the top 20%</h3><p class="note">Score now 80+, below 80 four weeks ago. '
            f'Largest gains first; stocks already in the long list are left out.</p>{top}</div>'
            f'<div><h3>Dropped into the bottom 20%</h3><p class="note">Score now 20 or less, above 20 four weeks ago. '
            f'Largest falls first; stocks already in the short list are left out.</p>{bottom}</div>'
            f'<div><h3>Analyst coverage rising</h3><p class="note">More analysts on Yahoo than 13 weeks ago. A big jump can also '
            f'be a Yahoo data refresh.</p>{cov}</div>'
            f'<div><h3>Steady upgraders</h3><p class="note">Consensus EPS raised in most of the last 13 weeks.</p>{per}</div></div>'
            f'<h3>Broker reports since {f_date(M["prev_week"])}</h3><p class="note">Verified (yellow) rows of AnalystsReco '
            f'only.</p>{actions}</section>')


# ---------------------------------------------------------------- scoreboard
def sec_scoreboard(M):
    S, hist, lm, C = M["stats"], M["hist"], M["last_month"], M["cfg"]
    if not S:
        return '<section class="card" id="scoreboard"><h2>Scoreboard</h2><p>Not enough history yet.</p></section>'
    tiles = [
        tile("Average weekly IC", f_num(S["mean_ic"], 3), f"{S['weeks']} weeks, {f_pct(S['pct_pos'], 0, False)} positive",
             "Rank correlation between the score and the next 4-week return, averaged over weeks"),
        tile("Last 12 weeks", f_num(S["recent_ic"], 3), "average IC"),
        tile("Top 20% vs bottom 20%", f_pct(S["spread"]), "return gap per 4 weeks",
             "Average 4-week return of the best-scored fifth minus the worst-scored fifth"),
        tile("t-statistic", f_num(S["t_stat"], 1), f"{S['n_independent']} non-overlapping 4-week periods",
             "Above about 2 means the average IC is unlikely to be luck"),
    ]
    chart = (ic_chart(hist) + legend([("lr", 1, "Weekly IC"), ("lk", 2, "12-week average")])
             + details_table("IC data", ["Week", "IC", "Top fifth", "Bottom fifth", "Stocks"],
                             [[f_date(d), f_num(r.ic, 3, True), f_pct(r.q5), f_pct(r.q1), f_num(r.n)]
                              for d, r in hist.iloc[::-1].iterrows()]))
    lm_html = ""
    if lm:
        def lm_table(df):
            rows = ["<tr>" + td_stock(tk, r["name"]) + td_num(r["pct"], text=f_num(r["pct"])) + td_pct(r["ret"])
                    + td_pct(r["rel"]) + "</tr>" for tk, r in df.iterrows()]
            rows.append('<tr class="sum">' + td_txt("<b>Average</b>") + "<td></td>" + td_pct(df["ret"].mean())
                        + td_pct(df["rel"].mean()) + "</tr>")
            return table([("Stock", ""), ("Score then", "num"), ("Return", "num"),
                          ("vs average", "num", "Return minus the equal-weight average of all stocks scored that week")],
                         rows, cls="compact")
        spread = lm["long"]["ret"].mean() - lm["short"]["ret"].mean()
        lm_html = (f'<h3>How the lists from {f_date(lm["date"])} did</h3><p class="note">The {lm["n_universe"]} stocks '
                   f'scored that week returned {f_pct(lm["bench"])} on average (equal weight) over the next 4 weeks. '
                   f'Long list minus short list: {f_pct(spread)}.</p>'
                   f'<div class="grid2"><div><h3>Top 10 long</h3>{lm_table(lm["long"])}</div>'
                   f'<div><h3>Bottom 10 short (F&amp;O)</h3>{lm_table(lm["short"])}</div></div>')
    how = (f'<details class="drill" open><summary>How to read this</summary><ul class="defs">'
           f'<li>Every week all scored stocks (about {n_scored(M)}: your Yahoo consensus list with {C.MIN_ANALYSTS}+ '
           f'analysts and a price) are ranked by the score. Four weeks later we check whether the higher-ranked stocks '
           f'actually did better.</li>'
           f'<li><b>IC</b> is the rank correlation between that week\'s scores and the next 4-week returns: +1 means the '
           f'ranking was perfect, 0 means no better than chance. Above 0.05 on average is useful.</li>'
           f'<li><b>Top 20% vs bottom 20%</b> is the average 4-week return of the best-scored fifth minus the worst-scored '
           f'fifth.</li>'
           f'<li>Everything here compares stocks with each other inside your list. It does not compare with NIFTY or any '
           f'index, because the list is a subset of stocks tracked by AMBIT and MOSL, not the index.</li>'
           f'<li>The last 4 weeks are missing from the chart because their 4-week result is not known yet.</li></ul></details>')
    return (f'<section class="card" id="scoreboard"><h2>Scoreboard</h2><p class="lede">Is the score still working? Each '
            f'bar compares one week\'s scores with the following 4-week returns ({f_date(S["first"])} to {f_date(S["last"])}).</p>'
            f'{how}<div class="tiles">{"".join(tiles)}</div><h3>Weekly IC</h3>{chart}{lm_html}</section>')


def sec_method(M):
    C = M["cfg"]
    items = [
        ("Scored stocks", f"Tickers in YahooFinanceTickers.csv with {C.MIN_ANALYSTS}+ analysts on Yahoo and a recent price "
                          "in nse_price_history."),
        ("Composite score", "Equal weight of four ranks across scored stocks: 4-week change in consensus EPS, 4-week change "
                            "in consensus TP, minus the 4-week price return (short-term reversal), and the price return "
                            "from 12 months ago to 1 month ago (12-1 momentum). A stock listed for less than a year, or with "
                            "an unadjusted corporate action in that window, is scored on the other three. Shown as a "
                            "percentile: 100 = best, 80+ = top 20%, 20 or less = bottom 20%. Nothing is fitted."),
        ("Score inputs", "Numbers only: Yahoo consensus estimates and prices from nse_price_history. Broker ratings "
                         "(AnalystsReco) are shown next to each stock but are not part of the score: tested as a fifth "
                         "part they did not improve the 4-week results. Qualitative notes in AnalystVeiwInsight.xlsx are "
                         "not read; only its AMBIT Industry / Sub-Industry labels are used, to group stocks."),
        ("EPS 1W / 4W / 13W", "(consensus EPS now minus consensus EPS 1, 4 or 13 weeks ago) / |EPS then|, for the same "
                              "fiscal year on both dates, then averaged over the two fiscal years Yahoo shows. Comparing "
                              "the same year keeps the April-May roll (FY27/FY28 replacing FY26/FY27) from looking like a "
                              "revision. Uses the average estimate unless the Focus Five toggle says Lowest or Highest."),
        ("TP 4W / 13W", "Consensus target price now / target price 4 or 13 weeks ago, minus 1."),
        ("1M / 3M vs NIFTY", "Stock price return over 1 or 3 months minus the NIFTY return over the same dates."),
        ("Industry figures", "Stocks are grouped by the AMBIT Industry and Sub-Industry in AnalystVeiwInsight.xlsx (Tracker "
                             "sheet; ConsolidatedView carries the same classification). Each figure is the median across "
                             "the group's scored stocks, every stock counting once. Net EPS upgrades = share of stocks "
                             "with 4-week EPS change above +0.5% minus share below -0.5%."),
        ("Splits and bonuses", "Prices come from nse_price_history, where you correct splits. Consensus TP and EPS history "
                               "is rescaled in memory using stock_split_probability notes; unexplained jumps are left out "
                               "of revisions. A pending split (is_processed = NO) with a clean ratio is adjusted for this "
                               "report only and listed in Data checks."),
        ("Read only", "The dashboard only reads the database and the Excel / CSV files. It writes just its own outputs: "
                      "the two HTML files and the history merge."),
        ("Broker views", "Only AnalystsReco rows whose Ticker cell is yellow (verified). Each rating is compared with "
                         "that broker's average rating over 18 months, so a MOSL Neutral counts as negative and an AMBIT "
                         "Sell is less unusual. Views older than 9 months are ignored."),
        ("Consensus age", "Weeks since the Yahoo TP or EPS last changed. Yahoo updates many Indian stocks in bursts, so a "
                          "fresh revision after a long gap can carry months of analyst changes."),
        ("IC", "Information coefficient: the rank correlation between the score one week and the return over the next "
               "4 weeks, within the scored stocks. Above 0.05 is useful; above 0.15 week after week would suggest a data error."),
    ]
    dl = "".join(f"<dt>{E(k)}</dt><dd>{E(v)}</dd>" for k, v in items)
    return f'<section class="card" id="method"><h2>Method</h2><dl class="method">{dl}</dl></section>'


# ================================================================ page
CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
--pos:#006300;--neg:#d03b3b;
--good:#0ca30c;--warn:#fab219;--bad:#d03b3b;--divp:#2a78d6;--divn:#e34948;--divm:#f0efec;--chip:rgba(11,11,11,.045);
--hover:rgba(11,11,11,.035);--shadow:0 6px 18px rgba(0,0,0,.12)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;
--s2:#d95926;--s3:#199e70;--s4:#c98500;--pos:#0ca30c;--neg:#e66767;--divp:#3987e5;--divn:#e66767;--divm:#383835;
--chip:rgba(255,255,255,.07);
--hover:rgba(255,255,255,.05);--shadow:0 6px 18px rgba(0,0,0,.5)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--pos:#0ca30c;--neg:#e66767;
--divp:#3987e5;--divn:#e66767;--divm:#383835;--chip:rgba(255,255,255,.07);--hover:rgba(255,255,255,.05);
--shadow:0 6px 18px rgba(0,0,0,.5)}
*{box-sizing:border-box}html{scroll-padding-top:96px}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1320px;margin:0 auto;padding:0 16px}
header.top{position:sticky;top:0;z-index:10;background:var(--page);border-bottom:1px solid var(--border)}
.hdr{display:flex;justify-content:space-between;align-items:center;gap:12px;padding-top:10px}
h1{font-size:20px;margin:0}.sub{margin:2px 0 0;color:var(--ink2);font-size:13px}
nav{display:flex;flex-wrap:wrap;gap:4px 14px;padding-bottom:8px;padding-top:6px}
nav a{color:var(--ink2);text-decoration:none;font-size:13px}nav a:hover{color:var(--ink);text-decoration:underline}
main{padding-bottom:40px}
section.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 20px;margin:18px 0}
h2{font-size:18px;margin:0 0 4px}h3{font-size:14px;margin:20px 0 8px}
.lede{color:var(--ink2);margin:0 0 12px;max-width:90ch}.note{color:var(--ink2);margin:-2px 0 8px;font-size:13px;max-width:90ch}
.muted{color:var(--muted)}.small{font-size:12px}
.btn{font:inherit;font-size:13px;color:var(--ink);background:var(--chip);border:1px solid var(--border);border-radius:8px;
padding:4px 10px;cursor:pointer}.btn:hover{background:var(--hover)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin:8px 0 6px}
.tile{border:1px solid var(--border);border-radius:10px;padding:10px 12px}
.tile .tl{color:var(--ink2);font-size:12px}.tile .tv{font-size:22px;font-weight:600;margin:2px 0}.tile .ts{color:var(--muted);font-size:12px}
.tw{overflow-x:auto;max-width:100%}
table.data{border-collapse:collapse;width:100%;font-size:13px}
table.data th{text-align:left;font-weight:600;color:var(--ink2);padding:6px 8px;border-bottom:1px solid var(--axis);
white-space:nowrap;cursor:pointer;user-select:none;background:var(--surface);vertical-align:bottom}
table.data th[aria-sort=ascending]::after{content:" \\2191"}table.data th[aria-sort=descending]::after{content:" \\2193"}
table.data td{padding:5px 8px;border-bottom:1px solid var(--grid);white-space:nowrap;vertical-align:middle}
table.data tbody tr:hover{background:var(--hover)}table.compact td{padding:4px 8px}
table.data tr.sum td{border-top:1px solid var(--axis);font-weight:600}
.num,table.data th.num{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:var(--pos)}.neg{color:var(--neg)}
td.tk b{font-weight:600}td.tk .nm{display:block;color:var(--ink2);font-size:12px;white-space:normal;max-width:26ch}
.badge{display:inline-block;margin-left:6px;padding:0 5px;border:1px solid var(--border);border-radius:5px;font-size:10px;
color:var(--ink2);vertical-align:1px}
.chip{display:inline-flex;align-items:center;gap:5px;padding:1px 8px 1px 6px;border-radius:999px;background:var(--chip);
font-size:12px;white-space:nowrap;margin:1px 0}.chip .ic{font-size:9px}
.chip.good .ic,.arr.good{color:var(--good)}.chip.bad .ic,.arr.bad{color:var(--bad)}.chip.warn .ic{color:var(--warn)}
.chip.neutral .ic{color:var(--muted)}.arr{font-size:10px;margin-left:3px}
table.data td.flags{white-space:normal;min-width:180px}table.data td.wrap{white-space:normal;min-width:320px}
table.data td.ind{white-space:normal;min-width:110px;max-width:170px}table.data td.drv{white-space:normal;min-width:150px}
.rd{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;vertical-align:1px;background:var(--muted)}
.rd.good{background:var(--good)}.rd.bad{background:var(--bad)}
td .sub{display:block;color:var(--ink2);font-size:12px}
.meter{display:inline-block;width:34px;height:6px;border-radius:3px;background:var(--grid);margin-right:6px;vertical-align:1px;overflow:hidden}
.meter i{display:block;height:100%;background:var(--s1);border-radius:3px}
table.rets .abs{display:none}table.rets.show-abs .abs{display:inline}table.rets.show-abs .rel{display:none}
.h3row{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}.h3row h3{margin-bottom:8px}
td.heat{background:var(--divm);color:var(--ink)}
td.heat.hp{background:color-mix(in oklab,var(--divp) calc(var(--h)*100%),var(--divm))}
td.heat.hn{background:color-mix(in oklab,var(--divn) calc(var(--h)*100%),var(--divm))}
table.heatmap td{border-bottom:2px solid var(--surface)}table.heatmap td.heat{border-left:2px solid var(--surface)}
.scale{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--ink2);margin:0 0 8px}
.scale .sw{width:28px;height:10px;border-radius:3px;display:inline-block}
.scale .hn{background:color-mix(in oklab,var(--divn) 60%,var(--divm))}.scale .mid{background:var(--divm)}
.scale .hp{background:color-mix(in oklab,var(--divp) 60%,var(--divm))}
details.drill{margin:6px 0}details.drill>summary,details.tv>summary{cursor:pointer;color:var(--ink2);font-size:13px;padding:3px 0}
.drills{columns:2 420px;column-gap:24px}.drills details{break-inside:avoid}
#checks summary{display:flex;align-items:baseline;gap:12px;cursor:pointer}#checks summary h2{display:inline}
ul.checks{margin:12px 0 0;padding:0;list-style:none}ul.checks li{display:flex;gap:10px;align-items:flex-start;padding:4px 0;
border-bottom:1px solid var(--grid)}ul.checks li .chip{min-width:62px}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px;color:var(--ink2);margin:2px 0 6px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend .lk{display:inline-block;width:14px;height:2px;border-radius:1px}.legend .lb{display:inline-block;width:14px;height:10px;border-radius:2px;opacity:.25}
.legend .lr{display:inline-block;width:10px;height:10px;border-radius:2px}
.lk.s1,.lr.s1{background:var(--s1)}.lk.s2,.lb.s2{background:var(--s2)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}
article.fc{border:1px solid var(--border);border-radius:10px;padding:10px 12px}
article.fc header{display:flex;flex-wrap:wrap;gap:4px 8px;align-items:baseline;margin-bottom:4px}
.fc-meta{width:100%;color:var(--ink2);font-size:12px}.fc-sub{font-size:12px;color:var(--ink2);margin-top:6px}
svg{display:block;width:100%;height:auto;overflow:visible}
svg .grid{stroke:var(--grid);stroke-width:1}svg .base{stroke:var(--axis);stroke-width:1}
svg .ax{fill:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}
svg .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .ln.s1{stroke:var(--s1)}svg .ln.s2{stroke:var(--s2)}svg .band.s2{fill:var(--s2);opacity:.12}
svg .dot,svg .hd{stroke:var(--surface);stroke-width:2}svg .s1.dot,svg .s1.hd{fill:var(--s1)}svg .s2.dot,svg .s2.hd{fill:var(--s2)}
svg .xh{stroke:var(--axis);stroke-width:1}svg .hit,svg .hitbar{fill:transparent}
svg .bar{fill:var(--s1)}svg .hitbar:hover{fill:var(--hover)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:4px 24px}
.filters{display:flex;gap:16px;align-items:center;margin:4px 0 8px;font-size:13px}
dl.method{display:grid;grid-template-columns:minmax(140px,200px) 1fr;gap:8px 18px;margin:8px 0 0}
dl.method dt{font-weight:600}dl.method dd{margin:0;color:var(--ink2)}
#tt{position:fixed;z-index:50;pointer-events:none;visibility:hidden;background:var(--surface);color:var(--ink);
border:1px solid var(--border);border-radius:8px;box-shadow:var(--shadow);padding:7px 10px;font-size:12px;max-width:340px}
#tt .h{color:var(--ink2);margin-bottom:3px}#tt .r{display:flex;align-items:center;gap:7px;white-space:nowrap}
#tt .k{width:12px;height:2px;border-radius:1px}#tt .v{font-weight:600}#tt .l{color:var(--ink2)}
#tt .k.s1{background:var(--s1)}#tt .k.s2{background:var(--s2)}#tt .k.s3{background:var(--s3)}#tt .k.s4{background:var(--s4)}
svg .ln.s3{stroke:var(--s3)}svg .ln.s4{stroke:var(--s4)}svg .s3.dot,svg .s3.hd{fill:var(--s3)}svg .s4.dot,svg .s4.hd{fill:var(--s4)}
.lk.s3{background:var(--s3)}.lk.s4{background:var(--s4)}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.seg button{font:inherit;font-size:13px;background:transparent;color:var(--ink2);border:0;border-left:1px solid var(--border);
padding:4px 10px;cursor:pointer}.seg button:first-child{border-left:0}.seg button:hover{background:var(--hover)}
.seg button[aria-pressed=true]{background:var(--chip);color:var(--ink);font-weight:600}
table.st .smin,table.st .smax{display:none}table.st.show-min .savg,table.st.show-max .savg{display:none}
table.st.show-min .smin,table.st.show-max .smax{display:inline}
ul.defs{margin:6px 0 10px;padding-left:18px;color:var(--ink2);font-size:13px;max-width:100ch}ul.defs li{margin:3px 0}
:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
@media (max-width:640px){section.card{padding:14px 12px}.grid2{grid-template-columns:1fr}dl.method{grid-template-columns:1fr}
.cards{grid-template-columns:1fr}h1{font-size:18px}}
"""

JS = r"""
(function(){
const tt=document.getElementById('tt');
function place(x,y){const r=tt.getBoundingClientRect();let px=x+14,py=y+14;
if(px+r.width>innerWidth-8)px=x-r.width-14;if(py+r.height>innerHeight-8)py=y-r.height-14;
tt.style.left=Math.max(8,px)+'px';tt.style.top=Math.max(8,py)+'px';tt.style.visibility='visible';}
function hide(){tt.style.visibility='hidden';}
function tipText(el){tt.textContent='';el.dataset.tip.split('\n').forEach((line,i)=>{const d=document.createElement('div');
d.textContent=line;d.className=i===0?'v':'l';tt.appendChild(d);});}
document.addEventListener('pointermove',e=>{const el=e.target.closest('[data-tip]');
if(el){tipText(el);place(e.clientX,e.clientY);}else if(!e.target.closest('svg.lc'))hide();});
document.addEventListener('focusin',e=>{const el=e.target.closest('[data-tip]');if(el){tipText(el);
const r=el.getBoundingClientRect();place(r.left,r.bottom);}});
document.addEventListener('focusout',hide);
const fmt=(v,k)=>{if(v===null||v===undefined)return '–';if(k==='pct')return (v*100).toFixed(1).replace('-','−')+'%';
const d=k==='eps'?2:(Math.abs(v)<100?1:0);return (k==='price'||k==='eps'?'₹':'')+v.toLocaleString('en-IN',{minimumFractionDigits:d,maximumFractionDigits:d});};
document.querySelectorAll('svg.lc').forEach(svg=>{let c;try{c=JSON.parse(svg.dataset.c);}catch(e){return;}
const xh=svg.querySelector('.xh');const ns='http://www.w3.org/2000/svg';
const dots=c.s.map(s=>{const d=document.createElementNS(ns,'circle');d.setAttribute('r','4');d.setAttribute('class','hd s'+s.slot);
d.setAttribute('visibility','hidden');svg.appendChild(d);return d;});
function idx(e){const pt=svg.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;const p=pt.matrixTransform(svg.getScreenCTM().inverse());
let b=0,bd=1e9;c.x.forEach((x,i)=>{const d=Math.abs(x-p.x);if(d<bd){bd=d;b=i;}});return b;}
svg.addEventListener('pointermove',e=>{const i=idx(e);xh.setAttribute('x1',c.x[i]);xh.setAttribute('x2',c.x[i]);xh.setAttribute('visibility','visible');
dots.forEach((d,k)=>{const y=c.s[k].y[i];if(y===null){d.setAttribute('visibility','hidden');}else{d.setAttribute('cx',c.x[i]);d.setAttribute('cy',y);d.setAttribute('visibility','visible');}});
tt.textContent='';const h=document.createElement('div');h.className='h';h.textContent=c.d[i];tt.appendChild(h);
c.s.forEach(s=>{const r=document.createElement('div');r.className='r';const k=document.createElement('span');k.className='k s'+s.slot;
const v=document.createElement('span');v.className='v';v.textContent=fmt(s.v[i],c.k);const l=document.createElement('span');l.className='l';l.textContent=s.n;
r.append(k,v,l);tt.appendChild(r);});
if(c.b&&c.b.lo[i]!==null){const r=document.createElement('div');r.className='l';r.textContent=c.b.n+': '+fmt(c.b.lo[i],c.k)+' to '+fmt(c.b.hi[i],c.k);tt.appendChild(r);}
place(e.clientX,e.clientY);});
svg.addEventListener('pointerleave',()=>{xh.setAttribute('visibility','hidden');dots.forEach(d=>d.setAttribute('visibility','hidden'));hide();});});
function val(td){if(!td)return null;const v=td.dataset.v;if(v===undefined)return td.textContent.trim().toLowerCase();if(v==='')return null;
const n=Number(v);return isNaN(n)?v.toLowerCase():n;}
document.querySelectorAll('table.data').forEach(tbl=>{const ths=tbl.tHead?Array.from(tbl.tHead.rows[0].cells):[];
ths.forEach((th,i)=>{th.tabIndex=0;const go=()=>{const asc=th.getAttribute('aria-sort')!=='ascending';ths.forEach(h=>h.removeAttribute('aria-sort'));
th.setAttribute('aria-sort',asc?'ascending':'descending');const tb=tbl.tBodies[0];const rows=Array.from(tb.rows).filter(r=>!r.classList.contains('sum'));
const sums=Array.from(tb.rows).filter(r=>r.classList.contains('sum'));
rows.sort((a,b)=>{const x=val(a.cells[i]),y=val(b.cells[i]);if(x===null&&y===null)return 0;if(x===null)return 1;if(y===null)return -1;
return (x<y?-1:x>y?1:0)*(asc?1:-1);});rows.concat(sums).forEach(r=>tb.appendChild(r));};
th.addEventListener('click',go);th.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();go();}});});});
function limit(tbl){const n=Number(tbl.dataset.limit||0);const fno=tbl.dataset.fnoOnly==='1';
const rows=Array.from(tbl.tBodies[0].rows).sort((a,b)=>Number(a.dataset.rank)-Number(b.dataset.rank));let shown=0;
rows.forEach(r=>{const okr=!fno||r.dataset.fno==='1';const vis=okr&&(!n||shown<n);r.hidden=!vis;if(vis)shown++;});}
document.querySelectorAll('[data-fno-filter]').forEach(cb=>cb.addEventListener('change',()=>{const t=document.getElementById(cb.dataset.fnoFilter);
t.dataset.fnoOnly=cb.checked?'1':'0';limit(t);}));
document.querySelectorAll('[data-toggle-abs]').forEach(b=>b.addEventListener('click',()=>{const t=document.getElementById(b.dataset.toggleAbs);
const on=t.classList.toggle('show-abs');b.setAttribute('aria-pressed',on);b.textContent=on?'Showing absolute returns':b.dataset.relLabel;
t.querySelectorAll('td[data-abs]').forEach(td=>{const a=td.dataset.abs;td.dataset.abs=td.dataset.v;td.dataset.v=a;});}));
document.querySelectorAll('[data-toggle-abs]').forEach(b=>b.dataset.relLabel=b.textContent);
document.querySelectorAll('[data-stat-for]').forEach(b=>b.addEventListener('click',()=>{const t=document.getElementById(b.dataset.statFor);
const st=b.dataset.stat;t.classList.remove('show-min','show-max');if(st!=='avg')t.classList.add('show-'+st);
b.parentElement.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===b?'true':'false'));
t.querySelectorAll('td[data-avg]').forEach(td=>{td.dataset.v=td.dataset[st];});}));
const root=document.documentElement;try{const s=localStorage.getItem('wd-theme');if(s)root.dataset.theme=s;}catch(e){}
document.getElementById('theme').addEventListener('click',()=>{const cur=root.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
root.dataset.theme=cur==='dark'?'light':'dark';try{localStorage.setItem('wd-theme',root.dataset.theme);}catch(e){}});
})();
"""


def render(M):
    body = "".join([sec_checks(M), sec_market(M), sec_focus(M), sec_shortlist(M), sec_long(M), sec_short(M),
                    sec_divergence(M), sec_emerging(M), sec_scoreboard(M), sec_method(M)])
    title = f"Weekly Dashboard {M['as_of']:%d %b %Y}"
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{E(title)}</title>'
            f'<style>{CSS}</style></head><body>{sec_header(M)}<main class="wrap">{body}</main>'
            f'<div id="tt" role="tooltip"></div><script>{JS}</script></body></html>')
