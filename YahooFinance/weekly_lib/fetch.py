"""Yahoo Finance consensus download (same logic and CSV columns as the old yahoo_consensus.ipynb, so the weekly
history stays comparable): target price, revenue and EPS estimates for the current and next fiscal year."""
import time

import pandas as pd

COLUMNS = ["YAHOO Ticker", "Date", "FIN Yr", "TP(Avg)", "TP(Min)", "TP(Max)", "Rev.(Avg)", "Rev.(Min)", "Rev.(Max)",
           "EPS(Avg)", "EPS(Min)", "EPS(Max)", "#Count", "Year Ago Rev.", "Year Ago EPS",
           "Rev_Growth_(Min)", "Rev_Growth_(Avg)", "Rev_Growth_(Max)", "EPS_Growth_(Min)", "EPS_Growth_(Avg)",
           "EPS_Growth_(Max)", "Rev_Max_Avg_Change", "Rev_Avg_Min_Change", "EPS_Max_Avg_Change", "EPS_Avg_Min_Change",
           "TP_Max_Avg_Change", "TP_Avg_Min_Change", "Current Price", "TP_Appreciation_(Min)",
           "TP_Appreciation_(Avg)", "TP_Appreciation_(Max)"]


def _safe(d, k):
    try:
        v = d.get(k, 0)
        return 0 if v is None or pd.isna(v) else v
    except Exception:
        return 0


def _growth(est, base):
    try:
        return round(est / base - 1, 4) if base else 0
    except Exception:
        return 0


def _disp(avg, value):
    try:
        return round((value - avg) / avg, 4) if avg else 0
    except Exception:
        return 0


def _try(fn, default):
    try:
        value = fn()
        return default if value is None else value
    except Exception as exc:                       # yfinance raises on unknown symbols and rate limits
        if "Too Many Requests" in str(exc) or "Rate" in type(exc).__name__:
            raise
        return default


def _merge_rows(primary, fallback):
    """Fill missing fields of the primary exchange row from the other exchange."""
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    out = {}
    for key in ("avg", "low", "high", "yearAgoRevenue", "yearAgoEps", "numberOfAnalysts"):
        pv, fv = _safe(primary, key), _safe(fallback, key)
        out[key] = pv if pv not in (0, None) else fv
    return out


def fetch_one(yf, base, date_str):
    """Two rows (current and next fiscal year) for one NSE / BSE symbol."""
    t_bo, t_ns = yf.Ticker(base + ".BO"), yf.Ticker(base + ".NS")
    earn_bo = _try(lambda: t_bo.earnings_estimate, pd.DataFrame())
    earn_ns = _try(lambda: t_ns.earnings_estimate, pd.DataFrame())
    rev_bo = _try(t_bo.get_revenue_estimate, pd.DataFrame())
    rev_ns = _try(t_ns.get_revenue_estimate, pd.DataFrame())
    info_bo = _try(lambda: t_bo.info, {})
    info_ns = _try(lambda: t_ns.info, {})
    main = t_bo if not earn_bo.empty else (t_ns if not earn_ns.empty else t_bo)
    info = info_bo if _safe(info_bo, "currentPrice") != 0 else info_ns
    stmt = _try(lambda: main.income_stmt, pd.DataFrame())
    latest_fy = pd.to_datetime(stmt.columns[0]).year if len(getattr(stmt, "columns", [])) else None

    price = _safe(info, "currentPrice")
    tp_avg, tp_min, tp_max = _safe(info, "targetMeanPrice"), _safe(info, "targetLowPrice"), _safe(info, "targetHighPrice")
    rows = []
    for i, period in enumerate(("0y", "+1y")):
        fin_year = f"FY{(latest_fy + i + 1) % 100}" if latest_fy else f"Y{i + 1}"
        rev_row = _merge_rows(rev_bo.loc[period] if period in rev_bo.index else None,
                              rev_ns.loc[period] if period in rev_ns.index else None)
        eps_row = _merge_rows(earn_bo.loc[period] if period in earn_bo.index else None,
                              earn_ns.loc[period] if period in earn_ns.index else None)
        rev_avg, rev_min, rev_max = (_safe(rev_row, k) / 1e7 for k in ("avg", "low", "high"))
        eps_avg, eps_min, eps_max = (_safe(eps_row, k) for k in ("avg", "low", "high"))
        ya_rev, ya_eps = _safe(rev_row, "yearAgoRevenue") / 1e7, _safe(eps_row, "yearAgoEps")
        rows.append({
            "YAHOO Ticker": base, "Date": date_str, "FIN Yr": fin_year,
            "TP(Avg)": round(tp_avg), "TP(Min)": round(tp_min), "TP(Max)": round(tp_max),
            "Rev.(Avg)": round(rev_avg, 1), "Rev.(Min)": round(rev_min, 1), "Rev.(Max)": round(rev_max, 1),
            "EPS(Avg)": round(eps_avg, 2), "EPS(Min)": round(eps_min, 2), "EPS(Max)": round(eps_max, 2),
            "#Count": int(_safe(rev_row, "numberOfAnalysts")),
            "Year Ago Rev.": round(ya_rev, 1), "Year Ago EPS": round(ya_eps, 2),
            "Rev_Growth_(Min)": _growth(rev_min, ya_rev), "Rev_Growth_(Avg)": _growth(rev_avg, ya_rev),
            "Rev_Growth_(Max)": _growth(rev_max, ya_rev),
            "EPS_Growth_(Min)": _growth(eps_min, ya_eps), "EPS_Growth_(Avg)": _growth(eps_avg, ya_eps),
            "EPS_Growth_(Max)": _growth(eps_max, ya_eps),
            "Rev_Max_Avg_Change": _disp(rev_avg, rev_max), "Rev_Avg_Min_Change": _disp(rev_avg, rev_min),
            "EPS_Max_Avg_Change": _disp(eps_avg, eps_max), "EPS_Avg_Min_Change": _disp(eps_avg, eps_min),
            "TP_Max_Avg_Change": _disp(tp_avg, tp_max), "TP_Avg_Min_Change": _disp(tp_avg, tp_min),
            "Current Price": round(price, 2),
            "TP_Appreciation_(Min)": _growth(tp_min, price), "TP_Appreciation_(Avg)": _growth(tp_avg, price),
            "TP_Appreciation_(Max)": _growth(tp_max, price),
        })
    return rows


def download(tickers, date_str, partial_file, pause, retries, log=print):
    """Fetch every ticker, saving progress to `partial_file` so an interrupted run resumes where it stopped."""
    import yfinance as yf
    done = pd.read_csv(partial_file) if partial_file.exists() else pd.DataFrame(columns=COLUMNS)
    done_set = set(done["YAHOO Ticker"].astype(str))
    todo = [t for t in tickers if t not in done_set]
    if done_set:
        log(f"Resuming: {len(done_set)} tickers already fetched, {len(todo)} to go")
    rows, failed, start = [], [], time.time()
    for n, base in enumerate(todo, start=1):
        for attempt in range(1, retries + 1):
            try:
                rows += fetch_one(yf, base, date_str)
                break
            except Exception as exc:              # rate limit or network trouble: back off and retry
                wait = 20 * attempt
                log(f"  {base}: {type(exc).__name__} {str(exc)[:80]} - retry {attempt}/{retries} in {wait}s")
                time.sleep(wait)
        else:
            failed.append(base)
        time.sleep(pause)
        if n % 25 == 0 or n == len(todo):
            pd.concat([done, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True).to_csv(partial_file, index=False)
            rate = (time.time() - start) / n
            log(f"  {n}/{len(todo)} done, about {rate * (len(todo) - n) / 60:.0f} min left")
    result = pd.concat([done, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
    return result[COLUMNS], failed
