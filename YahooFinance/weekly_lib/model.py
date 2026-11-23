"""Builds everything the dashboard shows: one data model from the consensus files, the database and the Excel inputs."""
from datetime import datetime

import numpy as np
import pandas as pd

import re

from . import calc
from . import config as C
from . import db
from . import load


def _fmt_date(d):
    return pd.Timestamp(d).strftime("%d-%b-%Y") if pd.notna(d) else "n/a"


def _raw(v):
    return v.strftime("%Y-%m-%d") if isinstance(v, (pd.Timestamp, datetime)) else str(v)


def idea_lists(m, top_n, min_analysts):
    """The dashboard's idea lists from one row per stock. Needs: eligible, score, pct, pct_4w, fno, eps4, eps13, mom,
    count, count_chg13, persist, price. Shared with dashboard_db so the website shows the same lists."""
    el = m[m["eligible"]].copy()
    el["jump"] = el["pct"] - el["pct_4w"].fillna(50)        # newly covered stocks count from the middle
    shown_long = el.sort_values("score", ascending=False).head(top_n).index
    shown_short = el[el["fno"]].sort_values("score").head(top_n).index
    return {
        "long": el.sort_values("score", ascending=False).head(60),
        "short": el[el["fno"]].sort_values("score").head(top_n),
        "div_short": el[(el["eps4"] < -0.01) & (el["mom"] > 0.03)]
        .assign(gap=lambda d: d["mom"] - d["eps4"]).sort_values("gap", ascending=False).head(40),
        "div_long": el[(el["eps4"] > 0.01) & (el["mom"] < -0.03)]
        .assign(gap=lambda d: d["eps4"] - d["mom"]).sort_values("gap", ascending=False).head(40),
        "new_top": el[(el["pct"] >= 80) & ~(el["pct_4w"] >= 80) & ~el.index.isin(shown_long)]
        .sort_values("jump", ascending=False).head(12),
        "new_bottom": el[(el["pct"] <= 20) & ~(el["pct_4w"] <= 20) & ~el.index.isin(shown_short)]
        .sort_values("jump").head(12),
        "coverage": m[(m["count_chg13"] >= 3) & m["price"].notna()]
        .sort_values(["count_chg13", "count"], ascending=False).head(12),
        "persistent": m[(m["persist"] >= 6) & (m["count"] >= min_analysts)]
        .sort_values(["persist", "eps13"], ascending=False).head(12),
    }


def load_core(engine, extra_tickers=(), consensus_dir=None):
    """Consensus history, DB prices (renames joined, pending splits adjusted), index levels and the weekly scores.
    Needs no Excel file, so the website laptop can run it too (dashboard_db)."""
    checks = []
    renames = db.load_ticker_renames(engine)        # old NSE symbol -> current symbol (only suggested, never applied)
    cons_long, notes = load.load_consensus(consensus_dir or C.CONSENSUS_DIR)
    checks += [("info", n) for n in notes]
    cons = calc.Consensus(cons_long)
    universe = set(cons.tickers) | set(extra_tickers)

    # 12-1 momentum needs a year of prices before the first scored week
    price_start = min(pd.Timestamp(C.PRICE_HISTORY_START), cons.dates[0] - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
    renamed_from = {new: old for old, new in renames.items() if new in universe and old not in universe}
    with_old = universe | set(renamed_from.values())
    prices = db.load_prices(engine, with_old, price_start)
    prices, joined, notes = calc.join_renamed(prices, renamed_from)
    if joined:
        checks.append(("info", "Renamed stocks: nse_price_history keeps their earlier prices under the old symbol, "
                               "so the two are joined here (read only): " + "; ".join(joined)))
    checks += [("warn", n) for n in notes]
    events = db.load_split_events(engine, with_old, price_start)
    events["ticker"] = events["ticker"].replace({old: new for new, old in renamed_from.items()})
    prices, notes = calc.adjust_unprocessed_splits(prices, events)
    checks += [("warn", n) for n in notes]
    breaks = calc.price_breaks(events)
    prices_ff = prices.ffill()
    idx = db.load_index_levels(engine, C.INDICES, price_start)

    cons.apply_corporate_actions(events)            # consensus clean-up (splits, odd jumps)
    cons.flag_suspect_jumps()
    checks += [("info", n) for n in cons.notes]
    scores, hist = calc.score_history(cons, prices, prices_ff, C.MIN_ANALYSTS, C.SCORE_WEEKS, C.FORWARD_WEEKS, breaks)
    return {"checks": checks, "renames": renames, "cons_long": cons_long, "cons": cons, "universe": universe,
            "prices": prices, "prices_ff": prices_ff, "events": events, "idx": idx, "scores": scores, "hist": hist}


def build_model(today=None):
    today = pd.Timestamp(today or datetime.now().date())
    engine = db.get_engine(C.DB_ENV_FILE)

    # ------------------------------------------------ watchlists and Focus Five (from the DB portfolio)
    watch = load.load_watchlists(C.WATCHLISTS_FILE)
    holdings = db.load_focus_holdings(engine, C.FOCUS_MEMBER_ID, C.FOCUS_COMPOSITE_ID)
    holdings["ticker"] = holdings["ticker"].str.upper()
    focus = [t for t in holdings["ticker"] if t not in C.CASH_TICKERS]
    shortlist = {}
    for section in C.SHORTLIST_SECTIONS:
        label = section.split("_", 1)[-1].replace("Shortlisted", "").strip() or "Shortlisted"
        for t in watch.get(section, []):
            shortlist.setdefault(t, []).append(label)

    # ------------------------------------------------ consensus, prices and scores through time
    core = load_core(engine, set(focus) | set(shortlist))
    checks = core["checks"]                         # (level, text) shown in the Data checks panel
    renames, cons_long, cons, universe = core["renames"], core["cons_long"], core["cons"], core["universe"]
    prices, prices_ff, idx, scores, hist = core["prices"], core["prices_ff"], core["idx"], core["scores"], core["hist"]
    as_of = cons.dates[-1]
    prev_week = cons.dates[-2] if len(cons.dates) > 1 else as_of
    watch_all = {t for tickers in watch.values() for t in tickers}
    rename_hints = sorted({f"{t} → {renames[re.sub(r'-(EQ|BE)$', '', t)]}"
                           for t in set(cons.tickers) | watch_all
                           if renames.get(re.sub(r"-(EQ|BE)$", "", t), t) != t})
    idx = idx.loc[pd.Timestamp(C.PRICE_HISTORY_START):]
    idx_ff = idx.ffill()
    meta = db.load_stock_meta(engine, universe)
    port_twrr = db.load_portfolio_twrr(engine, C.FOCUS_MEMBER_ID, C.FOCUS_PORTFOLIO_ID)
    idx_twrr = db.load_index_twrr(engine)
    px_date = prices.index.max()

    # ------------------------------------------------ Excel inputs (read from temp copies; Excel may be open)
    reco, reco_stats = load.load_analyst_reco(load.temp_copy(C.ANALYST_TRACKER_XLSX), C.ANALYST_RECO_SHEET, today)
    ambit_book = pd.ExcelFile(load.temp_copy(C.AMBIT_XLSX))   # opened once: each open of this 7 MB file takes ~25 s
    cls = load.load_ambit_classification(ambit_book)   # Industry / Sub-Industry labels only; no qualitative notes

    # ------------------------------------------------ scoreboard
    stats = calc.scoreboard_stats(hist, C.FORWARD_WEEKS)
    breadth = calc.breadth_history(scores)
    sc = scores[as_of]

    # ------------------------------------------------ one row per stock (latest week)
    t = as_of
    rev1, rev4, rev13 = (cons.revisions(t, w) for w in (1, 4, 13))
    fresh = calc.fresh_mask(prices, px_date)
    price_now = calc.asof_row(prices_ff, px_date).where(fresh)
    rets = calc.returns_table(prices_ff, px_date, C.RETURN_WINDOWS)
    rets[~fresh.reindex(rets.index, fill_value=False)] = np.nan
    idx_rets = calc.returns_table(idx_ff, px_date, C.RETURN_WINDOWS)
    rel = rets.sub(idx_rets.loc[C.MAIN_INDEX], axis=1)
    year = idx.loc[px_date - pd.DateOffset(years=1):px_date, [c for c in C.INDICES if c in idx.columns]].ffill()
    idx_hist = year / year.iloc[0] - 1                # cumulative return over the last 12 months
    bv = calc.broker_views(reco, today, price_now, C.RECO_SCORE, C.BROKER_VIEW_MAX_AGE_DAYS, prev_week)

    m = pd.DataFrame(index=sorted(universe))
    m["name"] = meta["short_name"].reindex(m.index)
    m["name"] = m["name"].fillna(cls["Stock"].reindex(m.index)).fillna(pd.Series(m.index, index=m.index))
    m["fno"] = meta["is_fno"].reindex(m.index).fillna(0).astype(int) == 1
    m["mcap"] = meta["marketcap"].reindex(m.index)
    m["industry"] = cls["Industry"].reindex(m.index).fillna("Unclassified")
    m["subindustry"] = cls["SubIndustry"].reindex(m.index).fillna("Unclassified")
    m["price"] = price_now.reindex(m.index)
    for w in C.RETURN_WINDOWS:
        m["ret_" + w] = rets[w].reindex(m.index)
        m["rel_" + w] = rel[w].reindex(m.index)
    m["from_high"] = calc.from_52w_high(prices, px_date).reindex(m.index)
    m["count"] = cons.count.loc[t].reindex(m.index)
    m["count_chg4"] = rev4["count_chg"].reindex(m.index)
    m["count_chg13"] = rev13["count_chg"].reindex(m.index)
    for col, mat in (("tp", cons.tp), ("tp_min", cons.tp_min), ("tp_max", cons.tp_max)):
        m[col] = mat.loc[t].reindex(m.index)
    m["upside"] = m["tp"] / m["price"] - 1
    m["upside_min"] = m["tp_min"] / m["price"] - 1
    m["upside_max"] = m["tp_max"] / m["price"] - 1
    m["disp"] = (m["tp_max"] - m["tp_min"]) / m["tp"]
    for lab, rv in (("1", rev1), ("4", rev4), ("13", rev13)):
        m["eps" + lab] = rv["eps"].reindex(m.index)
        m["tp" + lab] = rv["tp"].reindex(m.index)
        m["rev" + lab] = rv["rev"].reindex(m.index)
    for stat in ("min", "max"):                      # lowest / highest analyst estimate, for the holdings toggle
        for w in (1, 4, 13):
            rv = cons.revisions(t, w, stat)
            m[f"eps{w}_{stat}"] = rv["eps"].reindex(m.index)
            m[f"tp{w}_{stat}"] = rv["tp"].reindex(m.index)
    m["persist"] = cons.persistence(t, 13).reindex(m.index)
    change = cons.change_stats(t)
    m["age"] = change["age"].reindex(m.index)
    m["gap"] = change["gap"].reindex(m.index)
    for col in ("score", "pct", "z_eps", "z_tp", "z_rev", "z_mom12", "mom", "mom12"):
        m[col] = sc[col].reindex(m.index)
    m["eligible"] = sc["eligible"].reindex(m.index).fillna(False).astype(bool)
    m["rank"] = m["score"].rank(ascending=False)
    for lag, col in ((1, "pct_1w"), (4, "pct_4w")):
        p = cons.past_date(t, lag)
        m[col] = scores[p]["pct"].reindex(m.index) if p in scores else np.nan
    m = m.join(bv["agg"].add_prefix("b_"), how="left")

    latest_tickers = set(cons_long.loc[cons_long["Date"] == as_of, "Ticker"])

    # ------------------------------------------------ sectors
    sector_metrics = m.rename(columns={"rel_1M": "rel1m", "rel_3M": "rel3m"})[calc.SECTOR_COLS + ["eligible"]]
    ind_tbl, sub_tbl = calc.sector_tables(sector_metrics, cls)

    # ------------------------------------------------ idea lists
    lists = idea_lists(m, C.TOP_N, C.MIN_ANALYSTS)

    # ------------------------------------------------ last month's lists, marked to market
    t0 = cons.past_date(t, C.FORWARD_WEEKS)
    last_month = None
    if t0 in scores:
        s0 = scores[t0].join(m[["name", "fno"]], how="left")
        p0 = calc.asof_row(prices_ff, t0)
        ret = (price_now / p0 - 1).reindex(s0.index)
        bench = ret[s0["eligible"]].mean()          # equal-weight average of every stock scored that week
        s0 = s0.assign(ret=ret, rel=ret - bench)
        last_month = {"date": t0, "bench": bench, "n_universe": int(ret[s0["eligible"]].notna().sum()),
                      "long": s0[s0["eligible"]].sort_values("score", ascending=False).head(10),
                      "short": s0[s0["eligible"] & s0["fno"].fillna(False)].sort_values("score").head(10)}

    # ------------------------------------------------ Focus Five chart series
    dates = cons.dates
    px_on_dates = prices_ff.reindex(prices_ff.index.union(pd.DatetimeIndex(dates))).ffill().loc[dates]
    charts = {}
    for tk in focus:
        labels = cons.fy_labels(t, tk) if tk in cons.tickers else []
        charts[tk] = {
            "dates": dates,
            "price": px_on_dates[tk].tolist() if tk in px_on_dates.columns else [np.nan] * len(dates),
            "tp": cons.tp[tk].tolist() if tk in cons.tickers else [],
            "tp_min": cons.tp_min[tk].tolist() if tk in cons.tickers else [],
            "tp_max": cons.tp_max[tk].tolist() if tk in cons.tickers else [],
            "eps": [(lab, cons.eps[lab][tk].tolist()) for lab in labels[:2]],
        }

    # ------------------------------------------------ data checks
    n_cov = int((cons.count.loc[t] >= C.MIN_ANALYSTS).sum())
    checks.insert(0, ("ok", f"Consensus: {len(cons.dates)} weekly snapshots ({_fmt_date(cons.dates[0])} to "
                            f"{_fmt_date(as_of)}); latest has {int(cons.count.loc[t].notna().sum())} tickers, "
                            f"{n_cov} with {C.MIN_ANALYSTS}+ analysts, {int(m['eligible'].sum())} scored"))
    checks.insert(1, ("ok", f"Prices: nse_price_history to {_fmt_date(px_date)}; indices to {_fmt_date(idx.index.max())}"))
    if rename_hints:
        checks.append(("warn", "stock_universe shows these symbols as renamed. Nothing was changed: run "
                               "2_Ticker_Change_or_Refetch.bat to change them in every consensus CSV (and edit "
                               "watchlists.properties yourself): " + ", ".join(rename_hints)))
    no_price = sorted(latest_tickers - set(prices.columns[prices.loc[px_date].notna()]))
    scored = m[m["eligible"]]
    moved = scored[scored["age"] <= 1]
    burst = moved[moved["gap"] >= 8]
    checks.append(("info", f"Yahoo consensus moved in the last 2 weeks for {len(moved)} of {len(scored)} scored stocks; "
                           f"{len(burst)} of those had been frozen for 8+ weeks before (median "
                           f"{moved['gap'].median():.0f} weeks). Yahoo updates in bursts, so a 4-week revision can hold "
                           f"several months of analyst changes. {int((scored['age'] >= 8).sum())} scored stocks are "
                           f"unchanged for 8+ weeks now."))
    if no_price:
        checks.append(("warn", f"{len(no_price)} consensus tickers have no price on {_fmt_date(px_date)} in "
                               f"nse_price_history (delisted, REIT or not loaded?): {', '.join(no_price)}"))
    fixed = reco[reco["DateStatus"] == "fixed"]
    if len(fixed):
        checks.append(("warn", f"AnalystsReco: {len(fixed)} report dates had a wrong year and were repaired from the "
                               f"Quarter column, e.g. " + ", ".join(f"{r.Ticker} {r.Broker} {_raw(r.RawDate)} to "
                                                                    f"{r.Date:%Y-%m-%d}" for r in fixed.head(4).itertuples())))
    bad = reco[reco["DateStatus"] == "unresolved"]
    if len(bad):
        checks.append(("warn", "AnalystsReco: dates that need a manual fix: " + ", ".join(
            f"{r.Ticker} {r.Broker} {r.Quarter} {_raw(r.RawDate)}" for r in bad.itertuples())))
    checks.append(("info", f"AnalystsReco: using the {reco_stats['verified']} rows whose Ticker cell is yellow "
                           f"(verified); skipped {reco_stats['pending']} unmarked rows, {reco_stats['pending_tail']} of them "
                           f"at the bottom of the sheet. {reco_stats['zero_tp']} verified rows have no target price."))
    checks.append(("ok", f"AnalystsReco: {len(reco)} reports, latest {_fmt_date(reco['Date'].max())}; "
                         f"{len(bv['week'])} new since {_fmt_date(prev_week)}"))
    unclassified = sorted(set(m.index[m['eligible'] & (m['industry'] == 'Unclassified')]))
    if unclassified:
        checks.append(("info", f"{len(unclassified)} scored tickers are not in the AMBIT Tracker classification: "
                               f"{', '.join(unclassified[:25])}{' ...' if len(unclassified) > 25 else ''}"))
    focus_missing = [tk for tk in focus if tk not in cons.tickers]
    if focus_missing:
        checks.append(("warn", f"Focus Five tickers without Yahoo consensus: {', '.join(focus_missing)}"))

    return {
        "today": today, "as_of": as_of, "px_date": px_date, "prev_week": prev_week,
        "m": m, "lists": lists, "last_month": last_month, "hist": hist, "stats": stats, "breadth": breadth,
        "ind_tbl": ind_tbl, "sub_tbl": sub_tbl,
        "holdings": holdings, "focus": focus, "shortlist": shortlist, "charts": charts,
        "port_twrr": port_twrr, "idx_twrr": idx_twrr, "idx_rets": idx_rets, "idx_hist": idx_hist, "bv": bv,
        "checks": checks, "cfg": C,
    }
