"""Write the weekly dashboard numbers into MySQL tables for the website (timelineofwealth):

  consensus_weekly_score   one row per week_date + ticker
  consensus_weekly_group   one row per week_date + watchlist_group (101+ sections of watchlists.properties)
  consensus_scoreboard     one row per week_date
  consensus_broker_view    latest verified AnalystsReco report per ticker + broker (needs AnalystViewTracker.xlsx;
                           left unchanged if that file is missing)

Same scores and idea lists as the HTML dashboard (model.load_core / model.idea_lists), from the consensus CSVs and
nse_price_history; no portfolio data. Called by 4_Load_Consensus_To_DB.bat right
after consensus_db.load. A full load rewrites every week in one transaction; otherwise only the last few weeks
(their forward returns fill in over 4 weeks). Re-running is always safe.
    python -m weekly_lib.dashboard_db [--full]
"""
import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import bindparam, text

from weekly_lib import calc, consensus_db, db, model
from weekly_lib import load as inputs
from weekly_lib import config as C

OTHER = "Other"
GROUP_MIN_SECTION = 101          # sector watchlists start at [101_...]

DDL = [
    """CREATE TABLE IF NOT EXISTS consensus_weekly_score (
  week_date date NOT NULL COMMENT 'PK consensus week (Friday)',
  ticker varchar(30) COLLATE utf8_unicode_ci NOT NULL COMMENT 'PK stock_universe ticker',
  analyst_count int(5) DEFAULT NULL,
  watchlist_group varchar(80) COLLATE utf8_unicode_ci NOT NULL COMMENT 'first 101+ section of watchlists.properties, else Other',
  tp_rev_1w decimal(12,6) DEFAULT NULL COMMENT 'consensus TP change vs 1 week before (0.05 = +5%)',
  tp_rev_4w decimal(12,6) DEFAULT NULL,
  eps_fy1_rev_4w decimal(12,6) DEFAULT NULL COMMENT 'nearer fiscal year, same FY label on both dates',
  eps_fy2_rev_4w decimal(12,6) DEFAULT NULL,
  rev_fy1_rev_4w decimal(12,6) DEFAULT NULL COMMENT 'revenue estimate, nearer fiscal year',
  persistence_13w int(4) DEFAULT NULL COMMENT 'weeks with EPS upgrade minus weeks with downgrade, last 13',
  mom_4w decimal(12,6) DEFAULT NULL COMMENT 'price return over the 4 weeks to week_date',
  mom_12_1 decimal(12,6) DEFAULT NULL COMMENT 'price return from 12 months to 1 month before week_date',
  score decimal(12,6) DEFAULT NULL COMMENT 'composite score (mean rank z); NULL if not scored',
  score_rank int(5) DEFAULT NULL COMMENT '1 = best among the week''s scored stocks',
  is_long tinyint(1) NOT NULL DEFAULT 0,
  is_short tinyint(1) NOT NULL DEFAULT 0,
  is_divergence tinyint(1) NOT NULL DEFAULT 0,
  is_emerging tinyint(1) NOT NULL DEFAULT 0,
  price_at_score decimal(20,3) DEFAULT NULL COMMENT 'nse_price_history close on or before week_date',
  {new_cols},
  PRIMARY KEY (week_date, ticker),
  KEY index_consensus_weekly_score_ticker (ticker)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci COMMENT='Weekly composite score per stock (dashboard_db.py)'""",
    """CREATE TABLE IF NOT EXISTS consensus_weekly_group (
  week_date date NOT NULL COMMENT 'PK',
  watchlist_group varchar(80) COLLATE utf8_unicode_ci NOT NULL COMMENT 'PK',
  stock_count int(5) NOT NULL COMMENT 'stocks with consensus that week',
  avg_eps_rev_4w decimal(12,6) DEFAULT NULL,
  up_count int(5) NOT NULL DEFAULT 0 COMMENT 'EPS 4W revision above +0.5%',
  down_count int(5) NOT NULL DEFAULT 0 COMMENT 'EPS 4W revision below -0.5%',
  avg_return_1m decimal(12,6) DEFAULT NULL COMMENT 'mean price return over the month to week_date',
  rel_return_1m_vs_nifty decimal(12,6) DEFAULT NULL COMMENT 'avg_return_1m minus NIFTY over the same dates',
  PRIMARY KEY (week_date, watchlist_group)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci COMMENT='Weekly revisions by watchlist group (dashboard_db.py)'""",
    """CREATE TABLE IF NOT EXISTS consensus_scoreboard (
  week_date date NOT NULL COMMENT 'PK',
  ic_spearman decimal(8,6) DEFAULT NULL COMMENT 'rank corr of score with next 4-week return; NULL until known',
  long_fwd_ret decimal(12,6) DEFAULT NULL COMMENT 'long list next 4-week return minus the scored-stock average',
  short_fwd_ret decimal(12,6) DEFAULT NULL COMMENT 'short list next 4-week return minus the scored-stock average',
  spread_fwd_ret decimal(12,6) DEFAULT NULL COMMENT 'long_fwd_ret minus short_fwd_ret',
  hit_rate decimal(8,6) DEFAULT NULL COMMENT 'share of long and short picks that beat / lagged the average',
  breadth_up decimal(8,6) DEFAULT NULL COMMENT 'share of scored stocks with EPS 4W revision above +0.5%',
  breadth_down decimal(8,6) DEFAULT NULL COMMENT 'share below -0.5%',
  PRIMARY KEY (week_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci COMMENT='Weekly score track record (dashboard_db.py)'""",
    """CREATE TABLE IF NOT EXISTS consensus_broker_view (
  ticker varchar(30) COLLATE utf8_unicode_ci NOT NULL COMMENT 'PK stock_universe ticker',
  broker varchar(10) COLLATE utf8_unicode_ci NOT NULL COMMENT 'PK MOSL / AMBIT / PL / AXIS',
  reco varchar(20) COLLATE utf8_unicode_ci DEFAULT NULL,
  target decimal(20,3) DEFAULT NULL,
  report_date date DEFAULT NULL,
  quarter varchar(10) COLLATE utf8_unicode_ci DEFAULT NULL,
  PRIMARY KEY (ticker, broker)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci
  COMMENT='Latest verified AnalystsReco report per ticker and broker (dashboard_db.py)'""",
]
# columns added to consensus_weekly_score after its first release: in the CREATE above for new installs, and
# added by ALTER TABLE to an existing table (MySQL 5.7 has no ADD COLUMN IF NOT EXISTS)
NEW_SCORE_COLS = [
    ("eps_rev_4w", "decimal(12,6) DEFAULT NULL COMMENT 'EPS 4W revision, both FYs averaged (dashboard EPS 4W)'"),
    ("eps_rev_13w", "decimal(12,6) DEFAULT NULL COMMENT 'EPS 13W revision, both FYs averaged'"),
    ("score_pct", "decimal(8,4) DEFAULT NULL COMMENT 'score percentile, 100 = best'"),
    ("z_eps", "decimal(10,6) DEFAULT NULL COMMENT 'score component: EPS revision rank z'"),
    ("z_tp", "decimal(10,6) DEFAULT NULL COMMENT 'score component: TP revision rank z'"),
    ("z_rev", "decimal(10,6) DEFAULT NULL COMMENT 'score component: minus 4W price rank z (reversal)'"),
    ("z_mom12", "decimal(10,6) DEFAULT NULL COMMENT 'score component: 12-1 momentum rank z'"),
    ("count_chg_13w", "int(5) DEFAULT NULL COMMENT 'change in analyst count over 13 weeks'"),
    ("consensus_age_weeks", "int(5) DEFAULT NULL COMMENT 'weeks since the Yahoo TP or EPS last changed'"),
]
DDL[0] = DDL[0].replace("{new_cols}", ",\n  ".join(f"{c} {d}" for c, d in NEW_SCORE_COLS))
TABLES = ["consensus_weekly_score", "consensus_weekly_group", "consensus_scoreboard"]


def watchlist_groups(path):
    """ticker -> first 101+ section name (in section-number order)."""
    if not path.exists():
        return {}
    out = {}
    for section, tickers in sorted(inputs.load_watchlists(path).items()):
        m = re.match(r"(\d+)_", section)
        if m and int(m.group(1)) >= GROUP_MIN_SECTION:
            for t in tickers:
                out.setdefault(t, section)
    return out


def fy_revisions(cons, mats, t, weeks):
    """(FY1, FY2) % change vs `weeks` ago, same fiscal-year label on both dates; FY1 = nearest year shown at t."""
    nan = pd.Series(np.nan, index=cons.tickers)
    p = cons.past_date(t, weeks)
    if p is None or not mats:
        return nan, nan
    labels = sorted(mats)
    now = np.column_stack([mats[lab].loc[t].to_numpy(dtype=float) for lab in labels])
    before = np.column_stack([mats[lab].loc[p].to_numpy(dtype=float) for lab in labels])
    with np.errstate(divide="ignore", invalid="ignore"):
        chg = np.where(np.abs(before) > 0.01, (now - before) / np.abs(before), np.nan)
    has = ~np.isnan(now)
    rows = np.arange(len(cons.tickers))
    first = np.where(has.any(axis=1), has.argmax(axis=1), -1)
    has2 = has.copy()
    has2[rows[first >= 0], first[first >= 0]] = False
    second = np.where(has2.any(axis=1), has2.argmax(axis=1), -1)
    ok = cons._no_break(p, t).to_numpy()
    pick = lambda pos: pd.Series(np.where((pos >= 0) & ok, chg[rows, np.maximum(pos, 0)], np.nan), index=cons.tickers)
    return pick(first), pick(second)


def week_rows(core, t, fno, groups, nifty):
    """One week's per-stock rows, per-group rows and scoreboard row."""
    cons, scores, prices, pff = core["cons"], core["scores"], core["prices"], core["prices_ff"]
    sc = scores[t]
    count = cons.count.loc[t]
    tk = count.index[count.notna()]
    rev1, rev4, rev13 = (cons.revisions(t, w) for w in (1, 4, 13))
    eps1, eps2 = fy_revisions(cons, cons.eps, t, 4)
    rv1, _ = fy_revisions(cons, cons.rev, t, 4)
    p4 = cons.past_date(t, 4)
    px = calc.asof_row(pff, t).where(calc.fresh_mask(prices, t))
    m = pd.DataFrame(index=tk)
    m["count"] = count.reindex(tk)
    m["eligible"] = sc["eligible"].reindex(tk).fillna(False).astype(bool)
    for col in ("score", "pct", "mom", "mom12", "z_eps", "z_tp", "z_rev", "z_mom12"):
        m[col] = sc[col].reindex(tk)
    m["consensus_age_weeks"] = cons.change_stats(t)["age"].reindex(tk)
    m["pct_4w"] = scores[p4]["pct"].reindex(tk) if p4 in scores else np.nan
    m["fno"] = fno.reindex(tk, fill_value=False).astype(bool)
    m["eps4"], m["eps13"] = rev4["eps"].reindex(tk), rev13["eps"].reindex(tk)
    m["count_chg13"] = rev13["count_chg"].reindex(tk)
    m["persist"] = cons.persistence(t, 13).reindex(tk)
    m["price"] = px.reindex(tk)
    lists = model.idea_lists(m, C.TOP_N, C.MIN_ANALYSTS)
    flag = lambda *names: m.index.isin(set().union(*(lists[n].index for n in names))).astype(int)

    m["score_rank"] = m["score"].rank(ascending=False, method="first")
    m["is_long"] = m.index.isin(lists["long"].head(C.TOP_N).index).astype(int)
    m["is_short"] = flag("short")
    m["is_divergence"] = flag("div_short", "div_long")
    m["is_emerging"] = flag("new_top", "new_bottom", "coverage", "persistent")   # broker-report part left out
    m["tp_rev_1w"], m["tp_rev_4w"] = rev1["tp"].reindex(tk), rev4["tp"].reindex(tk)
    m["eps_fy1_rev_4w"] = eps1.reindex(tk).where(lambda s: s.abs() <= 1.0)     # tiny base or bad data
    m["eps_fy2_rev_4w"] = eps2.reindex(tk).where(lambda s: s.abs() <= 1.0)
    m["rev_fy1_rev_4w"] = rv1.reindex(tk)
    m["watchlist_group"] = [groups.get(x, OTHER) for x in tk]

    # group rows: every stock with consensus that week, price return over the month to t
    start = t - pd.DateOffset(months=1)
    m["ret_1m"] = (px / calc.asof_row(pff, start) - 1).reindex(tk)
    nifty_1m = calc.asof_row(nifty, t).iloc[0] / calc.asof_row(nifty, start).iloc[0] - 1 if nifty is not None else np.nan
    grp = []
    for name, g in m.groupby("watchlist_group"):
        e = g["eps4"].dropna()
        r = g["ret_1m"].mean()
        grp.append({"watchlist_group": name, "stock_count": len(g), "avg_eps_rev_4w": e.mean() if len(e) else np.nan,
                    "up_count": int((e > 0.005).sum()), "down_count": int((e < -0.005).sum()),
                    "avg_return_1m": r, "rel_return_1m_vs_nifty": r - nifty_1m})

    # scoreboard row: next 4-week returns vs the equal-weight average of the week's scored stocks
    board = {"ic_spearman": np.nan, "long_fwd_ret": np.nan, "short_fwd_ret": np.nan, "spread_fwd_ret": np.nan,
             "hit_rate": np.nan}
    e = m.loc[m["eligible"], "eps4"].dropna()
    board["breadth_up"] = (e > 0.005).mean() if len(e) else np.nan
    board["breadth_down"] = (e < -0.005).mean() if len(e) else np.nan
    end = t + pd.Timedelta(weeks=C.FORWARD_WEEKS)
    if end <= pff.index.max():
        fwd = (calc.asof_row(pff, end) / calc.asof_row(pff, t) - 1).reindex(tk)
        d = pd.DataFrame({"score": m["score"], "fwd": fwd}).dropna()
        d = d[d["fwd"].abs() < 1.0]
        if len(d) >= 30:
            xs = d["fwd"] - d["fwd"].mean()
            lo, sh = xs.reindex(m.index[m["is_long"] == 1]).dropna(), xs.reindex(m.index[m["is_short"] == 1]).dropna()
            board.update(ic_spearman=spearmanr(d["score"], d["fwd"]).correlation,
                         long_fwd_ret=lo.mean() if len(lo) else np.nan, short_fwd_ret=sh.mean() if len(sh) else np.nan)
            board["spread_fwd_ret"] = board["long_fwd_ret"] - board["short_fwd_ret"]
            picks = len(lo) + len(sh)
            board["hit_rate"] = ((lo > 0).sum() + (sh < 0).sum()) / picks if picks else np.nan
    return m, grp, board


SCORE_COLS = ["analyst_count", "watchlist_group", "tp_rev_1w", "tp_rev_4w", "eps_fy1_rev_4w", "eps_fy2_rev_4w",
              "rev_fy1_rev_4w", "persistence_13w", "mom_4w", "mom_12_1", "score", "score_rank", "is_long", "is_short",
              "is_divergence", "is_emerging", "price_at_score"] + [c for c, _ in NEW_SCORE_COLS]
BROKER_COLS = ["reco", "target", "report_date", "quarter"]
GROUP_COLS = ["stock_count", "avg_eps_rev_4w", "up_count", "down_count", "avg_return_1m", "rel_return_1m_vs_nifty"]
BOARD_COLS = ["ic_spearman", "long_fwd_ret", "short_fwd_ret", "spread_fwd_ret", "hit_rate", "breadth_up", "breadth_down"]


def _clean(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if np.isfinite(v) else None
    return v


def _insert(conn, table, keys, cols, rows):
    sql = text(f"INSERT INTO {table} ({', '.join(keys + cols)}) VALUES ({', '.join(':' + c for c in keys + cols)})")
    rows = [{k: _clean(v) for k, v in r.items()} for r in rows]
    for i in range(0, len(rows), 2000):
        conn.execute(sql, rows[i:i + 2000])


def add_missing_columns(engine):
    """ALTER an existing consensus_weekly_score for the columns added later (checked in information_schema)."""
    with engine.begin() as conn:
        have = {r[0].lower() for r in conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = 'consensus_weekly_score'"))}
        for col, definition in NEW_SCORE_COLS:
            if col not in have:
                conn.execute(text(f"ALTER TABLE consensus_weekly_score ADD COLUMN {col} {definition}"))


def broker_rows(engine, log):
    """Latest verified (yellow) AnalystsReco report per ticker and broker, at most BROKER_VIEW_MAX_AGE_DAYS old.
    None if the Excel file is missing or unreadable (the table is then left as it is)."""
    today = pd.Timestamp.now().normalize()
    try:
        reco, _ = inputs.load_analyst_reco(inputs.temp_copy(C.ANALYST_TRACKER_XLSX), C.ANALYST_RECO_SHEET, today)
        latest = calc.broker_views(reco, today, pd.Series(dtype=float), C.RECO_SCORE, C.BROKER_VIEW_MAX_AGE_DAYS,
                                   today - pd.Timedelta(days=7))["latest"].reset_index()
    except Exception as exc:
        log(f"WARNING consensus_broker_view not updated: could not read {C.ANALYST_TRACKER_XLSX} "
            f"({type(exc).__name__}: {str(exc)[:120]})")
        return None
    latest["Broker"] = latest["Broker"].astype(str).str.strip().str.upper()
    latest = latest[latest["Broker"].isin(C.BROKERS)].copy()
    mapping = consensus_db.ticker_map(engine, latest["Ticker"].dropna().unique())
    latest["ticker"] = latest["Ticker"].map(mapping)
    latest = latest.dropna(subset=["ticker"]).sort_values("Date").drop_duplicates(["ticker", "Broker"], keep="last")
    return [{"ticker": r.ticker, "broker": r.Broker, "reco": None if pd.isna(r.Reco) else str(r.Reco).strip(),
             "target": r.Target, "report_date": r.Date.date() if pd.notna(r.Date) else None,
             "quarter": None if pd.isna(r.Quarter) else str(r.Quarter).strip()} for r in latest.itertuples()]


def load(engine, full=False, consensus_dir=None, watchlists_file=None, log=print):
    """Write the three tables. full=True rewrites every week; otherwise the last FORWARD_WEEKS + 1 weeks."""
    core = model.load_core(engine, consensus_dir=consensus_dir)
    cons, scores = core["cons"], core["scores"]
    weeks = [t for t in cons.dates if t in scores]
    if not full:
        weeks = weeks[-(C.FORWARD_WEEKS + 1):]
    wl = watchlists_file or C.WATCHLISTS_FILE
    groups = watchlist_groups(wl)
    if not groups:
        log(f"dashboard tables: {wl} not found or has no 101+ sections; every stock goes to '{OTHER}'")
    universe = core["universe"]
    fno = db.load_stock_meta(engine, universe)["is_fno"].fillna(0).astype(int) == 1
    nifty = core["idx"][[C.MAIN_INDEX]].ffill() if C.MAIN_INDEX in core["idx"].columns else None
    mapping = consensus_db.ticker_map(engine, list(universe))

    score_rows, group_rows, board_rows = [], [], []
    for t in weeks:
        m, grp, board = week_rows(core, t, fno, groups, nifty)
        wd = t.date()
        m = m.rename(columns={"count": "analyst_count", "persist": "persistence_13w", "mom": "mom_4w",
                              "mom12": "mom_12_1", "price": "price_at_score", "eps4": "eps_rev_4w",
                              "eps13": "eps_rev_13w", "pct": "score_pct", "count_chg13": "count_chg_13w"})
        m["ticker"] = [mapping.get(x, x) for x in m.index]
        m = m.drop_duplicates("ticker")
        for col in ("analyst_count", "persistence_13w", "score_rank", "count_chg_13w", "consensus_age_weeks"):
            m[col] = m[col].astype("Int64").astype(object).where(m[col].notna(), None)
        score_rows += [{"week_date": wd, "ticker": r["ticker"], **{c: r[c] for c in SCORE_COLS}}
                       for r in m.to_dict("records")]
        group_rows += [{"week_date": wd, **g} for g in grp]
        board_rows.append({"week_date": wd, **board})

    brokers = broker_rows(engine, log)
    with engine.begin() as conn:                    # table definitions (MySQL commits DDL on its own)
        for ddl in DDL:
            conn.execute(text(ddl))
    add_missing_columns(engine)
    with engine.begin() as conn:                    # one transaction: the website never sees a half load
        if brokers is not None:
            conn.execute(text("DELETE FROM consensus_broker_view"))
            _insert(conn, "consensus_broker_view", ["ticker", "broker"], BROKER_COLS, brokers)
        for table in TABLES:
            if full:
                conn.execute(text(f"DELETE FROM {table}"))
            else:
                conn.execute(text(f"DELETE FROM {table} WHERE week_date IN :w").bindparams(
                    bindparam("w", expanding=True)), {"w": [t.date() for t in weeks]})
        _insert(conn, "consensus_weekly_score", ["week_date", "ticker"], SCORE_COLS, score_rows)
        _insert(conn, "consensus_weekly_group", ["week_date", "watchlist_group"], GROUP_COLS, group_rows)
        _insert(conn, "consensus_scoreboard", ["week_date"], BOARD_COLS, board_rows)
    log(f"dashboard tables: {len(weeks)} week(s) ({weeks[0]:%Y-%m-%d} to {weeks[-1]:%Y-%m-%d}): "
        f"{len(score_rows)} stock rows, {len(group_rows)} group rows, {len(board_rows)} scoreboard rows"
        + (f", {len(brokers)} broker views" if brokers is not None else ""))
    return len(score_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="rewrite every week, not only the last few")
    args = ap.parse_args()
    load(db.get_engine(C.DB_ENV_FILE), full=args.full)
