"""Load YahooConsensusHistory.csv into MySQL table yahoo_consensus (read by sudhirkulaye.com's stock page).

Yahoo symbols are mapped to stock_universe tickers:
  1. the symbol itself, when it is a stock_universe ticker
  2. a renamed symbol (stock_universe.ticker_old -> ticker), e.g. ICICIPRULI -> ICICILIFE
  3. a BSE-only listing tracked on Yahoo by its symbol, which stock_universe keeps in short_name for the BSE code
Unmapped symbols are stored under the Yahoo symbol so nothing is lost.

A separate step from the download: run 4_Load_Consensus_To_DB.bat (YahooConsensus folder) on each laptop
whose database should be updated; it reloads every week. Or:  python -m weekly_lib.consensus_db [--full]
"""
import argparse
import math

import pandas as pd
from sqlalchemy import text

from weekly_lib import db

DDL = """
CREATE TABLE IF NOT EXISTS yahoo_consensus (
  ticker varchar(30) COLLATE utf8_unicode_ci NOT NULL COMMENT 'PK stock_universe ticker (mapped from yahoo_ticker)',
  date date NOT NULL COMMENT 'PK weekly snapshot date',
  fin_year varchar(6) COLLATE utf8_unicode_ci NOT NULL COMMENT 'PK estimate fiscal year e.g. FY27',
  yahoo_ticker varchar(30) COLLATE utf8_unicode_ci NOT NULL COMMENT 'symbol as downloaded from Yahoo Finance',
  tp_avg decimal(20,3) DEFAULT NULL COMMENT 'consensus target price average',
  tp_min decimal(20,3) DEFAULT NULL COMMENT 'consensus target price lowest',
  tp_max decimal(20,3) DEFAULT NULL COMMENT 'consensus target price highest',
  rev_avg decimal(20,3) DEFAULT NULL COMMENT 'consensus revenue average (Cr)',
  rev_min decimal(20,3) DEFAULT NULL COMMENT 'consensus revenue lowest (Cr)',
  rev_max decimal(20,3) DEFAULT NULL COMMENT 'consensus revenue highest (Cr)',
  eps_avg decimal(20,3) DEFAULT NULL COMMENT 'consensus EPS average',
  eps_min decimal(20,3) DEFAULT NULL COMMENT 'consensus EPS lowest',
  eps_max decimal(20,3) DEFAULT NULL COMMENT 'consensus EPS highest',
  analyst_count int(5) DEFAULT NULL COMMENT 'number of analysts',
  year_ago_rev decimal(20,3) DEFAULT NULL COMMENT 'revenue of the year before fin_year',
  year_ago_eps decimal(20,3) DEFAULT NULL COMMENT 'EPS of the year before fin_year',
  current_price decimal(20,3) DEFAULT NULL COMMENT 'price on the snapshot date',
  PRIMARY KEY (ticker, date, fin_year),
  KEY index_yahoo_consensus_date (date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci
  COMMENT='Weekly Yahoo Finance analyst consensus (loaded by YahooFinance/weekly_lib/consensus_db.py)'
"""

COLUMNS = {  # csv column -> table column
    "TP(Avg)": "tp_avg", "TP(Min)": "tp_min", "TP(Max)": "tp_max",
    "Rev.(Avg)": "rev_avg", "Rev.(Min)": "rev_min", "Rev.(Max)": "rev_max",
    "EPS(Avg)": "eps_avg", "EPS(Min)": "eps_min", "EPS(Max)": "eps_max",
    "#Count": "analyst_count", "Year Ago Rev.": "year_ago_rev", "Year Ago EPS": "year_ago_eps",
    "Current Price": "current_price",
}
VALUE_COLS = list(COLUMNS.values())
INSERT = text(
    "INSERT INTO yahoo_consensus (ticker, date, fin_year, yahoo_ticker, " + ", ".join(VALUE_COLS) + ") "
    "VALUES (:ticker, :date, :fin_year, :yahoo_ticker, " + ", ".join(":" + c for c in VALUE_COLS) + ") "
    "ON DUPLICATE KEY UPDATE yahoo_ticker = VALUES(yahoo_ticker), "
    + ", ".join(f"{c} = VALUES({c})" for c in VALUE_COLS))


def ticker_map(engine, symbols):
    """Yahoo symbol -> stock_universe ticker."""
    known = db.load_known_tickers(engine)
    renames = db.load_ticker_renames(engine)
    bse_by_symbol = {sym.upper(): code for code, sym in db.load_bse_symbols(engine).items()}
    out = {}
    for s in symbols:
        u = s.strip().upper()
        if u in known:
            out[s] = u
        elif renames.get(u) in known:
            out[s] = renames[u]
        elif u in bse_by_symbol:
            out[s] = bse_by_symbol[u]
        else:
            out[s] = u
    return out


def _clean(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return v


def load(engine, history_file, full=False, log=print):
    """Upsert history rows into yahoo_consensus. Only weeks newer than the table's latest date unless full=True.
    Returns the number of rows written."""
    with engine.begin() as conn:
        conn.execute(text(DDL))
        last = conn.execute(text("SELECT MAX(date) FROM yahoo_consensus")).scalar()

    df = pd.read_csv(history_file, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    if last is not None and not full:
        df = df[df["Date"] > last]
    if df.empty:
        log(f"yahoo_consensus: already up to date (latest {last})")
        return 0

    mapping = ticker_map(engine, df["YAHOO Ticker"].dropna().unique())
    unmapped = sorted(s for s, t in mapping.items() if t == s.strip().upper() and t not in db.load_known_tickers(engine))
    df = df.rename(columns=COLUMNS)
    df["yahoo_ticker"] = df["YAHOO Ticker"].str.strip().str.upper()
    df["ticker"] = df["YAHOO Ticker"].map(mapping)
    df["fin_year"] = df["FIN Yr"].astype(str).str.strip()
    df["date"] = df["Date"]
    df = df.drop_duplicates(["ticker", "date", "fin_year"], keep="last")

    rows = [{k: _clean(v) for k, v in r.items()}
            for r in df[["ticker", "date", "fin_year", "yahoo_ticker"] + VALUE_COLS].to_dict("records")]
    with engine.begin() as conn:
        for i in range(0, len(rows), 2000):
            conn.execute(INSERT, rows[i:i + 2000])
    weeks = df["date"].nunique()
    log(f"yahoo_consensus: wrote {len(rows)} rows for {weeks} week(s) ({df['date'].min()} to {df['date'].max()})"
        + (f"; {len(unmapped)} symbols not in stock_universe: {', '.join(unmapped[:15])}"
           + (" ..." if len(unmapped) > 15 else "") if unmapped else ""))
    return len(rows)


if __name__ == "__main__":
    from weekly_lib import config as C
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="re-load every week, not only new ones")
    args = ap.parse_args()
    load(db.get_engine(C.DB_ENV_FILE), C.HISTORY_FILE, full=args.full)
