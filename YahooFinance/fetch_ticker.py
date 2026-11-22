"""Fetch one or a few tickers from Yahoo and put them into the latest weekly file, without a full download.

Replaces (or adds) those tickers' rows in the latest WeeklyConsensus/yyyymmdd_YahooConsensus.csv and in the same week
of YahooConsensusHistory.csv, and adds any symbol missing from YahooFinanceTickers.csv so next week's download includes
it. Use it after ticker_change.py, or when a ticker failed in the weekly download. Rebuild the dashboard afterwards.

Run with the Anaconda Python:
    python fetch_ticker.py ICICILIFE
    python fetch_ticker.py ICICILIFE TCS --date 20261120     put them into a specific week instead of the latest
    python fetch_ticker.py                                   asks which tickers
"""
import argparse
import sys
from datetime import datetime

import pandas as pd

from weekly_lib import config as C
from weekly_lib import fetch, history, tickers


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="*", help="ticker(s) to fetch")
    ap.add_argument("--date", help="week to update (yyyymmdd); default = latest weekly file")
    args = ap.parse_args(argv)
    symbols = [s.strip().upper() for s in (args.symbols or input("Ticker(s) to fetch: ").replace(",", " ").split()) if s.strip()]
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        print("No ticker given.")
        return 2

    stamp, week_file = history.latest_weekly_file(C.CONSENSUS_DIR, args.date)
    if week_file is None:
        print(f"No weekly file found{' for ' + args.date if args.date else ''} in {C.CONSENSUS_DIR}")
        return 1
    date_str = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"
    print(f"Fetching {', '.join(symbols)} from Yahoo for the week of {date_str} ...")

    import yfinance as yf
    rows, empty = [], []
    for sym in symbols:
        try:
            got = fetch.fetch_one(yf, sym, date_str)
        except Exception as exc:
            print(f"  {sym}: failed ({type(exc).__name__}: {str(exc)[:80]}) - try again in a few minutes")
            continue
        if not any(r["#Count"] or r["TP(Avg)"] or r["Current Price"] for r in got):
            print(f"  {sym}: Yahoo has no data at all for this symbol (spelling?) - nothing saved for it")
            continue
        if not any(r["#Count"] or r["TP(Avg)"] for r in got):
            empty.append(sym)
        rows += got
        for r in got:
            print(f"  {sym:<12} {r['FIN Yr']:<5} TP {r['TP(Avg)']:>8}  EPS {r['EPS(Avg)']:>9}  analysts {r['#Count']:>3}"
                  f"  price {r['Current Price']}")
    if not rows:
        print("Nothing fetched; no file changed.")
        return 1
    if empty:
        print(f"  Yahoo returned no estimates for {', '.join(empty)} (check the symbol); their rows are saved as empty.")

    new_rows = pd.DataFrame(rows, columns=fetch.COLUMNS)
    got = sorted(set(new_rows["YAHOO Ticker"]))
    replaced = history.upsert_rows(week_file, new_rows, got)
    print(f"{week_file.name}: {replaced} old row(s) replaced, {len(new_rows)} row(s) written")
    if C.HISTORY_FILE.exists():
        history.upsert_rows(C.HISTORY_FILE, new_rows, got, date=date_str, sort_by=("Date", "YAHOO Ticker"))
        print(f"{C.HISTORY_FILE.name}: week {date_str} updated")

    listed = {t.upper() for t in tickers.read_ticker_list(C.TICKERS_FILE)}
    missing = [s for s in got if s not in listed]
    if missing:
        tickers.write_ticker_list(C.TICKERS_FILE, sorted(listed | set(missing)), C.OLD_DIR)
        tickers.log_changes(C.LOG_DIR / "TickerChanges.csv",
                            [{"action": "ADDED", "ticker": s, "detail": f"by fetch_ticker.py {datetime.now():%d-%b-%Y}"}
                             for s in missing])
        print(f"Added to YahooFinanceTickers.csv: {', '.join(missing)}")
    print("Done. Next: 3_Build_WeeklyDashboard.bat for the dashboard, 4_Load_Consensus_To_DB.bat for the database.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
