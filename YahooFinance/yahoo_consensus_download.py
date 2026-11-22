"""Weekly step 1 of 4 - Yahoo consensus download (CSV files only; the database load is step 4).

  1. Adds new MOSL / AMBIT coverage to YahooFinanceTickers.csv (report PDFs added to CompanyResearchReports in the
     last 10 days). Symbol changes are only suggested; apply them with ticker_change.py.
  2. Downloads this week's consensus to WeeklyConsensus/yyyymmdd_YahooConsensus.csv. Skipped when that file already
     exists, because the download takes a long time and hits Yahoo thousands of times.
  3. Appends the new week to WeeklyConsensus/YahooConsensusHistory.csv.

Run with the Anaconda Python:
    python yahoo_consensus_download.py                  normal weekly run
    python yahoo_consensus_download.py --force          download again even if this week's file exists
    python yahoo_consensus_download.py --tickers-only   only update the ticker list
    python yahoo_consensus_download.py --dry-run        show ticker-list changes without saving or downloading
    python yahoo_consensus_download.py --test TCS,INFY --out C:\\temp   try a few tickers into another folder
"""
import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from weekly_lib import config as C
from weekly_lib import db, fetch, history, tickers


class Tee:
    """Print to the console and to a log file."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.file.write(f"{datetime.now():%H:%M:%S} {line}\n")
        self.file.flush()


def effective_date(text=None):
    if text:
        return datetime.strptime(text, "%Y%m%d").date()
    d = date.today()
    return d - timedelta(days={5: 1, 6: 2}.get(d.weekday(), 0))     # Saturday / Sunday -> Friday


def update_tickers(log, dry_run):
    engine = db.get_engine(C.DB_ENV_FILE)
    known, bse_symbols = db.load_known_tickers(engine), db.load_bse_symbols(engine)
    changes, skipped, result = tickers.update_ticker_list(
        C.TICKERS_FILE, known, bse_symbols, C.REPORTS_DIR, C.NEW_COVERAGE_DAYS, C.NEW_COVERAGE_BROKERS,
        C.OLD_DIR, C.LOG_DIR / "TickerChanges.csv", dry_run=dry_run)
    verb = "Would change" if dry_run else "Changed"
    if changes:
        log(f"{verb} YahooFinanceTickers.csv ({len(result)} tickers):")
        for c in changes:
            log(f"  {c['action']:<17} {c['ticker']:<12} {c['detail']}")
    else:
        log(f"YahooFinanceTickers.csv is up to date ({len(result)} tickers).")
    if skipped:
        log(f"  Report files not added (check the name): {', '.join(skipped[:10])}")
    hints = tickers.rename_suggestions(result, db.load_ticker_renames(engine))
    if hints:
        log("  stock_universe shows these symbols as renamed (nothing changed; use 2_Ticker_Change_or_Refetch.bat): "
            + ", ".join(f"{old} -> {new}" for old, new in hints))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download even if this week's file exists")
    ap.add_argument("--date", help="override the snapshot date (yyyymmdd); default = today, weekends -> Friday")
    ap.add_argument("--tickers-only", action="store_true", help="only update YahooFinanceTickers.csv")
    ap.add_argument("--skip-ticker-update", action="store_true", help="do not touch YahooFinanceTickers.csv")
    ap.add_argument("--dry-run", action="store_true", help="show ticker-list changes, change nothing")
    ap.add_argument("--test", help="comma-separated tickers to fetch as a test (needs --out)")
    ap.add_argument("--out", help="folder for --test output")
    args = ap.parse_args(argv)

    snap = effective_date(args.date)
    stamp = snap.strftime("%Y%m%d")
    log = Tee(C.LOG_DIR / f"download_{stamp}.log")
    log(f"=== Yahoo consensus download for {snap:%d-%b-%Y} (run {datetime.now():%d-%b-%Y %H:%M}) ===")

    if args.test:                                             # quick check of the Yahoo connection, nothing else
        out_dir = Path(args.out or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        names = [t.strip().upper() for t in args.test.split(",") if t.strip()]
        df, failed = fetch.download(names, snap.isoformat(), out_dir / f"{stamp}_test.partial.csv",
                                    C.DOWNLOAD_PAUSE_SECONDS, C.DOWNLOAD_RETRIES, log)
        target = out_dir / f"{stamp}_YahooConsensus_test.csv"
        df.to_csv(target, index=False)
        (out_dir / f"{stamp}_test.partial.csv").unlink(missing_ok=True)
        log(f"Test file written: {target} ({len(df)} rows; failed: {failed or 'none'})")
        return 0

    if not args.dry_run:
        moved = history.tidy_stray_weekly_files(C.YC_DIR, C.CONSENSUS_DIR)
        if moved:
            log(f"Moved {len(moved)} weekly file(s) from the top folder into WeeklyConsensus")

    ticker_list = None
    if not args.skip_ticker_update:
        ticker_list = update_tickers(log, args.dry_run)
    if args.tickers_only or args.dry_run:
        return 0

    target = C.CONSENSUS_DIR / f"{stamp}_YahooConsensus.csv"
    if target.exists() and not args.force:
        log(f"{target.name} already exists - download skipped (use --force to download again).")
    else:
        ticker_list = ticker_list or tickers.read_ticker_list(C.TICKERS_FILE)
        ticker_list = list(dict.fromkeys(t.upper() for t in ticker_list))
        log(f"Downloading {len(ticker_list)} tickers from Yahoo Finance ...")
        partial = C.CONSENSUS_DIR / f"{stamp}_YahooConsensus.partial.csv"
        start = time.time()
        df, failed = fetch.download(ticker_list, snap.isoformat(), partial, C.DOWNLOAD_PAUSE_SECONDS,
                                    C.DOWNLOAD_RETRIES, log)
        df.to_csv(target, index=False)
        partial.unlink(missing_ok=True)
        log(f"Saved {target.name}: {df['YAHOO Ticker'].nunique()} tickers in {(time.time() - start) / 60:.0f} min")
        if failed:
            log(f"Could not fetch after retries: {', '.join(failed)}")

    added = history.merge_history(C.CONSENSUS_DIR, C.HISTORY_FILE)
    log(f"YahooConsensusHistory.csv: {'added ' + ', '.join(added) if added else 'already up to date'}")
    log("Done. Next: 2_Ticker_Change_or_Refetch.bat only if a ticker needs fixing, then 3_Build_WeeklyDashboard.bat "
        "and 4_Load_Consensus_To_DB.bat (database)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
