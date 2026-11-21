"""Change a ticker symbol in every consensus CSV, e.g. after NSE renames ICICIPRULI to ICICILIFE.

Updates YahooFinanceTickers.csv, every WeeklyConsensus/yyyymmdd_YahooConsensus.csv and YahooConsensusHistory.csv.
Each changed file is copied to Old/TickerChange_<OLD>_to_<NEW>_<time>/ first. In a week that already has rows for the
new symbol, the old symbol's rows for that week are dropped instead of renamed. Only the symbol text changes.

Run with the Anaconda Python:
    python ticker_change.py ICICIPRULI ICICILIFE             show what will change, ask, then apply
    python ticker_change.py ICICIPRULI ICICILIFE --yes       apply without asking
    python ticker_change.py ICICIPRULI ICICILIFE --dry-run   only show what would change
Afterwards fetch the new symbol into the latest week:  python fetch_ticker.py ICICILIFE
"""
import argparse
import re
import sys
from datetime import datetime

from weekly_lib import config as C
from weekly_lib import tickers


def watchlist_sections(symbol):
    """Sections of watchlists.properties that still list the symbol (that file is edited by hand)."""
    hits, section = [], None
    if not C.WATCHLISTS_FILE.exists():
        return hits
    for line in C.WATCHLISTS_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\s*\[([^\]]+)\]", line)
        if m:
            section = m.group(1)
        elif line.strip().lower().startswith("tickers") and symbol in [t.strip().upper() for t in line.split("=", 1)[-1].split(",")]:
            hits.append(section)
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", help="symbol to replace")
    ap.add_argument("new", help="new symbol")
    ap.add_argument("--yes", action="store_true", help="apply without asking")
    ap.add_argument("--dry-run", action="store_true", help="only show what would change")
    args = ap.parse_args(argv)
    old, new = args.old.strip().upper(), args.new.strip().upper()
    if not old or not new or old == new:
        print("Give two different symbols, e.g.  ticker_change.py ICICIPRULI ICICILIFE")
        return 2

    plan = tickers.plan_rename(old, new, C.TICKERS_FILE, C.CONSENSUS_DIR, C.HISTORY_FILE)
    sections = watchlist_sections(old)
    if not plan:
        print(f"{old} does not appear in YahooFinanceTickers.csv or any consensus CSV. Nothing to change.")
    else:
        rows = sum(p[2] for p in plan)
        print(f"Change {old} -> {new} in {len(plan)} file(s), {rows} row(s):")
        for path, _, n, clash in plan:
            extra = f"  ({clash} dropped: {new} already present for that week)" if clash else ""
            print(f"  {path.name:<36} {n:>4} row(s){extra}")
    if sections:
        print(f"Note: watchlists.properties still lists {old} in {', '.join(sections)} - edit that file by hand.")
    if not plan or args.dry_run:
        return 0
    if not args.yes and input("Apply these changes? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Nothing changed.")
        return 1

    backup = C.OLD_DIR / f"TickerChange_{old}_to_{new}_{datetime.now():%Y%m%d_%H%M%S}"
    tickers.rename_everywhere(old, new, plan, backup)
    tickers.log_changes(C.LOG_DIR / "TickerChanges.csv", [{
        "action": "MANUAL_RENAME", "ticker": new,
        "detail": f"{old} -> {new}: {sum(p[2] for p in plan)} rows in {len(plan)} files (backup in Old/{backup.name})"}])
    print(f"Done. Original files copied to {backup}")
    print(f"Next: fetch {new} into the latest week with fetch_ticker.py {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
