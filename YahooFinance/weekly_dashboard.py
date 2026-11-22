"""Weekly step 3 of 4 - stock dashboard.

Builds Dashboard/WeeklyDashboard.html (always the latest) and Dashboard/yyyymmdd_WeeklyDashboard.html (a copy kept for
that consensus week) from the files step 1 downloaded. Nothing is fetched from the internet, so it can be re-run as
often as you like, for example after editing watchlists.properties; a re-run in the same week replaces that week's copy.

Run with the Anaconda Python:
    python weekly_dashboard.py            build both files
    python weekly_dashboard.py --open     ...and open the latest one in the browser
"""
import argparse
import sys
import webbrowser

from weekly_lib import config as C
from weekly_lib import history, model, render


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", action="store_true", help="open the latest dashboard in the default browser")
    args = ap.parse_args(argv)

    moved = history.tidy_stray_weekly_files(C.YC_DIR, C.CONSENSUS_DIR)
    added = history.merge_history(C.CONSENSUS_DIR, C.HISTORY_FILE)
    print("Building the dashboard ...", flush=True)
    M = model.build_model()
    if moved:
        M["checks"].append(("info", f"Moved weekly files from the top folder into WeeklyConsensus: {', '.join(moved)}"))
    if added:
        M["checks"].append(("ok", f"YahooConsensusHistory.csv: added {', '.join(added)}"))

    page = render.render(M)
    stamp = M["as_of"].strftime("%Y%m%d")
    C.DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    archive = C.DASHBOARD_DIR / C.ARCHIVE_NAME.format(date=stamp)
    for path in (C.LATEST_DASHBOARD, archive):
        path.write_text(page, encoding="utf-8")
    print(f"Dashboard written:\n  {C.LATEST_DASHBOARD}\n  {archive}")
    if args.open:
        webbrowser.open(C.LATEST_DASHBOARD.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
