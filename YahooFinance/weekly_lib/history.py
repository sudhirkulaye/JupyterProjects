"""YahooConsensusHistory.csv upkeep (the merged file the Cowork scorer uses) and tidying of stray weekly files."""
import re
import shutil
from pathlib import Path

import pandas as pd

WEEKLY_RE = re.compile(r"^(\d{8})_YahooConsensus\.csv$", re.IGNORECASE)


def tidy_stray_weekly_files(yc_dir, consensus_dir):
    """Move weekly CSVs written to the top folder (e.g. by the old notebook) into WeeklyConsensus/."""
    moved = []
    consensus_dir.mkdir(parents=True, exist_ok=True)
    for path in Path(yc_dir).glob("*_YahooConsensus.csv"):
        if WEEKLY_RE.match(path.name):
            target = consensus_dir / path.name
            if target.exists():
                continue
            shutil.move(str(path), str(target))
            moved.append(path.name)
    return moved


def latest_weekly_file(consensus_dir, stamp=None):
    """(yyyymmdd, path) of the newest weekly file, or of the given week."""
    weekly = sorted((m.group(1), p) for p in Path(consensus_dir).iterdir() if (m := WEEKLY_RE.match(p.name)))
    if stamp:
        weekly = [w for w in weekly if w[0] == stamp]
    return weekly[-1] if weekly else (None, None)


def upsert_rows(path, new_rows, symbols, date=None, sort_by=("YAHOO Ticker",)):
    """Replace the rows of `symbols` (only for `date` when given) with new_rows. Existing rows are kept as text,
    so nothing else in the file changes."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    target = df["YAHOO Ticker"].str.strip().str.upper().isin(symbols)
    if date is not None:
        target &= df["Date"] == date
    out = pd.concat([df[~target], new_rows.reindex(columns=df.columns)], ignore_index=True)
    keys = pd.DataFrame({f"k{i}": out[c].astype(str).str.upper() for i, c in enumerate(sort_by)})
    out = out.loc[keys.sort_values(list(keys.columns), kind="stable").index]
    out.to_csv(path, index=False)
    return int(target.sum())


def merge_history(consensus_dir, history_file):
    """Append every weekly file whose date is not yet in the history file. Safe to run any number of times."""
    weekly = sorted((m.group(1), p) for p in Path(consensus_dir).iterdir() if (m := WEEKLY_RE.match(p.name)))
    if not weekly:
        return []
    if history_file.exists():
        hist_cols = list(pd.read_csv(history_file, nrows=0).columns)
        have = set(pd.read_csv(history_file, usecols=["Date"], dtype=str)["Date"].str.replace("-", "").str[:8])
    else:
        hist_cols, have = None, set()
    added = []
    for stamp, path in weekly:
        if stamp in have:
            continue
        df = pd.read_csv(path)
        if hist_cols is None:
            hist_cols = list(df.columns)
            df.to_csv(history_file, index=False)
        else:
            df.reindex(columns=hist_cols).to_csv(history_file, mode="a", header=False, index=False)
        have.add(stamp)
        added.append(path.name)
    return added
