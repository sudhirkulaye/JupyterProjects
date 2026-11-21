"""YahooFinanceTickers.csv upkeep and manual ticker changes.

- New MOSL / AMBIT coverage is added automatically from CompanyResearchReports file names such as
  FY27Q2_TICKER_MOSL.pdf or FY27Q2_TICKER_AMBIT_1.pdf (BSE-only listings use their BSE code: 544937 -> NSE).
- Symbol changes are never applied automatically. They are only suggested; rename_everywhere() applies one when you
  run ticker_change.py.
"""
import csv
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

REPORT_RE = re.compile(r"^FY\d{2}Q[1-4](?:_[A-Z]+)?_(?P<ticker>.+?)_(?P<broker>[A-Z]+)(?:_\d+)?\.pdf$", re.IGNORECASE)
TICKER_COLUMN = "Yahoo_Ticker"
CONSENSUS_TICKER_COLUMN = "YAHOO Ticker"


def read_ticker_list(path):
    df = pd.read_csv(path, dtype=str)
    return [t.strip() for t in df[TICKER_COLUMN].dropna() if t.strip()]


def write_ticker_list(path, tickers, backup_dir):
    """Back up the current file to Old/ and write the new list (sorted, one per line)."""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"YahooFinanceTickers_{datetime.now():%Y%m%d_%H%M%S}.csv"
    shutil.copy2(path, backup)
    pd.DataFrame({TICKER_COLUMN: sorted(tickers)}).to_csv(path, index=False)
    return backup


def recent_report_tickers(reports_dir, days, brokers):
    """Tickers in report PDFs from the given brokers created or changed in the last `days` days."""
    cutoff = time.time() - days * 86400
    found = {}
    for pdf in Path(reports_dir).rglob("*.pdf"):
        m = REPORT_RE.match(pdf.name)
        if not m or m.group("broker").upper() not in brokers:
            continue
        try:
            stamp = max(pdf.stat().st_ctime, pdf.stat().st_mtime)
        except OSError:
            continue
        if stamp < cutoff:
            continue
        ticker = m.group("ticker").strip().upper().replace(" ", "")
        found.setdefault(ticker, []).append((pdf.name, datetime.fromtimestamp(stamp)))
    return found


def rename_suggestions(tickers, renames):
    """[(listed symbol, current symbol)] for symbols stock_universe shows as renamed (a stray -EQ suffix ignored)."""
    out = []
    for t in tickers:
        base = re.sub(r"-(EQ|BE)$", "", t.upper())
        new = renames.get(base)
        if new and new != t.upper():
            out.append((t, new))
    return out


def update_ticker_list(path, known, bse_symbols, reports_dir, days, brokers, backup_dir, log_file, dry_run=False):
    """Tidy the list (case, duplicates) and add new MOSL / AMBIT coverage. Returns (changes, skipped, list)."""
    changes, result, seen = [], [], set()
    for raw in read_ticker_list(path):
        up = raw.upper()
        if up != raw:
            changes.append({"action": "CASE_FIXED", "ticker": up, "detail": f"{raw} -> {up}"})
        if up in seen:
            changes.append({"action": "DUPLICATE_REMOVED", "ticker": up, "detail": f"{raw} listed more than once"})
            continue
        seen.add(up)
        result.append(up)
    skipped = []
    for code, files in sorted(recent_report_tickers(reports_dir, days, brokers).items()):
        symbol = bse_symbols.get(code, code) if code.isdigit() else code
        if symbol in seen:
            continue
        if symbol.isdigit():
            skipped.append(f"{code} (BSE code without a symbol in stock_universe.short_name; {files[0][0]})")
            continue
        if not code.isdigit() and symbol not in known:
            skipped.append(f"{code} (not in stock_universe; {files[0][0]})")
            continue
        seen.add(symbol)
        result.append(symbol)
        via = f" via BSE code {code}" if code != symbol else ""
        changes.append({"action": "ADDED", "ticker": symbol,
                        "detail": f"new coverage{via}: {files[0][0]} ({files[0][1]:%d-%b-%Y})"})
    if changes and not dry_run:
        write_ticker_list(path, result, backup_dir)
        log_changes(log_file, changes)
    return changes, skipped, result


def log_changes(log_file, changes):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_file.exists()
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["RunTime", "Action", "Ticker", "Detail"])
        stamp = f"{datetime.now():%Y-%m-%d %H:%M}"
        for c in changes:
            w.writerow([stamp, c["action"], c["ticker"], c["detail"]])


# ---------------------------------------------------------------- manual ticker change
def _csv_targets(tickers_file, consensus_dir, history_file):
    """(path, ticker column) for every CSV that carries ticker symbols."""
    targets = [(Path(tickers_file), TICKER_COLUMN)]
    targets += [(p, CONSENSUS_TICKER_COLUMN) for p in sorted(Path(consensus_dir).glob("*_YahooConsensus.csv"))]
    if Path(history_file).exists():
        targets.append((Path(history_file), CONSENSUS_TICKER_COLUMN))
    return targets


def plan_rename(old, new, tickers_file, consensus_dir, history_file):
    """What a rename would touch: [(path, column, rows with old, old rows dropped because new already has that week)]."""
    old, new = old.upper(), new.upper()
    plan = []
    for path, col in _csv_targets(tickers_file, consensus_dir, history_file):
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if col not in df.columns:
            continue
        is_old = df[col].str.strip().str.upper() == old
        if not is_old.any():
            continue
        clash = 0
        if col == CONSENSUS_TICKER_COLUMN:      # a week that already has the new symbol keeps its own rows
            new_dates = set(df.loc[df[col].str.strip().str.upper() == new, "Date"])
            clash = int((is_old & df["Date"].isin(new_dates)).sum())
        elif (df[col].str.strip().str.upper() == new).any():
            clash = int(is_old.sum())
        plan.append((path, col, int(is_old.sum()), clash))
    return plan


def rename_everywhere(old, new, plan, backup_root):
    """Apply a planned rename. Files are read and written as text, so nothing but the symbol changes.
    Every file is copied to backup_root first."""
    old, new = old.upper(), new.upper()
    backup_root = Path(backup_root)
    backup_root.mkdir(parents=True, exist_ok=True)
    for path, col, _, _ in plan:
        shutil.copy2(path, backup_root / path.name)
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        sym = df[col].str.strip().str.upper()
        is_old = sym == old
        if col == CONSENSUS_TICKER_COLUMN:
            new_dates = set(df.loc[sym == new, "Date"])
            drop = is_old & df["Date"].isin(new_dates)
        else:
            drop = is_old & (sym == new).any()
        df.loc[is_old & ~drop, col] = new
        df = df[~drop]
        if col == TICKER_COLUMN:
            df = df.sort_values(col)
        df.to_csv(path, index=False)
