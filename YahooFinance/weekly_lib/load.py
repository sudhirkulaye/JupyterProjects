"""Loaders for the file-based inputs: weekly consensus CSVs, AnalystsReco, AMBIT classification, watchlists."""
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

CONSENSUS_FILE_RE = re.compile(r"^(\d{8})_YahooConsensus\.csv$", re.IGNORECASE)
CONSENSUS_NUMERIC = ["TP(Avg)", "TP(Min)", "TP(Max)", "Rev.(Avg)", "Rev.(Min)", "Rev.(Max)",
                     "EPS(Avg)", "EPS(Min)", "EPS(Max)", "#Count", "Year Ago Rev.", "Year Ago EPS",
                     "Current Price"]


# ---------------------------------------------------------------- consensus CSVs
def list_consensus_files(folder):
    files = []
    for path in Path(folder).iterdir():
        m = CONSENSUS_FILE_RE.match(path.name)
        if m:
            files.append((pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%d")), path))
    return sorted(files)


def load_consensus(folder):
    """All weekly snapshots in one long frame. Zeros (the old 'missing' marker) become NaN."""
    frames, notes = [], []
    for file_date, path in list_consensus_files(folder):
        df = pd.read_csv(path)
        df["Date"] = file_date
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No *_YahooConsensus.csv files in {folder}")
    cons = pd.concat(frames, ignore_index=True)
    raw_tickers = cons["YAHOO Ticker"].astype(str).str.strip()
    cons["Ticker"] = raw_tickers.str.upper()
    case_dups = sorted(set(raw_tickers[raw_tickers != cons["Ticker"]]))
    if case_dups:
        notes.append(f"Ticker list has lower-case duplicates, merged: {', '.join(case_dups)}")
    for col in CONSENSUS_NUMERIC:
        cons[col] = pd.to_numeric(cons[col], errors="coerce").replace(0, np.nan)
    cons["FIN Yr"] = cons["FIN Yr"].astype(str).str.strip().replace({"nan": np.nan})
    before = len(cons)
    cons = cons.drop_duplicates(["Date", "Ticker", "FIN Yr"], keep="first")
    if len(cons) < before:
        notes.append(f"Dropped {before - len(cons)} duplicate consensus rows")
    return cons.sort_values(["Date", "Ticker", "FIN Yr"]).reset_index(drop=True), notes


# ---------------------------------------------------------------- Excel helpers
def temp_copy(path):
    """Copy a workbook to a temp folder so a file that is open in Excel can still be read."""
    target = Path(tempfile.mkdtemp(prefix="wd_")) / Path(path).name
    shutil.copy2(path, target)
    return target


def _expected_report_year(quarter):
    m = re.match(r"FY(\d{2})Q([1-4])", str(quarter))
    if not m:
        return None
    fy, qn = 2000 + int(m.group(1)), int(m.group(2))
    return fy - 1 if qn in (1, 2) else fy        # Q1/Q2 results come out in the previous calendar year


def _fix_report_date(raw, quarter, today):
    """Repair year typos (2025-01-16 for FY26Q3, 2027-11-08 for FY27Q2, '0202-07-31', ...)."""
    expected = _expected_report_year(quarter)
    parsed = raw if isinstance(raw, (pd.Timestamp, datetime)) else pd.to_datetime(raw, errors="coerce")
    parsed = pd.Timestamp(parsed) if pd.notna(parsed) else pd.NaT
    if pd.notna(parsed) and (expected is None or parsed.year == expected) and parsed <= today + pd.Timedelta(days=1):
        return parsed, "ok"
    month_day = None
    if pd.notna(parsed):
        month_day = (parsed.month, parsed.day)
    else:
        m = re.search(r"(\d{1,2})[-./](\d{1,2})\s*(?:00:00:00)?\s*$", str(raw))
        if m:
            month_day = (int(m.group(1)), int(m.group(2)))
    if month_day is None or expected is None:
        return parsed, ("unresolved" if pd.isna(parsed) or parsed > today else "ok")
    try:
        fixed = pd.Timestamp(year=expected, month=month_day[0], day=month_day[1])
    except ValueError:
        return parsed, "unresolved"
    if fixed > today:
        return parsed, "unresolved"   # a repair would land in the future: keep the date as typed
    return fixed, "fixed"


VERIFIED_FILLS = {"FFFFFF00"}   # yellow Ticker cell = record entered, cross-checked and done


def _read_sheet_with_marks(xlsx, sheet):
    """Sheet values plus whether each row's first (Ticker) cell carries the 'verified' yellow fill."""
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        ws = wb[sheet]
        rows_iter = ws.iter_rows()
        header = [c.value for c in next(rows_iter)]
        rows, marks = [], []
        for row in ws.iter_rows(min_row=2, max_col=len(header)):
            values = [c.value for c in row]
            if not values or values[0] is None:
                continue
            try:
                fill = row[0].fill
                rgb = fill.fgColor.rgb if fill is not None and fill.fill_type == "solid" else None
            except AttributeError:
                rgb = None
            rows.append(values + [None] * (len(header) - len(values)))
            marks.append(isinstance(rgb, str) and rgb.upper() in VERIFIED_FILLS)
    finally:
        wb.close()
    return pd.DataFrame(rows, columns=[h if h is not None else f"col{i}" for i, h in enumerate(header)]), marks


def load_analyst_reco(xlsx, sheet, today):
    """Verified (yellow) AnalystsReco rows only; the sheet is work in progress and unmarked rows may be partial."""
    raw, verified = _read_sheet_with_marks(xlsx, sheet)
    stats = {"rows": len(raw), "verified": int(sum(verified))}
    tail_pending = 0
    for flag in reversed(verified):
        if flag:
            break
        tail_pending += 1
    stats["pending"] = stats["rows"] - stats["verified"]
    stats["pending_tail"] = tail_pending
    raw = raw[verified].copy()
    cols = {"Ticker": "Ticker", "Quarter": "Quarter", "Date": "RawDate", "Broker": "Broker", "Reco": "Reco",
            "Target": "Target", "Price": "ReportPrice", "EPS Y0": "EPS_Y0", "EPS Y1": "EPS_Y1", "EPS Y2": "EPS_Y2",
            "Rev Y0": "Rev_Y0", "Rev. Y1": "Rev_Y1", "Rev. Y2": "Rev_Y2", "ROCE% Y1": "ROCE_Y1",
            "Fwd Multiple": "FwdMultiple"}
    df = raw[[c for c in cols if c in raw.columns]].rename(columns=cols)
    df = df[df["Ticker"].notna() & df["Broker"].notna()].copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Broker"] = df["Broker"].astype(str).str.strip().str.upper()
    df["Reco"] = df["Reco"].astype(str).str.strip().str.title().replace({"Nan": np.nan})
    for col in ("Target", "ReportPrice", "EPS_Y0", "EPS_Y1", "EPS_Y2", "Rev_Y0", "Rev_Y1", "Rev_Y2", "ROCE_Y1"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)   # 0 = not filled in yet
    stats["zero_tp"] = int(df["Target"].isna().sum())
    fixed = [_fix_report_date(r, q, today) for r, q in zip(df["RawDate"], df["Quarter"])]
    df["Date"] = [f[0] for f in fixed]
    df["DateStatus"] = [f[1] for f in fixed]
    return df, stats


# ---------------------------------------------------------------- AMBIT classification
HEADER_MARK = "═"   # the '═══ INDUSTRY ═══' rows in the Tracker / ConsolidatedView sheets


def load_ambit_classification(xlsx, sheets=("Tracker", "ConsolidatedView")):
    """Industry / Sub-Industry by ticker from the AMBIT-style sheets (both sheets carry the same layout)."""
    frames = []
    for sheet in sheets:
        try:
            frames.append(_classification_from_sheet(xlsx, sheet))
        except ValueError:          # sheet missing
            continue
    if not frames:
        raise ValueError(f"No classification sheet found in {xlsx}")
    m = pd.concat(frames)
    return m[~m.index.duplicated(keep="first")]


def _classification_from_sheet(xlsx, sheet):
    t = pd.read_excel(xlsx, sheet_name=sheet, header=0, usecols=range(4))
    is_header = t["Industry"].astype(str).str.contains(HEADER_MARK)
    industry = t["Industry"].where(is_header).str.replace(HEADER_MARK, "", regex=False).str.strip().ffill()
    sub = t["Sub-Industry"].ffill()
    m = pd.DataFrame({"Industry": industry, "SubIndustry": sub, "Stock": t["Stock"], "Ticker": t["Ticker"]})
    m = m[m["Ticker"].notna() & m["Stock"].notna()]
    m = m[~m["Ticker"].astype(str).str.fullmatch(r"\d+")]
    m = m[m["Industry"] != "ETFs"].copy()
    m["Ticker"] = m["Ticker"].astype(str).str.strip().str.upper()
    m["Industry"] = m["Industry"].map(nice_case)
    m["SubIndustry"] = m["SubIndustry"].astype(str).str.strip().map(nice_case)
    return m.drop_duplicates("Ticker").set_index("Ticker")


ACRONYMS = {"NBFC", "CV", "PSU", "MFI", "IT", "EMS", "QSR", "AMC", "EPC", "PEB", "R&D", "ER&D", "EV", "2W", "4W",
            "CDMO", "BPO", "HPC", "BPC", "OMC", "CGD", "PV", "FMCG", "IPP", "T&D", "LPG", "API", "SFB", "HFC",
            "MSME", "ITES", "OEM", "BFSI", "HVAC", "AC", "TV", "LED", "PCB", "CNG", "LNG", "ODM", "ETF", "REIT",
            "OTA", "B2B", "B2C", "D2C", "RE", "EPS"}


def nice_case(text):
    """'NBFCS - AUTO/CV FINANCE' -> 'NBFCs - Auto/CV Finance' (title case that keeps acronyms)."""
    def fix(token):
        up = token.upper()
        if up in ACRONYMS:
            return up
        if up.endswith("S") and up[:-1] in ACRONYMS:
            return up[:-1] + "s"
        return token[:1].upper() + token[1:].lower()
    parts = re.split(r"([\s/\-()]+)", str(text).strip())
    return "".join(fix(p) if p and not re.fullmatch(r"[\s/\-()]+", p) else p for p in parts)


# ---------------------------------------------------------------- watchlists.properties
def load_watchlists(path):
    sections, current = {}, None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^\[([^\]]+)\]", line)
        if m:
            current = m.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current and line.lower().startswith("tickers"):
            _, _, value = line.partition("=")
            sections[current] += [t.strip().upper() for t in value.split(",") if t.strip()]
    return sections
