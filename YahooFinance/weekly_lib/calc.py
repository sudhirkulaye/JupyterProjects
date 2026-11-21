"""Calculations: price returns, split-safe consensus revisions, composite score, scoreboard, brokers, sectors."""
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import spearmanr

NICE_RATIOS = np.array([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0])


def snap_ratio(ratio, tol=0.08):
    """Round an observed price ratio to the nearest clean split / bonus ratio (2 for 1:1 bonus, 5 for 1:5 ...)."""
    if ratio is None or not np.isfinite(ratio) or ratio < 1.3:
        return None
    i = int(np.argmin(np.abs(NICE_RATIOS / ratio - 1)))
    return float(NICE_RATIOS[i]) if abs(NICE_RATIOS[i] / ratio - 1) <= tol else None


# ================================================================ prices
def window_start(end, label):
    if label == "YTD":
        return pd.Timestamp(end.year - 1, 12, 31)
    n, unit = int(label[:-1]), label[-1]
    return end - pd.Timedelta(weeks=n) if unit == "W" else end - relativedelta(months=n)


def asof_row(panel_ff, when):
    sub = panel_ff.loc[:when]
    return sub.iloc[-1] if len(sub) else pd.Series(np.nan, index=panel_ff.columns)


def fresh_mask(prices, when, days=6):
    """True where the stock actually traded in the few days up to `when` (forward-filled prices can be stale)."""
    return prices.loc[when - pd.Timedelta(days=days):when].notna().any()


def returns_table(panel_ff, end, labels):
    p_end = asof_row(panel_ff, end)
    return pd.DataFrame({lab: p_end / asof_row(panel_ff, window_start(end, lab)) - 1 for lab in labels})


def from_52w_high(panel, end):
    window = panel.loc[end - pd.Timedelta(days=365):end]
    return asof_row(panel.ffill(), end) / window.max() - 1


def join_renamed(panel, renamed_from, max_jump=0.25):
    """nse_price_history keeps a renamed stock under two symbols (ICICIPRULI up to 22-Oct-2026, ICICILIFE after).
    Fill the new symbol's earlier dates from the old one when the two series meet without a jump. In memory only.
    renamed_from: {new symbol: old symbol}. Returns (panel, joined, warnings)."""
    joined, warnings = [], []
    for new, old in renamed_from.items():
        if old not in panel.columns:
            continue
        s_old = panel[old]
        if new not in panel.columns or panel[new].first_valid_index() is None:
            panel[new] = s_old
            warnings.append(f"{new}: no prices under the new symbol yet; using the old symbol {old}")
            continue
        first = panel[new].first_valid_index()
        before = s_old[s_old.index < first].dropna()
        if before.empty:
            continue
        jump = panel.at[first, new] / before.iloc[-1] - 1
        if abs(jump) > max_jump:
            warnings.append(f"{new}: price history not joined with the old symbol {old} ({jump:+.0%} jump at the rename)")
            continue
        early = panel.index < first
        panel.loc[early, new] = s_old[early]
        joined.append(f"{new} from {old} before {first:%d-%b-%Y}")
    return panel.drop(columns=[o for o in renamed_from.values() if o in panel.columns]), joined, warnings


def price_breaks(events):
    """Dates where a stock's DB price jumps for a corporate action that is still not adjusted (a demerger such as
    TATAMOTORS on 14-Oct-2025, or any pending move that is not a clean split ratio). Long price windows skip them."""
    out = {}
    pending = events[events["is_processed"].astype(str).str.upper() != "YES"]
    for ev in pending.itertuples(index=False):
        ratio = ev.previous_close_price / ev.close_price if ev.close_price else np.nan
        if snap_ratio(ratio) is None:
            out.setdefault(ev.ticker, []).append(ev.date)
    return out


def momentum_12_1(prices_ff, t, breaks=None, look=252, skip=21):
    """Price return from about 12 months ago to 1 month ago (in trading days), leaving out the latest month."""
    d = prices_ff.loc[:t]
    if len(d) <= look:
        return pd.Series(np.nan, index=prices_ff.columns)
    start, end = d.index[-1 - look], d.index[-1 - skip]
    mom = d.iloc[-1 - skip] / d.iloc[-1 - look] - 1
    for tk, dates in (breaks or {}).items():
        if tk in mom.index and any(start < x <= end for x in dates):
            mom[tk] = np.nan
    return mom


def adjust_unprocessed_splits(panel, events):
    """The DB is corrected for splits after ~3 months; adjust any still-pending clean split on the fly."""
    notes = []
    pending = events[events["is_processed"].astype(str).str.upper() != "YES"]
    for ev in pending.itertuples(index=False):
        if ev.ticker not in panel.columns or not ev.close_price:
            continue
        s = panel[ev.ticker]
        before = s.loc[:ev.date - pd.Timedelta(days=1)].dropna()
        after = s.loc[ev.date:].dropna()
        if before.empty or after.empty:
            continue
        factor = snap_ratio(ev.previous_close_price / ev.close_price)
        if factor is None:
            notes.append(f"{ev.ticker}: {ev.day_percent_change:.0%} move on {ev.date:%d-%b-%Y} is pending in "
                         f"stock_split_probability but is not a clean split ratio; prices left unchanged")
            continue
        if abs(after.iloc[0] / before.iloc[-1] * factor - 1) > 0.12:
            continue                                   # history already adjusted
        panel.loc[panel.index < ev.date, ev.ticker] = s[s.index < ev.date] / factor
        notes.append(f"{ev.ticker}: pending split on {ev.date:%d-%b-%Y} (factor {factor:g}) adjusted for this report")
    return panel, notes


# ================================================================ consensus panel
class Consensus:
    """Weekly Yahoo consensus as wide (date x ticker) matrices, with split rescaling and roll-safe revisions."""

    def __init__(self, cons):
        self.dates = [pd.Timestamp(d) for d in sorted(cons["Date"].unique())]
        base = cons.sort_values(["Date", "Ticker", "FIN Yr"]).drop_duplicates(["Date", "Ticker"], keep="first")

        def wide(frame, col):
            return frame.pivot(index="Date", columns="Ticker", values=col).reindex(self.dates)

        self.tp = wide(base, "TP(Avg)")
        self.tickers = self.tp.columns
        self.tp_min = wide(base, "TP(Min)").reindex(columns=self.tickers)
        self.tp_max = wide(base, "TP(Max)").reindex(columns=self.tickers)
        self.count = wide(base, "#Count").reindex(columns=self.tickers)
        fy = cons[cons["FIN Yr"].notna()]
        self.labels = sorted(fy["FIN Yr"].unique())
        def by_label(col):
            return {lab: wide(fy[fy["FIN Yr"] == lab], col).reindex(columns=self.tickers) for lab in self.labels}

        self.eps, self.eps_min, self.eps_max = by_label("EPS(Avg)"), by_label("EPS(Min)"), by_label("EPS(Max)")
        self.rev = by_label("Rev.(Avg)")
        self._tp_by_stat = {"avg": self.tp, "min": self.tp_min, "max": self.tp_max}
        self._eps_by_stat = {"avg": self.eps, "min": self.eps_min, "max": self.eps_max}
        self.breaks = {}
        self.notes = []
        self._rev_cache = {}

    # ---------------------------------------------------- corporate actions
    def _per_share(self):
        yield "TP", self.tp
        yield "TP min", self.tp_min
        yield "TP max", self.tp_max
        for stat, mats in (("avg", self.eps), ("min", self.eps_min), ("max", self.eps_max)):
            for lab, mat in mats.items():
                yield f"EPS {stat} {lab}", mat

    @staticmethod
    def _rescale(mat, tk, ev_date, factor):
        """Yahoo often applies a split months late (LICI's May bonus reached its TP only in November), so take
        the first later update whose size is near 1/factor (allowing for a genuine change in the same update)."""
        s = mat[tk]
        r = s / s.ffill().shift(1)
        later = r[(r.index >= ev_date - pd.Timedelta(days=14)) & ((r - 1).abs() > 0.001)]
        hit = later[(later * factor >= 0.6) & (later * factor <= 1.4)]
        if hit.empty:
            return None
        t_star = hit.index[0]
        mask = mat.index < t_star
        mat.loc[mask, tk] = s[mask] / factor
        return t_star

    def apply_corporate_actions(self, events):
        first = self.dates[0] - pd.Timedelta(days=45)
        for ev in events[events["date"] >= first].itertuples(index=False):
            tk = ev.ticker
            if tk not in self.tickers:
                continue
            note = str(ev.note or "").strip()
            ratio = ev.previous_close_price / ev.close_price if ev.close_price else np.nan
            named_split = any(w in note.lower() for w in ("bonus", "split"))
            pending = str(ev.is_processed).upper() != "YES"
            factor = snap_ratio(ratio, tol=0.10) if named_split else (snap_ratio(ratio, 0.06) if pending else None)
            if factor:
                touched = [name for name, mat in self._per_share() if self._rescale(mat, tk, ev.date, factor) is not None]
                if touched:
                    self.notes.append(f"{tk}: {note or 'split'} on {ev.date:%d-%b-%Y} - consensus TP/EPS history "
                                      f"before it divided by {factor:g}")
            else:
                self.breaks.setdefault(tk, []).append(ev.date)
                self.notes.append(f"{tk}: corporate action on {ev.date:%d-%b-%Y} (note: {note or 'none'}, "
                                  f"price move {ev.day_percent_change:+.0%}) - revisions across it ignored")

    def flag_suspect_jumps(self, lo=0.6, hi=1.67):
        r = self.tp / self.tp.ffill().shift(1)
        mask = ((r < lo) | (r > hi)).to_numpy()
        for i, j in zip(*np.nonzero(mask)):
            date, tk = r.index[i], r.columns[j]
            self.breaks.setdefault(tk, []).append(date - pd.Timedelta(days=1))
            self.notes.append(f"{tk}: consensus TP moved {r.iat[i, j] - 1:+.0%} in the week to {date:%d-%b-%Y} with "
                              f"no recorded split - revisions across it ignored")

    # ---------------------------------------------------- revisions
    def past_date(self, t, weeks):
        target = t - pd.Timedelta(weeks=weeks)
        best = min(self.dates, key=lambda d: abs((d - target).days))
        return best if abs((best - target).days) <= 4 and best < t else None

    def _no_break(self, p, t):
        ok = pd.Series(True, index=self.tickers)
        for tk, ds in self.breaks.items():
            if any(p < d <= t for d in ds):
                ok[tk] = False
        return ok

    @staticmethod
    def _rel_change(a, b):
        return ((a - b) / b.abs()).where(b.abs() > 0.01)

    def revisions(self, t, weeks, stat="avg"):
        """% change vs `weeks` ago of the consensus average (or lowest / highest) estimate. EPS and revenue compare
        the same fiscal-year label on both dates and average the two years Yahoo shows (roll-safe)."""
        key = (t, weeks, stat)
        if key in self._rev_cache:
            return self._rev_cache[key]
        p = self.past_date(t, weeks)
        if p is None:
            out = pd.DataFrame(np.nan, index=self.tickers, columns=["tp", "eps", "rev", "count_chg"])
        else:
            tp_mat, eps_mats = self._tp_by_stat[stat], self._eps_by_stat[stat]
            eps = pd.concat([self._rel_change(m.loc[t], m.loc[p]) for m in eps_mats.values()], axis=1).mean(axis=1)
            rev = pd.concat([self._rel_change(m.loc[t], m.loc[p]) for m in self.rev.values()], axis=1).mean(axis=1)
            out = pd.DataFrame({"tp": tp_mat.loc[t] / tp_mat.loc[p] - 1, "eps": eps, "rev": rev,
                                "count_chg": self.count.loc[t] - self.count.loc[p]})
            out.loc[~self._no_break(p, t), ["tp", "eps", "rev"]] = np.nan
        self._rev_cache[key] = out
        return out

    def persistence(self, t, weeks=13):
        """Weeks with an EPS upgrade minus weeks with a downgrade, over the last `weeks` weekly changes."""
        idx = self.dates.index(t)
        net = pd.Series(0, index=self.tickers)
        for k in range(weeks):
            if idx - k < 1:
                break
            r = self.revisions(self.dates[idx - k], 1)["eps"]
            net += (r > 0.001).astype(int) - (r < -0.001).astype(int)
        return net

    def fy_labels(self, t, tk):
        return sorted(lab for lab, m in self.eps.items() if pd.notna(m.at[t, tk]))

    def change_stats(self, t):
        """age: weeks since the TP or any EPS estimate last moved. gap: weeks it had been frozen before that move
        (Yahoo data for many Indian stocks sits unchanged for months, then updates in a burst)."""
        changed = self.tp.pct_change(fill_method=None).abs() > 0.001
        for mat in self.eps.values():
            changed |= mat.pct_change(fill_method=None).abs() > 0.001
        changed = changed.loc[:t]
        rows = {}
        for tk in changed.columns:
            ds = changed.index[changed[tk].to_numpy()]
            last = ds[-1] if len(ds) else self.dates[0]
            prev = ds[-2] if len(ds) > 1 else self.dates[0]
            rows[tk] = ((t - last).days / 7, (last - prev).days / 7 if len(ds) else np.nan)
        return pd.DataFrame.from_dict(rows, orient="index", columns=["age", "gap"]).round()


# ================================================================ composite score and scoreboard
def rank_z(s):
    r = s.rank()
    return (r - r.mean()) / r.std()


def composite_score(rev, mom, count, min_analysts, mom12=None):
    """Equal-weight rank z-scores: EPS revision + TP revision - 4-week price move (short-term reversal)
    + 12-1 month price momentum. A stock listed under a year has no momentum and is scored on the other three."""
    df = pd.DataFrame({"eps": rev["eps"], "tp": rev["tp"]})
    df["mom"] = mom.reindex(df.index)
    df["mom12"] = mom12.reindex(df.index) if mom12 is not None else np.nan
    df["count"] = count.reindex(df.index)
    df.loc[df["eps"].abs() > 1.0, "eps"] = np.nan          # tiny base or bad data
    df.loc[df["tp"].abs() > 0.5, "tp"] = np.nan
    elig = (df["count"] >= min_analysts) & df["mom"].notna() & (df["eps"].notna() | df["tp"].notna())
    sub = df[elig]
    z = pd.DataFrame({"z_eps": rank_z(sub["eps"]), "z_tp": rank_z(sub["tp"]), "z_rev": -rank_z(sub["mom"])})
    if mom12 is not None:
        z["z_mom12"] = rank_z(sub["mom12"])
    out = pd.DataFrame(index=df.index)
    out["score"] = z.mean(axis=1).where(z.notna().sum(axis=1) >= 2)
    out["pct"] = out["score"].rank(pct=True) * 100
    out = out.join(z)
    out["eligible"] = out["score"].notna()
    return out


def score_history(cons, prices, prices_ff, min_analysts, score_weeks, fwd_weeks, breaks=None):
    last_px = prices_ff.index.max()
    scores, rows = {}, []
    for t in cons.dates:
        rev = cons.revisions(t, score_weeks)
        if rev[["eps", "tp"]].notna().to_numpy().sum() == 0:
            continue
        p_t = asof_row(prices_ff, t)
        fresh = fresh_mask(prices, t)
        mom = (p_t / asof_row(prices_ff, t - pd.Timedelta(weeks=score_weeks)) - 1).where(fresh)
        mom12 = momentum_12_1(prices_ff, t, breaks).where(fresh)
        sc = composite_score(rev, mom, cons.count.loc[t], min_analysts, mom12)
        sc["mom"] = mom.reindex(sc.index)
        sc["mom12"] = mom12.reindex(sc.index)
        sc["eps"], sc["tp"] = rev["eps"], rev["tp"]
        scores[t] = sc
        end = t + pd.Timedelta(weeks=fwd_weeks)
        if end > last_px:
            continue
        fwd = (asof_row(prices_ff, end) / p_t - 1).reindex(sc.index)
        d = pd.DataFrame({"score": sc["score"], "fwd": fwd}).dropna()
        d = d[d["fwd"].abs() < 1.0]
        if len(d) < 30:
            continue
        q = pd.qcut(d["score"].rank(method="first"), 5, labels=False)
        xs = d["fwd"] - d["fwd"].mean()
        rows.append({"date": t, "n": len(d), "ic": spearmanr(d["score"], d["fwd"]).correlation,
                     "q5": xs[q == 4].mean(), "q1": xs[q == 0].mean()})
    hist = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["n", "ic", "q5", "q1"])
    return scores, hist


def scoreboard_stats(hist, step):
    if hist.empty:
        return {}
    ic = hist["ic"]
    non_overlap = ic.iloc[::-step][::-1]
    t_stat = non_overlap.mean() / (non_overlap.std() / np.sqrt(len(non_overlap))) if len(non_overlap) > 2 else np.nan
    return {"weeks": len(ic), "mean_ic": ic.mean(), "recent_ic": ic.tail(12).mean(), "pct_pos": (ic > 0).mean(),
            "spread": (hist["q5"] - hist["q1"]).mean(), "q5": hist["q5"].mean(), "q1": hist["q1"].mean(),
            "t_stat": t_stat, "n_independent": len(non_overlap), "first": hist.index.min(), "last": hist.index.max()}


def breadth_history(scores):
    rows = []
    for t, sc in scores.items():
        e = sc.loc[sc["eligible"], "eps"].dropna()
        if len(e) < 30:
            continue
        up, down = (e > 0.005).mean(), (e < -0.005).mean()
        rows.append({"date": t, "up": up, "down": down, "net": up - down, "n": len(e)})
    return pd.DataFrame(rows).set_index("date")


# ================================================================ brokers
def broker_views(reco, today, price_now, reco_score, max_age_days, week_start):
    r = reco.dropna(subset=["Date"]).copy()
    r = r[r["Date"] <= today + pd.Timedelta(days=1)]
    r["score"] = r["Reco"].str.upper().map(reco_score)
    r = r.sort_values(["Ticker", "Broker", "Date"])
    g = r.groupby(["Ticker", "Broker"], sort=False)
    for col in ("Reco", "score", "Target", "Date", "Quarter", "EPS_Y2"):
        r["prev_" + col] = g[col].shift(1)
    r["tp_chg"] = r["Target"] / r["prev_Target"] - 1
    r["reco_chg"] = r["score"] - r["prev_score"]
    now_q = r["Quarter"].astype(str).str.extract(r"FY(\d{2})Q([1-4])")
    prev_q = r["prev_Quarter"].astype(str).str.extract(r"FY(\d{2})Q([1-4])")
    same_basis = (now_q[0] == prev_q[0]) & ~((now_q[1] == "4") & (prev_q[1] == "3"))
    r["eps2_chg"] = (r["EPS_Y2"] / r["prev_EPS_Y2"] - 1).where(same_basis)
    broker_mean = r[r["Date"] >= today - pd.Timedelta(days=548)].groupby("Broker")["score"].mean()
    r["stance"] = r["score"] - r["Broker"].map(broker_mean)
    latest = r.groupby(["Ticker", "Broker"], sort=False).tail(1)
    latest = latest[latest["Date"] >= today - pd.Timedelta(days=max_age_days)]
    agg = latest.groupby("Ticker").agg(
        n=("Broker", "size"), stance=("stance", "mean"),
        n_pos=("score", lambda s: int((s > 0).sum())), n_neu=("score", lambda s: int((s == 0).sum())),
        n_neg=("score", lambda s: int((s < 0).sum())), avg_tp=("Target", "mean"), last_date=("Date", "max"))
    agg["tp_upside"] = agg["avg_tp"] / price_now.reindex(agg.index) - 1
    month = r[r["Date"] >= today - pd.Timedelta(days=30)]
    agg["up_30d"] = month[month["reco_chg"] > 0].groupby("Ticker").size().reindex(agg.index).fillna(0)
    agg["down_30d"] = month[month["reco_chg"] < 0].groupby("Ticker").size().reindex(agg.index).fillna(0)
    agg["tpcut_30d"] = month[month["tp_chg"] <= -0.10].groupby("Ticker").size().reindex(agg.index).fillna(0)
    week = r[(r["Date"] > week_start) & (r["Date"] <= today + pd.Timedelta(days=1))]
    return {"all": r, "latest": latest.set_index(["Ticker", "Broker"]), "agg": agg, "week": week,
            "broker_mean": broker_mean}


# ================================================================ sectors
SECTOR_COLS = ["eps4", "eps13", "tp4", "rel1m", "rel3m"]


def _summarise(g):
    e = g["eps4"].dropna()
    n = len(e)
    return pd.Series({"n": n, "eps4": e.median() if n else np.nan,
                      "breadth": ((e > 0.005).sum() - (e < -0.005).sum()) / n if n else np.nan,
                      "eps13": g["eps13"].median(), "tp4": g["tp4"].median(),
                      "rel1m": g["rel1m"].median(), "rel3m": g["rel3m"].median()})


def sector_tables(metrics, cls):
    df = metrics[metrics["eligible"]].join(cls[["Industry", "SubIndustry"]], how="left")
    df["Industry"] = df["Industry"].fillna("Unclassified")
    df["SubIndustry"] = df["SubIndustry"].fillna("Unclassified")
    ind = df.groupby("Industry")[SECTOR_COLS].apply(_summarise).sort_values("breadth", ascending=False)
    sub = df.groupby(["Industry", "SubIndustry"])[SECTOR_COLS].apply(_summarise)
    return ind, sub
