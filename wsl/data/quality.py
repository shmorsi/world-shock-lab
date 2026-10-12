"""Data-quality report (``wsl data report`` -> docs/DATA_QUALITY.md).

Reports counts, dates and discrepancy statistics only. No raw values of restricted series
(policy L or D) are written, so the report can be committed and published.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from wsl.data.cache import IntegrityError, RawCache
from wsl.data.registry import load_registry
from wsl.data.sources import nasdaq
from wsl.data.sources.base import STATUS_INVALID, STATUS_MISSING, STATUS_OK
from wsl.data.sources.yahoo import parse_chart
from wsl.paths import CLEAN_DIR, DATA_QUALITY_REPORT, MANIFEST, ROOT

SEED = 20261012
STALE_DAYS = {"D": 45, "M": 150, "A": 3 * 366}


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_\n"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def _fmt(v: object) -> str:
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, float):
        return "" if np.isnan(v) else f"{v:,.4g}"
    return str(v)


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(CLEAN_DIR / f"{name}.parquet")


# ----------------------------------------------------------------------------- sections
def section_sources(cache: RawCache) -> str:
    manifest = json.loads(MANIFEST.read_text())["artifacts"]
    rows = []
    for key, entry in manifest.items():
        rec = next(r for r in entry["retrievals"] if r["sha256"] == entry["pinned"])
        rows.append({"group": key.split("/")[0], "bytes": rec["size"],
                     "retrieved": rec["retrieved_at"][:10]})
    df = pd.DataFrame(rows).groupby("group").agg(
        artifacts=("bytes", "size"), MB=("bytes", lambda b: round(b.sum() / 1e6, 2)),
        retrieved=("retrieved", "max")).reset_index()
    return "## 1. Raw artifacts (pinned in data/manifest.json)\n\n" + _table(df)


def section_coverage(obs: pd.DataFrame, series: pd.DataFrame) -> str:
    g = obs.groupby("series_id")
    cov = pd.DataFrame({
        "first": g["date"].min(), "last": g["date"].max(),
        "ok": g["status"].agg(lambda s: int((s == STATUS_OK).sum())),
        "missing": g["status"].agg(lambda s: int((s == STATUS_MISSING).sum())),
        "invalid": g["status"].agg(lambda s: int((s == STATUS_INVALID).sum())),
    }).reset_index()
    meta = series[["series_id", "source", "freq", "policy"]]
    cov = meta.merge(cov, on="series_id")
    # Only list price-level series for Yahoo (close/tr are representative; dividends sparse).
    cov = cov[~cov["series_id"].str.endswith((".adjclose", ".dividend"))]
    return ("## 2. Coverage by series\n\n`missing` = the source reported no value "
            "(e.g. a market holiday); `invalid` = an impossible value we refused to use. "
            "Nothing is filled.\n\n" + _table(cov.sort_values(["source", "series_id"])))


def section_pending(series: pd.DataFrame) -> str:
    specs = load_registry()
    built = set(series["parent_id"])
    pending = [s for s in specs if s.id not in built]
    if not pending:
        return "## 3. Pending sources\n\nNone: every registered series was built.\n"
    df = pd.DataFrame([{"series": s.id, "source": s.source, "key": s.key} for s in pending])
    reason = {"fred": "FRED_API_KEY not set in .env"}
    df["reason"] = df["source"].map(reason).fillna("artifact not fetched")
    return (f"## 3. Pending sources\n\n{len(pending)} registered series are not built yet.\n\n"
            + _table(df.groupby(["source", "reason"])["series"].agg(
                lambda s: ", ".join(s)).reset_index()))


def section_invalid(obs: pd.DataFrame) -> str:
    bad = obs[obs["status"] == STATUS_INVALID][["series_id", "date"]]
    return ("## 4. Invalid values reported by sources\n\nImpossible values (e.g. a stock price "
            "of 0) are set to missing and listed here; they never enter returns.\n\n"
            + _table(bad))


def section_holidays(obs: pd.DataFrame) -> str:
    def trading_days(sid: str) -> set[pd.Timestamp]:
        s = obs[(obs["series_id"] == sid) & (obs["status"] == STATUS_OK)]
        return set(s["date"])
    us, ca = trading_days("SPY.close"), trading_days("TSX.close")
    lo = max(min(us), min(ca))
    us_only = sorted(d for d in us - ca if d >= lo)
    ca_only = sorted(d for d in ca - us if d >= lo)
    yr = pd.DataFrame({
        "year": range(lo.year, max(max(us), max(ca)).year + 1)})
    yr["US open, TSX closed"] = yr["year"].map(pd.Series([d.year for d in us_only]).value_counts())
    yr["TSX open, US closed"] = yr["year"].map(pd.Series([d.year for d in ca_only]).value_counts())
    yr = yr.fillna(0).astype(int)
    return ("## 5. US vs Canadian trading calendars\n\nDays on which one market traded and "
            "the other did not (SPY vs S&P/TSX closes). Event windows use each series' own "
            "calendar; nothing is forward-filled across these days.\n\n"
            f"Total: {len(us_only)} US-only days, {len(ca_only)} TSX-only days since "
            f"{lo.date()}.\n\n" + _table(yr.tail(12)))


def section_tr_check(cache: RawCache) -> str:
    rows = []
    for s in load_registry():
        if s.source != "yahoo" or s.key.startswith("^"):
            continue
        daily, divs, splits, _ = parse_chart(cache.read(f"yahoo/{s.key}")[0])
        from wsl.data.transforms import total_return_index
        close = daily["close"].mask(daily["close"] <= 0)
        tr = total_return_index(close, divs)
        a = np.log(tr.dropna()).diff()
        b = np.log(daily["adjclose"].where(daily["adjclose"] > 0).dropna()).diff()
        d = (a - b).dropna().abs() * 1e4
        rows.append({"ticker": s.key, "days": len(d), "median |Δ| bp": float(d.median()),
                     "max |Δ| bp": float(d.max()), "worst date": d.idxmax(),
                     "days > 5bp": int((d > 5).sum()), "dividends": len(divs),
                     "splits": len(splits)})
    return ("## 6. Our total-return index vs Yahoo's adjusted close\n\nDaily log-return "
            "differences. Small differences are expected (Yahoo adjusts by "
            "(1 − d/P) factors; we use (P + d)/P). Large ones flag dividend or split "
            "problems at the source.\n\n" + _table(pd.DataFrame(rows)))


def section_nasdaq(cache: RawCache) -> str:
    rng = np.random.default_rng(SEED)
    rows = []
    for s in nasdaq.us_listed([x for x in load_registry() if x.source == "yahoo"]):
        try:
            nq = nasdaq.parse_closes(cache.read(f"nasdaq/{s.key}")[0])
        except IntegrityError:
            continue
        daily, _, _, _ = parse_chart(cache.read(f"yahoo/{s.key}")[0])
        # Both sources are split-adjusted and not dividend-adjusted (established on the
        # 2016-11-07 EWU/EWI/EWJ and 2025-12-05 XLE splits), so closes compare directly.
        y_unadj = daily["close"].where(daily["close"] > 0).dropna()
        common = y_unadj.index.intersection(nq.index)
        if len(common) == 0:
            rows.append({"ticker": s.key, "overlap days": 0})
            continue
        sample = pd.DatetimeIndex(sorted(rng.choice(common, size=min(250, len(common)),
                                                    replace=False)))
        rel = (y_unadj[sample] / nq[sample] - 1).abs() * 100
        full = (y_unadj[common] / nq[common] - 1).abs() * 100
        rows.append({"ticker": s.key, "overlap days": len(common), "sampled": len(sample),
                     "max |diff| % (sample)": float(rel.max()),
                     "days > 0.5% (all overlap)": int((full > 0.5).sum()),
                     "worst date": full.idxmax()})
    return ("## 7. Yahoo vs Nasdaq (second source, US-listed ETFs only)\n\nSplit-adjusted, "
            "dividend-unadjusted closes from both; "
            f"250 dates sampled per ticker (seed {SEED}) plus a full-overlap count. "
            "TSX-listed tickers have no free second source (data/SOURCES.md §3.4).\n\n"
            + _table(pd.DataFrame(rows)))


def _robust_jumps(level: pd.Series, k: float = 8.0) -> pd.Series:
    d = level.dropna().diff().dropna() * 100  # bp
    mad = (d - d.rolling(120, min_periods=60).median()).abs().rolling(120, min_periods=60).median()
    z = d / (1.4826 * mad)
    return d[z.abs() > k]


def section_yield_jumps(obs: pd.DataFrame, events: dict[str, pd.DataFrame]) -> str:
    known = set(events["boc"]["date"]) | set(events["fomc"]["date"])
    rows = []
    for sid in ("CA2Y", "CA3Y", "CA5Y", "CA10Y"):
        s = obs[(obs["series_id"] == sid) & (obs["status"] == STATUS_OK)].set_index("date")
        jumps = _robust_jumps(s["value"])
        off_event = [d for d in jumps.index if d not in known]
        rows.append({"series": sid, "robust z > 8": len(jumps),
                     "on BoC/FOMC days": len(jumps) - len(off_event),
                     "off-event (candidate switches or other news)": len(off_event),
                     "latest off-event dates": ", ".join(d.strftime("%Y-%m-%d")
                                                         for d in off_event[-4:])})
    return ("## 8. Large daily moves in GoC benchmark yields\n\nBenchmark bonds are switched "
            "after auctions, which can create artificial jumps (BoC notes). We cannot tell a "
            "switch from news with Valet data alone, so large moves are listed for review; "
            "BoC-proxy windows that coincide with one will be flagged in M3.\n\n"
            + _table(pd.DataFrame(rows)))


def section_events(events: dict[str, pd.DataFrame]) -> str:
    fomc, boc, opec, ecb = events["fomc"], events["boc"], events["opec"], events["ecb"]
    t = fomc["ts"].dt.strftime("%H:%M")
    collisions = sorted(set(fomc["date"]) & set(boc["date"]))
    rows = [
        {"table": "events_fomc (JK)", "events": len(fomc),
         "range": f"{fomc['date'].min().date()} → {fomc['date'].max().date()}",
         "notes": "most common times ET: " + ", ".join(
             f"{k} ({v})" for k, v in t.value_counts().head(3).items())
                  + f"; {int((fomc['ts'].dt.dayofweek >= 5).sum())} on weekends"},
        {"table": "events_boc (constructed)", "events": len(boc),
         "range": f"{boc['date'].min().date()} → {boc['date'].max().date()}",
         "notes": f"{int(boc['scheduled'].sum())} scheduled, {int((~boc['scheduled']).sum())} "
                  f"unscheduled; {boc['decision'].value_counts().to_dict()}"},
        {"table": "events_opec (Känzig)", "events": len(opec),
         "range": f"{opec['date'].min().date()} → {opec['date'].max().date()}",
         "notes": f"{int(opec['weekend'].sum())} on weekends (map to next trading day)"},
        {"table": "events_ecb (JK)", "events": len(ecb),
         "range": f"{ecb['date'].min().date()} → {ecb['date'].max().date()}", "notes": ""},
    ]
    return ("## 9. Event tables\n\n" + _table(pd.DataFrame(rows)) +
            f"\n**FOMC and BoC on the same day:** {len(collisions)} historical dates, "
            f"{sum(d.year >= 2019 for d in collisions)} of them in the pre-registered test "
            "window (2019+)"
            + (f" ({', '.join(d.strftime('%Y-%m-%d') for d in collisions)})" if collisions
               else "") + ", plus 2026-10-28 and 2026-12-09 ahead. On these days a daily "
            "BoC proxy also contains the Fed news; they are excluded or flagged in M3.\n")


def section_boc_times(boc: pd.DataFrame) -> str:
    sched = boc[boc["scheduled"]]
    g = sched.groupby(sched["date"].dt.year).agg(
        time_et=("time_et", lambda s: ", ".join(sorted(set(s)))),
        basis=("time_basis", lambda s: ", ".join(sorted({str(x) for x in s if pd.notna(x)}))))
    g = g.reset_index().rename(columns={"date": "year"}).sort_values("year")
    # Collapse contiguous runs of years with the same time and basis.
    run = (g[["time_et", "basis"]] != g[["time_et", "basis"]].shift()).any(axis=1).cumsum()
    collapsed = g.groupby(run).agg(time_et=("time_et", "first"), basis=("basis", "first"),
                                   first=("year", "min"), last=("year", "max"))
    collapsed["years"] = [f"{a}–{b}" if a != b else str(a)
                          for a, b in zip(collapsed["first"], collapsed["last"], strict=True)]
    collapsed = collapsed[["years", "time_et", "basis"]]
    return ("## 10. BoC announcement times (from yearly schedule releases)\n\nAll documented "
            "times are 09:00–10:00 ET, i.e. before the 12:00 H.10 FX fix and before market "
            "closes, so the announcement day is day 0 for every closing-price series.\n\n"
            + _table(collapsed))


def section_cross_source(obs: pd.DataFrame) -> str:
    def yoy(sid: str) -> pd.Series:
        s = obs[(obs["series_id"] == sid) & (obs["status"] == STATUS_OK)].set_index("date")
        v = s["value"]
        return (100 * (v / v.shift(12) - 1)).dropna()
    rows = []
    pairs = [("CA_CPI_NSA", "CAN_CPI", "Canada CPI YoY: StatCan vs OECD")]
    for a, b, label in pairs:
        if not {a, b} <= set(obs["series_id"]):
            continue
        ya, yb = yoy(a), yoy(b)
        common = ya.index.intersection(yb.index)
        d = (ya[common] - yb[common]).abs()
        rows.append({"check": label, "months": len(common), "max |diff| pp": float(d.max()),
                     "months > 0.15pp": int((d > 0.15).sum())})
    return ("## 11. Cross-source consistency\n\n" + _table(pd.DataFrame(rows)) +
            "\nFurther checks (FRED vs OECD US CPI; VIXCLS vs Yahoo ^VIX; TSX vs EWC × "
            "USD/CAD) run once FRED data is available.\n")


def section_stale(obs: pd.DataFrame, series: pd.DataFrame) -> str:
    asof = pd.Timestamp(datetime.now(UTC).date())
    last = obs[obs["status"] == STATUS_OK].groupby("series_id")["date"].max()
    freq_of = dict(zip(series["series_id"], series["freq"], strict=True))
    rows = []
    for sid, d in last.items():
        if str(sid).endswith((".dividend", ".adjclose")):
            continue
        freq = freq_of[str(sid)]
        age = (asof - d).days
        if age > STALE_DAYS[str(freq)]:
            rows.append({"series": sid, "freq": freq, "last value": d, "days old": age})
    return ("## 12. Stale or discontinued series\n\nLast observation older than 45 days "
            "(daily), 150 days (monthly) or 3 years (annual). The OECD CPI gaps are in the "
            "source itself: both OECD CPI dataflows end at these dates for these countries "
            "(open item O-8 in data/SOURCES.md).\n\n" + _table(pd.DataFrame(rows)))


def _lines(parts: Iterable[str]) -> str:
    return "\n".join(parts)


def write_report(path: Path = DATA_QUALITY_REPORT) -> Path:
    cache = RawCache()
    obs, series = _read("obs"), _read("series")
    info = _read("build_info").iloc[0]
    events = {"fomc": _read("events_fomc"), "boc": _read("events_boc"),
              "opec": _read("events_opec"), "ecb": _read("events_ecb")}
    header = (
        "# Data quality report\n\n"
        "Generated by `uv run wsl data report` from the built data. Do not edit by hand.\n\n"
        f"- built at: {info['built_at']} (git {info['git_sha'][:12]}"
        f"{' with uncommitted changes' if info['git_sha'].endswith('-dirty') else ''})\n"
        f"- manifest sha256: `{info['manifest_sha256'][:16]}…`\n"
        f"- series registry sha256: `{info['series_registry_sha256'][:16]}…`\n"
        f"- observations: {len(obs):,} rows, {obs['series_id'].nunique()} series\n\n"
        "Only counts, dates and discrepancy statistics appear here; no values of restricted "
        "series are reported.\n"
    )
    body = _lines([
        header,
        section_sources(cache),
        section_coverage(obs, series),
        section_pending(series),
        section_invalid(obs),
        section_holidays(obs),
        section_tr_check(cache),
        section_nasdaq(cache),
        section_yield_jumps(obs, events),
        section_events(events),
        section_boc_times(events["boc"]),
        section_cross_source(obs),
        section_stale(obs, series),
    ])
    path.write_text(body)
    return path.relative_to(ROOT)
