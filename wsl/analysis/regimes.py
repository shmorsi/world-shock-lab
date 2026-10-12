"""Regime indicators measured before each announcement (protocol §9 and v3).

* high_vix: VIX at the last close before the announcement is above the median of the 252
  closes ending there. VIX is CBOE-copyrighted (O-9): computed locally, never published raw.
* cycle: sign of the change in the policy-rate target over the 182 days before the
  announcement -> dummies "hiking" and "cutting" (holding is the baseline). Fed: target until
  2008-12-15, upper end of the target range after; BoC: target overnight rate.
* high_inflation: year-on-year CPI (NSA) of the month two months before the announcement
  month is above 3.0% (that month is always published by then). US CPI for Fed and OPEC
  events, Canadian CPI for BoC events.
* oil_exporter (by country and year): WDI net energy imports below 0 in the latest year
  available before the event year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wsl.paths import CLEAN_DIR
from wsl.timing import utc_ns

INFLATION_THRESHOLD = 3.0
VIX_WINDOW = 252
CYCLE_DAYS = 182


def _series(series_id: str) -> pd.Series:
    obs = pd.read_parquet(CLEAN_DIR / "obs.parquet", filters=[("series_id", "==", series_id)])
    out: pd.Series = obs[obs["status"] == "ok"].set_index("date")["value"].sort_index()
    return out


def _before(values: pd.Series, ts: pd.Series, tz: str) -> np.ndarray:
    """Position of the last daily value dated strictly before each event's local date."""
    dates = pd.DatetimeIndex(values.index).tz_localize(tz)
    event_days = ts.dt.tz_convert(tz).dt.normalize()
    return np.searchsorted(utc_ns(dates), utc_ns(event_days), side="left") - 1


def high_vix(ts: pd.Series) -> np.ndarray:
    vix = _series("VIX")
    med = vix.rolling(VIX_WINDOW, min_periods=VIX_WINDOW).median()
    pos = _before(vix, ts, "America/New_York")
    ok = pos >= 0
    out = np.full(len(ts), np.nan)
    v, m = vix.to_numpy(), med.to_numpy()
    out[ok] = np.where(np.isnan(m[pos[ok]]), np.nan, (v[pos[ok]] > m[pos[ok]]).astype(float))
    return out


def policy_target(family: str) -> pd.Series:
    if family == "boc":
        return _series("BOC_TARGET")
    old, upper = _series("FF_TARGET"), _series("FF_TARGET_UPPER")
    out: pd.Series = pd.concat([old[old.index < upper.index.min()], upper]).sort_index()
    return out


def cycle(ts: pd.Series, family: str) -> np.ndarray:
    """(N, 2) dummies [hiking, cutting]; NaN where the target history is too short."""
    rate = policy_target(family)
    tz = "America/Toronto" if family == "boc" else "America/New_York"
    now = _before(rate, ts, tz)
    then = _before(rate, ts - pd.Timedelta(days=CYCLE_DAYS), tz)
    out = np.full((len(ts), 2), np.nan)
    ok = (now >= 0) & (then >= 0)
    r = rate.to_numpy()
    change = r[now[ok]] - r[then[ok]]
    out[ok, 0] = (change > 1e-9).astype(float)
    out[ok, 1] = (change < -1e-9).astype(float)
    return out


def high_inflation(ts: pd.Series, family: str) -> np.ndarray:
    cpi = _series("CA_CPI_NSA" if family == "boc" else "US_CPI_NSA")
    yoy = 100.0 * (cpi / cpi.shift(12) - 1.0)
    yoy.index = pd.DatetimeIndex(yoy.index).to_period("M")
    months = ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M") - 2
    vals = yoy.reindex(months.to_numpy()).to_numpy()
    return np.where(np.isnan(vals), np.nan, (vals > INFLATION_THRESHOLD).astype(float))


def regime_matrix(ts: pd.Series, family: str, regime: str) -> tuple[np.ndarray, list[str]]:
    if regime == "high_vix":
        return high_vix(ts)[:, None], ["high VIX"]
    if regime == "cycle":
        return cycle(ts, family), ["hiking", "cutting"]
    if regime == "high_inflation":
        return high_inflation(ts, family)[:, None], ["inflation > 3%"]
    raise ValueError(f"unknown regime {regime!r}")


def exporter_flags(country_years: pd.DataFrame) -> np.ndarray:
    """``country_years`` has columns country (ISO3) and year; returns 1 if the latest WDI net
    energy imports value from an earlier year is below zero, 0 if at or above, NaN if none."""
    obs = pd.read_parquet(CLEAN_DIR / "obs.parquet")
    wdi = obs[obs["series_id"].str.endswith("_NET_ENERGY_IMPORTS") & (obs["status"] == "ok")]
    wdi = wdi.assign(country=wdi["series_id"].str.split("_").str[0], year=wdi["date"].dt.year)
    out = np.full(len(country_years), np.nan)
    for i, (c, y) in enumerate(zip(country_years["country"], country_years["year"], strict=True)):
        prior = wdi[(wdi["country"] == c) & (wdi["year"] < y)]
        if not prior.empty:
            out[i] = float(prior.sort_values("year")["value"].iloc[-1] < 0)
    return out
