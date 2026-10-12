"""Event windows: how much a series moved from just before an announcement to h observations
after it, using each series' own observation time (data/SOURCES.md §5, wsl/timing.py).

For an announcement at time A and a series observed at local time T each day:
  before  = last observation strictly before A
  day 0   = first observation strictly after A; day h = the h-th observation after that
  d_h     = x(day h) - x(before), where x is 100·level for yields (so changes are in bp)
            or 100·log(level) for prices, FX and indices (changes in %)
  pre_l   = x(before - l + 1) - x(before - l), the l-th daily change before the window,
            used as controls

Windows whose before/day-0 observation is more than ``max_gap_days`` from the announcement
(e.g. before a series starts) are left empty, as are windows that touch a non-positive
price (log undefined, e.g. WTI on 2020-04-20). Nothing is filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from wsl.paths import CLEAN_DIR
from wsl.timing import observation_timestamps, utc_ns

Kind = Literal["bp", "logpct"]


@dataclass(frozen=True)
class WindowSeries:
    id: str
    dates: pd.DatetimeIndex
    x: np.ndarray  # transformed level: 100·level (bp) or 100·log(level) (%)
    obs_ts: pd.DatetimeIndex  # tz-aware observation timestamps
    kind: Kind
    obs_time: str
    assumed_time: str | None


def kind_for(units: str) -> Kind:
    return "bp" if units == "pct" else "logpct"


def transform(values: np.ndarray, kind: Kind) -> np.ndarray:
    if kind == "bp":
        return 100.0 * values
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(values > 0, 100.0 * np.log(np.where(values > 0, values, 1.0)), np.nan)


def make_series(series_id: str, values: pd.Series, obs_time: str, tz: str, kind: Kind,
                assumed_time: str | None = None) -> WindowSeries:
    values = values.dropna().sort_index()
    dates = pd.DatetimeIndex(values.index)
    ts = observation_timestamps(dates, obs_time, tz, assumed_time=assumed_time)
    keep = ~np.asarray(ts.isna())
    return WindowSeries(series_id, dates[keep], transform(values.to_numpy()[keep], kind),
                        ts[keep], kind, obs_time, assumed_time)


def load_series(series_id: str, assumed_time: str | None = None) -> WindowSeries:
    """A built daily series (status ok rows only) ready for event windows."""
    obs = pd.read_parquet(CLEAN_DIR / "obs.parquet", filters=[("series_id", "==", series_id)])
    meta = pd.read_parquet(CLEAN_DIR / "series.parquet").set_index("series_id").loc[series_id]
    if meta["freq"] != "D":
        raise ValueError(f"{series_id} is not daily")
    if not bool(meta["event_window_ok"]):
        raise ValueError(f"{series_id} is flagged as unusable for event windows")
    ok = obs[obs["status"] == "ok"].set_index("date")["value"]
    return make_series(series_id, ok, str(meta["obs_time"]), str(meta["tz"]),
                       kind_for(str(meta["units"])), assumed_time)


def window_changes(s: WindowSeries, events: pd.Series, horizons: int, pre_lags: int = 5,
                   max_gap_days: float = 7.0) -> pd.DataFrame:
    """One row per event (index preserved): d0..dH, pre1..preL, before/day-0 dates."""
    if events.dt.tz is None:
        raise ValueError("event timestamps must be timezone-aware")
    obs_ns = utc_ns(s.obs_ts)
    ev_ns = utc_ns(events)
    n = len(obs_ns)
    after = np.searchsorted(obs_ns, ev_ns, side="right")
    before = np.searchsorted(obs_ns, ev_ns, side="left") - 1
    gap = int(max_gap_days * 86400e9)
    valid = (before >= 0) & (after < n)
    valid &= np.where(valid, ev_ns - obs_ns[np.clip(before, 0, n - 1)] <= gap, False)
    valid &= np.where(valid, obs_ns[np.clip(after, 0, n - 1)] - ev_ns <= gap, False)

    def take(pos: np.ndarray) -> np.ndarray:
        ok = valid & (pos >= 0) & (pos < n)
        return np.where(ok, s.x[np.clip(pos, 0, n - 1)], np.nan)

    base = take(before)
    out: dict[str, np.ndarray] = {}
    for h in range(horizons + 1):
        out[f"d{h}"] = take(after + h) - base
    for lag in range(1, pre_lags + 1):
        out[f"pre{lag}"] = take(before - lag + 1) - take(before - lag)
    df = pd.DataFrame(out, index=events.index)
    nat = np.full(len(events), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    before_dates, day0_dates = nat.copy(), nat.copy()
    dates = s.dates.to_numpy(dtype="datetime64[ns]")
    before_dates[valid] = dates[before[valid]]
    day0_dates[valid] = dates[after[valid]]
    df.insert(0, "day0_date", day0_dates)
    df.insert(0, "before_date", before_dates)
    return df
