"""Timestamp-aware event windows (data/SOURCES.md §5).

A daily series observed at a known local time (e.g. the 12:00 ET H.10 FX fix) is turned into
timestamps. For an announcement at time A, the event window of that series runs from the last
observation strictly before A to the first observation strictly after A. This is what makes an
FOMC statement at 14:00 ET show up in noon-fixed FX one day later than in 16:00 equity closes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class UnknownObservationTime(ValueError):
    pass


def observation_timestamps(
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    obs_time: str,
    tz: str,
    assumed_time: str | None = None,
) -> pd.DatetimeIndex:
    """Local observation timestamps (tz-aware) for a daily series.

    ``obs_time`` must be "HH:MM". For series whose time is not documented ("close_unknown")
    the caller must pass an explicit ``assumed_time``; that assumption is the caller's to
    document and sensitivity-test. Non-market series ("daily_rate", "intraday_average") are
    rejected: they cannot define an event window.
    """
    time = obs_time
    if obs_time == "close_unknown":
        if assumed_time is None:
            raise UnknownObservationTime("observation time undocumented; pass assumed_time")
        time = assumed_time
    elif obs_time in {"daily_rate", "intraday_average"}:
        raise UnknownObservationTime(f"{obs_time} series cannot define an event window")
    hh, mm = (int(x) for x in time.split(":"))
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
    stamped = idx + pd.Timedelta(hours=hh, minutes=mm)
    return stamped.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")


def window_positions(
    obs_ts: pd.DatetimeIndex, announcement: pd.Timestamp
) -> tuple[int | None, int | None]:
    """(last position strictly before, first position strictly after) the announcement.

    ``obs_ts`` must be tz-aware and strictly increasing; ``announcement`` must be tz-aware.
    Either element is None when no such observation exists.
    """
    if obs_ts.tz is None or announcement.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    if not obs_ts.is_monotonic_increasing or obs_ts.has_duplicates:
        raise ValueError("observation timestamps must be strictly increasing")
    after = int(obs_ts.searchsorted(announcement, side="right"))
    before = int(obs_ts.searchsorted(announcement, side="left")) - 1
    return (before if before >= 0 else None, after if after < len(obs_ts) else None)


def utc_ns(ts: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Timezone-aware timestamps as int64 nanoseconds since the epoch (UTC), whatever the
    stored resolution (pandas 3 may store s/ms/us)."""
    idx = pd.DatetimeIndex(ts)
    if idx.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    arr = idx.tz_convert("UTC").as_unit("ns").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    return arr.astype("int64")
