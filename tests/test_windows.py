"""Event windows on synthetic series with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from wsl.shocks.windows import make_series, window_changes
from wsl.timing import utc_ns

NY = "America/New_York"


def _linear(n: int = 60, unit: str = "ns") -> pd.Series:
    dates = pd.bdate_range("2024-01-02", periods=n).as_unit(unit)
    return pd.Series(np.arange(n, dtype=float), index=dates)  # level i on day i (pct)


def test_linear_series_gives_known_changes() -> None:
    s = make_series("X", _linear(), "16:00", NY, "bp")
    ev = pd.Series([pd.Timestamp("2024-01-10 10:00", tz=NY)])  # day index 6
    w = window_changes(s, ev, 3, pre_lags=2)
    # before = day 5 close, day 0 = day 6 close: d_h = 100 * (h + 1) bp; each pre change = 100
    np.testing.assert_allclose(w.loc[0, ["d0", "d1", "d2", "d3"]].to_numpy(float),
                               [100, 200, 300, 400])
    np.testing.assert_allclose(w.loc[0, ["pre1", "pre2"]].to_numpy(float), [100, 100])
    assert w.loc[0, "day0_date"] == pd.Timestamp("2024-01-10")


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_timestamp_resolution_does_not_matter(unit: str) -> None:
    """pandas 3 can store dates at s/ms/us resolution; windows must not change."""
    ref = window_changes(make_series("X", _linear(), "16:00", NY, "bp"),
                         pd.Series([pd.Timestamp("2024-01-10 10:00", tz=NY)]), 2)
    s = make_series("X", _linear(unit=unit), "16:00", NY, "bp")
    ev = pd.Series([pd.Timestamp("2024-01-10 10:00", tz=NY).as_unit(unit)])
    pd.testing.assert_frame_equal(window_changes(s, ev, 2), ref)


def test_event_timezone_is_respected() -> None:
    s = make_series("X", _linear(), "12:00", NY, "bp")  # noon fix, like H.10 FX
    in_ny = pd.Series([pd.Timestamp("2024-01-10 14:00", tz=NY)])
    in_utc = pd.Series([pd.Timestamp("2024-01-10 19:00", tz="UTC")])  # the same instant
    for ev in (in_ny, in_utc):
        w = window_changes(s, ev, 1)
        assert w.loc[0, "day0_date"] == pd.Timestamp("2024-01-11")  # 14:00 is after noon


def test_events_outside_the_series_are_left_empty() -> None:
    s = make_series("X", _linear(20), "16:00", NY, "bp")
    ev = pd.Series([pd.Timestamp("2023-06-01 10:00", tz=NY), pd.Timestamp("2025-06-01", tz=NY)])
    w = window_changes(s, ev, 2)
    assert w[["d0", "d1", "d2"]].isna().all().all()
    assert w["before_date"].isna().all()


def test_non_positive_prices_leave_log_windows_empty() -> None:
    v = _linear(15) + 10.0
    v.iloc[7] = -5.0  # e.g. WTI on 2020-04-20
    s = make_series("P", v, "16:00", NY, "logpct")
    ev = pd.Series([pd.Timestamp(v.index[7], tz=NY) + pd.Timedelta(hours=10),
                    pd.Timestamp(v.index[10], tz=NY) + pd.Timedelta(hours=10)])
    w = window_changes(s, ev, 1)
    assert np.isnan(w.loc[0, "d0"])
    assert w.loc[1, "d0"] == pytest.approx(100 * np.log(v.iloc[10] / v.iloc[9]))


@given(st.integers(min_value=0, max_value=59 * 24 * 60))
def test_day0_is_first_observation_after_the_event(minutes: int) -> None:
    s = make_series("X", _linear(), "16:00", NY, "bp")
    t = pd.Timestamp("2024-01-02 00:00", tz=NY) + pd.Timedelta(minutes=minutes)
    w = window_changes(s, pd.Series([t]), 0, max_gap_days=1e9)
    if pd.isna(w.loc[0, "day0_date"]):
        return
    day0 = pd.Timestamp(w.loc[0, "day0_date"]).tz_localize(NY) + pd.Timedelta(hours=16)
    before = pd.Timestamp(w.loc[0, "before_date"]).tz_localize(NY) + pd.Timedelta(hours=16)
    assert before < t < day0
    assert (day0 - before) <= pd.Timedelta(days=4)  # adjacent business days


def test_utc_ns_is_resolution_and_zone_invariant() -> None:
    a = pd.DatetimeIndex([pd.Timestamp("2024-03-10 14:00", tz=NY)]).as_unit("s")
    b = pd.DatetimeIndex([pd.Timestamp("2024-03-10 18:00", tz="UTC")]).as_unit("ns")
    assert utc_ns(a)[0] == utc_ns(b)[0]
    with pytest.raises(ValueError):
        utc_ns(pd.DatetimeIndex(["2024-01-01"]))
