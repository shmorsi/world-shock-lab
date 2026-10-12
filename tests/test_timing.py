"""Event-window alignment: properties and the FOMC-vs-noon-FX case from data/SOURCES.md §5."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from wsl.timing import UnknownObservationTime, observation_timestamps, window_positions

NY = "America/New_York"


def test_fomc_at_14h_is_after_noon_fx_fix_same_day() -> None:
    dates = pd.bdate_range("2022-03-14", "2022-03-18")
    fx = observation_timestamps(dates, "12:00", NY)
    eq = observation_timestamps(dates, "16:00", NY)
    fomc = pd.Timestamp("2022-03-16 14:00", tz=NY)
    b, a = window_positions(fx, fomc)
    assert (dates[b].date(), dates[a].date()) == (pd.Timestamp("2022-03-16").date(),
                                                  pd.Timestamp("2022-03-17").date())
    b, a = window_positions(eq, fomc)
    assert (dates[b].date(), dates[a].date()) == (pd.Timestamp("2022-03-15").date(),
                                                  pd.Timestamp("2022-03-16").date())


def test_sunday_announcement_maps_to_monday() -> None:
    dates = pd.bdate_range("2020-03-09", "2020-03-20")
    eq = observation_timestamps(dates, "16:00", NY)
    b, a = window_positions(eq, pd.Timestamp("2020-03-15 17:00", tz=NY))
    assert dates[b] == pd.Timestamp("2020-03-13") and dates[a] == pd.Timestamp("2020-03-16")


def test_unknown_close_requires_explicit_assumption() -> None:
    dates = pd.bdate_range("2020-01-01", periods=3)
    with pytest.raises(UnknownObservationTime):
        observation_timestamps(dates, "close_unknown", "America/Toronto")
    ts = observation_timestamps(dates, "close_unknown", "America/Toronto", assumed_time="16:00")
    assert ts[0].hour == 16


@pytest.mark.parametrize("kind", ["daily_rate", "intraday_average"])
def test_non_market_series_cannot_define_windows(kind: str) -> None:
    with pytest.raises(UnknownObservationTime):
        observation_timestamps(pd.bdate_range("2020-01-01", periods=2), kind, NY)


@given(st.integers(min_value=0, max_value=60 * 24 * 30))
def test_window_brackets_announcement(minutes: int) -> None:
    dates = pd.bdate_range("2021-01-04", periods=25)
    obs = observation_timestamps(dates, "16:00", NY)
    ann = pd.Timestamp("2021-01-01 00:00", tz=NY) + pd.Timedelta(minutes=minutes)
    b, a = window_positions(obs, ann)
    if b is not None:
        assert obs[b] < ann
        assert b == len(obs) - 1 or obs[b + 1] >= ann
    if a is not None:
        assert obs[a] > ann
        assert a == 0 or obs[a - 1] <= ann
    if b is not None and a is not None:
        assert a - b in (1, 2)  # 2 only when an observation coincides with the announcement


def test_rejects_naive_timestamps() -> None:
    obs = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=3))
    with pytest.raises(ValueError):
        window_positions(obs, pd.Timestamp("2021-01-05 10:00"))
