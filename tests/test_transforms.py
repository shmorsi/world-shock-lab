"""Property tests for the pure data transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wsl.data.transforms import datetimes_to_ns, diff_bp, log_change_pct, total_return_index

prices = st.lists(st.floats(min_value=0.5, max_value=500.0, allow_nan=False), min_size=2,
                  max_size=60)


def _series(vals: list[float]) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=len(vals))
    return pd.Series(vals, index=idx, dtype="float64")


def _no_divs() -> pd.Series:
    return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))


@given(prices)
def test_tr_without_dividends_tracks_price(vals: list[float]) -> None:
    close = _series(vals)
    tr = total_return_index(close, _no_divs())
    np.testing.assert_allclose(tr.to_numpy(), 100.0 * close.to_numpy() / close.iloc[0],
                               rtol=1e-10)


@given(prices, st.floats(min_value=0.1, max_value=1000.0))
def test_tr_is_invariant_to_price_scale(vals: list[float], k: float) -> None:
    close = _series(vals)
    divs = pd.Series([0.3], index=[close.index[len(vals) // 2]])
    a = total_return_index(close, divs)
    b = total_return_index(close * k, divs * k)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-9)


@given(prices, st.floats(min_value=0.01, max_value=5.0))
def test_dividend_raises_tr_relative_to_price(vals: list[float], d: float) -> None:
    close = _series(vals)
    ex = close.index[-1]
    tr = total_return_index(close, pd.Series([d], index=[ex]))
    base = total_return_index(close, _no_divs())
    assert tr.iloc[-1] > base.iloc[-1]
    np.testing.assert_allclose(tr.iloc[:-1].to_numpy(), base.iloc[:-1].to_numpy())


@settings(max_examples=50)
@given(prices, st.data())
def test_missing_closes_stay_missing_and_are_not_filled(vals: list[float],
                                                        data: st.DataObject) -> None:
    close = _series(vals)
    hole = data.draw(st.integers(min_value=1, max_value=len(vals) - 1))
    close.iloc[hole] = np.nan
    tr = total_return_index(close, _no_divs())
    assert np.isnan(tr.iloc[hole])
    assert tr.drop(close.index[hole]).notna().all()


def test_dividend_on_missing_day_credited_to_next_close() -> None:
    close = _series([10.0, np.nan, 10.0])
    divs = pd.Series([1.0], index=[close.index[1]])
    tr = total_return_index(close, divs)
    assert tr.iloc[2] == pytest.approx(110.0)


def test_rejects_non_positive_prices() -> None:
    with pytest.raises(ValueError):
        total_return_index(_series([1.0, 0.0]), _no_divs())


def test_log_change_undefined_for_non_positive_levels() -> None:
    s = _series([50.0, -10.0, 20.0, 25.0])  # e.g. negative WTI print
    out = log_change_pct(s)
    assert np.isnan(out.iloc[1]) and np.isnan(out.iloc[2])
    assert out.iloc[3] == pytest.approx(100 * np.log(25 / 20))


@given(st.lists(st.floats(min_value=-5, max_value=20, allow_nan=False), min_size=2, max_size=40))
def test_diff_bp_sums_to_total_change(vals: list[float]) -> None:
    s = _series(vals)
    assert diff_bp(s).sum() == pytest.approx(100 * (vals[-1] - vals[0]), abs=1e-6)


def test_datetimes_to_ns_keeps_timezone() -> None:
    df = pd.DataFrame({"ts": pd.to_datetime(["2020-01-01 14:00"]).tz_localize("America/New_York"),
                       "d": pd.to_datetime(["2020-01-01"]), "x": [1]})
    out = datetimes_to_ns(df)
    assert str(out["ts"].dtype) == "datetime64[ns, America/New_York]"
    assert str(out["d"].dtype) == "datetime64[ns]"
