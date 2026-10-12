"""Pure data transforms. Property-tested in tests/test_transforms.py."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd


def total_return_index(
    close: pd.Series, dividend: pd.Series, base: float = 100.0
) -> pd.Series:
    """Total-return index from (split-adjusted) closes and cash dividends per share.

    ``close`` is indexed by trading date and may contain NaN (missing at source).
    ``dividend`` is indexed by ex-date. Between two consecutive *available* closes at dates
    p < t, the gross return is ``(close_t + D) / close_p`` where D sums the dividends with
    ex-date in (p, t]. A dividend whose ex-date has no close is therefore credited to the next
    available close. Missing closes stay NaN in the output; nothing is filled.
    """
    if not close.index.is_monotonic_increasing:
        raise ValueError("close must be sorted by date")
    if not isinstance(dividend.index, pd.DatetimeIndex):
        dividend = pd.Series(dividend.to_numpy(dtype="float64"),
                             index=pd.DatetimeIndex(dividend.index))
    values = close.to_numpy(dtype="float64")
    out = np.full(len(values), np.nan)
    avail = np.flatnonzero(~np.isnan(values))
    if avail.size == 0:
        return pd.Series(out, index=close.index)
    if (values[avail] <= 0).any():
        raise ValueError("total_return_index needs strictly positive prices")
    dates = close.index.to_numpy()
    divs = dividend.dropna()
    div_dates, div_amounts = divs.index.to_numpy(), divs.to_numpy(dtype="float64")
    level = base
    out[avail[0]] = level
    for prev, cur in pairwise(avail):
        in_window = (div_dates > dates[prev]) & (div_dates <= dates[cur])
        level *= (values[cur] + float(div_amounts[in_window].sum())) / values[prev]
        out[cur] = level
    return pd.Series(out, index=close.index)
    if (avail <= 0).any():
        raise ValueError("total_return_index needs strictly positive prices")
    divs = dividend.dropna()
    level = base
    out.loc[avail.index[0]] = level
    prev_date, prev_close = avail.index[0], float(avail.iloc[0])
    for date, px in avail.iloc[1:].items():
        d = float(divs[(divs.index > prev_date) & (divs.index <= date)].sum())
        level *= (float(px) + d) / prev_close
        out.loc[date] = level
        prev_date, prev_close = date, float(px)
    return out


def datetimes_to_ns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast every datetime column to nanosecond resolution (pandas 3 infers us/s from text)."""
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            out[col] = out[col].astype(pd.DatetimeTZDtype(unit="ns", tz=dtype.tz))
        elif pd.api.types.is_datetime64_dtype(dtype):
            out[col] = out[col].astype("datetime64[ns]")
    return out


def log_change_pct(level: pd.Series) -> pd.Series:
    """100 × log change between consecutive available values; undefined (NaN) for prices <= 0."""
    lvl = level.dropna()
    pos = lvl.where(lvl > 0)
    out: pd.Series = (100.0 * np.log(pos).diff()).reindex(level.index)
    return out


def diff_bp(level_pct: pd.Series) -> pd.Series:
    """Change in basis points between consecutive available values of a series in percent."""
    lvl = level_pct.dropna()
    return (100.0 * lvl.diff()).reindex(level_pct.index)
