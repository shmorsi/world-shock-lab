"""Placebo tests: the real surprises reassigned to random non-announcement days.

Design (protocol §9: 1,000 draws, seed 20261012, 5 days around events excluded). Each draw
picks as many eligible trading days as there are real events, assigns them the real
surprises in random order, and re-estimates the event-sample LP. Eligible days are the
outcome's trading days at least 5 trading days away from any announcement of the same
family and not the day-0 of any other family's announcement. If the real shock matters,
the real coefficient should sit far out in the placebo distribution; the share of placebo
draws at least as large in absolute value is a permutation p-value.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wsl.shocks.windows import WindowSeries, window_changes
from wsl.timing import utc_ns

DRAWS = 1000
SEED = 20261012
EXCLUDE = 5


@dataclass(frozen=True)
class PlaceboResult:
    horizon: int
    actual: float
    placebo: np.ndarray  # coefficients, one per draw

    @property
    def p_value(self) -> float:
        return float((np.abs(self.placebo) >= abs(self.actual)).mean())

    @property
    def placebo_mean(self) -> float:
        return float(self.placebo.mean())

    @property
    def placebo_sd(self) -> float:
        return float(self.placebo.std(ddof=1))


def _coef(y: np.ndarray, X: np.ndarray) -> float:
    keep = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    return float(beta[1])


def eligible_days(s: WindowSeries, same_family: pd.Series,
                  other_families: Sequence[pd.Series] = ()) -> pd.DatetimeIndex:
    """Trading days of ``s`` far enough from announcements (see module docstring), within
    the date range spanned by ``same_family``."""
    obs_ns = utc_ns(s.obs_ts)
    n = len(obs_ns)
    bad = np.zeros(n, dtype=bool)
    same_pos = np.searchsorted(obs_ns, utc_ns(same_family), side="right")
    for p in same_pos:
        bad[max(0, p - EXCLUDE):min(n, p + EXCLUDE + 1)] = True
    for fam in other_families:
        pos = np.searchsorted(obs_ns, utc_ns(fam), side="right")
        bad[pos[pos < n]] = True
    bad[: EXCLUDE + 1] = True
    fam_ns = utc_ns(same_family)
    bad |= (obs_ns < fam_ns.min()) | (obs_ns > fam_ns.max())
    return s.dates[~bad]


def placebo_test(
    s: WindowSeries,
    events: pd.DataFrame,
    shocks: Sequence[str],
    actual: dict[int, float],
    *,
    per: float = 1.0,
    other_families: Sequence[pd.Series] = (),
    draws: int = DRAWS,
    seed: int = SEED,
) -> list[PlaceboResult]:
    """``shocks[0]`` is the shock of interest; the rest are controls (e.g. CBI for MP).

    ``events`` must be the events actually used in the real regression (valid windows), so
    each placebo draw has the same number of observations."""
    horizons = sorted(actual)
    days = eligible_days(s, events["ts"], other_families)
    # A placebo "announcement" just before that day's observation: day 0 is that day.
    ts = pd.Series(s.obs_ts[s.dates.isin(days)] - pd.Timedelta(minutes=1))
    w = window_changes(s, ts, max(horizons), pre_lags=5)
    pre = w[[f"pre{lag}" for lag in range(1, 6)]].to_numpy()
    scale = np.array([per] + [1.0] * (len(shocks) - 1))
    shock_mat = events[list(shocks)].dropna().to_numpy() / scale
    n_ev = shock_mat.shape[0]
    rng = np.random.default_rng(seed)
    out = {h: np.empty(draws) for h in horizons}
    for d in range(draws):
        pick = rng.choice(len(w), size=n_ev, replace=False)
        perm = shock_mat[rng.permutation(n_ev)]
        X = np.column_stack([np.ones(n_ev), perm, pre[pick]])
        for h in horizons:
            out[h][d] = _coef(w[f"d{h}"].to_numpy()[pick], X)
    return [PlaceboResult(h, actual[h], out[h]) for h in horizons]
