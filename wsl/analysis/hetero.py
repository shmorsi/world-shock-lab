"""Heteroskedasticity-based checks of a daily-window policy proxy (Rigobon & Sack 2004).

Announcement days are compared with control days: the same weekday and clock time one week
earlier, unless that control falls within a day of another announcement, in which case the
pair is dropped. If announcements move the proxy, its variance is higher on announcement
days. The variance ratio measures that; 1 - 1/ratio is the share of announcement-day
variance attributable to the announcement.

The Rigobon-Sack estimator uses the same difference to identify the effect of the policy
news on an outcome without assuming the proxy is free of other news:
    β = [Cov_E(Δy, Δi) - Cov_C(Δy, Δi)] / [Var_E(Δi) - Var_C(Δi)]
Confidence intervals come from a pairs bootstrap (events resampled with their controls).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wsl.timing import utc_ns

BOOT = 5000
SEED = 20261012


def control_times(events: pd.Series, all_announcements: pd.Series,
                  weeks: int = 1) -> pd.Series:
    """Event time minus ``weeks`` weeks; NaT where that falls within 1 day of any
    announcement (of any family)."""
    ctrl = events - pd.Timedelta(weeks=weeks)
    ann = np.sort(utc_ns(all_announcements))
    c = utc_ns(ctrl)
    i = np.clip(np.searchsorted(ann, c), 1, len(ann) - 1)
    nearest = np.minimum(np.abs(ann[i] - c), np.abs(ann[i - 1] - c))
    return ctrl.where(nearest > pd.Timedelta(days=1).value)


@dataclass(frozen=True)
class VarianceRatio:
    n_pairs: int
    var_event: float
    var_control: float
    ratio: float
    ci90: tuple[float, float]

    @property
    def signal_share(self) -> float:
        return 1.0 - 1.0 / self.ratio


def variance_ratio(event_changes: np.ndarray, control_changes: np.ndarray,
                   boot: int = BOOT, seed: int = SEED) -> VarianceRatio:
    keep = ~np.isnan(event_changes) & ~np.isnan(control_changes)
    e, c = event_changes[keep], control_changes[keep]
    rng = np.random.default_rng(seed)
    draws = np.empty(boot)
    for b in range(boot):
        i = rng.integers(0, len(e), len(e))
        draws[b] = np.var(e[i], ddof=1) / np.var(c[i], ddof=1)
    ratio = float(np.var(e, ddof=1) / np.var(c, ddof=1))
    lo, hi = np.quantile(draws, [0.05, 0.95])
    return VarianceRatio(int(keep.sum()), float(np.var(e, ddof=1)), float(np.var(c, ddof=1)),
                         ratio, (float(lo), float(hi)))


@dataclass(frozen=True)
class RigobonSack:
    n_pairs: int
    beta: float
    ci90: tuple[float, float]
    ols_event_days: float


def rigobon_sack(i_e: np.ndarray, y_e: np.ndarray, i_c: np.ndarray, y_c: np.ndarray,
                 boot: int = BOOT, seed: int = SEED) -> RigobonSack:
    keep = ~np.isnan(i_e) & ~np.isnan(y_e) & ~np.isnan(i_c) & ~np.isnan(y_c)
    i_e, y_e, i_c, y_c = i_e[keep], y_e[keep], i_c[keep], y_c[keep]

    def est(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
        num = np.cov(b, a)[0, 1] - np.cov(d, c)[0, 1]
        den = np.var(a, ddof=1) - np.var(c, ddof=1)
        return float(num / den)

    rng = np.random.default_rng(seed)
    draws = np.empty(boot)
    for k in range(boot):
        j = rng.integers(0, len(i_e), len(i_e))
        draws[k] = est(i_e[j], y_e[j], i_c[j], y_c[j])
    lo, hi = np.quantile(draws, [0.05, 0.95])
    ols = float(np.cov(y_e, i_e)[0, 1] / np.var(i_e, ddof=1))
    return RigobonSack(int(keep.sum()), est(i_e, y_e, i_c, y_c), (float(lo), float(hi)), ols)
