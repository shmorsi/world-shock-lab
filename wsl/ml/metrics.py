"""Forecast metrics (protocol §6) and the event-block bootstrap (§7)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    """Gneiting & Raftery (2007) interval score for a central (1 - alpha) interval."""
    out = (hi - lo) + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)
    return np.asarray(out, dtype=float)


def cell_metrics(y: np.ndarray, pred: np.ndarray, half68: np.ndarray, half90: np.ndarray,
                 zero_mae_train: float, p25_abs_train: float) -> dict[str, float]:
    err = pred - y
    big = np.abs(y) > p25_abs_train
    return {
        "n": float(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "scaled_mae": float(np.mean(np.abs(err)) / zero_mae_train),
        "oos_r2_vs_zero": float(1.0 - np.sum(err**2) / np.sum(y**2)),
        "directional_accuracy": float(np.mean(np.sign(pred[big]) == np.sign(y[big])))
        if big.any() else float("nan"),
        "directional_accuracy_all": float(np.mean(np.sign(pred) == np.sign(y))),
        "coverage68": float(np.mean(np.abs(err) <= half68)),
        "coverage90": float(np.mean(np.abs(err) <= half90)),
        "interval_score90_scaled": float(np.mean(
            interval_score(y, pred - half90, pred + half90, 0.10)) / zero_mae_train),
    }


def pooled_scaled_mae(abs_err: pd.DataFrame, scale: pd.Series) -> float:
    """``abs_err``: events x cells (NaN where the cell has no target). Equal weight per cell."""
    per_cell = abs_err.mean(axis=0) / scale[abs_err.columns]
    return float(per_cell.mean())


def bootstrap_difference(err_a: pd.DataFrame, err_b: pd.DataFrame, scale: pd.Series,
                         resamples: int, seed: int, ci: float = 0.90
                         ) -> tuple[float, tuple[float, float], np.ndarray]:
    """Event-block bootstrap of pooled scaled MAE(a) - pooled scaled MAE(b): events are
    resampled with replacement, keeping every cell of an event together."""
    a, b = err_a.to_numpy(), err_b.to_numpy()
    s = scale[err_a.columns].to_numpy()
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    diffs = np.empty(resamples)
    for r in range(resamples):
        i = rng.integers(0, n, n)
        diffs[r] = np.mean(np.nanmean(a[i], axis=0) / s) - np.mean(np.nanmean(b[i], axis=0) / s)
    point = pooled_scaled_mae(err_a, scale) - pooled_scaled_mae(err_b, scale)
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return point, (float(lo), float(hi)), diffs


def verdict(ci: tuple[float, float]) -> str:
    if ci[1] < 0:
        return "ML adds value"
    if ci[0] > 0:
        return "ML is worse"
    return "No evidence ML adds value"
