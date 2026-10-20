"""Scores for the ML study. The mean-response bootstrap is the M5 one, reused as-is."""

from __future__ import annotations

import numpy as np

from wsl.ml.metrics import verdict as _mean_verdict


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """Pinball (quantile) loss. Lower is better. Proper for the τ-quantile."""
    err = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    t = float(tau)
    return np.asarray(np.where(err >= 0.0, t * err, (t - 1.0) * err), dtype=float)


def fissler_ziegel(y: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float = 0.05
                   ) -> np.ndarray:
    """Strictly consistent score for the lower-tail pair (VaR, expected shortfall). Lower is better.

    Fissler and Ziegel (2016), with an increasing g(x) = x and φ(e) = exp(e). The exponential
    keeps φ' positive when the shortfall is negative, which it is for returns. An additive
    piece that depends only on y is left out; it does not change which forecast ranks better.
    Secondary metric, not the decision.
    """
    a = float(alpha)
    yy = np.asarray(y, dtype=float)
    v = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    ind = (yy <= v).astype(float)
    pe = np.exp(e)
    score = (ind - a) * (v - yy) + pe * (e - v + (1.0 / a) * ind * (v - yy)) - pe
    return np.asarray(score, dtype=float)


def tail_verdict(ci: tuple[float, float]) -> str:
    base = _mean_verdict(ci)
    if base == "ML adds value":
        return "ML adds value on the 5% tail"
    if base == "ML is worse":
        return "ML is worse on the 5% tail"
    return "No evidence ML adds value on the 5% tail"
