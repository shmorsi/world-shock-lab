"""Least squares with robust covariance estimators.

Covariances are sandwich estimators V = B⁻¹ S B⁻¹ with bread B = X'X and meat S built from
the scores x_i u_i:

* HC0 / HC1: Eicker-Huber-White (HC1 scales by n / (n - k));
* HC3: scores scaled by 1 / (1 - h_ii) (MacKinnon-White), the default for small event samples;
* NW: Newey-West with Bartlett weights w_l = 1 - l / (L + 1), l = 1..L;
* CL: clustered by group (CR1: G/(G-1) · (n-1)/(n-k)), e.g. one cluster per announcement in
  an event panel. With one observation per time period and no lags this equals Driscoll-Kraay
  with zero lags.

Rows with any missing value in y or X are dropped (listwise) and the count is reported;
nothing is imputed. Cross-checked against statsmodels in tests/test_ols.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

CovType = Literal["HC0", "HC1", "HC3", "NW", "CL"]
Array = NDArray[np.float64]


@dataclass(frozen=True)
class OLSResult:
    params: Array
    cov: Array
    nobs: int
    resid: Array
    names: tuple[str, ...]
    cov_type: str

    @property
    def se(self) -> Array:
        return np.sqrt(np.diag(self.cov))

    def coef(self, name: str) -> float:
        return float(self.params[self.names.index(name)])

    def stderr(self, name: str) -> float:
        return float(self.se[self.names.index(name)])


def newey_west_meat(scores: Array, lags: int) -> Array:
    """Σ_t s_t s_t' + Σ_{l=1..L} w_l Σ_t (s_t s_{t-l}' + s_{t-l} s_t'), Bartlett weights.

    ``scores`` must be in time order with no gaps other than dropped rows; with
    ``lags = 0`` this is the HC0 meat.
    """
    if lags < 0:
        raise ValueError("lags must be >= 0")
    meat = scores.T @ scores
    n = scores.shape[0]
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += w * (gamma + gamma.T)
    return meat


def complete_rows(*arrays: Array) -> NDArray[np.bool_]:
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for a in arrays:
        mask &= ~np.isnan(a).reshape(a.shape[0], -1).any(axis=1)
    return mask


def ols(
    y: Array,
    X: Array,
    *,
    cov_type: CovType = "HC3",
    nw_lags: int | None = None,
    names: tuple[str, ...] | None = None,
    df_correction: bool = False,
    groups: NDArray[np.int64] | None = None,
) -> OLSResult:
    """OLS of y on X (X must already contain a constant if one is wanted)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    keep = complete_rows(y, X)
    y, X = y[keep], X[keep]
    g = np.asarray(groups)[keep] if groups is not None else None
    n, k = X.shape
    if n <= k:
        raise ValueError(f"not enough observations: n={n}, k={k}")
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    scores = X * u[:, None]
    if cov_type in ("HC0", "HC1"):
        meat = scores.T @ scores
        if cov_type == "HC1":
            meat *= n / (n - k)
    elif cov_type == "HC3":
        h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
        s3 = scores / (1.0 - h)[:, None]
        meat = s3.T @ s3
    elif cov_type == "CL":
        if g is None:
            raise ValueError("CL covariance needs groups")
        ids, inv = np.unique(g, return_inverse=True)
        summed = np.zeros((len(ids), k))
        np.add.at(summed, inv, scores)
        G = len(ids)
        meat = summed.T @ summed * (G / (G - 1)) * ((n - 1) / (n - k))
    elif cov_type == "NW":
        if nw_lags is None:
            raise ValueError("NW covariance needs nw_lags")
        meat = newey_west_meat(scores, nw_lags)
        if df_correction:
            meat *= n / (n - k)
    else:
        raise ValueError(f"unknown cov_type {cov_type!r}")
    cov = xtx_inv @ meat @ xtx_inv
    return OLSResult(params=beta, cov=cov, nobs=n, resid=u,
                     names=names or tuple(f"x{i}" for i in range(k)), cov_type=cov_type)
