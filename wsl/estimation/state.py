"""State dependence: does the same shock have a different effect in different regimes?

Stacked event-sample LP. For every horizon h = 0..H, on the same set of events,
    Δ_h y_e = a_h + b_h s_e + Σ_k c_hk (s_e · R_ke) + Σ_k d_hk R_ke + W_e γ_h + u_he,
where R_k are regime dummies measured before the event (one baseline regime omitted).
The c_hk are the differences in the shock's effect between regimes.

Two tests, both with p-values from a wild cluster bootstrap (Rademacher weights drawn per
event and shared across horizons, so the cross-horizon correlation is kept; the null of no
interaction is imposed):
  * joint: all c_hk = 0 for every h and k;
  * headline: c_hk = 0 at the headline horizon.
Asymptotic Wald tests with many highly correlated restrictions and a few hundred clusters
over-reject; the bootstrap keeps the size (checked in tests/test_state.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
BOOT = 999
SEED = 20261012


@dataclass(frozen=True)
class StateResult:
    regimes: tuple[str, ...]  # the non-baseline regime names, one per interaction
    horizons: tuple[int, ...]
    base_effect: Array  # (H+1,) effect in the baseline regime
    interaction: Array  # (H+1, K) difference vs baseline
    interaction_se: Array  # (H+1, K) cluster-robust
    n_events: int
    n_by_regime: dict[str, int]
    p_joint: float
    p_headline: float
    headline: int

    def effect_in(self, regime_index: int) -> Array:
        """Effect of the shock in non-baseline regime k (base + interaction)."""
        return self.base_effect + self.interaction[:, regime_index]

    @property
    def mde_headline(self) -> Array:
        """Minimum detectable interaction at the headline horizon (10% two-sided test,
        80% power): (1.645 + 0.842) · s.e."""
        h = self.horizons.index(self.headline)
        return np.asarray(2.487 * self.interaction_se[h], dtype=float)


def _fit(Y: Array, X: Array, xtx_inv: Array) -> tuple[Array, Array]:
    B = xtx_inv @ X.T @ Y
    return B, Y - X @ B


def _wald(B: Array, U: Array, X: Array, xtx_inv: Array, idx: Sequence[int],
          cols: Sequence[int]) -> float:
    """Cluster-robust (one cluster per event) Wald statistic for B[idx, cols] = 0 jointly."""
    n, k = X.shape
    theta = np.concatenate([B[list(idx), c] for c in cols])
    # Influence of event e on theta: (X'X)^-1 x_e u_ec, for the selected rows and columns.
    infl = np.concatenate([(X @ xtx_inv[:, list(idx)]) * U[:, [c]] for c in cols], axis=1)
    V = infl.T @ infl * (n / (n - k))
    return float(theta @ np.linalg.pinv(V) @ theta)


def state_lp(
    Y: Array,
    shock: Array,
    regimes: Array,
    regime_names: Sequence[str],
    controls: Array | None,
    horizons: Sequence[int],
    headline: int,
    boot: int = BOOT,
    seed: int = SEED,
) -> StateResult:
    """``Y``: (N, H+1) outcome changes; ``regimes``: (N, K) 0/1 dummies for the non-baseline
    regimes; rows with any missing value are dropped (common sample across horizons)."""
    shock = shock.reshape(-1, 1)
    parts = [np.ones_like(shock), shock, shock * regimes, regimes]
    if controls is not None:
        parts.append(controls)
    X = np.column_stack(parts)
    keep = ~np.isnan(Y).any(axis=1) & ~np.isnan(X).any(axis=1)
    Y, X, R = Y[keep], X[keep], regimes[keep]
    n, K = X.shape[0], R.shape[1]
    xtx_inv = np.linalg.inv(X.T @ X)
    B, U = _fit(Y, X, xtx_inv)
    inter_idx = list(range(2, 2 + K))
    all_cols = list(range(Y.shape[1]))
    h_col = list(horizons).index(headline)
    w_joint = _wald(B, U, X, xtx_inv, inter_idx, all_cols)
    w_head = _wald(B, U, X, xtx_inv, inter_idx, [h_col])

    # Restricted model (no interactions) for the bootstrap null.
    Xr = np.delete(X, inter_idx, axis=1)
    xr_inv = np.linalg.inv(Xr.T @ Xr)
    Br, Ur = _fit(Y, Xr, xr_inv)
    fitted_r = Xr @ Br
    rng = np.random.default_rng(seed)
    exceed_joint = exceed_head = 0
    for _ in range(boot):
        w = rng.choice([-1.0, 1.0], size=(n, 1))
        Ys = fitted_r + w * Ur
        Bs, Us = _fit(Ys, X, xtx_inv)
        exceed_joint += _wald(Bs, Us, X, xtx_inv, inter_idx, all_cols) >= w_joint
        exceed_head += _wald(Bs, Us, X, xtx_inv, inter_idx, [h_col]) >= w_head

    infl = X @ xtx_inv
    se = np.empty((Y.shape[1], K))
    for c in range(Y.shape[1]):
        scores = infl * U[:, [c]]
        V = scores.T @ scores * (n / (n - X.shape[1]))
        se[c] = np.sqrt(np.diag(V)[inter_idx])
    counts = {name: int(R[:, k].sum()) for k, name in enumerate(regime_names)}
    counts["baseline"] = int((R.sum(axis=1) == 0).sum())
    return StateResult(
        regimes=tuple(regime_names), horizons=tuple(horizons), base_effect=B[1],
        interaction=B[2:2 + K].T, interaction_se=se, n_events=n, n_by_regime=counts,
        p_joint=(1 + exceed_joint) / (1 + boot), p_headline=(1 + exceed_head) / (1 + boot),
        headline=headline,
    )
