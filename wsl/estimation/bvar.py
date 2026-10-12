"""Bayesian VAR with a Minnesota prior, identified with an external instrument (proxy SVAR).

Prior: Minnesota, implemented with dummy observations (Banbura, Giannone & Reichlin 2010).
Each variable's own first lag is centred on ``own_lag_mean`` (1 = random walk, for series in
levels), other coefficients on zero, with tightness ``lam`` and lag decay 1/l. The point
estimate is the posterior mean, i.e. least squares on the data stacked with the dummies.

Identification (Stock & Watson 2012; Mertens & Ravn 2013; Gertler & Karadi 2015): an
instrument z_t correlated with the structural shock of interest and uncorrelated with the
others identifies its impact column up to scale, b ∝ Cov(u_t, z_t). It is normalized so the
first variable moves by ``impact`` on impact. Responses are Ψ_h b.

Bands: moving-block bootstrap of residuals and instrument jointly, with the position-wise
centring of Jentsch & Lunsford (2019); the VAR is re-simulated from the bootstrap residuals,
re-estimated and re-identified in every draw. The bootstrap is vectorised across draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def lag_matrix(Y: Array, p: int) -> Array:
    """[y_{t-1}, ..., y_{t-p}, 1] for t = p..T-1."""
    T = Y.shape[0]
    cols = [Y[p - lag:T - lag] for lag in range(1, p + 1)]
    return np.column_stack([*cols, np.ones(T - p)])


def minnesota_dummies(Y: Array, p: int, lam: float, own_lag_mean: float,
                      eps: float = 1e-3) -> tuple[Array, Array]:
    """Dummy observations (Yd, Xd) for the Minnesota prior; sigma from AR(1) residuals."""
    n = Y.shape[1]
    sigma = np.empty(n)
    for i in range(n):
        y, x = Y[1:, i], np.column_stack([Y[:-1, i], np.ones(len(Y) - 1)])
        b, *_ = np.linalg.lstsq(x, y, rcond=None)
        sigma[i] = np.std(y - x @ b, ddof=2)
    k = n * p + 1
    Yd = np.vstack([np.diag(own_lag_mean * sigma) / lam, np.zeros((n * (p - 1), n)),
                    np.diag(sigma), np.zeros((1, n))])
    Xd = np.zeros((n * p + n + 1, k))
    Xd[: n * p, : n * p] = np.kron(np.diag(np.arange(1, p + 1)), np.diag(sigma)) / lam
    Xd[-1, -1] = eps
    return Yd, Xd


@dataclass(frozen=True)
class BVAR:
    B: Array  # (n*p + 1, n): rows = lag 1 block, ..., lag p block, constant
    U: Array  # (T - p, n) residuals on the actual data
    p: int
    n: int

    def ma(self, horizons: int) -> Array:
        """Reduced-form MA coefficients Ψ_h, shape (H+1, n, n)."""
        A = [self.B[i * self.n:(i + 1) * self.n].T for i in range(self.p)]
        psi = [np.eye(self.n)]
        for h in range(1, horizons + 1):
            acc = np.zeros((self.n, self.n))
            for lag in range(1, min(h, self.p) + 1):
                acc += A[lag - 1] @ psi[h - lag]
            psi.append(acc)
        return np.stack(psi)


def fit(Y: Array, p: int, lam: float = 0.2, own_lag_mean: float = 1.0) -> BVAR:
    X = lag_matrix(Y, p)
    Yt = Y[p:]
    Yd, Xd = minnesota_dummies(Y, p, lam, own_lag_mean)
    Xs, Ys = np.vstack([X, Xd]), np.vstack([Yt, Yd])
    B = np.linalg.solve(Xs.T @ Xs, Xs.T @ Ys)
    return BVAR(B=B, U=Yt - X @ B, p=p, n=Y.shape[1])


@dataclass(frozen=True)
class ProxyID:
    impact: Array  # (n,) impact column, normalized
    first_stage_f: float  # robust (HC1) F of u_1 on z
    n_used: int


def identify(U: Array, z: Array, impact: float = 1.0) -> ProxyID:
    """Impact column from Cov(u, z), normalized so variable 0 moves by ``impact``."""
    keep = ~np.isnan(z)
    u, zz = U[keep], z[keep] - z[keep].mean()
    cov = (u - u.mean(axis=0)).T @ zz / (len(zz) - 1)
    b = cov / cov[0] * impact
    x = np.column_stack([np.ones(len(zz)), zz])
    coef, *_ = np.linalg.lstsq(x, u[:, 0], rcond=None)
    e = u[:, 0] - x @ coef
    xtx_inv = np.linalg.inv(x.T @ x)
    meat = (x * e[:, None]).T @ (x * e[:, None]) * len(zz) / (len(zz) - 2)
    var = (xtx_inv @ meat @ xtx_inv)[1, 1]
    return ProxyID(impact=b, first_stage_f=float(coef[1] ** 2 / var), n_used=int(keep.sum()))


def irf(model: BVAR, b: Array, horizons: int) -> Array:
    """(H+1, n) responses to the identified shock."""
    return np.asarray(np.einsum("hij,j->hi", model.ma(horizons), b), dtype=float)


def _jl_blocks(T: int, ell: int, rng: np.random.Generator) -> NDArray[np.int64]:
    """Indices of a moving-block resample of length T (blocks of length ell)."""
    n_blocks = int(np.ceil(T / ell))
    starts = rng.integers(0, T - ell + 1, size=n_blocks)
    out: NDArray[np.int64] = (starts[:, None] + np.arange(ell)[None, :]).reshape(-1)[:T]
    return out


def _jl_centre(x: Array, ell: int) -> Array:
    """Position-wise means for Jentsch-Lunsford centring: mean of x over all blocks at each
    within-block position s = 0..ell-1."""
    T = x.shape[0]
    return np.asarray(np.stack([x[s:T - ell + 1 + s].mean(axis=0) for s in range(ell)]))


def default_block_length(T: int) -> int:
    length: int = round(5.03 * float(T) ** 0.25)
    return max(2, length)


@dataclass(frozen=True)
class ProxyBVARResult:
    irf: Array  # (H+1, n)
    lo68: Array
    hi68: Array
    lo90: Array
    hi90: Array
    first_stage_f: float
    n_used: int
    draws: int
    block_length: int
    boot_f_median: float


def proxy_bvar(Y: Array, z: Array, p: int, horizons: int, *, impact: float = 1.0,
               lam: float = 0.2, own_lag_mean: float = 1.0, draws: int = 500,
               block_length: int | None = None, seed: int = 20261012) -> ProxyBVARResult:
    """``z`` aligned with Y (length T); NaN where the instrument is unavailable."""
    model = fit(Y, p, lam, own_lag_mean)
    zt = z[p:]
    pid = identify(model.U, zt, impact)
    point = irf(model, pid.impact, horizons)

    T_eff, n = model.U.shape
    ell = block_length or default_block_length(T_eff)
    rng = np.random.default_rng(seed)
    joint = np.column_stack([model.U, zt])
    centre = _jl_centre(np.nan_to_num(joint), ell)
    A = [model.B[i * n:(i + 1) * n].T for i in range(p)]
    const = model.B[-1]
    sims = np.empty((draws, horizons + 1, n))
    fs = np.empty(draws)
    # Draw all bootstrap residual paths, then simulate the VAR for every draw at once.
    idx = np.stack([_jl_blocks(T_eff, ell, rng) for _ in range(draws)])  # (draws, T_eff)
    pos = np.tile(np.arange(T_eff) % ell, (draws, 1))
    Ustar = joint[idx][..., :n] - centre[pos][..., :n]
    zstar = joint[idx][..., n] - np.where(np.isnan(joint[idx][..., n]), 0.0, centre[pos][..., n])
    Ysim = np.empty((draws, Y.shape[0], n))
    Ysim[:, :p] = Y[:p]
    for t in range(p, Y.shape[0]):
        acc = np.broadcast_to(const, (draws, n)).copy()
        for lag in range(1, p + 1):
            acc += Ysim[:, t - lag] @ A[lag - 1].T
        Ysim[:, t] = acc + Ustar[:, t - p]
    for d in range(draws):
        m = fit(Ysim[d], p, lam, own_lag_mean)
        pid_d = identify(m.U, zstar[d], impact)
        sims[d] = irf(m, pid_d.impact, horizons)
        fs[d] = pid_d.first_stage_f
    q = np.quantile(sims, [0.05, 0.16, 0.84, 0.95], axis=0)
    return ProxyBVARResult(irf=point, lo90=q[0], lo68=q[1], hi68=q[2], hi90=q[3],
                           first_stage_f=pid.first_stage_f, n_used=pid.n_used, draws=draws,
                           block_length=ell, boot_f_median=float(np.median(fs)))
