"""LP-IV: local projections with an external instrument (Stock & Watson 2018).

For each horizon h, regress y_{t+h} - y_{t-1} on the policy variable x_t, instrumenting x_t with
the shock series z_t, with lags of y, x and z as controls. The coefficient is the response of
y at horizon h to a shock that moves x by one unit on impact.

Single instrument, single endogenous regressor. Controls are partialled out first
(Frisch-Waugh-Lovell), after which the estimator is β = z'y / z'x.

Weak-instrument diagnostics, reported for every horizon:
* effective first-stage F (Montiel Olea & Pflueger 2013). With one instrument it equals the
  robust (HAC) Wald F of the first stage;
* Anderson-Rubin confidence sets, which remain valid however weak the instrument. Computed in
  closed form: {β0 : (a - β0 b)² <= c (V_aa - 2 β0 V_ab + β0² V_bb)} where a, b are the
  reduced-form and first-stage coefficients on z and V their joint robust covariance. The set
  is an interval if the first stage is strong enough, otherwise two rays or the whole line.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from wsl.estimation.irf import IRF, LEVELS, Label, z_value
from wsl.estimation.lp import Dependent, lag_frame, lead_dependent
from wsl.estimation.ols import complete_rows, newey_west_meat

IVCov = Literal["HC0", "HC1", "NW"]


@dataclass(frozen=True)
class ARSet:
    kind: Literal["interval", "two_rays", "real_line", "empty"]
    lo: float = float("nan")  # for two_rays: the set is (-inf, lo] ∪ [hi, inf)
    hi: float = float("nan")

    def contains(self, b: float) -> bool:
        if self.kind == "interval":
            return self.lo <= b <= self.hi
        if self.kind == "two_rays":
            return b <= self.lo or b >= self.hi
        return self.kind == "real_line"


def ar_set(a: float, b: float, V: np.ndarray, level: float) -> ARSet:
    """Anderson-Rubin set for β = a / b given the 2x2 robust covariance of (a, b)."""
    c = z_value(level) ** 2
    qa = b * b - c * V[1, 1]
    qb = -2.0 * (a * b - c * V[0, 1])
    qc = a * a - c * V[0, 0]
    disc = qb * qb - 4.0 * qa * qc
    if abs(qa) < 1e-14:
        return ARSet("real_line")
    if disc < 0:
        return ARSet("empty") if qa > 0 else ARSet("real_line")
    r1, r2 = sorted(((-qb - np.sqrt(disc)) / (2 * qa), (-qb + np.sqrt(disc)) / (2 * qa)))
    return ARSet("interval", r1, r2) if qa > 0 else ARSet("two_rays", r1, r2)


@dataclass(frozen=True)
class IVDiagnostics:
    horizons: tuple[int, ...]
    first_stage_coef: np.ndarray
    effective_f: np.ndarray
    ar: dict[float, tuple[ARSet, ...]]


def _resid(v: np.ndarray, W: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(W, v, rcond=None)
    return np.asarray(v - W @ coef, dtype=float)


def _meat(scores: np.ndarray, cov_type: IVCov, lags: int, n: int, k: int) -> np.ndarray:
    m = newey_west_meat(scores, lags if cov_type == "NW" else 0)
    return m * (n / (n - k)) if cov_type == "HC1" else m


def iv_just_identified(
    y: np.ndarray, x: np.ndarray, z: np.ndarray, W: np.ndarray, *, cov_type: IVCov = "NW",
    nw_lags: int = 0,
) -> tuple[float, float, int, float, float, np.ndarray, float]:
    """-> (beta, se, n, first-stage pi, effective F, joint cov of (a, pi), reduced-form a)."""
    keep = complete_rows(y, x, z, W)
    y, x, z, W = y[keep], x[keep], z[keep], W[keep]
    n, k = W.shape[0], W.shape[1] + 1
    yt, xt, zt = _resid(y, W), _resid(x, W), _resid(z, W)
    zz, zx = float(zt @ zt), float(zt @ xt)
    beta = float(zt @ yt) / zx
    u = yt - beta * xt
    se = float(np.sqrt(_meat((zt * u)[:, None], cov_type, nw_lags, n, k)[0, 0]) / abs(zx))
    pi = zx / zz
    a = float(zt @ yt) / zz
    v, e = xt - pi * zt, yt - a * zt
    joint = _meat(np.column_stack([zt * e, zt * v]), cov_type, nw_lags, n, k) / zz**2
    f_eff = pi * pi / joint[1, 1]
    return beta, se, n, pi, f_eff, joint, a


def lp_iv(
    y: pd.Series,
    x: pd.Series,
    z: pd.Series,
    horizons: Iterable[int],
    *,
    lags: int = 4,
    controls: pd.DataFrame | None = None,
    dependent: Dependent = "change",
    cov_type: IVCov = "NW",
    nw_lags: int | Literal["h+1"] = "h+1",
    label: Label = "identified_causal",
    outcome: str = "",
    shock_name: str = "",
) -> tuple[IRF, IVDiagnostics]:
    if not (y.index.equals(x.index) and y.index.equals(z.index)):
        raise ValueError("y, x and z must share the same (time-ordered) index")
    parts: list[pd.Series | pd.DataFrame] = [
        pd.Series(1.0, index=y.index, name="const"), lag_frame(y, lags, "y"),
        lag_frame(x, lags, "x"), lag_frame(z, lags, "z")]
    if controls is not None:
        parts.append(controls.reindex(y.index))
    W = pd.concat(parts, axis=1).to_numpy()
    hs, coef, se, nobs, pis, fs = [], [], [], [], [], []
    ar: dict[float, list[ARSet]] = {lv: [] for lv in LEVELS}
    for h in horizons:
        dep = lead_dependent(y, h, dependent).to_numpy()
        L = (h + 1) if nw_lags == "h+1" else nw_lags
        beta, s, n, pi, f_eff, joint, a = iv_just_identified(
            dep, x.to_numpy(), z.to_numpy(), W, cov_type=cov_type, nw_lags=L)
        hs.append(h)
        coef.append(beta)
        se.append(s)
        nobs.append(n)
        pis.append(pi)
        fs.append(f_eff)
        for lv in LEVELS:
            ar[lv].append(ar_set(a, pi, joint, lv))
    irf = IRF(horizons=tuple(hs), coef=np.array(coef), se=np.array(se), nobs=tuple(nobs),
              method="lp_iv", cov_type=cov_type if cov_type != "NW" else f"NW({nw_lags})",
              label=label, outcome=outcome, shock=shock_name,
              meta={"lags": lags, "dependent": dependent, "min_effective_f": float(min(fs))})
    diag = IVDiagnostics(horizons=tuple(hs), first_stage_coef=np.array(pis),
                         effective_f=np.array(fs), ar={lv: tuple(v) for lv, v in ar.items()})
    return irf, diag
