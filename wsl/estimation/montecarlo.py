"""Monte Carlo checks of the estimators against data with known impulse responses.

Used by tests/test_simulation_recovery.py (assertions) and by ``wsl validate simulation``
(docs/SIMULATION.md). Every experiment is seeded, so both always report the same numbers.

Experiments
-----------
dense      monthly-like: Gaussian shock observed every period, T = 500.
sparse     daily-like: the shock is nonzero on about 3% of days with Student-t(4) sizes
           (about 240 events in T = 8000), like announcement surprises.
lpiv       LP-IV with a noisy instrument z = s·ε + ν (strong and weak first stages).
placebo    the "shock" is independent noise; the true response is zero at every horizon.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wsl.estimation.irf import IRF
from wsl.estimation.lp import event_lp, lp
from wsl.estimation.lpiv import lp_iv
from wsl.estimation.simulate import StructuralVAR, sparse_shock, textbook_var

SEED = 20261012
H = 12
OUTCOME = 1  # "activity" in textbook_var; responses to shock 0 ("monetary")


@dataclass
class MCResult:
    name: str
    truth: np.ndarray
    estimates: np.ndarray  # (R, H+1)
    covered68: np.ndarray  # (R, H+1) bool
    covered90: np.ndarray
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def R(self) -> int:
        return int(self.estimates.shape[0])

    @property
    def bias(self) -> np.ndarray:
        return np.asarray(self.estimates.mean(axis=0) - self.truth)

    @property
    def mc_se(self) -> np.ndarray:
        return np.asarray(self.estimates.std(axis=0, ddof=1) / np.sqrt(self.R))

    @property
    def coverage90(self) -> np.ndarray:
        return self.covered90.mean(axis=0)

    @property
    def coverage68(self) -> np.ndarray:
        return self.covered68.mean(axis=0)

    def summary(self) -> dict[str, float]:
        peak = float(np.max(np.abs(self.truth))) or 1.0
        return {
            "R": self.R,
            "max |bias| / peak": float(np.max(np.abs(self.bias)) / peak),
            "max |bias| / MC s.e.": float(np.max(np.abs(self.bias) / self.mc_se)),
            "coverage 68 (mean)": float(self.coverage68.mean()),
            "coverage 90 (mean)": float(self.coverage90.mean()),
            "coverage 90 (min h)": float(self.coverage90.min()),
            **self.extra,
        }


def _run(name: str, truth: np.ndarray, R: int,
         one: Callable[[np.random.Generator], IRF], seed: int) -> MCResult:
    rng = np.random.default_rng(seed)
    est, c68, c90 = [], [], []
    for _ in range(R):
        irf = one(rng)
        est.append(irf.coef)
        c68.append(irf.covers(truth, 0.68))
        c90.append(irf.covers(truth, 0.90))
    return MCResult(name, truth, np.array(est), np.array(c68), np.array(c90))


def dense(cov_type: str = "NW", R: int = 200, T: int = 500, var: StructuralVAR | None = None,
          outcome: int = OUTCOME, seed: int = SEED) -> MCResult:
    var = var or textbook_var()
    truth = var.irf(H)[:, outcome, 0]

    def one(rng: np.random.Generator) -> IRF:
        Y, eps = var.simulate(T, rng)
        idx = pd.RangeIndex(T)
        return lp(pd.Series(Y[:, outcome], index=idx), pd.Series(eps[:, 0], index=idx),
                  range(H + 1), cov_type=cov_type)  # type: ignore[arg-type]

    return _run(f"dense LP, {cov_type}, outcome {outcome}", truth, R, one, seed)


def _sparse_draw(var: StructuralVAR, T: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s0 = sparse_shock(T, rng)
    Y, eps = var.simulate(T, rng, shock0=s0)
    return Y, eps, s0


def sparse(spec: str = "event_HC3", R: int = 200, T: int = 8000, seed: int = SEED + 1,
           var: StructuralVAR | None = None) -> MCResult:
    """spec: 'event_HC3' (primary daily spec) or 'zerofill_NW' / 'zerofill_HC1' / 'zerofill_HC3'."""
    var = var or textbook_var()
    truth = var.irf(H)[:, OUTCOME, 0]
    n_events: list[int] = []

    def one(rng: np.random.Generator) -> IRF:
        Y, eps, s0 = _sparse_draw(var, T, rng)
        y = Y[:, OUTCOME]
        if spec.startswith("zerofill_"):
            idx = pd.RangeIndex(T)
            return lp(pd.Series(y, index=idx), pd.Series(eps[:, 0], index=idx), range(H + 1),
                      cov_type=spec.split("_")[1])  # type: ignore[arg-type]
        ev = np.flatnonzero(s0 != 0)
        ev = ev[(ev > 2) & (ev < T - H - 1)]
        n_events.append(len(ev))
        df = pd.DataFrame({"s": eps[ev, 0]}, index=ev)
        for h in range(H + 1):
            df[f"d{h}"] = y[ev + h] - y[ev - 1]
        for lag in (1, 2):  # pre-event changes as controls
            df[f"pre{lag}"] = y[ev - lag] - y[ev - lag - 1]
        return event_lp(df, {h: f"d{h}" for h in range(H + 1)}, "s", ["pre1", "pre2"],
                        cov_type=spec.split("_")[1])  # type: ignore[arg-type]

    res = _run(f"sparse shocks, {spec}", truth, R, one, seed)
    if n_events:
        res.extra["events per draw (mean)"] = float(np.mean(n_events))
    return res


def lpiv(strength: float = 1.0, R: int = 200, T: int = 500, seed: int = SEED + 2,
         var: StructuralVAR | None = None) -> MCResult:
    """Instrument z = strength·ε₀ + ν, ν ~ N(0,1). Truth: response of the outcome per unit
    impact on the policy variable (variable 0)."""
    var = var or textbook_var()
    irf_true = var.irf(H)
    truth = irf_true[:, OUTCOME, 0] / irf_true[0, 0, 0]
    rng = np.random.default_rng(seed)
    est, c68, c90, ar90, fs = [], [], [], [], []
    for _ in range(R):
        Y, eps = var.simulate(T, rng)
        idx = pd.RangeIndex(T)
        z = pd.Series(strength * eps[:, 0] + rng.standard_normal(T), index=idx)
        irf, diag = lp_iv(pd.Series(Y[:, OUTCOME], index=idx), pd.Series(Y[:, 0], index=idx),
                          z, range(H + 1), cov_type="NW")
        est.append(irf.coef)
        c68.append(irf.covers(truth, 0.68))
        c90.append(irf.covers(truth, 0.90))
        ar90.append([s.contains(float(t)) for s, t in zip(diag.ar[0.90], truth, strict=True)])
        fs.append(float(np.median(diag.effective_f)))
    res = MCResult(f"LP-IV, instrument strength {strength}", truth, np.array(est),
                   np.array(c68), np.array(c90))
    res.extra = {"median effective F": float(np.median(fs)),
                 "Anderson-Rubin coverage 90 (mean)": float(np.mean(ar90)),
                 "Anderson-Rubin coverage 90 (min h)": float(np.min(np.mean(ar90, axis=0)))}
    return res


def placebo(spec: str = "event_HC3", R: int = 200, T: int = 8000, seed: int = SEED + 3,
            var: StructuralVAR | None = None) -> MCResult:
    """Placebo: estimate responses to an independent sparse series; truth is zero."""
    var = var or textbook_var()
    truth = np.zeros(H + 1)

    def one(rng: np.random.Generator) -> IRF:
        Y, _, _ = _sparse_draw(var, T, rng)
        fake = sparse_shock(T, rng)
        y = Y[:, OUTCOME]
        ev = np.flatnonzero(fake != 0)
        ev = ev[(ev > 2) & (ev < T - H - 1)]
        df = pd.DataFrame({"s": fake[ev]}, index=ev)
        for h in range(H + 1):
            df[f"d{h}"] = y[ev + h] - y[ev - 1]
        for lag in (1, 2):
            df[f"pre{lag}"] = y[ev - lag] - y[ev - lag - 1]
        return event_lp(df, {h: f"d{h}" for h in range(H + 1)}, "s", ["pre1", "pre2"],
                        cov_type="HC3")

    return _run(f"placebo, {spec}", truth, R, one, seed)


def consistency(T_small: int = 250, T_large: int = 2000, R: int = 100,
                seed: int = SEED + 4) -> tuple[float, float]:
    """RMSE of dense LP (HC1) over all horizons at two sample sizes."""
    small = dense("HC1", R=R, T=T_small, seed=seed)
    large = dense("HC1", R=R, T=T_large, seed=seed)

    def rmse(r: MCResult) -> float:
        return float(np.sqrt(np.mean((r.estimates - r.truth) ** 2)))

    return rmse(small), rmse(large)
