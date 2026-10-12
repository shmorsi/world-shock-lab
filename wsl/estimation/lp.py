"""Local projections (Jordà 2005).

Two forms:

* ``lp``: time-series LP. For each horizon h, regress y_{t+h} - y_{t-1} (``dependent="change"``)
  or y_{t+h} (``"level"``) on the shock at t, a constant, p lags of y, q lags of the shock and
  optional (already-lagged) controls. With lag augmentation, heteroskedasticity-robust (HC)
  standard errors are valid at every horizon (Montiel Olea & Plagborg-Møller 2021); Newey-West
  with h+1 lags is the conventional alternative. Both are checked by simulation in
  tests/test_simulation_recovery.py.
* ``event_lp``: event-sample LP. One row per announcement; for each h, regress the outcome's
  change over the event window [last obs before, h-th obs after] on the surprise and controls.
  Window construction is the caller's job (wsl.timing); this function only estimates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from wsl.estimation.irf import IRF, Label
from wsl.estimation.ols import CovType, ols

Dependent = Literal["change", "level"]


def lag_frame(s: pd.Series, lags: int, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({f"{prefix}_l{lag}": s.shift(lag) for lag in range(1, lags + 1)},
                        index=s.index)


def lead_dependent(y: pd.Series, h: int, dependent: Dependent) -> pd.Series:
    lead = y.shift(-h)
    return lead - y.shift(1) if dependent == "change" else lead


def lp(
    y: pd.Series,
    shock: pd.Series,
    horizons: Iterable[int],
    *,
    y_lags: int = 4,
    shock_lags: int = 4,
    controls: pd.DataFrame | None = None,
    dependent: Dependent = "change",
    cov_type: CovType = "NW",
    nw_lags: int | Literal["h+1"] = "h+1",
    label: Label = "identified_causal",
    outcome: str = "",
    shock_name: str = "",
) -> IRF:
    if not y.index.equals(shock.index):
        raise ValueError("y and shock must share the same (time-ordered) index")
    base = pd.concat(
        [pd.Series(1.0, index=y.index, name="const"), shock.rename("shock"),
         lag_frame(y, y_lags, "y"), lag_frame(shock, shock_lags, "shock")]
        + ([controls.reindex(y.index)] if controls is not None else []),
        axis=1,
    )
    names = tuple(base.columns)
    hs, coef, se, nobs = [], [], [], []
    for h in horizons:
        dep = lead_dependent(y, h, dependent)
        lags = (h + 1) if nw_lags == "h+1" else nw_lags
        r = ols(dep.to_numpy(), base.to_numpy(), cov_type=cov_type,
                nw_lags=lags if cov_type == "NW" else None, names=names)
        hs.append(h)
        coef.append(r.coef("shock"))
        se.append(r.stderr("shock"))
        nobs.append(r.nobs)
    return IRF(horizons=tuple(hs), coef=np.array(coef), se=np.array(se), nobs=tuple(nobs),
               method="lp", cov_type=cov_type if cov_type != "NW" else f"NW({nw_lags})",
               label=label, outcome=outcome, shock=shock_name,
               meta={"y_lags": y_lags, "shock_lags": shock_lags, "dependent": dependent,
                     "controls": list(controls.columns) if controls is not None else []})


def event_lp(
    data: pd.DataFrame,
    outcomes: Mapping[int, str],
    shock: str,
    controls: Sequence[str] = (),
    *,
    cov_type: CovType = "HC3",
    label: Label = "identified_causal",
    outcome: str = "",
) -> IRF:
    """``outcomes`` maps horizon -> column holding that horizon's event-window change."""
    regressors = ["const", shock, *controls]
    X = pd.concat([pd.Series(1.0, index=data.index, name="const"), data[[shock, *controls]]],
                  axis=1)
    hs, coef, se, nobs = [], [], [], []
    for h, col in sorted(outcomes.items()):
        r = ols(data[col].to_numpy(), X.to_numpy(), cov_type=cov_type,
                names=tuple(regressors))
        hs.append(h)
        coef.append(r.coef(shock))
        se.append(r.stderr(shock))
        nobs.append(r.nobs)
    return IRF(horizons=tuple(hs), coef=np.array(coef), se=np.array(se), nobs=tuple(nobs),
               method="event_lp", cov_type=cov_type, label=label, outcome=outcome, shock=shock,
               meta={"controls": list(controls)})
