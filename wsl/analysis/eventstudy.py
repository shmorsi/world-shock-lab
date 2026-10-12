"""Event-study and monthly LP runners on the built data (the pre-registered primary specs,
docs/EVALUATION_PROTOCOL.md §9)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from wsl.estimation.irf import IRF, Label
from wsl.estimation.lp import event_lp, lp
from wsl.paths import CLEAN_DIR
from wsl.shocks.windows import load_series, window_changes

DAILY_H = 20
MONTHLY_H = 24
PRE_LAGS = 5


def event_irf(
    events: pd.DataFrame,
    shock: str,
    outcome: str,
    *,
    other_shocks: Sequence[str] = (),
    per: float = 1.0,
    horizons: int = DAILY_H,
    assumed_time: str | None = None,
    label: Label = "identified_causal",
) -> IRF:
    """Event-sample LP: Δ_h outcome on the shock (per ``per`` units), controlling for the
    five pre-event daily changes and any other shocks of the same announcement; HC3."""
    s = load_series(outcome, assumed_time=assumed_time)
    w = window_changes(s, events["ts"], horizons, PRE_LAGS)
    df = w.join(events[[shock, *other_shocks]])
    df[shock] = df[shock] / per
    controls = [f"pre{lag}" for lag in range(1, PRE_LAGS + 1)] + list(other_shocks)
    irf = event_lp(df, {h: f"d{h}" for h in range(horizons + 1)}, shock, controls,
                   cov_type="HC3", label=label, outcome=outcome)
    used = df.dropna(subset=["d0", shock, *controls])
    return IRF(irf.horizons, irf.coef, irf.se, irf.nobs, irf.method, irf.cov_type, label,
               outcome=outcome, shock=shock,
               meta={**irf.meta, "per": per, "units": "bp" if s.kind == "bp" else "%",
                     "obs_time": s.obs_time, "assumed_time": assumed_time,
                     "first_event": str(used.index.min()), "last_event": str(used.index.max())})


def monthly_series(series_id: str) -> pd.Series:
    obs = pd.read_parquet(CLEAN_DIR / "obs.parquet", filters=[("series_id", "==", series_id)])
    s = obs[obs["status"] == "ok"].set_index("date")["value"].sort_index()
    return pd.Series(100.0 * np.log(s.to_numpy()), index=s.index)


def monthly_shock(shock_id: str) -> pd.Series:
    m = pd.read_parquet(CLEAN_DIR / "shocks_monthly.parquet")
    s = m[m["shock_id"] == shock_id].set_index("month")["value"].sort_index()
    return s.dropna()


def monthly_irf(shock_id: str, outcome: str, *, per: float = 1.0, horizons: int = MONTHLY_H,
                lags: int = 12, label: Label = "identified_causal") -> IRF:
    """Monthly LP of 100·log(outcome) on a monthly shock series, 12 lags of both, NW(h+1),
    on the months where the shock series exists."""
    shock = monthly_shock(shock_id) / per
    y = monthly_series(outcome).reindex(shock.index)
    irf = lp(y, shock, range(horizons + 1), y_lags=lags, shock_lags=lags, cov_type="NW",
             label=label, outcome=outcome, shock_name=shock_id)
    return IRF(irf.horizons, irf.coef, irf.se, irf.nobs, irf.method, irf.cov_type, label,
               outcome=outcome, shock=shock_id,
               meta={**irf.meta, "per": per, "units": "%",
                     "first_month": str(shock.index.min().date()),
                     "last_month": str(shock.index.max().date())})
