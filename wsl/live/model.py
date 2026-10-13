"""The live forecasting model (config/live.yaml, protocol v4 addendum).

For each bank, outcome and horizon: OLS of the outcome's change on the proxy (the bank's own
2-year yield change on the announcement day) over that bank's past announcements. Prediction
intervals are the forecast plus or minus quantiles of the absolute leave-one-out residuals,
so they are intervals for the realized outcome. The model is deliberately simple: M5 found
no forecasting gain from anything richer, and a linear model is what makes the grid exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wsl.shocks.windows import WindowSeries, window_changes


@dataclass(frozen=True)
class CellModel:
    outcome: str
    h: int
    a: float
    b: float
    n: int
    half: dict[str, float]  # interval level ("68", "90") -> half-width
    first: str
    last: str

    def predict(self, s: float) -> dict[str, float]:
        m = self.a + self.b * s
        out = {"mean": m}
        for lvl, w in self.half.items():
            out[f"lo{lvl}"] = m - w
            out[f"hi{lvl}"] = m + w
        return out


def event_frame(events: pd.Series, proxy: WindowSeries, outcomes: dict[str, WindowSeries],
                horizons: list[int]) -> pd.DataFrame:
    """Per event: the proxy (bp) and each outcome's change at each horizon."""
    ts = pd.Series(pd.to_datetime(list(events), utc=True)).sort_values().reset_index(drop=True)
    out = pd.DataFrame({"ts": ts})
    out["proxy"] = window_changes(proxy, ts, 0)["d0"].to_numpy()
    for name, s in outcomes.items():
        w = window_changes(s, ts, max(horizons) - 1)
        for h in horizons:
            out[f"{name}|h{h}"] = w[f"d{h - 1}"].to_numpy()
    return out


def fit_cell(x: np.ndarray, y: np.ndarray, levels: list[float]) -> tuple[float, float,
                                                                         dict[str, float]]:
    """OLS of y on [1, x]; half-widths from |leave-one-out residuals| quantiles."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    hat = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)
    loo = resid / (1.0 - hat)
    half = {f"{round(100 * p)}": float(np.quantile(np.abs(loo), p)) for p in levels}
    return float(beta[0]), float(beta[1]), half


def fit(frame: pd.DataFrame, outcomes: list[str], horizons: list[int], levels: list[float],
        min_events: int) -> dict[str, CellModel]:
    models: dict[str, CellModel] = {}
    for o in outcomes:
        for h in horizons:
            col = f"{o}|h{h}"
            d = frame[["ts", "proxy", col]].dropna()
            if len(d) < min_events:
                raise ValueError(f"{col}: only {len(d)} usable events (minimum {min_events})")
            a, b, half = fit_cell(d["proxy"].to_numpy(), d[col].to_numpy(), levels)
            models[col] = CellModel(o, h, a, b, len(d), half,
                                    d["ts"].min().date().isoformat(),
                                    d["ts"].max().date().isoformat())
    return models
