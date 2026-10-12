"""Impulse-response containers shared by every estimator (LP, LP-IV, and later SVAR/ML)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

Label = Literal["identified_causal", "reduced_form", "model_forecast", "hypothetical_scenario"]
LEVELS = (0.68, 0.90)


def z_value(level: float) -> float:
    return float(norm.ppf(0.5 + level / 2.0))


@dataclass(frozen=True)
class IRF:
    """Pointwise estimates and normal bands for one outcome and one shock."""

    horizons: tuple[int, ...]
    coef: np.ndarray
    se: np.ndarray
    nobs: tuple[int, ...]
    method: str  # e.g. "lp", "event_lp", "lp_iv"
    cov_type: str
    label: Label
    outcome: str = ""
    shock: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def band(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        z = z_value(level)
        return self.coef - z * self.se, self.coef + z * self.se

    def covers(self, truth: np.ndarray, level: float) -> np.ndarray:
        lo, hi = self.band(level)
        return (lo <= truth) & (truth <= hi)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"h": self.horizons, "coef": self.coef, "se": self.se,
                           "nobs": self.nobs})
        for lv in LEVELS:
            lo, hi = self.band(lv)
            df[f"lo{int(lv * 100)}"], df[f"hi{int(lv * 100)}"] = lo, hi
        return df

    def to_dict(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "shock": self.shock, "method": self.method,
                "cov_type": self.cov_type, "label": self.label, "meta": self.meta,
                "rows": self.to_frame().to_dict(orient="records")}
