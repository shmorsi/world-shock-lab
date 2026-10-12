"""The Bank of Canada announcement-day proxy and its validation (docs/PLAN.md P9).

Proxy: the change in the GoC 2-year benchmark yield from the close before a scheduled BoC
announcement to the close on announcement day, in bp. Events follow protocol v2: scheduled
announcements with a known time, excluding FOMC days.

The Valet "closing" time is undocumented (open item O-1). Every BoC announcement is at
09:00-10:00 ET, so any afternoon close gives the same window; 16:00 is assumed.

Pass criteria, fixed in this file before the validation was first run on real data:
  V1  announcement days are more volatile: variance-ratio 90% CI lower bound > 1;
  V2  still true after removing the same-day US 2-year move (US-purged proxy);
  V3  a hawkish proxy strengthens the Canadian dollar (USD/CAD falls on day 0), and the real
      coefficient beats the placebo distribution (permutation p < 0.10).
All three pass -> the proxy is labelled `identified_causal` (as a daily-window proxy, weaker
than an intraday surprise). Otherwise it is labelled `reduced_form`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wsl.analysis.eventstudy import event_irf
from wsl.analysis.hetero import (
    RigobonSack,
    VarianceRatio,
    control_times,
    rigobon_sack,
    variance_ratio,
)
from wsl.analysis.placebo import PlaceboResult, placebo_test
from wsl.estimation.irf import IRF
from wsl.paths import CLEAN_DIR
from wsl.shocks import events as ev
from wsl.shocks.windows import load_series, window_changes

ASSUMED_CA_CLOSE = "16:00"
PROXY_SERIES = "CA2Y"
PERIODS = {"2001-2014 (train)": ("2001-01-01", "2014-12-31"),
           "2015-2018 (validation)": ("2015-01-01", "2018-12-31"),
           "2019-2026 (test)": ("2019-01-01", "2026-09-30")}
OUTCOMES = ["USDCAD", "CA10Y", "TSX.tr", "CA_BANKS_EW6"]
# Valet GoC yields have an undocumented close (O-1); same assumption as for the proxy.
ASSUMED = {"CA2Y": ASSUMED_CA_CLOSE, "CA10Y": ASSUMED_CA_CLOSE}
PER = 10.0  # responses per 10 bp of proxy


def all_announcements() -> pd.Series:
    """Every announcement time we know of, for excluding control and placebo days."""
    ecb_dates = pd.read_parquet(CLEAN_DIR / "events_ecb.parquet")["date"]
    parts = [ev.fomc()["ts"], ev.boc(scheduled_only=False, exclude_fomc_same_day=False)["ts"],
             ev.opec()["ts"], (ecb_dates + pd.Timedelta(hours=13, minutes=45))
             .dt.tz_localize("Europe/Berlin")]
    out: pd.Series = pd.concat([p.dt.tz_convert("UTC") for p in parts], ignore_index=True)
    return out


def proxy_events() -> pd.DataFrame:
    """BoC events (protocol v2) with proxy_bp = day-0 change in the 2-year yield."""
    b = ev.boc()
    w = window_changes(load_series(PROXY_SERIES, ASSUMED_CA_CLOSE), b["ts"], 0)
    b["proxy_bp"] = w["d0"]
    return b.dropna(subset=["proxy_bp"])


@dataclass
class ProxyValidation:
    n_events: int
    first: str
    last: str
    proxy_sd: float
    vr: VarianceRatio
    vr_by_period: dict[str, VarianceRatio]
    purge_beta: float
    vr_purged: VarianceRatio
    corr_raw_purged: float
    by_decision: pd.DataFrame
    slope_on_decision: tuple[float, float, float]  # coef, se, r2
    irfs: dict[str, IRF] = field(default_factory=dict)
    rs: dict[str, RigobonSack] = field(default_factory=dict)
    placebo: dict[str, list[PlaceboResult]] = field(default_factory=dict)

    @property
    def criteria(self) -> dict[str, bool]:
        usd = self.irfs["USDCAD"]
        p0 = next(r for r in self.placebo["USDCAD"] if r.horizon == 0)
        return {
            "V1 announcement-day variance ratio > 1 (90% CI)": self.vr.ci90[0] > 1.0,
            "V2 same after removing the US 2-year move": self.vr_purged.ci90[0] > 1.0,
            "V3 hawkish proxy strengthens CAD, beats placebo (p < 0.10)":
                bool(usd.coef[0] < 0 and p0.p_value < 0.10),
        }

    @property
    def label(self) -> str:
        return "identified_causal" if all(self.criteria.values()) else "reduced_form"


def _changes(series_id: str, times: pd.Series, assumed: str | None = None) -> np.ndarray:
    return np.asarray(window_changes(load_series(series_id, assumed), times, 0)["d0"],
                      dtype=float)


def validate() -> ProxyValidation:
    b = proxy_events()
    ctrl = control_times(b["ts"], all_announcements())
    pair = ctrl.notna()
    i_e = b["proxy_bp"].to_numpy()
    i_c = np.full(len(b), np.nan)
    i_c[pair.to_numpy()] = _changes(PROXY_SERIES, ctrl[pair], ASSUMED_CA_CLOSE)
    vr = variance_ratio(i_e, i_c)
    by_period = {}
    for name, (a, z) in PERIODS.items():
        m = ((b.index >= a) & (b.index <= z))
        by_period[name] = variance_ratio(i_e[m], i_c[m])

    # Purge the same-day US 2-year move, with the loading estimated on control days only.
    us_e = _changes("US2Y", b["ts"])
    us_c = np.full(len(b), np.nan)
    us_c[pair.to_numpy()] = _changes("US2Y", ctrl[pair])
    k = ~np.isnan(i_c) & ~np.isnan(us_c)
    beta_us = float(np.cov(i_c[k], us_c[k])[0, 1] / np.var(us_c[k], ddof=1))
    vr_purged = variance_ratio(i_e - beta_us * us_e, i_c - beta_us * us_c)
    both = ~np.isnan(us_e)
    corr = float(np.corrcoef(i_e[both], (i_e - beta_us * us_e)[both])[0, 1])

    g = b.groupby("decision")["proxy_bp"]
    by_decision = pd.DataFrame({"events": g.size(), "mean_bp": g.mean(),
                                "se_bp": g.std(ddof=1) / np.sqrt(g.size())})
    X = np.column_stack([np.ones(len(b)), b["change_bp"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, i_e, rcond=None)
    resid = i_e - X @ coef
    se = float(np.sqrt(np.sum(resid**2) / (len(b) - 2) / np.sum((X[:, 1] - X[:, 1].mean())**2)))
    r2 = float(1 - resid.var() / i_e.var())

    out = ProxyValidation(len(b), str(b.index.min()), str(b.index.max()),
                          float(np.std(i_e, ddof=1)), vr, by_period, beta_us, vr_purged, corr,
                          by_decision, (float(coef[1]), se, r2))
    others = [ev.fomc()["ts"], ev.opec()["ts"]]
    for o in OUTCOMES:
        assumed: str | None = ASSUMED.get(o)
        irf = event_irf(b, "proxy_bp", o, per=PER, label="reduced_form", assumed_time=assumed)
        out.irfs[o] = irf
        y_e = _changes(o, b["ts"], assumed)
        y_c = np.full(len(b), np.nan)
        y_c[pair.to_numpy()] = _changes(o, ctrl[pair], assumed)
        out.rs[o] = rigobon_sack(i_e, y_e, i_c, y_c)
        s = load_series(o, assumed)
        used = b[~np.isnan(y_e)]
        out.placebo[o] = placebo_test(s, used, ["proxy_bp"], {0: irf.coef[0], 5: irf.coef[5]},
                                      per=PER, other_families=others)
    return out
