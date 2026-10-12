"""The response catalog behind the scenario engine and the web app.

For every shock and every daily outcome: responses at h = 0..20 trading days with standard
errors, the number of observations, the evidence label, and the shock's historical size
distribution (so sizes can be expressed in standard deviations and checked against the range
ever observed).

Shocks
  fed_mp   Fed policy surprise (JK MP), per 10 bp            identified_causal
  fed_cbi  Fed information surprise (JK CBI), per 10 bp      identified_causal
  fed_pc1  Fed total surprise (JK pc1), per 10 bp            identified_causal
  boc      BoC announcement-day proxy, per 10 bp             identified_causal (daily proxy)
  opec     OPEC oil supply news (Känzig), per 1%             identified_causal
  gpr      daily jump in the Geopolitical Risk index, 1 sd   reduced_form
  tpu      daily jump in Trade Policy Uncertainty, 1 sd      reduced_form
Event shocks use the pre-registered event-sample LP; the two index shocks use a daily
time-series LP (Newey-West), because they are not announcements and not identified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from wsl.analysis import regimes
from wsl.analysis.boc_proxy import ASSUMED, proxy_events
from wsl.analysis.eventstudy import PRE_LAGS
from wsl.analysis.m4 import network_outcomes
from wsl.estimation.lp import lp
from wsl.estimation.ols import ols
from wsl.paths import CLEAN_DIR
from wsl.shocks import events as ev
from wsl.shocks.windows import load_series, window_changes

H = 20


@dataclass(frozen=True)
class ShockSpec:
    id: str
    name: str
    family: str  # fomc | boc | opec | index
    column: str
    controls: tuple[str, ...]
    per: float
    unit: str
    label: str
    description: str


SHOCKS: tuple[ShockSpec, ...] = (
    ShockSpec("fed_mp", "Fed policy surprise", "fomc", "mp_bp", ("cbi_bp",), 10.0, "bp",
              "identified_causal", "Policy component of the 30-minute rate-futures surprise "
              "around FOMC announcements (Jarociński & Karadi). Positive = hawkish."),
    ShockSpec("fed_cbi", "Fed information surprise", "fomc", "cbi_bp", ("mp_bp",), 10.0, "bp",
              "identified_causal", "Information component: rates and stocks move together, "
              "read as the Fed revealing news about the economy (Jarociński & Karadi)."),
    ShockSpec("fed_pc1", "Fed total surprise", "fomc", "pc1_bp", (), 10.0, "bp",
              "identified_causal", "The full 30-minute surprise (first principal component of "
              "rate futures), before splitting policy from information."),
    ShockSpec("boc", "Bank of Canada surprise", "boc", "proxy_bp", (), 10.0, "bp",
              "identified_causal", "Change in the Canadian 2-year yield on scheduled BoC "
              "announcement days (a daily-window proxy validated in M3; carries some common "
              "North American rate news)."),
    ShockSpec("opec", "Oil supply news", "opec", "pc_pct", (), 1.0, "%",
              "identified_causal", "Oil-futures surprise on OPEC announcement days (Känzig). "
              "Positive = news of tighter supply, oil up."),
    ShockSpec("gpr", "Geopolitical risk jump", "index", "GPRD", (), 1.0, "sd",
              "reduced_form", "One-standard-deviation daily rise in the Geopolitical Risk "
              "index (Caldara & Iacoviello). Association, not an identified shock."),
    ShockSpec("tpu", "Trade-policy uncertainty jump", "index", "TPUD", (), 1.0, "sd",
              "reduced_form", "One-standard-deviation daily rise in Trade Policy Uncertainty "
              "(daily, from 2015). Association, not an identified shock."),
)
REGIME_OUTCOMES = ("US2Y", "SPY.tr", "USD_BROAD", "WTI", "CA2Y", "CA10Y", "USDCAD", "TSX.tr",
                   "CA_BANKS_EW6", "XEG.tr")


def events_for(spec: ShockSpec) -> pd.DataFrame:
    if spec.family == "fomc":
        return ev.fomc().dropna(subset=[spec.column, *spec.controls])
    if spec.family == "boc":
        return proxy_events()
    return ev.opec()


def _event_irf(spec: ShockSpec, events: pd.DataFrame, sid: str) -> dict[str, Any]:
    """Event-sample LP for h = 0..H with HC3; also the covariance with the partner Fed shock
    (MP/CBI come from one regression, so a scenario using both needs it)."""
    s = load_series(sid, ASSUMED.get(sid))
    w = window_changes(s, events["ts"], H, PRE_LAGS)
    cols = [spec.column, *spec.controls]
    df = w.join(events[cols])
    df[cols] = df[cols] / spec.per
    pre = [f"pre{lag}" for lag in range(1, PRE_LAGS + 1)]
    X = np.column_stack([np.ones(len(df)), df[cols].to_numpy(), df[pre].to_numpy()])
    coef, se, n, cov_partner = [], [], [], []
    for h in range(H + 1):
        r = ols(df[f"d{h}"].to_numpy(), X, cov_type="HC3")
        coef.append(float(r.params[1]))
        se.append(float(np.sqrt(r.cov[1, 1])))
        n.append(r.nobs)
        cov_partner.append(float(r.cov[1, 2]) if spec.controls else 0.0)
    return {"coef": coef, "se": se, "n": n, "cov_partner": cov_partner,
            "units": "bp" if s.kind == "bp" else "%"}


def _index_shock(spec: ShockSpec) -> pd.Series:
    obs = pd.read_parquet(CLEAN_DIR / "obs.parquet", filters=[("series_id", "==", spec.column)])
    lvl = obs[obs["status"] == "ok"].set_index("date")["value"].sort_index()
    return lvl


def _index_irf(spec: ShockSpec, sid: str, level: pd.Series) -> tuple[dict[str, Any], float]:
    s = load_series(sid, ASSUMED.get(sid))
    x = pd.Series(s.x, index=s.dates)
    # Index change since the previous trading day of this outcome (weekend news included).
    lv = level.reindex(level.index.union(x.index)).ffill().reindex(x.index)
    shock = lv.diff()
    sd = float(shock.std())
    shock = shock / sd
    m = shock.notna()
    irf = lp(x[m], shock[m], range(H + 1), y_lags=PRE_LAGS, shock_lags=PRE_LAGS,
             cov_type="NW", label="reduced_form")
    return ({"coef": irf.coef.tolist(), "se": irf.se.tolist(), "n": list(irf.nobs),
             "cov_partner": [0.0] * (H + 1), "units": "bp" if s.kind == "bp" else "%"}, sd)


def distribution(values: np.ndarray, per: float) -> dict[str, float]:
    v = values[~np.isnan(values)]
    q = np.quantile(v, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {"n": float(len(v)), "sd": float(np.std(v, ddof=1)), "min": float(v.min()),
            "max": float(v.max()), "p01": float(q[0]), "p05": float(q[1]), "p50": float(q[2]),
            "p95": float(q[3]), "p99": float(q[4]), "per": per}


def build_catalog(outcomes: list[str] | None = None) -> dict[str, Any]:
    outs = outcomes or network_outcomes()
    meta = pd.read_parquet(CLEAN_DIR / "series.parquet").set_index("series_id")
    shocks_out, irfs = [], {}
    for spec in SHOCKS:
        if spec.family == "index":
            level = _index_shock(spec)
            per_out = {}
            sd = float("nan")
            for sid in outs:
                per_out[sid], sd = _index_irf(spec, sid, level)
            dist_vals = (level.diff() / level.diff().std()).to_numpy()
            dist = distribution(dist_vals, 1.0) | {"index_sd": sd}
        else:
            events = events_for(spec)
            per_out = {sid: _event_irf(spec, events, sid) for sid in outs}
            dist = distribution(events[spec.column].to_numpy(dtype=float), spec.per)
        irfs[spec.id] = per_out
        shocks_out.append({**spec.__dict__, "controls": list(spec.controls),
                           "distribution": dist})
    outcome_meta = [{"id": sid, "name": str(meta.loc[sid, "name"]),
                     "units": "bp" if str(meta.loc[sid, "units"]) == "pct" else "%",
                     "country": str(meta.loc[sid, "country"]),
                     "asset_class": str(meta.loc[sid, "asset_class"]),
                     "policy": str(meta.loc[sid, "policy"])} for sid in outs]
    return {"horizons": list(range(H + 1)), "shocks": shocks_out, "outcomes": outcome_meta,
            "irfs": irfs}


def build_regime_irfs(include_vix: bool) -> dict[str, Any]:
    """Responses estimated separately within each regime (subsample event LPs), for the
    explorer's regime toggle. The pre-registered tests of differences are in docs/RESULTS.md."""
    out: dict[str, Any] = {}
    regime_defs = [("cycle", ["hiking", "cutting", "hold"]),
                   ("high_inflation", ["inflation > 3%", "inflation ≤ 3%"])]
    if include_vix:
        regime_defs.append(("high_vix", ["high VIX", "low VIX"]))
    for spec in SHOCKS:
        if spec.family == "index" or spec.id == "fed_pc1":
            continue
        events = events_for(spec)
        fam = "boc" if spec.family == "boc" else "fomc"
        for regime, names in regime_defs:
            R, _ = regimes.regime_matrix(events["ts"], fam, regime)
            if regime == "cycle":
                groups = {"hiking": R[:, 0] == 1, "cutting": R[:, 1] == 1,
                          "hold": (R[:, 0] == 0) & (R[:, 1] == 0)}
            else:
                groups = {names[0]: R[:, 0] == 1, names[1]: R[:, 0] == 0}
            for gname, mask in groups.items():
                sub = events[mask]
                if len(sub) < 25:
                    continue
                for sid in REGIME_OUTCOMES:
                    out.setdefault(spec.id, {}).setdefault(regime, {}).setdefault(gname, {})[
                        sid] = _event_irf(spec, sub, sid)
    return out
