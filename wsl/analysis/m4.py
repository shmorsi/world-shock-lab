"""M4 analyses on the built data, following protocol v3 (config/evaluation.yaml)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from wsl.analysis import regimes
from wsl.analysis.boc_proxy import ASSUMED, all_announcements, proxy_events
from wsl.analysis.eventstudy import PRE_LAGS, event_irf, monthly_series, monthly_shock
from wsl.analysis.hetero import control_times, rigobon_sack
from wsl.estimation import bvar
from wsl.estimation.irf import IRF, z_value
from wsl.estimation.lp import event_lp, lp
from wsl.estimation.lpiv import iv_just_identified, lp_iv
from wsl.estimation.multiple import bh_qvalues
from wsl.estimation.ols import ols
from wsl.estimation.state import StateResult, state_lp
from wsl.paths import CLEAN_DIR, EVALUATION_CONFIG
from wsl.shocks import events as ev
from wsl.shocks.windows import load_series, window_changes

H = 20
Z90 = z_value(0.90)
SPEC: dict[str, Any] = yaml.safe_load(EVALUATION_CONFIG.read_text())["in_sample_specs"]
HEADLINE = int(SPEC["headline_horizon"]["daily"])
HEADLINE_M = int(SPEC["headline_horizon"]["monthly"])
Q = float(SPEC["multiple_testing"]["q"])
PER = {"fomc": 10.0, "boc": 10.0, "opec": 1.0}
SHOCK_LABEL = {"fomc": "Fed policy surprise (JK MP), per 10 bp",
               "boc": "BoC announcement-day proxy, per 10 bp",
               "opec": "OPEC oil supply surprise, per 1%"}


def series_id(name: str) -> str:
    """Config names Yahoo tickers by their id (e.g. XEG); the built series is XEG.tr."""
    meta = pd.read_parquet(CLEAN_DIR / "series.parquet", columns=["series_id"])
    ids = set(meta["series_id"])
    return name if name in ids else f"{name}.tr"


def family_events(family: str) -> tuple[pd.DataFrame, str, list[str]]:
    if family == "fomc":
        return ev.fomc().dropna(subset=["mp_bp", "cbi_bp"]), "mp_bp", ["cbi_bp"]
    if family == "boc":
        return proxy_events(), "proxy_bp", []
    return ev.opec(), "pc_pct", []


# ------------------------------------------------------------------------ state dependence
@dataclass
class StateCell:
    family: str
    outcome: str
    regime: str
    result: StateResult
    q_joint: float = float("nan")
    q_headline: float = float("nan")


def run_state_dependence(boot: int = 999) -> list[StateCell]:
    cells: list[StateCell] = []
    names = {"fomc_mp": "fomc", "boc_proxy": "boc", "opec": "opec"}
    for key, spec in SPEC["state_dependence"]["cells"].items():
        family = names[key]
        events, shock, controls = family_events(family)
        for outcome in spec["outcomes"]:
            sid = series_id(outcome)
            s = load_series(sid, ASSUMED.get(sid))
            w = window_changes(s, events["ts"], H, PRE_LAGS)
            Y = w[[f"d{h}" for h in range(H + 1)]].to_numpy()
            ctrl = np.column_stack([w[[f"pre{lag}" for lag in range(1, PRE_LAGS + 1)]].to_numpy(),
                                    events[controls].to_numpy()]) if controls else \
                w[[f"pre{lag}" for lag in range(1, PRE_LAGS + 1)]].to_numpy()
            shock_v = events[shock].to_numpy() / PER[family]
            for regime in spec["regimes"]:
                R, rnames = regimes.regime_matrix(events["ts"], family, regime)
                res = state_lp(Y, shock_v, R, rnames, ctrl, list(range(H + 1)), HEADLINE,
                               boot=boot)
                cells.append(StateCell(family, sid, regime, res))
    p = np.array([c.result.p_joint for c in cells] + [c.result.p_headline for c in cells])
    q = bh_qvalues(p)
    for i, c in enumerate(cells):
        c.q_joint, c.q_headline = float(q[i]), float(q[len(cells) + i])
    return cells


# ---------------------------------------------------------------------- Canada comparison
@dataclass
class CanadaRow:
    outcome: str
    units: str
    fed: dict[int, tuple[float, float, int]]  # h -> (coef, se, n), per 10 bp US 2y move
    fed_first_stage_f: float
    boc: dict[int, tuple[float, float, int]]  # per 10 bp CA 2y move (the proxy)
    boc_rs: dict[int, tuple[float, float, float]]  # h -> (beta, lo90, hi90)


def run_canada(horizons: Sequence[int] = (0, HEADLINE)) -> list[CanadaRow]:
    fomc, _, _ = family_events("fomc")
    us2y = window_changes(load_series("US2Y"), fomc["ts"], 0)["d0"].to_numpy()
    boc = proxy_events()
    ctrl = control_times(boc["ts"], all_announcements())
    pair = ctrl.notna().to_numpy()
    ca2y = load_series("CA2Y", ASSUMED["CA2Y"])
    i_c = np.full(len(boc), np.nan)
    i_c[pair] = window_changes(ca2y, ctrl[pair], 0)["d0"].to_numpy()
    rows = []
    for outcome in SPEC["canada_comparison"]["outcomes"]:
        sid = series_id(outcome)
        s = load_series(sid, ASSUMED.get(sid))
        wf = window_changes(s, fomc["ts"], max(horizons), PRE_LAGS)
        pre = wf[[f"pre{lag}" for lag in range(1, PRE_LAGS + 1)]].to_numpy()
        W = np.column_stack([np.ones(len(fomc)), fomc["cbi_bp"].to_numpy(), pre])
        fed, f_stat = {}, float("nan")
        for h in horizons:
            beta, se, n, _pi, f_eff, _j, _a = iv_just_identified(
                wf[f"d{h}"].to_numpy(), us2y / 10.0, fomc["mp_bp"].to_numpy(), W,
                cov_type="HC1")
            fed[h] = (beta, se, n)
            f_stat = f_eff if h == 0 else f_stat
        b_irf = event_irf(boc, "proxy_bp", sid, per=10.0, horizons=max(horizons),
                          assumed_time=ASSUMED.get(sid))
        boc_d = {h: (float(b_irf.coef[h]), float(b_irf.se[h]), int(b_irf.nobs[h]))
                 for h in horizons}
        we = window_changes(s, boc["ts"], max(horizons))
        wc = pd.DataFrame(np.nan, index=boc.index, columns=we.columns)
        wc.loc[pair] = window_changes(s, ctrl[pair], max(horizons)).to_numpy()
        rs = {}
        for h in horizons:
            r = rigobon_sack(boc["proxy_bp"].to_numpy(), we[f"d{h}"].to_numpy().astype(float),
                             i_c, wc[f"d{h}"].to_numpy().astype(float))
            rs[h] = (r.beta * 10, r.ci90[0] * 10, r.ci90[1] * 10)
        rows.append(CanadaRow(sid, "bp" if s.kind == "bp" else "%", fed, f_stat, boc_d, rs))
    return rows


# ------------------------------------------------------------------------ exporter panel
@dataclass
class PanelResult:
    horizon: int
    importer: tuple[float, float]  # coef, se (per 1% oil surprise)
    exporter: tuple[float, float]
    difference: tuple[float, float, float]  # coef, se, p
    n_rows: int
    n_events: int
    countries: dict[str, float]  # country -> share of events classified exporter


def run_exporter_panel(horizons: Sequence[int] = (0, HEADLINE)) -> tuple[list[PanelResult],
                                                                         pd.DataFrame]:
    meta = pd.read_parquet(CLEAN_DIR / "series.parquet")
    etfs = meta[(meta["asset_class"] == "equity_country") & meta["series_id"].str.endswith(".tr")]
    opec = ev.opec()
    frames, per_country = [], []
    for sid, country in zip(etfs["series_id"], etfs["country"], strict=True):
        s = load_series(sid)
        w = window_changes(s, opec["ts"], max(horizons), PRE_LAGS)
        w["shock"] = opec["pc_pct"].to_numpy()
        w["country"] = country
        w["event"] = np.arange(len(opec))
        w["exporter"] = regimes.exporter_flags(pd.DataFrame({
            "country": country, "year": opec["ts"].dt.year.to_numpy()}))
        frames.append(w)
        irf = event_irf(opec, "pc_pct", sid, horizons=max(horizons))
        per_country.append({"country": country, "etf": sid.removesuffix(".tr"),
                            **{f"h{h}": (float(irf.coef[h]), float(irf.se[h]),
                                         int(irf.nobs[h])) for h in horizons},
                            "exporter_share": float(np.nanmean(w["exporter"]))})
    panel = pd.concat(frames, ignore_index=True)
    countries = sorted(panel["country"].unique())
    results = []
    for h in horizons:
        d = panel.dropna(subset=[f"d{h}", "shock", "exporter"] +
                         [f"pre{lag}" for lag in range(1, PRE_LAGS + 1)])
        fe = np.column_stack([(d["country"] == c).to_numpy(float) for c in countries])
        X = np.column_stack([d["shock"], d["shock"] * d["exporter"], d["exporter"],
                             d[[f"pre{lag}" for lag in range(1, PRE_LAGS + 1)]].to_numpy(), fe])
        r = ols(d[f"d{h}"].to_numpy(), X, cov_type="CL", groups=d["event"].to_numpy())
        V = r.cov
        b, c = r.params[0], r.params[1]
        se_exp = float(np.sqrt(V[0, 0] + V[1, 1] + 2 * V[0, 1]))
        p = float(2 * (1 - norm.cdf(abs(c / np.sqrt(V[1, 1])))))
        share = {str(k): float(v) for k, v in d.groupby("country")["exporter"].mean().items()}
        results.append(PanelResult(h, (float(b), float(np.sqrt(V[0, 0]))), (float(b + c), se_exp),
                                   (float(c), float(np.sqrt(V[1, 1])), p), len(d),
                                   int(d["event"].nunique()), share))
    return results, pd.DataFrame(per_country)


# ------------------------------------------------------------------------------- network
def network_outcomes() -> list[str]:
    """Every daily series usable for event windows, except: VIX and Moody's Baa (O-9, never
    published); FRED's copyrighted breakevens (the public-domain versions are used); Yahoo
    price fields other than total return; and series with an unknown closing time, apart from
    the GoC yields whose close M3 established (O-1)."""
    meta = pd.read_parquet(CLEAN_DIR / "series.parquet")
    m = meta[(meta["freq"] == "D") & meta["event_window_ok"].astype(bool)]
    m = m[~m["series_id"].str.endswith((".close", ".adjclose", ".dividend"))]
    m = m[(m["obs_time"] != "close_unknown") | m["series_id"].isin(list(ASSUMED))]
    return sorted(set(m["series_id"]) - {"VIX", "BAA10Y", "US_BE5Y", "US_BE10Y"})


def run_network(horizons: Sequence[int] = tuple(SPEC["spillover_network_horizons"])
                ) -> pd.DataFrame:
    meta = pd.read_parquet(CLEAN_DIR / "series.parquet").set_index("series_id")
    rows = []
    for family in ("fomc", "boc", "opec"):
        events, shock, controls = family_events(family)
        for sid in network_outcomes():
            irf = event_irf(events, shock, sid, other_shocks=controls, per=PER[family],
                            horizons=max(horizons), assumed_time=ASSUMED.get(sid))
            for h in horizons:
                t = irf.coef[h] / irf.se[h]
                rows.append({"shock": family, "outcome": sid, "h": h,
                             "coef": float(irf.coef[h]), "se": float(irf.se[h]),
                             "p": float(2 * (1 - norm.cdf(abs(t)))), "n": int(irf.nobs[h]),
                             "units": irf.meta["units"], "country": str(meta.loc[sid, "country"]),
                             "asset_class": str(meta.loc[sid, "asset_class"])})
    df = pd.DataFrame(rows)
    df["q"] = bh_qvalues(df["p"].to_numpy())
    df["edge"] = df["q"] <= Q
    return df


# ---------------------------------------------------------------------------- robustness
@dataclass
class RobustRow:
    family: str
    outcome: str
    h: int
    variants: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # coef, lo, hi

    @property
    def verdict(self) -> str:
        signs = {np.sign(c) for c, _, _ in self.variants.values()}
        sig = [lo > 0 or hi < 0 for _, lo, hi in self.variants.values()]
        if len(signs) == 1 and sum(sig) >= (2 / 3) * len(sig):
            return "robust"
        return "fragile"


def _band(irf: IRF, h: int) -> tuple[float, float, float]:
    c, s = float(irf.coef[h]), float(irf.se[h])
    return c, c - Z90 * s, c + Z90 * s


def _subset(events: pd.DataFrame, which: str) -> pd.DataFrame:
    year = events["ts"].dt.year
    keep = {"pre_2008": year <= 2007, "post_2008": year >= 2009}.get(which, year != 2020)
    out: pd.DataFrame = events[keep]
    return out


def _zero_filled(events: pd.DataFrame, shock: str, sid: str, per: float, h: int) -> IRF:
    s = load_series(sid, ASSUMED.get(sid))
    day0 = window_changes(s, events["ts"], 0)["day0_date"]
    x = pd.Series(s.x, index=s.dates)
    shock_daily = pd.Series(0.0, index=s.dates)
    vals = (events[shock] / per).groupby(day0.to_numpy()).sum()
    shock_daily.loc[shock_daily.index.intersection(vals.index)] = vals
    first, last = events["ts"].min().tz_localize(None), events["ts"].max().tz_localize(None)
    m = (x.index >= first.normalize()) & (x.index <= last.normalize())
    return lp(x[m], shock_daily[m], range(h + 1), y_lags=5, shock_lags=5, cov_type="NW")


def run_robustness(h_list: Sequence[int] = (0, HEADLINE)) -> list[RobustRow]:
    names = {"fomc_mp": "fomc", "boc_proxy": "boc", "opec": "opec"}
    rows = []
    for key, spec in SPEC["state_dependence"]["cells"].items():
        family = names[key]
        events, shock, controls = family_events(family)
        per = PER[family]
        for outcome in spec["outcomes"]:
            sid = series_id(outcome)
            a = ASSUMED.get(sid)

            def run(evs: pd.DataFrame, sh: str = shock, ctl: Sequence[str] = tuple(controls),
                    pre: int = PRE_LAGS, sid: str = sid, a: str | None = a,
                    per: float = per) -> IRF:
                s = load_series(sid, a)
                w = window_changes(s, evs["ts"], max(h_list), pre).join(evs[[sh, *ctl]])
                w[sh] = w[sh] / per
                cols = [f"pre{lag}" for lag in range(1, pre + 1)] + list(ctl)
                return event_lp(w, {h: f"d{h}" for h in range(max(h_list) + 1)}, sh, cols)

            variants: dict[str, IRF] = {
                "baseline": run(events), "no_controls": run(events, pre=0),
                "ten_pre_lags": run(events, pre=10),
                "pre_2008": run(_subset(events, "pre_2008")),
                "post_2008": run(_subset(events, "post_2008")),
                "ex_2020": run(_subset(events, "ex_2020")),
                "zero_filled_daily_nw": _zero_filled(events, shock, sid, per, max(h_list)),
            }
            if family == "fomc":
                variants["fed_shock_pc1"] = run(ev.fomc().dropna(subset=["pc1_bp"]), "pc1_bp", ())
                variants["fed_shock_bs_orth"] = run(ev.bauer_swanson(), "mps_orth_bp", ())
            for h in h_list:
                row = RobustRow(family, sid, h, {k: _band(v, h) for k, v in variants.items()})
                if family == "boc":
                    ctrl = control_times(events["ts"], all_announcements())
                    pair = ctrl.notna().to_numpy()
                    s = load_series(sid, a)
                    ye = window_changes(s, events["ts"], h)[f"d{h}"].to_numpy()
                    yc = np.full(len(events), np.nan)
                    yc[pair] = window_changes(s, ctrl[pair], h)[f"d{h}"].to_numpy()
                    ic = np.full(len(events), np.nan)
                    ic[pair] = window_changes(load_series("CA2Y", ASSUMED["CA2Y"]), ctrl[pair],
                                              0)["d0"].to_numpy()
                    r = rigobon_sack(events["proxy_bp"].to_numpy(), ye, ic, yc)
                    row.variants["rigobon_sack"] = (r.beta * per, r.ci90[0] * per,
                                                    r.ci90[1] * per)
                rows.append(row)
    return rows


# ------------------------------------------------------------------------- VAR vs LP

def monthly_frame(names: Sequence[str]) -> pd.DataFrame:
    cols = {}
    for n in names:
        if n in ("US1Y", "CA2Y", "USDCAD"):
            obs = pd.read_parquet(CLEAN_DIR / "obs.parquet", filters=[("series_id", "==", n)])
            s = obs[obs["status"] == "ok"].set_index("date")["value"]
            months = pd.DatetimeIndex(s.index).to_period("M")
            m = s.groupby(months).mean()
            m.index = pd.PeriodIndex(m.index).to_timestamp()
            cols[n] = m if n != "USDCAD" else 100 * np.log(m)
        else:
            cols[n] = monthly_series(n)
    return pd.DataFrame(cols)


@dataclass
class VarLpResult:
    instrument: str
    variables: list[str]
    sample: tuple[str, str]
    T: int
    var: bvar.ProxyBVARResult
    lp: dict[str, IRF]
    lp_first_stage_f: float


def run_var_vs_lp(instrument: str, draws: int | None = None) -> VarLpResult:
    vspec = SPEC["var"]
    names = list(vspec["variables"])
    df = monthly_frame(names)
    z = monthly_shock(instrument)
    df = df.join(z.rename("z"), how="inner").dropna()
    Y, zz = df[names].to_numpy(), df["z"].to_numpy()
    impact = 0.25
    res = bvar.proxy_bvar(Y, zz, p=int(vspec["lags"]), horizons=24, impact=impact,
                          lam=float(vspec["prior"]["overall_tightness"]),
                          own_lag_mean=float(vspec["prior"]["own_lag_mean"]),
                          draws=draws or int(vspec["bootstrap"]["draws"]),
                          seed=int(vspec["bootstrap"]["seed"]))
    lps, f_min = {}, float("inf")
    for n in names:
        irf, diag = lp_iv(df[n], df["US1Y"], df["z"], range(25), lags=int(vspec["lags"]),
                          cov_type="NW")
        lps[n] = IRF(irf.horizons, irf.coef * impact, irf.se * impact, irf.nobs, "lp_iv",
                     irf.cov_type, "identified_causal", outcome=n, shock=instrument,
                     meta={**irf.meta, "first_stage_f_min": float(diag.effective_f.min())})
        f_min = min(f_min, float(diag.effective_f[0]))
    return VarLpResult(instrument, names, (str(df.index.min().date()), str(df.index.max().date())),
                       len(df), res, lps, f_min)
