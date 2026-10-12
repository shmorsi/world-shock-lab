"""The M5 event dataset (docs/EVALUATION_PROTOCOL.md §2-§4, config/evaluation.yaml).

One row per announcement (FOMC, BoC, OPEC). Features use only information available before
the announcement, plus the announcement's own surprise:
  * surprise components, raw (never the full-sample PCs): FOMC 30-minute futures moves
    (MP1, FF4, ED2-ED4, TFUT02, TFUT10, in bp); BoC the day-0 2-year move (bp); OPEC the
    oil futures moves (front, 3, 6, 12 months, %);
  * state at t-1: VIX close, US and Canadian 10y-2y slopes, the asset's realized volatility
    over the 20 prior days, the 6-month change in the own central bank's policy rate,
    year-on-year CPI of the month two months back, and WTI's 20-day return.
Targets: for h in {1, 5}, the change from the last observation before the announcement to
the h-th observation after it (bp for yields, 100·log for prices and FX), per primary asset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from wsl.analysis import regimes
from wsl.analysis.boc_proxy import ASSUMED, proxy_events
from wsl.paths import CLEAN_DIR, EVALUATION_CONFIG
from wsl.shocks import events as ev
from wsl.shocks.windows import WindowSeries, load_series, window_changes
from wsl.timing import utc_ns

CFG = yaml.safe_load(EVALUATION_CONFIG.read_text())
HORIZONS: list[int] = list(CFG["targets"]["horizons_trading_days"])
ASSET_NAMES: list[str] = list(CFG["targets"]["primary_assets"])
ASSETS = {"US2Y": "US2Y", "CA2Y": "CA2Y", "USDCAD": "USDCAD", "SPY": "SPY.tr",
          "TSX": "TSX.tr", "CA_BANKS_EW6": "CA_BANKS_EW6", "XEG": "XEG.tr", "WTI": "WTI"}
FAMILIES = ("fomc", "boc", "opec")
SHOCK_COLS = {
    "fomc": [f"fomc_{c}" for c in CFG["event_families"]["fomc"]["realtime_features"]],
    "boc": ["boc_d_ca2y"],
    "opec": [f"opec_{c}" for c in CFG["event_families"]["opec"]["realtime_features"]],
}
STATE_COLS = ["vix", "us_slope", "ca_slope", "policy_change_6m", "cpi_yoy", "wti_ret_20d"]
VOL_WINDOW = 20


@dataclass
class EventData:
    events: pd.DataFrame  # one row per event: family, ts, year, shocks, state
    targets: pd.DataFrame  # index = events index; columns "<asset>|h<h>"
    asset_vol: pd.DataFrame  # realized vol per asset, columns = asset names

    def target_cols(self) -> list[str]:
        return [f"{a}|h{h}" for a in ASSET_NAMES for h in HORIZONS]


def _value_before(s: WindowSeries, ts: pd.Series, lag: int = 0) -> np.ndarray:
    """x (transformed level) at the last observation strictly before each event, ``lag``
    observations further back; NaN when unavailable."""
    pos = np.searchsorted(utc_ns(s.obs_ts), utc_ns(ts), side="left") - 1 - lag
    ok = pos >= 0
    out = np.full(len(ts), np.nan)
    out[ok] = s.x[pos[ok]]
    return out


def _realized_vol(s: WindowSeries, ts: pd.Series) -> np.ndarray:
    pos = np.searchsorted(utc_ns(s.obs_ts), utc_ns(ts), side="left") - 1
    out = np.full(len(ts), np.nan)
    d = np.diff(s.x)
    for i, p in enumerate(pos):
        if p >= VOL_WINDOW:
            out[i] = float(np.std(d[p - VOL_WINDOW:p], ddof=1))
    return out


def _events() -> pd.DataFrame:
    f = ev.fomc()
    raw = pd.read_parquet(CLEAN_DIR / "events_fomc.parquet")
    fomc = pd.DataFrame({"family": "fomc", "ts": f["ts"].to_numpy()})
    for c in CFG["event_families"]["fomc"]["realtime_features"]:
        fomc[f"fomc_{c}"] = 100.0 * raw[f"raw_{c}"].to_numpy()
    fomc.index = "fomc:" + f.index
    b = proxy_events(exclude_fomc_same_day=False)
    boc = pd.DataFrame({"family": "boc", "ts": b["ts"].to_numpy(),
                        "boc_d_ca2y": b["proxy_bp"].to_numpy(),
                        "same_day": b["same_day_fomc"].to_numpy()}, index="boc:" + b.index)
    o = ev.opec()
    raw_o = pd.read_parquet(CLEAN_DIR / "events_opec.parquet")
    opec = pd.DataFrame({"family": "opec", "ts": o["ts"].to_numpy()}, index="opec:" + o.index)
    for c in CFG["event_families"]["opec"]["realtime_features"]:
        opec[f"opec_{c}"] = raw_o[c].to_numpy()
    out = pd.concat([fomc, boc, opec])
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out["year"] = out["ts"].dt.tz_convert("America/New_York").dt.year
    for fam, cols in SHOCK_COLS.items():  # other families' surprise columns are exactly zero
        for c in cols:
            if c not in out:
                out[c] = np.nan
            out.loc[out["family"] != fam, c] = 0.0
    out["boc_same_day_fomc"] = out.pop("same_day").fillna(False).astype(bool)
    return out.sort_values("ts")


def build() -> EventData:
    e = _events()
    ts = e["ts"]
    us10, us2 = load_series("US10Y"), load_series("US2Y")
    ca10, ca2 = load_series("CA10Y", ASSUMED["CA10Y"]), load_series("CA2Y", ASSUMED["CA2Y"])
    vix = load_series("VIX")
    wti = load_series("WTI")
    e["vix"] = np.exp(_value_before(vix, ts) / 100.0)  # back to index points
    e["us_slope"] = _value_before(us10, ts) - _value_before(us2, ts)  # bp
    e["ca_slope"] = _value_before(ca10, ts) - _value_before(ca2, ts)
    e["wti_ret_20d"] = _value_before(wti, ts) - _value_before(wti, ts, lag=20)
    pc = np.full(len(e), np.nan)
    cpi = np.full(len(e), np.nan)
    for fam in FAMILIES:
        m = (e["family"] == fam).to_numpy()
        key = "boc" if fam == "boc" else "fomc"
        rate = regimes.policy_target(key)
        tz = "America/Toronto" if key == "boc" else "America/New_York"
        now = regimes._before(rate, ts[m], tz)
        then = regimes._before(rate, ts[m] - pd.Timedelta(days=regimes.CYCLE_DAYS), tz)
        r = rate.to_numpy()
        ok = (now >= 0) & (then >= 0)
        vals = np.full(m.sum(), np.nan)
        vals[ok] = 100.0 * (r[now[ok]] - r[then[ok]])
        pc[m] = vals
        series = regimes._series("CA_CPI_NSA" if key == "boc" else "US_CPI_NSA")
        yoy = 100.0 * (series / series.shift(12) - 1.0)
        yoy.index = pd.DatetimeIndex(yoy.index).to_period("M")
        months = ts[m].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M") - 2
        cpi[m] = yoy.reindex(months.to_numpy()).to_numpy()
    e["policy_change_6m"] = pc
    e["cpi_yoy"] = cpi

    targets, vols = {}, {}
    for name in ASSET_NAMES:
        sid = ASSETS[name]
        s = load_series(sid, ASSUMED.get(sid))
        w = window_changes(s, ts, max(HORIZONS) - 1)
        for h in HORIZONS:
            targets[f"{name}|h{h}"] = w[f"d{h - 1}"].to_numpy()
        vols[name] = _realized_vol(s, ts)
    return EventData(events=e, targets=pd.DataFrame(targets, index=e.index),
                     asset_vol=pd.DataFrame(vols, index=e.index))


def split_of(year: pd.Series) -> pd.Series:
    """train / validation / test / live by announcement year (config splits)."""
    sp = CFG["splits"]
    y = year.astype(int)
    out = pd.Series("train", index=year.index)
    out[(y >= int(sp["validation"]["start"][:4])) & (y <= int(sp["validation"]["end"][:4]))] = \
        "validation"
    out[y >= int(sp["test"]["start"][:4])] = "test"
    return out


def in_test_window(ts: pd.Series) -> pd.Series:
    """The test split ends 2026-09-30 (config); later events belong to the live period."""
    end = pd.Timestamp(CFG["splits"]["test"]["end"], tz="America/New_York") + pd.Timedelta(days=1)
    return ts < end
