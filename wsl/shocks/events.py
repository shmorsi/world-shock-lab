"""Announcement event sets with timezone-aware timestamps and shock columns in common units.

Units: Fed and BoC surprises in basis points; OPEC surprises in percent (oil futures).
"""

from __future__ import annotations

import pandas as pd

from wsl.paths import CLEAN_DIR

NY = "America/New_York"
TORONTO = "America/Toronto"


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(CLEAN_DIR / f"{name}.parquet")


def fomc() -> pd.DataFrame:
    """Jarociński-Karadi FOMC events: pc1, MP and CBI (median rotation) in bp."""
    f = _read("events_fomc")
    out = pd.DataFrame({
        "ts": f["ts"],
        "pc1_bp": 100 * f["pc1"],
        "mp_bp": 100 * f["MP_median"],
        "cbi_bp": 100 * f["CBI_median"],
        "event": f["event"],
    })
    out.index = f["ts"].dt.strftime("%Y-%m-%d %H:%M").rename("event_id")
    return out


def bauer_swanson() -> pd.DataFrame:
    """Bauer-Swanson (2023) FOMC surprises, 1988-2023: MPS and MPS_ORTH in bp."""
    b = _read("events_fomc_bs")
    out = pd.DataFrame({"ts": b["ts"], "mps_bp": 100 * b["MPS"],
                        "mps_orth_bp": 100 * b["MPS_ORTH"], "unscheduled": b["Unscheduled"]})
    out.index = b["ts"].dt.strftime("%Y-%m-%d %H:%M").rename("event_id")
    return out


def opec() -> pd.DataFrame:
    """Känzig OPEC announcements. The time of day is not in the data; the announcement day
    is treated as starting at 00:00 New York time, so day 0 is that day's close for closing
    series and the next trading day's close for weekend announcements, matching the daily
    futures window the surprises are measured on."""
    o = _read("events_opec")
    ts = o["date"].dt.tz_localize(NY)
    out = pd.DataFrame({"ts": ts, "pc_pct": o["pc_fullsample"], "front_pct": o["Front"],
                        "weekend": o["weekend"]})
    out.index = o["date"].dt.strftime("%Y-%m-%d").rename("event_id")
    return out


def boc(scheduled_only: bool = True, exclude_fomc_same_day: bool = True) -> pd.DataFrame:
    """BoC rate announcements with their announcement time (ET). Defaults follow protocol v2
    (config/evaluation.yaml: scheduled only, FOMC same-day events excluded)."""
    b = _read("events_boc")
    fomc_days = set(_read("events_fomc")["date"].dt.normalize())
    b = b[b["time_et"] != "unknown"].copy()
    b["same_day_fomc"] = b["date"].isin(fomc_days)
    if scheduled_only:
        b = b[b["scheduled"]]
    if exclude_fomc_same_day:
        b = b[~b["same_day_fomc"]]
    ts = (b["date"] + pd.to_timedelta(b["time_et"] + ":00")).dt.tz_localize(TORONTO)
    out = pd.DataFrame({"ts": ts, "decision": b["decision"], "change_bp": b["change_bp"],
                        "scheduled": b["scheduled"], "same_day_fomc": b["same_day_fomc"]})
    out.index = b["date"].dt.strftime("%Y-%m-%d").rename("event_id")
    return out
