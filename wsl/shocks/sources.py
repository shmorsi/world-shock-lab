"""Published shock series -> typed event tables and a long monthly shock table.

Pinned versions (data/SOURCES.md §1): GitHub files are fetched by commit SHA, so the bytes
can only change if we change the SHA here.
"""

from __future__ import annotations

import io

import pandas as pd

from wsl.data.cache import RawCache
from wsl.data.sources.base import Artifact

JK_FED_SHA = "f788aef7d573f87f28b70ffc3763e73c4886a53f"
JK_ECB_SHA = "f7ffc821b0ade71dd38539e4044e25368dfb4dc1"
KAENZIG_SHA = "3496b6e848ba760d4be25289fc68c1e989998790"
KAENZIG_VINTAGE = "2025M12"

_GH = "https://raw.githubusercontent.com"
_JK_FED = f"{_GH}/marekjarocinski/jkshocks_update_fed/{JK_FED_SHA}"
_JK_ECB = f"{_GH}/marekjarocinski/jkshocks_update_ecb/{JK_ECB_SHA}"
_KAENZIG = f"{_GH}/dkaenzig/oilsupplynews/{KAENZIG_SHA}"
_FRBSF = "https://www.frbsf.org/wp-content/uploads"
_GDRIVE = "https://drive.google.com/uc?export=download&id="
FOMC_TZ = "America/New_York"

ARTIFACTS = [
    Artifact("jk_fed/shocks_t", f"{_JK_FED}/shocks_fed_jk_t.csv", "csv"),
    Artifact("jk_fed/shocks_m", f"{_JK_FED}/shocks_fed_jk_m.csv", "csv"),
    Artifact("jk_fed/surprises", f"{_JK_FED}/source_data/fomc_surprises_jk.csv", "csv"),
    Artifact("jk_ecb/shocks_d", f"{_JK_ECB}/shocks_ecb_mpd_me_d.csv", "csv"),
    Artifact("jk_ecb/shocks_m", f"{_JK_ECB}/shocks_ecb_mpd_me_m.csv", "csv"),
    Artifact("kaenzig/oil_supply_news",
             f"{_KAENZIG}/oilSupplyNewsShocks_{KAENZIG_VINTAGE}.xlsx", "xlsx"),
    Artifact("bauer_swanson/surprises", f"{_FRBSF}/monetary-policy-surprises-data.xlsx", "xlsx"),
    Artifact("baumeister_hamilton/supply", f"{_GDRIVE}1OsA8btgm2rmDucUFngiLkwv4uywTDmya", "xlsx"),
    Artifact("baumeister_hamilton/demand", f"{_GDRIVE}1neFXLrIvGwggebQRwjmtrWK-dfQZ9NH8", "xlsx"),
]


def artifacts() -> list[Artifact]:
    return list(ARTIFACTS)


def _csv(cache: RawCache, key: str) -> pd.DataFrame:
    content, _ = cache.read(key)
    return pd.read_csv(io.BytesIO(content))


def _xlsx(cache: RawCache, key: str, header: int | None = 0) -> dict[str, pd.DataFrame]:
    content, _ = cache.read(key)
    book: dict[str, pd.DataFrame] = pd.read_excel(io.BytesIO(content), sheet_name=None,
                                                  header=header)
    return book


# ------------------------------------------------------------------------------------- FOMC
def fomc_events(cache: RawCache) -> pd.DataFrame:
    """JK shocks per announcement, joined with JK's raw 30-minute surprises.

    ``ts`` is the announcement time in America/New_York (JK timestamps; Bloomberg from 1997,
    Bauer-Swanson before). Raw surprise columns are prefixed ``raw_``.
    """
    shocks = _csv(cache, "jk_fed/shocks_t")
    raw = _csv(cache, "jk_fed/surprises")
    shocks["ts_naive"] = pd.to_datetime(shocks["start"])
    raw["ts_naive"] = pd.to_datetime(raw["start"])
    if raw["ts_naive"].duplicated().any():
        dup = raw.loc[raw["ts_naive"].duplicated(keep=False), "start"].unique().tolist()
        raise ValueError(f"duplicate timestamps in JK raw surprises: {dup[:5]}")
    raw = raw.drop(columns=["start"]).rename(
        columns={c: f"raw_{c}" for c in raw.columns if c not in ("ts_naive", "Event")}
    )
    out = shocks.drop(columns=["start"]).merge(raw, on="ts_naive", how="left", validate="1:1")
    if out["Event"].isna().any():
        raise ValueError("some JK shocks have no matching raw-surprise row")
    out["ts"] = out["ts_naive"].dt.tz_localize(FOMC_TZ)
    out["date"] = out["ts_naive"].dt.normalize()
    out = out.rename(columns={"Event": "event", "SP500": "sp500_hf"})
    out["source"] = f"jk_fed@{JK_FED_SHA[:12]}"
    cols = ["ts", "date", "event", "pc1", "sp500_hf", "MP_pm", "CBI_pm", "MP_median",
            "CBI_median"] + [c for c in out.columns if c.startswith("raw_")] + ["source"]
    return out[cols].sort_values("ts").reset_index(drop=True)


def fomc_events_bs(cache: RawCache) -> pd.DataFrame:
    """Bauer-Swanson (2023) announcement-level surprises, 'FOMC (update 2023)' sheet."""
    df = _xlsx(cache, "bauer_swanson/surprises")["FOMC (update 2023)"].copy()
    times = pd.to_datetime(df["Time"].str.strip(), format="%I:%M%p").dt.time
    naive = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + times.astype(str))
    df.insert(0, "ts", naive.dt.tz_localize(FOMC_TZ))
    df.insert(1, "date", df["Date"].dt.normalize())
    df["Unscheduled"] = df["Unscheduled"].astype(bool)
    df["source"] = "bauer_swanson@frbsf"
    return df.drop(columns=["Date", "Time"]).sort_values("ts").reset_index(drop=True)


# -------------------------------------------------------------------------------------- ECB
def ecb_events(cache: RawCache) -> pd.DataFrame:
    df = _csv(cache, "jk_ecb/shocks_d")
    df["date"] = pd.to_datetime(df["date"])
    df["source"] = f"jk_ecb@{JK_ECB_SHA[:12]}"
    return df.rename(columns={"STOXX50": "stoxx50_hf"}).sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------------------------------- OPEC
def opec_events(cache: RawCache) -> pd.DataFrame:
    """Känzig daily OPEC surprises (% change in WTI futures, front to 12M) + full-sample PC."""
    df = _xlsx(cache, "kaenzig/oil_supply_news")["Daily"].copy()
    df = df.rename(columns={"Date/Contract": "date", "PC": "pc_fullsample"})
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.day_name()
    df["weekend"] = df["date"].dt.dayofweek >= 5
    df["source"] = f"kaenzig@{KAENZIG_SHA[:12]}:{KAENZIG_VINTAGE}"
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------------- monthly
def _month(year: pd.Series, month: pd.Series) -> pd.Series:
    return pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": 1}))


def _long(df: pd.DataFrame, cols: dict[str, str], source: str) -> pd.DataFrame:
    parts = [
        pd.DataFrame({"shock_id": sid, "month": df["month"], "value": df[col], "source": source})
        for col, sid in cols.items()
    ]
    return pd.concat(parts, ignore_index=True)


def monthly_shocks(cache: RawCache) -> pd.DataFrame:
    frames = []

    jk = _csv(cache, "jk_fed/shocks_m")
    jk["month"] = _month(jk["year"], jk["month"])
    frames.append(_long(jk, {"pc1_hf": "JK_FED_PC1", "MP_median": "JK_FED_MP",
                             "CBI_median": "JK_FED_CBI", "MP_pm": "JK_FED_MP_PM",
                             "CBI_pm": "JK_FED_CBI_PM"}, f"jk_fed@{JK_FED_SHA[:12]}"))

    ecb = _csv(cache, "jk_ecb/shocks_m")
    ecb["month"] = _month(ecb["year"], ecb["month"])
    frames.append(_long(ecb, {"pc1_hf": "JK_ECB_PC1", "MP_median": "JK_ECB_MP",
                              "CBI_median": "JK_ECB_CBI"}, f"jk_ecb@{JK_ECB_SHA[:12]}"))

    bs = _xlsx(cache, "bauer_swanson/surprises")["Monthly (update 2023)"].copy()
    bs["month"] = _month(bs["Year"], bs["Month"])
    frames.append(_long(bs, {"MPS": "BS_MPS", "MPS_ORTH": "BS_MPS_ORTH"}, "bauer_swanson@frbsf"))

    kz = _xlsx(cache, "kaenzig/oil_supply_news")["Monthly"].copy()
    ym = kz["Date"].str.extract(r"^(\d{4})M(\d{2})$").astype(int)
    kz["month"] = _month(ym[0], ym[1])
    frames.append(_long(kz, {"Oil supply surprise series": "KAENZIG_OIL_SURPRISE",
                             "Oil supply news shock": "KAENZIG_OIL_NEWS"},
                        f"kaenzig@{KAENZIG_SHA[:12]}:{KAENZIG_VINTAGE}"))

    frames.append(_bh(cache, "baumeister_hamilton/supply", {"oil supply shocks": "BH_OIL_SUPPLY"}))
    frames.append(_bh(cache, "baumeister_hamilton/demand", {
        "economic activity shocks": "BH_ECON_ACTIVITY",
        "oil consumption demand shocks": "BH_OIL_CONSUMPTION_DEMAND",
        "oil inventory demand shocks": "BH_OIL_INVENTORY_DEMAND",
    }))
    out = pd.concat(frames, ignore_index=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.sort_values(["shock_id", "month"]).reset_index(drop=True)


def _bh(cache: RawCache, key: str, labels: dict[str, str]) -> pd.DataFrame:
    """Baumeister-Hamilton sheets: header labels on row 1, dates in column 0 from row 2."""
    sheet = next(iter(_xlsx(cache, key, header=None).values()))
    header = sheet.iloc[1].tolist()
    body = sheet.iloc[2:]
    body = body[pd.to_datetime(body[0], errors="coerce").notna()]
    df = pd.DataFrame({"month": pd.to_datetime(body[0]).dt.to_period("M").dt.to_timestamp()})
    for label, sid in labels.items():
        if label not in header:
            raise ValueError(f"{key}: expected column {label!r}, found {header}")
        df[sid] = body[header.index(label)].astype("float64").to_numpy()
    return _long(df, {sid: sid for sid in labels.values()}, f"{key}@sha-pinned")

