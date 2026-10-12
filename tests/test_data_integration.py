"""Checks on the built data. Run with `uv run pytest -m data` after `make data`."""

from __future__ import annotations

import pandas as pd
import pytest

from wsl.data import schemas
from wsl.data.cache import RawCache
from wsl.paths import CLEAN_DIR, ROOT
from wsl.shocks import sources as shock_sources
from wsl.shocks.boc_fad import load_target_rate, validate

pytestmark = pytest.mark.data


def _read(name: str) -> pd.DataFrame:
    path = CLEAN_DIR / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"{path.name} not built")
    return pd.read_parquet(path)


def test_observations_pass_schema() -> None:
    schemas.OBS.validate(_read("obs"))


def test_every_series_has_provenance() -> None:
    obs, series = _read("obs"), _read("series")
    assert set(obs["series_id"]) == set(series["series_id"])
    cache = RawCache()
    for _, row in series[series["source"] != "derived"].iterrows():
        assert cache.pinned_record(row["artifact_key"]).sha256 == row["sha256"]
    derived = series[series["source"] == "derived"]
    assert set(derived["series_id"]) == {"US_BE5Y_PD", "US_BE10Y_PD", "CA_BANKS_EW6"}
    for _, row in derived.iterrows():  # every input is itself a built series
        assert set(row["url"].removeprefix("derived:").split(",")) <= set(series["series_id"])


def test_event_counts_match_m0_verification() -> None:
    assert len(_read("events_fomc")) == 334  # JK, 1990-02-08 .. 2026-09-16
    assert len(_read("events_opec")) == 169  # Känzig 2025M12 vintage
    assert len(_read("events_ecb")) == 312
    assert len(_read("events_fomc_bs")) == 361


def test_boc_fad_list_validates_against_target_rate() -> None:
    ev = _read("events_boc")
    assert validate(ev, load_target_rate()) == []
    assert ev["url"].str.startswith("https://www.bankofcanada.ca/").all()


def test_restricted_series_never_marked_redistributable() -> None:
    series = _read("series")
    yahoo = series[series["source"] == "yahoo"]
    assert (yahoo["policy"] == "L").all()


def test_monthly_shocks_cover_expected_ranges() -> None:
    m = _read("shocks_monthly")
    last = m.dropna(subset=["value"]).groupby("shock_id")["month"].max()
    assert last["JK_FED_MP"] >= pd.Timestamp("2026-09-01")
    assert last["KAENZIG_OIL_NEWS"] == pd.Timestamp("2025-12-01")
    assert last["BH_OIL_SUPPLY"] >= pd.Timestamp("2026-06-01")


def test_pinned_shas_are_documented() -> None:
    text = (ROOT / "data" / "SOURCES.md").read_text()
    for sha in (shock_sources.JK_FED_SHA, shock_sources.JK_ECB_SHA, shock_sources.KAENZIG_SHA):
        assert sha in text


def test_boc_event_set_follows_protocol_v2() -> None:
    from wsl.shocks import events as ev

    b = ev.boc()
    assert b["scheduled"].all() and not b["same_day_fomc"].any()
    assert len(b) == 195  # 207 scheduled - 11 on FOMC days - 1 with no announcement time
    assert b["ts"].dt.tz is not None


def test_results_files_carry_provenance() -> None:
    import json

    for path in sorted((ROOT / "results").rglob("*.json")):
        doc = json.loads(path.read_text())
        prov = doc["provenance"]
        assert prov["git_sha"] and prov["evaluation_config_sha256"] and prov["series"], path
        assert all(len(v["sha256"]) == 64 for v in prov["series"].values()), path
