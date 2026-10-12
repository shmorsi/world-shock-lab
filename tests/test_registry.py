from __future__ import annotations

import pytest

from wsl.data.fetch import SERIES_SOURCES
from wsl.data.registry import SeriesSpec, load_registry, validate_registry


def test_registry_loads_and_is_consistent() -> None:
    specs = load_registry()
    assert len({s.id for s in specs}) == len(specs)
    assert {s.source for s in specs} <= set(SERIES_SOURCES)


def test_restricted_series_are_never_redistributable() -> None:
    by_id = {s.id: s for s in load_registry()}
    assert by_id["BAA10Y"].policy == "L"  # Moody's
    assert by_id["SPY"].policy == "L"  # Yahoo
    assert by_id["CA2Y"].policy == "D"  # CanDeal via Valet


def test_averaged_series_are_excluded_from_event_windows() -> None:
    for s in load_registry():
        if s.obs_time in ("intraday_average", "daily_rate"):
            assert not s.event_window_ok, s.id


def test_noon_fx_timing_is_documented() -> None:
    by_id = {s.id: s for s in load_registry()}
    assert by_id["USDCAD"].obs_time == "12:00" and by_id["USDCAD"].obs_time_verified


def test_validation_rejects_bad_entries() -> None:
    good = dict(id="A", source="fred", key="K", freq="D", units="pct", asset_class="rates",
                obs_time="16:00", obs_time_verified=False, tz="UTC", policy="C",
                event_window_ok=True, name="a")
    with pytest.raises(ValueError, match="duplicate"):
        validate_registry([SeriesSpec(**good), SeriesSpec(**good)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="event windows"):
        validate_registry([SeriesSpec(**{**good, "obs_time": "intraday_average"})])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="obs_time"):
        validate_registry([SeriesSpec(**{**good, "obs_time": "25:00"})])  # type: ignore[arg-type]
