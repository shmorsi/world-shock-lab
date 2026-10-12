"""Parsers on small synthetic payloads in each source's format (no real data is committed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wsl.data import schemas
from wsl.data.cache import RawCache
from wsl.data.http import redact_url
from wsl.data.registry import SeriesSpec
from wsl.data.sources import fred, statcan, valet, worldbank, yahoo
from wsl.data.sources.base import STATUS_INVALID, STATUS_MISSING, STATUS_OK


def spec(**kw: object) -> SeriesSpec:
    base: dict[str, object] = dict(id="X", source="s", key="K", freq="D", units="pct",
                                   asset_class="rates", obs_time="16:00",
                                   obs_time_verified=False, tz="America/New_York", policy="C",
                                   event_window_ok=True, name="x")
    base.update(kw)
    return SeriesSpec(**base)  # type: ignore[arg-type]


@pytest.fixture
def cache(tmp_path: Path) -> RawCache:
    return RawCache(root=tmp_path)


def _validate(obs: pd.DataFrame) -> None:
    obs = obs.copy()
    obs["date"] = obs["date"].astype("datetime64[ns]")
    schemas.OBS.validate(obs)


def test_fred_missing_dot_becomes_flagged_nan(cache: RawCache) -> None:
    payload = {"observations": [{"date": "2020-12-24", "value": "0.12"},
                                {"date": "2020-12-25", "value": "."},
                                {"date": "2020-12-28", "value": "0.13"}]}
    cache.store("fred/observations/DGS2", json.dumps(payload).encode(), url="u", ext="json")
    p = fred.parse([spec(id="US2Y", source="fred", key="DGS2")], cache)
    assert p.obs["status"].tolist() == [STATUS_OK, STATUS_MISSING, STATUS_OK]
    assert np.isnan(p.obs["value"].iloc[1])
    assert p.artifact_of == {"US2Y": "fred/observations/DGS2"}
    _validate(p.obs)


def test_fred_artifacts_keep_key_out_of_params() -> None:
    arts = fred.artifacts([spec(source="fred", key="DGS2")])
    assert all("api_key" not in a.params and a.secret_params == {"api_key": "FRED_API_KEY"}
               for a in arts)


def test_redact_url_hides_api_key() -> None:
    url = "https://api.stlouisfed.org/fred/series?series_id=DGS2&api_key=SECRET123&file_type=json"
    red = redact_url(url)
    assert "SECRET123" not in red and "api_key=REDACTED" in red and "series_id=DGS2" in red


def test_valet_parses_and_flags_blank_values(cache: RawCache) -> None:
    payload = {"observations": [{"d": "2024-01-02", "CA2Y": {"v": "3.9"}},
                                {"d": "2024-01-03", "CA2Y": {"v": ""}},
                                {"d": "2024-01-04"}]}
    cache.store("valet/CA2Y", json.dumps(payload).encode(), url="u", ext="json")
    p = valet.parse([spec(id="CA2Y", source="valet", key="CA2Y")], cache)
    assert p.obs["value"].iloc[0] == pytest.approx(3.9)
    assert (p.obs["status"].iloc[1:] == STATUS_MISSING).all()
    _validate(p.obs)


def _chart(closes: list[float | None], adj: list[float | None],
           divs: dict[int, float]) -> dict[str, object]:
    # Timestamps at 14:30 UTC = 09:30 New York (market open), as Yahoo reports daily bars.
    ts = [int(pd.Timestamp(d, tz="UTC").timestamp()) + 14 * 3600 + 1800
          for d in pd.bdate_range("2024-01-02", periods=len(closes))]
    return {"chart": {"result": [{
        "meta": {"exchangeTimezoneName": "America/New_York", "symbol": "TEST",
                 "exchangeName": "PCX", "currency": "USD"},
        "timestamp": ts,
        "indicators": {"quote": [{"close": closes}], "adjclose": [{"adjclose": adj}]},
        "events": {"dividends": {str(ts[i]): {"amount": a, "date": ts[i]}
                                 for i, a in divs.items()}},
    }]}}


def test_yahoo_derives_total_return_and_flags_zero_price(cache: RawCache) -> None:
    payload = _chart([10.0, 0.0, 10.0, None, 11.0], [9.0, 0.0, 9.5, None, 10.5], {2: 0.5})
    cache.store("yahoo/TEST", json.dumps(payload).encode(), url="u", ext="json")
    p = yahoo.parse([spec(id="T", source="yahoo", key="TEST", policy="L")], cache)
    obs = p.obs.set_index(["series_id", "date"])
    close = obs.loc["T.close"]
    assert close["status"].tolist() == [STATUS_OK, STATUS_INVALID, STATUS_OK, STATUS_MISSING,
                                        STATUS_OK]
    tr = obs.loc["T.tr", "value"]
    assert tr.iloc[0] == pytest.approx(100.0)
    assert np.isnan(tr.iloc[1])  # the invalid zero price never enters the TR index
    assert tr.iloc[2] == pytest.approx(105.0)  # (10 + 0.5) / 10
    assert tr.iloc[4] == pytest.approx(105.0 * 11.0 / 10.0)
    assert obs.loc["T.dividend", "value"].tolist() == [0.5]
    assert pd.Timestamp(obs.loc["T.close"].index[0]) == pd.Timestamp("2024-01-02")
    _validate(p.obs)


def test_statcan_parse(cache: RawCache) -> None:
    payload = [{"status": "SUCCESS", "object": {"vectorDataPoint": [
        {"refPer": "2026-07-01", "value": 169.1, "releaseTime": "2026-08-19T08:30"},
        {"refPer": "2026-08-01", "value": 169.8, "releaseTime": "2026-09-14T08:30"}]}}]
    cache.store("statcan/v1", json.dumps(payload).encode(), url="u", ext="json")
    specs = [spec(id="CA_CPI", source="statcan", key="1", freq="M", obs_time="daily_rate",
                  event_window_ok=False)]
    p = statcan.parse(specs, cache)
    assert p.obs["value"].tolist() == [169.1, 169.8]
    rel = statcan.release_times(specs, cache)
    assert rel["release_time"].iloc[1] == pd.Timestamp("2026-09-14 08:30")


def test_statcan_rejects_failed_status(cache: RawCache) -> None:
    cache.store("statcan/v1", json.dumps([{"status": "FAILED"}]).encode(), url="u", ext="json")
    with pytest.raises(ValueError):
        statcan.parse([spec(source="statcan", key="1", freq="M")], cache)


def test_worldbank_parse_sorts_and_keeps_nulls(cache: RawCache) -> None:
    rows = [{"countryiso3code": "CAN", "date": "2024", "value": -95.0},
            {"countryiso3code": "CAN", "date": "2023", "value": None},
            {"countryiso3code": "JPN", "date": "2024", "value": 87.0}]
    cache.store("worldbank/EG.IMP.CONS.ZS/CAN", json.dumps([{"pages": 1}, rows]).encode(),
                url="u", ext="json")
    s = spec(id="CAN_NEI", source="worldbank", key="CAN", country="CAN",
             group_key="EG.IMP.CONS.ZS", freq="A")
    p = worldbank.parse([s], cache)
    assert p.obs["date"].dt.year.tolist() == [2023, 2024]
    assert p.obs["status"].tolist() == [STATUS_MISSING, STATUS_OK]
