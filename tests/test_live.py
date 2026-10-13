"""Live mode (M8): the model, the calendar, publication rules and the licence filter on scores.
Synthetic data only; no network."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wsl.live import run
from wsl.live.data import LiveData, calendar, config
from wsl.live.model import event_frame, fit, fit_cell
from wsl.shocks.windows import make_series


def test_fit_cell_recovers_line_and_intervals_cover() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 5, 400)
    y = 1.0 + 0.5 * x + rng.normal(0, 2, 400)
    a, b, half = fit_cell(x, y, [0.68, 0.90])
    assert a == pytest.approx(1.0, abs=0.3) and b == pytest.approx(0.5, abs=0.05)
    xn = rng.normal(0, 5, 4000)
    yn = 1.0 + 0.5 * xn + rng.normal(0, 2, 4000)
    err = np.abs(yn - (a + b * xn))
    assert np.mean(err <= half["68"]) == pytest.approx(0.68, abs=0.04)
    assert np.mean(err <= half["90"]) == pytest.approx(0.90, abs=0.03)


def test_calendar_flags_collisions() -> None:
    cal = calendar(config())
    col = cal.groupby("decision_date")["collision"].first()
    assert bool(col["2026-10-28"]) and bool(col["2026-12-09"])
    assert not bool(col["2027-03-03"]) and not bool(col["2027-03-17"])
    fomc = cal[cal["bank"] == "fomc"].iloc[0]["ts"]
    assert fomc.tz_convert("America/New_York").strftime("%H:%M") == "14:00"


# ---------------------------------------------------------------------- synthetic world
def _world(seed: int = 1) -> tuple[LiveData, pd.DatetimeIndex]:
    """Business days 2015-2026; announcements every 30 days; outcomes respond to the proxy."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-01", "2026-10-30")
    ann = days[40::30]
    us2 = 2.0 + np.cumsum(rng.normal(0, 0.03, len(days)))
    ca2 = 1.5 + np.cumsum(rng.normal(0, 0.03, len(days)))
    jump = pd.Series(0.0, index=days)
    jump[ann] = rng.normal(0, 0.06, len(ann))  # percentage points
    us2 = us2 + jump.cumsum().to_numpy()
    ca2 = ca2 + 0.6 * jump.cumsum().to_numpy()
    fx = 130.0 * np.exp(np.cumsum(rng.normal(0, 0.004, len(days))) + 0.03 * jump.cumsum())
    banks = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))

    def ws(sid: str, v: np.ndarray, obs_time: str, tz: str, units: str):  # type: ignore[no-untyped-def]
        from wsl.shocks.windows import kind_for

        return make_series(sid, pd.Series(v, index=days), obs_time, tz, kind_for(units))

    series = {
        "US2Y": ws("US2Y", us2, "15:30", "America/New_York", "pct"),
        "CA2Y": ws("CA2Y", ca2, "16:00", "America/Toronto", "pct"),
        "USDCAD": ws("USDCAD", fx, "12:00", "America/New_York", "fx"),
        "CA_BANKS_EW6": ws("CA_BANKS_EW6", banks, "16:00", "America/Toronto", "index"),
    }
    hist = pd.Series([pd.Timestamp(d.date().isoformat() + " 14:00", tz="America/New_York")
                      for d in ann])
    prov = {k: {"artifact": k, "sha256": "0" * 64, "retrieved_at": "x", "url": ""}
            for k in [*series, "JK"]}
    data = LiveData(series=series, provenance=prov,
                    policy={"US2Y": "C", "USDCAD": "C", "CA2Y": "D", "CA_BANKS_EW6": "L"},
                    fomc_history=hist, boc_history=hist,
                    jk_surprises=pd.DataFrame({"ts": hist, "TFUT02_bp": 1.0}))
    return data, ann


def test_event_frame_and_fit_on_synthetic_world() -> None:
    data, ann = _world()
    frame = event_frame(data.fomc_history, data.series["US2Y"],
                        {"CA2Y": data.series["CA2Y"]}, [1, 5])
    models = fit(frame, ["CA2Y"], [1, 5], [0.68, 0.9], 40)
    # The proxy also carries the day's ordinary noise (sd 3 bp vs 6 bp jumps), so the slope is
    # attenuated: 0.6 x 36 / (36 + 9) = 0.48. This is the proxy model's known errors-in-
    # variables bias, and why it is a forecast, not a causal estimate.
    assert models["CA2Y|h1"].b == pytest.approx(0.48, abs=0.08)
    assert models["CA2Y|h1"].n == len(ann)


@pytest.fixture
def live_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(run, "FORECASTS", tmp_path / "forecasts")
    monkeypatch.setattr(run, "SCORES", tmp_path / "scores")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    return tmp_path


def _cfg() -> dict:  # type: ignore[type-arg]
    cfg = config()
    cfg["calendar"] = {"fomc": {"time": "14:00", "tz": "America/New_York",
                                "dates": ["2026-10-07"]},
                       "boc": {"time": "09:45", "tz": "America/Toronto",
                               "dates": ["2026-10-07", "2026-10-14"]}}
    cfg["model"]["sample_start"] = "2015-01-01"
    return cfg


def test_publication_window_and_never_overwrite(live_dirs: Path) -> None:
    data, _ = _world()
    cfg = _cfg()
    early = datetime(2026, 9, 1, tzinfo=UTC)
    assert run.forecast_due(cfg, data, early) == []  # more than 10 days ahead
    late = datetime(2026, 10, 6, 19, 0, tzinfo=UTC)  # under 24 h before both 10-07 decisions
    assert [p.name for p in run.forecast_due(cfg, data, late)] == ["2026-10-14_boc.json"]
    ok = datetime(2026, 10, 1, tzinfo=UTC)
    paths = run.forecast_due(cfg, data, ok)
    assert sorted(p.name for p in paths) == ["2026-10-07_boc.json", "2026-10-07_fomc.json"]
    f = json.loads(paths[0].read_text())
    assert f["collision"] and f["label"] == "model_forecast"
    assert len(f["grid"]) == 3 * 3 * 2
    with pytest.raises(FileExistsError):
        run._write_new(paths[0], {})
    assert run.forecast_due(cfg, data, ok) == []  # already published: nothing rewritten


def test_scores_follow_the_licence_filter(live_dirs: Path) -> None:
    data, _ = _world()
    cfg = _cfg()
    run.forecast_due(cfg, data, datetime(2026, 10, 1, tzinfo=UTC))
    wrote = run.score_due(cfg, data, datetime(2026, 10, 20, tzinfo=UTC))
    names = sorted(p.name for p in wrote)
    assert "2026-10-07_fomc_score1.json" in names and "2026-10-07_boc_score1.json" in names
    assert "2026-10-07_fomc_score2.json" not in names  # no JK row for that date

    fomc = json.loads((live_dirs / "scores" / "2026-10-07_fomc_score1.json").read_text())
    assert fomc["realized_proxy_bp"] is not None  # US 2Y is public domain
    cells = {(c["outcome"], c["h"]): c for c in fomc["scores"]}
    assert "realized" in cells[("USDCAD", 1)]  # [C]: realized value published
    assert "realized" not in cells[("CA2Y", 1)]  # [D]: hit flags only
    assert "realized" not in cells[("CA_BANKS_EW6", 5)]  # [L]: hit flags only
    assert {"inside68", "inside90", "beats_no_change"} <= set(cells[("CA2Y", 1)])

    boc = json.loads((live_dirs / "scores" / "2026-10-07_boc_score1.json").read_text())
    assert boc["realized_proxy_bp"] is None  # the BoC proxy is a CA 2Y change (O-3)
    assert all("forecast" not in c and "realized" not in c for c in boc["scores"])
    assert next(c for c in boc["scores"] if c["outcome"] == "CA2Y" and c["h"] == 1)["mechanical"]

    # The score names the exact forecast bytes it used.
    fpath = live_dirs / fomc["forecast_file"]
    from wsl.data.cache import sha256_bytes

    assert fomc["forecast_sha256"] == sha256_bytes(fpath.read_bytes())


def test_no_score_before_outcomes_are_published(live_dirs: Path) -> None:
    data, _ = _world()
    cfg = _cfg()
    run.forecast_due(cfg, data, datetime(2026, 10, 5, tzinfo=UTC))
    assert (live_dirs / "forecasts" / "2026-10-14_boc.json").exists()
    # At 10-15 the decision has passed but its h = 5 day has not: nothing to score yet.
    cut = data.series["CA2Y"]
    keep = cut.dates <= pd.Timestamp("2026-10-15")
    data.series["CA2Y"] = type(cut)(cut.id, cut.dates[keep], cut.x[keep], cut.obs_ts[keep],
                                    cut.kind, cut.obs_time, cut.assumed_time)
    wrote = run.score_due(cfg, data, datetime(2026, 10, 15, 23, tzinfo=UTC))
    assert not any("2026-10-14" in p.name for p in wrote)
