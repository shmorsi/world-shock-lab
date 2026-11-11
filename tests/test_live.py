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
def _world(seed: int = 1, end: str = "2026-10-30") -> tuple[LiveData, pd.DatetimeIndex]:
    """Business days 2015-2026; announcements every 30 days; outcomes respond to the proxy."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-01", end)
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


def test_committed_forecasts_can_be_scored(live_dirs: Path) -> None:
    """The real 2026-10-28 forecast files (copied, never touched) score once the outcomes
    exist: score 1 for both banks, and score 2 for the FOMC once JK lists the meeting."""
    import shutil

    from wsl.data.cache import sha256_bytes
    from wsl.paths import ROOT

    committed = sorted((ROOT / "live" / "forecasts").glob("2026-10-28_*.json"))
    assert [p.name for p in committed] == ["2026-10-28_boc.json", "2026-10-28_fomc.json"]
    before = {p.name: sha256_bytes(p.read_bytes()) for p in committed}
    (live_dirs / "forecasts").mkdir()
    for p in committed:
        shutil.copy(p, live_dirs / "forecasts" / p.name)

    data, _ = _world(end="2026-11-20")
    jk_day = pd.Timestamp("2026-10-28 14:00", tz="America/New_York")
    data.jk_surprises = pd.DataFrame({"ts": [jk_day], "TFUT02_bp": [-3.0]})
    cfg = config()
    # Decision passed, but no 5-day outcome yet: nothing to score.
    assert run.score_due(cfg, _cut(data, "2026-10-30"), datetime(2026, 10, 30, 23,
                                                                  tzinfo=UTC)) == []
    wrote = run.score_due(cfg, data, datetime(2026, 11, 20, 23, tzinfo=UTC))
    assert sorted(p.name for p in wrote) == ["2026-10-28_boc_score1.json",
                                             "2026-10-28_fomc_score1.json",
                                             "2026-10-28_fomc_score2.json"]
    for p in wrote:
        s = json.loads(p.read_text())
        f = live_dirs / s["forecast_file"]
        assert s["forecast_sha256"] == before[f.name] == sha256_bytes(f.read_bytes())
        assert len(s["scores"]) == 6 and s["collision"]
    s2 = json.loads((live_dirs / "scores" / "2026-10-28_fomc_score2.json").read_text())
    assert s2["realized_proxy_bp"] == -3.0
    assert {p.name: sha256_bytes(p.read_bytes()) for p in committed} == before
    assert run.score_due(cfg, data, datetime(2026, 11, 21, tzinfo=UTC)) == []  # once only


def _cut(data: LiveData, last: str) -> LiveData:
    from dataclasses import replace

    series = {}
    for k, s in data.series.items():
        keep = s.dates <= pd.Timestamp(last)
        series[k] = type(s)(s.id, s.dates[keep], s.x[keep], s.obs_ts[keep], s.kind,
                            s.obs_time, s.assumed_time)
    return replace(data, series=series)


# ------------------------------------------------------------- FRED without an API key
FRED_ROWS = {"DGS2": "observation_date,DGS2\n2026-11-05,4.46\n2026-11-06,4.51\n2026-11-09,\n",
             "DEXCAUS": "observation_date,DEXCAUS\n2026-11-05,1.3980\n2026-11-06,1.4106\n"}


class _FakeHttp:
    """Answers every request from ``reply(url, params)`` and records what was asked."""

    def __init__(self, reply) -> None:  # type: ignore[no-untyped-def]
        self.reply = reply
        self.calls: list[tuple[str, dict[str, str]]] = []

    def request(self, method: str, url: str, *, params=None, json_body=None,  # type: ignore[no-untyped-def]
                headers=None):
        from urllib.parse import urlencode

        from wsl.data.http import Fetched, redact_url

        self.calls.append((url, dict(params or {})))
        body = self.reply(url, dict(params or {}))
        if isinstance(body, Exception):
            raise body
        full = url + ("?" + urlencode(params) if params else "")
        return Fetched(url=redact_url(full), content=body, content_type=None)

    def get(self, url: str, **kw):  # type: ignore[no-untyped-def]
        return self.request("GET", url, **kw)


def _reply(fred_csv=None):  # type: ignore[no-untyped-def]
    """FRED's public CSV for its two live series; a small JSON or CSV body for the rest."""
    def reply(url: str, params: dict[str, str]) -> bytes | Exception:
        if "fredgraph.csv" in url:
            got = (fred_csv or FRED_ROWS.get)(params["id"])
            return got if isinstance(got, Exception) else got.encode()
        if "api.stlouisfed.org" in url:
            return b'{"observations": [{"date": "2026-11-06", "value": "4.51"}]}'
        return b"start,TFUT02\n" if url.endswith(".csv") else b"{}"
    return reply


@pytest.fixture
def no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own .env must not decide which route a test takes."""
    from wsl.live import data as live_data

    monkeypatch.setattr(live_data, "load_dotenv", lambda *a, **k: False)


def test_live_reads_fred_from_the_public_csv_when_no_key_is_set(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_dotenv: None) -> None:
    from wsl.data.cache import RawCache, sha256_bytes
    from wsl.data.sources import fred
    from wsl.live import data as live_data

    monkeypatch.setenv("FRED_API_KEY", "")  # what GitHub Actions passes for a missing secret
    assert live_data.fred_route() == "csv"
    cache = RawCache(root=tmp_path, manifest=Path("manifest.json"))
    http = _FakeHttp(_reply())
    outcomes = live_data.download(config(), cache, http)  # type: ignore[arg-type]

    assert live_data.failed_downloads(outcomes) is None
    assert not any(o.status == "skipped" for o in outcomes)
    assert all("api.stlouisfed.org" not in url for url, _ in http.calls)
    asked = [p["id"] for url, p in http.calls if url == fred.CSV_URL]
    assert asked == ["DGS2", "DEXCAUS"]
    # Observations only: the API's metadata and tags artifacts are neither asked for nor read.
    fred_keys = [o.key for o in outcomes if o.key.startswith("fred/")]
    assert fred_keys == ["fred/observations/DGS2", "fred/observations/DEXCAUS"]
    assert outcomes[-1].key == live_data.JK_KEY  # the JK file is still fetched last

    content, rec = cache.read("fred/observations/DGS2")
    assert json.loads(content)["observations"][-1] == {"date": "2026-11-09", "value": "."}
    prov = live_data._provenance("fred/observations/DGS2", rec)
    assert prov["url"] == "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2"
    assert prov["via"] == fred.CSV_NOTE and "no API key" in prov["via"]
    assert prov["sha256"] == sha256_bytes(content)

    # The FRED parser reads what the CSV route stored, with nothing filled in.
    reg = {s.id: s for s in live_data.load_registry()}
    obs = fred.parse([reg["US2Y"], reg["USDCAD"]], cache).obs
    us2 = obs[obs["series_id"] == "US2Y"]
    assert us2["value"].tolist()[:2] == [4.46, 4.51] and us2["status"].tolist()[-1] != "ok"
    assert obs[obs["series_id"] == "USDCAD"]["value"].tolist() == [1.398, 1.4106]


def test_live_keeps_the_fred_api_when_a_key_is_set(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_dotenv: None) -> None:
    from wsl.data.cache import RawCache
    from wsl.live import data as live_data

    monkeypatch.setenv("FRED_API_KEY", "k-123")
    assert live_data.fred_route() == "api"
    cache = RawCache(root=tmp_path, manifest=Path("manifest.json"))
    http = _FakeHttp(_reply())
    outcomes = live_data.download(config(), cache, http)  # type: ignore[arg-type]

    assert live_data.failed_downloads(outcomes) is None
    assert all("fredgraph.csv" not in url for url, _ in http.calls)
    api = [(url, p) for url, p in http.calls if "api.stlouisfed.org" in url]
    assert len(api) == 6 and all(p["api_key"] == "k-123" for _, p in api)
    assert sum(o.key.startswith("fred/") for o in outcomes) == 6  # observations, series, tags
    prov = live_data._provenance("fred/observations/DGS2",
                                 cache.pinned_record("fred/observations/DGS2"))
    assert "via" not in prov and "api_key=REDACTED" in prov["url"] and "k-123" not in prov["url"]


@pytest.mark.parametrize(("bad", "why"), [
    ("<!DOCTYPE html><html><body>Access Denied</body></html>", "returned HTML"),
    ("observation_date,DGS10\n2026-11-06,4.10\n", "unexpected CSV header"),
    ("observation_date,DGS2\n", "no observations"),
])
def test_a_bad_public_csv_stops_the_live_run_before_anything_is_stored(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_dotenv: None, bad: str,
        why: str) -> None:
    from wsl.data.cache import RawCache
    from wsl.data.http import FetchError
    from wsl.live import data as live_data

    monkeypatch.delenv("FRED_API_KEY", raising=False)
    cache = RawCache(root=tmp_path, manifest=Path("manifest.json"))
    rows = dict(FRED_ROWS, DGS2=bad)
    outcomes = live_data.download(config(), cache, _FakeHttp(_reply(rows.get)))  # type: ignore[arg-type]
    failed = [o for o in outcomes if o.status == "failed"]
    assert [o.key for o in failed] == ["fred/observations/DGS2"] and why in failed[0].detail
    message = live_data.failed_downloads(outcomes)
    assert message is not None and why in message and "Nothing was published" in message
    assert cache.entry("fred/observations/DGS2") is None  # the bad bytes were never stored

    refused = dict(FRED_ROWS, DEXCAUS=FetchError("GET fredgraph.csv failed: HTTP 403"))
    outcomes = live_data.download(config(), RawCache(root=tmp_path / "b",
                                                     manifest=Path("manifest.json")),
                                  _FakeHttp(_reply(refused.get)))  # type: ignore[arg-type]
    message = live_data.failed_downloads(outcomes)
    assert message is not None and "fred/observations/DEXCAUS" in message and "HTTP 403" in message
    # A refused CSV download says what to do about it: the key, or another machine.
    assert "FRED_API_KEY" in message and "another machine" in message


def test_the_csv_route_gives_up_quickly_and_other_failures_keep_the_plain_hint() -> None:
    from wsl.data.fetch import FetchOutcome
    from wsl.live import data as live_data

    assert (live_data.CSV_TIMEOUT, live_data.CSV_RETRIES) == (30.0, 1)
    other = live_data.failed_downloads([FetchOutcome("yahoo/RY", "failed", "HTTP 429")])
    assert other is not None and "FRED_API_KEY" not in other and "Fix the downloads" in other


# ------------------------------------------------------------- the workflow's report
def test_live_run_stops_before_writing_when_a_download_fails() -> None:
    from wsl.data.fetch import FetchOutcome
    from wsl.live.data import failed_downloads

    assert failed_downloads([FetchOutcome("fred/observations/DGS2", "unchanged")]) is None
    blocked = failed_downloads([FetchOutcome("yahoo/RY", "failed", "HTTP 429")])
    assert blocked is not None and "HTTP 429" in blocked and "Nothing was published" in blocked
    # An input that was skipped is as incomplete as one that failed.
    skipped = failed_downloads([FetchOutcome("x/y", "skipped", "SOME_KEY not set")])
    assert skipped is not None and "SOME_KEY not set" in skipped


def test_pending_lists_what_a_run_still_owes(tmp_path: Path) -> None:
    from wsl.live import pending

    cfg = _cfg()
    fc, sc = tmp_path / "forecasts", tmp_path / "scores"
    fc.mkdir()
    sc.mkdir()
    (fc / "2026-10-07_fomc.json").write_text("{}")
    (fc / "2026-10-07_boc.json").write_text("{}")
    (sc / "2026-10-07_boc_score1.json").write_text("{}")

    o = pending.owed(cfg, datetime(2026, 10, 9, tzinfo=UTC), fc, sc)
    assert [x.split(":")[0] for x in o.due] == ["2026-10-14_boc"]
    assert o.missed == []
    assert [x.split(":")[0] for x in o.scores] == ["2026-10-07_fomc"]  # BoC needs no score 2
    assert "score1, score2" in o.scores[0]

    # A missing key is not reported as a problem by itself: the summary says which route the
    # run read FRED by (and that FRED's website refuses some networks), and a forecast that is
    # still due is an error.
    notes, md = pending.report(o, "csv")
    assert notes == [("error", o.due[0])]
    assert "public CSV" in md and "add the secret" in md and "Skipped" not in md
    assert "2026-10-14_boc" in md and "Score 1 waits" in md
    assert "the FRED API" in pending.report(o, "api")[1]
    assert "FRED data" not in pending.report(o)[1] and "FRED data" not in pending.report(o, "")[1]

    late = pending.owed(cfg, datetime(2026, 10, 14, tzinfo=UTC), fc, sc)
    assert late.due == [] and [x.split(":")[0] for x in late.missed] == ["2026-10-14_boc"]


def test_pending_exit_code_fails_only_when_a_forecast_is_due(monkeypatch: pytest.MonkeyPatch,
                                                             tmp_path: Path) -> None:
    from wsl.live import pending

    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    # The committed config: nothing is due on 2026-10-23 (the 10-28 forecasts exist), but the
    # 12-09 forecasts are due on 2026-11-30.
    assert pending.main(["--route", "csv", "--now", "2026-10-23T19:00:00Z"]) == 0
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(pending, "SCORES", tmp_path)  # no scores yet
    assert pending.main(["--route", "csv", "--now", "2026-11-30T22:30:00Z"]) == 1
    text = summary.read_text()
    assert "2026-12-09_fomc: forecast due" in text and "2026-10-28_boc" in text
    assert "fredgraph.csv" in text
    # The workflow passes an empty route if its first step did not run; that is not an error.
    assert pending.main(["--route", "", "--now", "2026-10-23T19:00:00Z"]) == 0
