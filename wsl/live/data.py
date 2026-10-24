"""Fresh data for live mode, kept apart from the pinned research cache.

Live mode needs today's observations, so it downloads the few series it uses into its own
cache (``data/live_cache/``, git-ignored) and always adopts what the source serves now. The
research cache and its pins are never touched. Every forecast and score file records the
SHA-256 and retrieval time of the bytes it used.

Event history comes from files that are either committed or fetched here:
  * FOMC: the JK surprise file (current ``main``), which lists every announcement JK has
    measured, plus calendar dates that have passed but JK has not yet added;
  * BoC: ``data/events/boc_fad.csv`` (scheduled decisions), plus passed calendar dates.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

from wsl.analysis.boc_proxy import ASSUMED
from wsl.data.cache import RawCache
from wsl.data.fetch import SERIES_SOURCES, FetchOutcome, fetch_artifact
from wsl.data.http import Http, redact_url
from wsl.data.registry import SeriesSpec, load_derived, load_registry
from wsl.data.sources.base import Artifact
from wsl.data.transforms import ew_return_index
from wsl.paths import ROOT
from wsl.shocks.windows import WindowSeries, kind_for, make_series

LIVE_CONFIG = ROOT / "config" / "live.yaml"
LIVE_CACHE = ROOT / "data" / "live_cache"
BOC_FAD = ROOT / "data" / "events" / "boc_fad.csv"
BASE_SERIES = ("US2Y", "CA2Y", "USDCAD")
BANKS = ("RY", "TD", "BNS", "BMO", "CM", "NA")
JK_KEY = "jk_live/surprises"


def config() -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(LIVE_CONFIG.read_text())
    return cfg


def calendar(cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per scheduled decision: bank, decision_date, ts (tz-aware), collision."""
    rows = []
    for bank, c in cfg["calendar"].items():
        for d in c["dates"]:
            day = d if isinstance(d, date) else date.fromisoformat(str(d))
            ts = pd.Timestamp(f"{day.isoformat()} {c['time']}").tz_localize(c["tz"])
            rows.append({"bank": bank, "decision_date": day.isoformat(), "ts": ts})
    out = pd.DataFrame(rows)
    counts = out["decision_date"].value_counts()
    out["collision"] = out["decision_date"].map(counts).gt(1)
    return out.sort_values(["ts", "bank"]).reset_index(drop=True)


@dataclass
class LiveData:
    series: dict[str, WindowSeries]
    provenance: dict[str, dict[str, str]]  # series id -> artifact, sha256, retrieved_at, policy
    policy: dict[str, str]  # series id -> licence policy (C / D / L)
    fomc_history: pd.Series  # tz-aware announcement times
    boc_history: pd.Series
    jk_surprises: pd.DataFrame  # ts (America/New_York) and TFUT02_bp
    outcomes: list[FetchOutcome] = field(default_factory=list)


def _specs() -> list[SeriesSpec]:
    reg = {s.id: s for s in load_registry()}
    return [reg[i] for i in (*BASE_SERIES, *BANKS)]


def _jk_artifact(cfg: dict[str, Any]) -> Artifact:
    return Artifact(JK_KEY, cfg["scoring"]["jk_surprises_url"], "csv")


def download(cfg: dict[str, Any], cache: RawCache, http: Http | None = None
             ) -> list[FetchOutcome]:
    """Fetch the current version of every live input; the newest bytes become the pin."""
    load_dotenv(ROOT / ".env")
    http = http or Http()
    arts: list[Artifact] = []
    specs = _specs()
    for source in sorted({s.source for s in specs}):
        arts += SERIES_SOURCES[source].artifacts([s for s in specs if s.source == source])
    arts.append(_jk_artifact(cfg))
    return [fetch_artifact(a, cache, http, repin=True) for a in arts]


def failed_downloads(outcomes: list[FetchOutcome]) -> str | None:
    """Why a live run must stop, or None when every download succeeded.

    A skipped FRED series means ``FRED_API_KEY`` is unset. A score file can never be edited
    once it is published, so the run stops before writing anything when a download failed.
    """
    bad = [o for o in outcomes if o.status in ("failed", "skipped")]
    if not bad:
        return None
    lines = [f"  {o.status:<8} {o.key}: {o.detail}" for o in bad]
    if any("FRED_API_KEY" in o.detail for o in bad):
        hint = ("FRED_API_KEY is not set. Add it to .env, or as the GitHub Actions secret, "
                "and rerun. Nothing was published or scored.")
    else:
        hint = "Nothing was published or scored. Fix the downloads and rerun."
    return "Live run stopped; these downloads did not succeed:\n" + "\n".join(lines) + "\n" + hint


def _series_from(spec_id: str, values: pd.Series, obs_time: str, tz: str, units: str
                 ) -> WindowSeries:
    return make_series(spec_id, values, obs_time, tz, kind_for(units), ASSUMED.get(spec_id))


def _ok_values(obs: pd.DataFrame, series_id: str) -> pd.Series:
    rows = obs[(obs["series_id"] == series_id) & (obs["status"] == "ok")]
    return rows.set_index("date")["value"].sort_index()


def _passed(cal: pd.DataFrame, bank: str, now: pd.Timestamp) -> pd.Series:
    rows = cal[(cal["bank"] == bank) & (cal["ts"] < now)]
    return pd.Series(rows["ts"].to_list(), dtype="object")


def load(cfg: dict[str, Any], cache: RawCache, now: datetime) -> LiveData:
    """Parse the live cache into event-window series and event histories."""
    specs = _specs()
    by_id = {s.id: s for s in specs}
    frames, provenance, policy = [], {}, {}
    for source in sorted({s.source for s in specs}):
        group = [s for s in specs if s.source == source]
        parsed = SERIES_SOURCES[source].parse(group, cache)
        frames.append(parsed.obs)
        for sid, key in parsed.artifact_of.items():
            rec = cache.pinned_record(key)
            provenance[sid] = {"artifact": key, "sha256": rec.sha256,
                               "retrieved_at": rec.retrieved_at, "url": redact_url(rec.url)}
    obs = pd.concat(frames, ignore_index=True)

    series: dict[str, WindowSeries] = {}
    for sid in BASE_SERIES:
        s = by_id[sid]
        series[sid] = _series_from(sid, _ok_values(obs, sid), s.obs_time, s.tz, s.units)
        policy[sid] = s.policy
    banks = next(d for d in load_derived() if d.id == "CA_BANKS_EW6")
    wide = pd.DataFrame({m: _ok_values(obs, m) for m in banks.inputs})
    ew = ew_return_index(wide, banks.min_inputs or len(banks.inputs)).dropna()
    series[banks.id] = _series_from(banks.id, ew, banks.obs_time, banks.tz, banks.units)
    policy[banks.id] = banks.policy
    provenance[banks.id] = {
        "artifact": "derived:" + ",".join(banks.inputs),
        "sha256": ",".join(provenance[m]["sha256"][:12] for m in banks.inputs),
        "retrieved_at": max(provenance[m]["retrieved_at"] for m in banks.inputs), "url": ""}
    for sid, ws in series.items():
        provenance[sid]["last_obs"] = ws.dates[-1].date().isoformat()

    stamp = pd.Timestamp(now)
    cal = calendar(cfg)
    content, rec = cache.read(JK_KEY)
    jk = pd.read_csv(io.BytesIO(content))
    jk_ts = pd.to_datetime(jk["start"]).dt.tz_localize("America/New_York")
    jk_s = pd.DataFrame({"ts": jk_ts, "TFUT02_bp": 100.0 * jk["TFUT02"].astype(float)})
    provenance["JK"] = {"artifact": JK_KEY, "sha256": rec.sha256,
                        "retrieved_at": rec.retrieved_at, "url": rec.url}
    jk_days = set(jk_ts.dt.date)
    extra_fomc = [t for t in _passed(cal, "fomc", stamp) if t.date() not in jk_days]
    fomc_hist = pd.Series(list(jk_ts) + extra_fomc, dtype="object")

    fad = pd.read_csv(BOC_FAD)
    fad = fad[fad["scheduled"].astype(bool) & (fad["time_et"] != "unknown")]
    boc_ts = [pd.Timestamp(f"{d} {t}").tz_localize("America/Toronto")
              for d, t in zip(fad["date"], fad["time_et"], strict=True)]
    fad_days = {t.date() for t in boc_ts}
    boc_ts += [t for t in _passed(cal, "boc", stamp) if t.date() not in fad_days]
    fomc_days = {pd.Timestamp(t).tz_convert("America/New_York").date() for t in fomc_hist}
    # Protocol v2: BoC decisions on FOMC days are not used to fit the BoC model.
    boc_hist = pd.Series([t for t in boc_ts if t.date() not in fomc_days], dtype="object")
    return LiveData(series=series, provenance=provenance, policy=policy,
                    fomc_history=fomc_hist, boc_history=boc_hist, jk_surprises=jk_s)


def live_cache() -> RawCache:
    return RawCache(root=LIVE_CACHE, manifest=Path("manifest.json"))
