"""Pinned raw files -> validated parquet (data/clean) -> DuckDB (``wsl data build``).

The build reads nothing from the network. Sources whose pinned artifacts are not in the local
cache are skipped and listed, never silently replaced.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from wsl.data import schemas
from wsl.data.cache import RawCache
from wsl.data.fetch import SERIES_SOURCES
from wsl.data.registry import DerivedSpec, SeriesSpec, by_source, load_derived, load_registry
from wsl.data.sources import statcan
from wsl.data.sources.base import obs_frame
from wsl.data.transforms import datetimes_to_ns, ew_return_index
from wsl.paths import (
    CLEAN_DIR,
    DUCKDB_PATH,
    EVALUATION_CONFIG,
    EVENTS_DIR,
    MANIFEST,
    ROOT,
    SERIES_REGISTRY,
)
from wsl.shocks import sources as shock_sources

EVENT_TABLES = {
    "events_fomc": (shock_sources.fomc_events, schemas.FOMC,
                    ["jk_fed/shocks_t", "jk_fed/surprises"]),
    "events_fomc_bs": (shock_sources.fomc_events_bs, None, ["bauer_swanson/surprises"]),
    "events_ecb": (shock_sources.ecb_events, schemas.ECB, ["jk_ecb/shocks_d"]),
    "events_opec": (shock_sources.opec_events, schemas.OPEC, ["kaenzig/oil_supply_news"]),
    "shocks_monthly": (shock_sources.monthly_shocks, schemas.MONTHLY_SHOCKS,
                       ["jk_fed/shocks_m", "jk_ecb/shocks_m", "bauer_swanson/surprises",
                        "kaenzig/oil_supply_news", "baumeister_hamilton/supply",
                        "baumeister_hamilton/demand"]),
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Files the pipeline itself writes; changing them does not make the code "dirty".
GENERATED = ("results", "docs/DATA_QUALITY.md", "docs/SIMULATION.md", "docs/REPLICATION.md",
             "web/preview/index.html")


def git_sha() -> str:
    """HEAD commit, suffixed "-dirty" if code, config or data definitions have uncommitted
    changes (generated outputs are ignored)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True)
        excludes = [f":(exclude){g}" for g in GENERATED]
        dirty = subprocess.run(["git", "status", "--porcelain", "--", ".", *excludes], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _series_rows(specs: list[SeriesSpec], parsed_artifacts: dict[str, str],
                 transformations: dict[str, list[str]], cache: RawCache) -> list[dict[str, object]]:
    by_id = {s.id: s for s in specs}
    rows = []
    for sid, akey in sorted(parsed_artifacts.items()):
        spec = by_id.get(sid) or by_id[sid.rsplit(".", 1)[0]]
        rec = cache.pinned_record(akey)
        rows.append({
            "series_id": sid, "parent_id": spec.id, "source": spec.source, "key": spec.key,
            "name": spec.name, "freq": spec.freq, "units": spec.units,
            "country": spec.country or "", "asset_class": spec.asset_class,
            "obs_time": spec.obs_time, "obs_time_verified": spec.obs_time_verified,
            "tz": spec.tz, "policy": spec.policy, "event_window_ok": spec.event_window_ok,
            "artifact_key": akey, "sha256": rec.sha256, "url": rec.url,
            "retrieved_at": rec.retrieved_at,
            "transformations": json.dumps(transformations.get(sid, [])),
        })
    return rows


def derive(obs: pd.DataFrame, series: pd.DataFrame,
           specs: list[DerivedSpec]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Compute derived series from built ones. Provenance: a composite hash of the inputs'
    pinned artifact hashes, the latest input retrieval time, and the formula."""
    ok = obs[obs["status"] == "ok"]
    meta = series.set_index("series_id")
    frames, rows = [], []
    for d in specs:
        missing = [i for i in d.inputs if i not in meta.index]
        if missing:
            continue
        wide = ok[ok["series_id"].isin(d.inputs)].pivot(index="date", columns="series_id",
                                                         values="value")[list(d.inputs)]
        if d.op == "spread":
            values = (wide[d.inputs[0]] - wide[d.inputs[1]]).dropna()
            formula = f"{d.inputs[0]} - {d.inputs[1]} on dates where both are available"
        else:
            values = ew_return_index(wide, d.min_inputs or len(d.inputs)).dropna()
            formula = (f"equal-weight daily-rebalanced index of {', '.join(d.inputs)}; "
                       f"at least {d.min_inputs} members per day; base 100")
        frames.append(obs_frame(d.id, values.index, values.to_numpy()))
        shas = sorted(str(meta.loc[i, "sha256"]) for i in d.inputs)
        rows.append({
            "series_id": d.id, "parent_id": d.id, "source": "derived", "key": d.op,
            "name": d.name, "freq": d.freq, "units": d.units, "country": d.country or "",
            "asset_class": d.asset_class, "obs_time": d.obs_time,
            "obs_time_verified": d.obs_time_verified, "tz": d.tz, "policy": d.policy,
            "event_window_ok": d.event_window_ok, "artifact_key": f"derived/{d.id}",
            "sha256": hashlib.sha256("".join(shas).encode()).hexdigest(),
            "url": "derived:" + ",".join(d.inputs),
            "retrieved_at": max(str(meta.loc[i, "retrieved_at"]) for i in d.inputs),
            "transformations": json.dumps([formula]),
        })
    out = pd.concat(frames, ignore_index=True) if frames else obs.iloc[0:0]
    return out, rows


def build(cache: RawCache | None = None) -> list[str]:
    cache = cache or RawCache()
    specs = load_registry()
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    obs_frames: list[pd.DataFrame] = []
    series_rows: list[dict[str, object]] = []

    for source, sspecs in by_source(specs).items():
        mod = SERIES_SOURCES[source]
        needed = [a.key for a in mod.artifacts(sspecs)]
        missing = [k for k in needed if not cache.has_pinned_locally(k)]
        if missing:
            lines.append(f"SKIP {source}: {len(missing)}/{len(needed)} artifacts not cached "
                         f"(e.g. {missing[0]})")
            continue
        parsed = mod.parse(sspecs, cache)
        obs_frames.append(parsed.obs)
        series_rows += _series_rows(sspecs, parsed.artifact_of, parsed.transformations, cache)
        lines.append(f"ok   {source}: {parsed.obs['series_id'].nunique()} series, "
                     f"{len(parsed.obs):,} rows")

    obs = pd.concat(obs_frames, ignore_index=True)
    series = pd.DataFrame(series_rows)
    derived_obs, derived_rows = derive(obs, series, load_derived())
    obs = pd.concat([obs, derived_obs], ignore_index=True)
    series = pd.concat([series, pd.DataFrame(derived_rows)], ignore_index=True)
    lines.append(f"ok   derived: {len(derived_rows)} series, {len(derived_obs):,} rows")
    obs["date"] = obs["date"].astype("datetime64[ns]")
    obs["value"] = obs["value"].astype("float64")
    schemas.OBS.validate(obs)
    schemas.SERIES.validate(series)
    obs.to_parquet(CLEAN_DIR / "obs.parquet", index=False)
    series.to_parquet(CLEAN_DIR / "series.parquet", index=False)
    tables = {"obs": CLEAN_DIR / "obs.parquet", "series": CLEAN_DIR / "series.parquet"}

    provenance = []
    for name, (fn, schema, keys) in EVENT_TABLES.items():
        if not all(cache.has_pinned_locally(k) for k in keys):
            lines.append(f"SKIP {name}: inputs not cached")
            continue
        df = datetimes_to_ns(fn(cache))
        if schema is not None:
            schema.validate(df)
        df.to_parquet(CLEAN_DIR / f"{name}.parquet", index=False)
        tables[name] = CLEAN_DIR / f"{name}.parquet"
        for k in keys:
            rec = cache.pinned_record(k)
            provenance.append({"table": name, "artifact_key": k, "sha256": rec.sha256,
                               "url": rec.url, "retrieved_at": rec.retrieved_at})
        lines.append(f"ok   {name}: {len(df):,} rows")

    fad_csv = EVENTS_DIR / "boc_fad.csv"
    if fad_csv.exists():
        fad = pd.read_csv(fad_csv, parse_dates=["date", "next_scheduled_stated",
                                                "target_effective_date"])
        fad["change_bp"] = fad["change_bp"].astype("float64")
        fad = datetimes_to_ns(fad)
        schemas.BOC_FAD.validate(fad)
        fad.to_parquet(CLEAN_DIR / "events_boc.parquet", index=False)
        tables["events_boc"] = CLEAN_DIR / "events_boc.parquet"
        provenance.append({"table": "events_boc", "artifact_key": "data/events/boc_fad.csv",
                           "sha256": file_sha256(fad_csv),
                           "url": "constructed from bankofcanada.ca press releases (per-row url)",
                           "retrieved_at": ""})
        lines.append(f"ok   events_boc: {len(fad):,} rows")
    else:
        lines.append("SKIP events_boc: data/events/boc_fad.csv missing (`wsl events boc-fad`)")

    statcan_specs = [s for s in specs if s.source == "statcan"]
    if all(cache.has_pinned_locally(a.key) for a in statcan.artifacts(statcan_specs)):
        rel = datetimes_to_ns(statcan.release_times(statcan_specs, cache))
        rel.to_parquet(CLEAN_DIR / "release_times.parquet", index=False)
        tables["release_times"] = CLEAN_DIR / "release_times.parquet"

    prov = pd.DataFrame(provenance)
    prov.to_parquet(CLEAN_DIR / "provenance.parquet", index=False)
    tables["provenance"] = CLEAN_DIR / "provenance.parquet"

    info = pd.DataFrame([{
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "manifest_sha256": file_sha256(MANIFEST),
        "series_registry_sha256": file_sha256(SERIES_REGISTRY),
        "evaluation_config_sha256": file_sha256(EVALUATION_CONFIG),
    }])
    info.to_parquet(CLEAN_DIR / "build_info.parquet", index=False)
    tables["build_info"] = CLEAN_DIR / "build_info.parquet"

    with duckdb.connect(str(DUCKDB_PATH)) as con:
        for name, path in tables.items():
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet(?)",
                        [str(path)])
    lines.append(f"duckdb: {DUCKDB_PATH.relative_to(ROOT)} ({len(tables)} tables)")
    return lines
