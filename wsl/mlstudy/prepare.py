"""Build the local tables the study reads, from public downloads.

The committed manifest pins are not changed. Where a source still serves the
pinned bytes, those bytes are written to the pinned path and the official
parsers read them. Where a source has revised the file, the new bytes are used
for this build only and ``data/manifest.json`` is restored afterwards. Raw
files stay git-ignored. The study results record the hashes that were used.
"""

from __future__ import annotations

import json
import os
import stat
from typing import Any

import pandas as pd

from wsl.data.build import _series_rows, derive
from wsl.data.cache import RawCache, sha256_bytes
from wsl.data.fetch import SERIES_SOURCES
from wsl.data.http import Http
from wsl.data.registry import load_derived, load_registry
from wsl.data.schemas import BOC_FAD, FOMC, OBS, OPEC, SERIES
from wsl.data.sources.base import Artifact
from wsl.data.transforms import datetimes_to_ns
from wsl.paths import CLEAN_DIR, EVENTS_DIR, MANIFEST, ROOT
from wsl.shocks import sources as shock_sources

NEEDED = (
    "US2Y", "US10Y", "USDCAD", "WTI", "VIX", "US_CPI_NSA",
    "FF_TARGET", "FF_TARGET_UPPER", "FF_TARGET_LOWER",
    "CA2Y", "CA10Y", "BOC_TARGET", "CA_CPI_NSA",
    "SPY", "TSX", "RY", "TD", "BNS", "BMO", "CM", "NA", "XEG",
)
SHOCK_KEYS = ("jk_fed/shocks_t", "jk_fed/surprises", "kaenzig/oil_supply_news")
INPUTS_PATH = CLEAN_DIR / "study_inputs.json"


def _fred_observations(http: Http, series_key: str) -> tuple[bytes, str]:
    """FRED's graph CSV needs no API key. Wrapped in the JSON shape the parser reads."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    got = http.get(url, params={"id": series_key})
    text = got.content.decode("utf-8-sig")
    if text.lstrip().startswith("<"):
        raise RuntimeError(f"FRED returned HTML for {series_key}")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    obs = []
    for line in lines[1:]:
        date, _, val = line.partition(",")
        val = val.strip().strip('"')
        obs.append({"date": date.strip(), "value": "." if val in {"", "."} else val})
    body = json.dumps({"observations": obs}).encode()
    return body, f"{url}?id={series_key}"


def _place(cache: RawCache, key: str, content: bytes, url: str, ext: str) -> str:
    """Make ``cache.read(key)`` succeed. Returns 'matched_pin' or 'revised'."""
    digest = sha256_bytes(content)
    entry = cache.entry(key)
    if entry is not None and entry.get("pinned") == digest:
        rec = cache.pinned_record(key)
        path = cache.root / rec.path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
            os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return "matched_pin"
    cache.store(key, content, url=url, ext=ext, repin=False)
    # Adopt the new bytes in memory only. The file manifest is restored by the caller.
    adopted = cache.entry(key)
    if adopted is None:
        raise RuntimeError(f"manifest has no entry for {key} after storing it")
    adopted["pinned"] = digest
    return "revised"


def _download_one(http: Http, art: Artifact, source: str) -> tuple[bytes, str]:
    if source == "fred" and "/observations/" in art.key:
        series_key = art.key.rsplit("/", 1)[-1]
        return _fred_observations(http, series_key)
    got = http.request(art.method, art.url, params=art.params or None, json_body=art.payload)
    if art.ext in {"json", "csv"} and got.content[:64].lstrip().lower().startswith(b"<"):
        raise RuntimeError(f"{art.key} returned HTML")
    return got.content, got.url


def prepare() -> dict[str, Any]:
    """Download what this study needs and write ``data/clean``. Returns a provenance dict."""
    saved = MANIFEST.read_bytes()
    cache = RawCache()
    http = Http()
    specs = [s for s in load_registry() if s.id in NEEDED]
    missing = sorted(set(NEEDED) - {s.id for s in specs})
    if missing:
        raise RuntimeError(f"series not in the registry: {missing}")
    inputs: list[dict[str, Any]] = []
    lines: list[str] = []
    try:
        for source, group in _by_source(specs).items():
            mod = SERIES_SOURCES[source]
            arts = [a for a in mod.artifacts(group)
                    if source != "fred" or "/observations/" in a.key]
            for art in arts:
                content, url = _download_one(http, art, source)
                status = _place(cache, art.key, content, url, art.ext)
                inputs.append({"key": art.key, "sha256": sha256_bytes(content),
                               "status": status, "bytes": len(content), "url": url})
                lines.append(f"{status:<12} {art.key}")
        for art in shock_sources.artifacts():
            if art.key not in SHOCK_KEYS:
                continue
            content, url = _download_one(http, art, "shocks")
            status = _place(cache, art.key, content, url, art.ext)
            inputs.append({"key": art.key, "sha256": sha256_bytes(content),
                           "status": status, "bytes": len(content), "url": url})
            lines.append(f"{status:<12} {art.key}")

        obs_frames: list[pd.DataFrame] = []
        series_rows: list[dict[str, object]] = []
        for source, group in _by_source(specs).items():
            parsed = SERIES_SOURCES[source].parse(group, cache)
            obs_frames.append(parsed.obs)
            series_rows += _series_rows(group, parsed.artifact_of, parsed.transformations, cache)
            lines.append(f"parsed {source}: {parsed.obs['series_id'].nunique()} series")
        obs = pd.concat(obs_frames, ignore_index=True)
        series = pd.DataFrame(series_rows)
        derived_obs, derived_rows = derive(obs, series, load_derived())
        obs = pd.concat([obs, derived_obs], ignore_index=True)
        series = pd.concat([series, pd.DataFrame(derived_rows)], ignore_index=True)
        obs["date"] = obs["date"].astype("datetime64[ns]")
        obs["value"] = obs["value"].astype("float64")
        OBS.validate(obs)
        SERIES.validate(series)
        CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        obs.to_parquet(CLEAN_DIR / "obs.parquet", index=False)
        series.to_parquet(CLEAN_DIR / "series.parquet", index=False)
        lines.append(f"obs rows {len(obs):,}; series {series['series_id'].nunique()}")

        fomc = datetimes_to_ns(shock_sources.fomc_events(cache))
        FOMC.validate(fomc)
        fomc.to_parquet(CLEAN_DIR / "events_fomc.parquet", index=False)
        opec = datetimes_to_ns(shock_sources.opec_events(cache))
        OPEC.validate(opec)
        opec.to_parquet(CLEAN_DIR / "events_opec.parquet", index=False)
        fad = pd.read_csv(EVENTS_DIR / "boc_fad.csv",
                          parse_dates=["date", "next_scheduled_stated", "target_effective_date"])
        fad["change_bp"] = fad["change_bp"].astype("float64")
        fad = datetimes_to_ns(fad)
        BOC_FAD.validate(fad)
        fad.to_parquet(CLEAN_DIR / "events_boc.parquet", index=False)
        lines.append(f"events fomc {len(fomc)} opec {len(opec)} boc {len(fad)}")

        doc = {"inputs": inputs, "lines": lines,
               "matched_pin": sum(1 for r in inputs if r["status"] == "matched_pin"),
               "revised": sum(1 for r in inputs if r["status"] == "revised")}
        INPUTS_PATH.write_text(json.dumps(doc, indent=1) + "\n")
        return doc
    finally:
        MANIFEST.write_bytes(saved)


def _by_source(specs: list[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for s in specs:
        out.setdefault(s.source, []).append(s)
    return out


def manifest_unchanged() -> bool:
    """True when prepare restored the committed manifest."""
    import subprocess
    diff = subprocess.run(["git", "diff", "--quiet", "--", "data/manifest.json"], cwd=ROOT)
    return diff.returncode == 0
