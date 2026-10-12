"""Results files with provenance. Every results JSON says exactly which inputs, code and
protocol produced it, so any number shown later can be traced back."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from wsl.data.build import file_sha256, git_sha
from wsl.paths import CLEAN_DIR, EVALUATION_CONFIG, MANIFEST, ROOT

RESULTS_DIR = ROOT / "results"


def provenance(series_ids: Iterable[str], tables: Iterable[str]) -> dict[str, Any]:
    series = pd.read_parquet(CLEAN_DIR / "series.parquet").set_index("series_id")
    prov = pd.read_parquet(CLEAN_DIR / "provenance.parquet")
    info = pd.read_parquet(CLEAN_DIR / "build_info.parquet").iloc[0]
    sids = sorted(set(series_ids))
    tabs = sorted(set(tables))
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "data_built_at": str(info["built_at"]),
        "evaluation_config_sha256": file_sha256(EVALUATION_CONFIG),
        "manifest_sha256": file_sha256(MANIFEST),
        "series": {s: {"sha256": str(series.loc[s, "sha256"]), "url": str(series.loc[s, "url"]),
                       "policy": str(series.loc[s, "policy"])} for s in sids},
        "event_tables": {t: prov[prov["table"] == t][["artifact_key", "sha256"]]
                         .to_dict(orient="records") for t in tabs},
    }


def write(name: str, payload: dict[str, Any], series_ids: Iterable[str],
          tables: Iterable[str]) -> Path:
    path = RESULTS_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"name": name, "provenance": provenance(series_ids, tables), **payload}
    path.write_text(json.dumps(doc, indent=1, default=float) + "\n")
    return path
