"""FRED API (api.stlouisfed.org). Requires FRED_API_KEY in .env.

Per series we store three artifacts: observations, series metadata (title, units, notes,
last update) and tags (FRED's copyright tags, see data/SOURCES.md §3.1).
"""

from __future__ import annotations

import json

from wsl.data.cache import RawCache
from wsl.data.registry import SeriesSpec
from wsl.data.sources.base import Artifact, Parsed, concat_obs, obs_frame

API = "https://api.stlouisfed.org/fred"
KEY_ENV = "FRED_API_KEY"


def _art(kind: str, key: str, endpoint: str, extra: dict[str, str] | None = None) -> Artifact:
    return Artifact(
        key=f"fred/{kind}/{key}",
        url=f"{API}/{endpoint}",
        ext="json",
        params={"series_id": key, "file_type": "json", **(extra or {})},
        requires_env=KEY_ENV,
        secret_params={"api_key": KEY_ENV},
    )


def artifacts(specs: list[SeriesSpec]) -> list[Artifact]:
    out: list[Artifact] = []
    for s in specs:
        out.append(_art("observations", s.key, "series/observations",
                        {"observation_start": "1900-01-01"}))
        out.append(_art("series", s.key, "series"))
        out.append(_art("tags", s.key, "series/tags"))
    return out


def parse(specs: list[SeriesSpec], cache: RawCache) -> Parsed:
    frames, artifact_of = [], {}
    for s in specs:
        key = f"fred/observations/{s.key}"
        content, _ = cache.read(key)
        rows = json.loads(content)["observations"]
        # FRED marks missing observations (e.g. market holidays) with ".".
        values = [None if r["value"] == "." else r["value"] for r in rows]
        frames.append(obs_frame(s.id, [r["date"] for r in rows], values))
        artifact_of[s.id] = key
    return Parsed(obs=concat_obs(frames), artifact_of=artifact_of)
