"""OECD SDMX REST API (rate-limited; licence for data pending, open item O-5 -> policy D)."""

from __future__ import annotations

import io

import pandas as pd

from wsl.data.cache import RawCache
from wsl.data.registry import SeriesSpec
from wsl.data.sources.base import Artifact, Parsed, concat_obs, obs_frame

API = "https://sdmx.oecd.org/public/rest/data"


def _artifact_key(group_key: str) -> str:
    """Key includes the country list, so editing the registry creates a new artifact instead
    of looking like a revision of the old one."""
    flow, _, dims = group_key.partition("/")
    countries, series_dims = dims.split(".")[0], ".".join(dims.split(".")[2:])
    return f"oecd/{flow.split(',')[1]}/{series_dims}/{countries}"


def artifacts(specs: list[SeriesSpec]) -> list[Artifact]:
    groups = sorted({s.group_key for s in specs if s.group_key})
    return [
        Artifact(key=_artifact_key(g), url=f"{API}/{g}", ext="csv",
                 params={"startPeriod": "1950-01", "format": "csvfilewithlabels"})
        for g in groups
    ]


def parse(specs: list[SeriesSpec], cache: RawCache) -> Parsed:
    frames, artifact_of = [], {}
    tables: dict[str, pd.DataFrame] = {}
    for s in specs:
        assert s.group_key is not None
        key = _artifact_key(s.group_key)
        if key not in tables:
            content, _ = cache.read(key)
            tables[key] = pd.read_csv(io.BytesIO(content), dtype={"OBS_VALUE": "float64"})
        t = tables[key]
        sub = t[t["REF_AREA"] == s.key].sort_values("TIME_PERIOD")
        if sub["TIME_PERIOD"].duplicated().any():
            raise ValueError(f"{s.id}: query matched more than one OECD series per period")
        frames.append(obs_frame(s.id, sub["TIME_PERIOD"], sub["OBS_VALUE"]))
        artifact_of[s.id] = key
    return Parsed(obs=concat_obs(frames), artifact_of=artifact_of)
