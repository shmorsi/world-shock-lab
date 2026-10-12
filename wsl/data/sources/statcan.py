"""Statistics Canada Web Data Service (Statistics Canada Open Licence).

Each data point also carries ``releaseTime``, kept in a separate table (``release_times``) for
real-time features later. It is the time of the *latest* revision, not the first release.
"""

from __future__ import annotations

import json

import pandas as pd

from wsl.data.cache import RawCache
from wsl.data.registry import SeriesSpec
from wsl.data.sources.base import Artifact, Parsed, concat_obs, obs_frame

API = "https://www150.statcan.gc.ca/t1/wds/rest"


def artifacts(specs: list[SeriesSpec]) -> list[Artifact]:
    return [
        Artifact(
            key=f"statcan/v{s.key}",
            url=f"{API}/getDataFromVectorByReferencePeriodRange",
            ext="json",
            params={"vectorIds": f'"{s.key}"', "startRefPeriod": "1900-01-01",
                    "endReferencePeriod": "2100-01-01"},
        )
        for s in specs
    ]


def _points(content: bytes) -> list[dict[str, object]]:
    payload = json.loads(content)
    if payload[0].get("status") != "SUCCESS":
        raise ValueError(f"StatCan returned status {payload[0].get('status')!r}")
    points: list[dict[str, object]] = payload[0]["object"]["vectorDataPoint"]
    return points


def parse(specs: list[SeriesSpec], cache: RawCache) -> Parsed:
    frames, artifact_of = [], {}
    for s in specs:
        key = f"statcan/v{s.key}"
        content, _ = cache.read(key)
        pts = _points(content)
        frames.append(obs_frame(s.id, [p["refPer"] for p in pts], [p["value"] for p in pts]))
        artifact_of[s.id] = key
    return Parsed(obs=concat_obs(frames), artifact_of=artifact_of)


def release_times(specs: list[SeriesSpec], cache: RawCache) -> pd.DataFrame:
    rows = []
    for s in specs:
        content, _ = cache.read(f"statcan/v{s.key}")
        rows += [{"series_id": s.id, "date": p["refPer"], "release_time": p["releaseTime"]}
                 for p in _points(content)]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["release_time"] = pd.to_datetime(df["release_time"])
    return df
