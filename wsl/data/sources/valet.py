"""Bank of Canada Valet API. No key; requests are rate-limited (see BoC terms, SOURCES.md §3.2)."""

from __future__ import annotations

import json

from wsl.data.cache import RawCache
from wsl.data.registry import SeriesSpec
from wsl.data.sources.base import Artifact, Parsed, concat_obs, obs_frame

API = "https://www.bankofcanada.ca/valet"


def artifacts(specs: list[SeriesSpec]) -> list[Artifact]:
    return [
        Artifact(key=f"valet/{s.key}", url=f"{API}/observations/{s.key}/json", ext="json",
                 params={"start_date": "1900-01-01"})
        for s in specs
    ]


def parse(specs: list[SeriesSpec], cache: RawCache) -> Parsed:
    frames, artifact_of = [], {}
    for s in specs:
        key = f"valet/{s.key}"
        content, _ = cache.read(key)
        rows = json.loads(content)["observations"]
        dates = [r["d"] for r in rows]
        values = [(r.get(s.key) or {}).get("v") or None for r in rows]
        frames.append(obs_frame(s.id, dates, values))
        artifact_of[s.id] = key
    return Parsed(obs=concat_obs(frames), artifact_of=artifact_of)
