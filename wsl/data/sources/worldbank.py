"""World Bank WDI API v2 (CC BY 4.0). Annual values, used to classify oil exporters."""

from __future__ import annotations

import json

from wsl.data.cache import RawCache
from wsl.data.registry import SeriesSpec
from wsl.data.sources.base import Artifact, Parsed, concat_obs, obs_frame

API = "https://api.worldbank.org/v2"


def artifacts(specs: list[SeriesSpec]) -> list[Artifact]:
    indicators: dict[str, list[str]] = {}
    for s in specs:
        assert s.country is not None
        indicators.setdefault(s.group_key or s.key, []).append(s.country)
    return [
        Artifact(key=_key(ind, codes),
                 url=f"{API}/country/{';'.join(sorted(codes))}/indicator/{ind}", ext="json",
                 params={"format": "json", "per_page": 20000, "date": "1960:2030"})
        for ind, codes in sorted(indicators.items())
    ]


def _key(indicator: str, codes: list[str]) -> str:
    return f"worldbank/{indicator}/{'+'.join(sorted(codes))}"


def parse(specs: list[SeriesSpec], cache: RawCache) -> Parsed:
    frames, artifact_of = [], {}
    codes: dict[str, list[str]] = {}
    for s in specs:
        assert s.country is not None
        codes.setdefault(s.group_key or s.key, []).append(s.country)
    for s in specs:
        indicator = s.group_key or s.key
        key = _key(indicator, codes[indicator])
        content, _ = cache.read(key)
        meta, rows = json.loads(content)
        if meta.get("pages", 1) != 1:
            raise ValueError(f"{key}: paginated response; raise per_page")
        mine = sorted((r for r in rows if r["countryiso3code"] == s.country),
                      key=lambda r: r["date"])
        frames.append(obs_frame(s.id, [f"{r['date']}-01-01" for r in mine],
                                [r["value"] for r in mine]))
        artifact_of[s.id] = key
    return Parsed(obs=concat_obs(frames), artifact_of=artifact_of)
