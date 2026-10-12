"""Shared types for source adapters.

An adapter does two things:
  * ``artifacts(specs)`` lists the downloads it needs (each becomes one raw-cache entry);
  * ``parse(specs, cache)`` turns pinned raw bytes into the long observation table.

Parsers never fill, interpolate or drop values. A value the source reports as missing is kept
as NaN with ``status = "missing_at_source"``; an impossible value (e.g. a zero stock price) is
kept as NaN with ``status = "invalid_at_source"`` and listed in the data-quality report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from wsl.data.cache import RawCache
from wsl.data.registry import SeriesSpec

OBS_COLUMNS = ["series_id", "date", "value", "status"]
STATUS_OK = "ok"
STATUS_MISSING = "missing_at_source"
STATUS_INVALID = "invalid_at_source"  # source reported an impossible value; kept as NaN, flagged


@dataclass(frozen=True)
class Artifact:
    key: str  # raw-cache key, "<source>/<name>"
    url: str
    ext: str
    params: dict[str, Any] = field(default_factory=dict)
    method: str = "GET"
    payload: Any = None
    requires_env: str | None = None  # e.g. "FRED_API_KEY"; the artifact is skipped without it
    secret_params: dict[str, str] = field(default_factory=dict)  # param -> env var


@dataclass
class Parsed:
    obs: pd.DataFrame  # OBS_COLUMNS
    artifact_of: dict[str, str]  # series_id -> raw-cache key it was parsed from
    transformations: dict[str, list[str]] = field(default_factory=dict)  # series_id -> steps


def obs_frame(series_id: str, dates: Any, values: Any) -> pd.DataFrame:
    """Build observation rows; NaN values are flagged, never dropped or filled."""
    df = pd.DataFrame({"date": pd.to_datetime(dates),
                       "value": pd.to_numeric(values, errors="coerce")})
    df.insert(0, "series_id", series_id)
    df["status"] = STATUS_OK
    df.loc[df["value"].isna(), "status"] = STATUS_MISSING
    return df[OBS_COLUMNS]


def concat_obs(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=OBS_COLUMNS)
    return pd.concat(frames, ignore_index=True)


ArtifactLister = Callable[[list[SeriesSpec]], list[Artifact]]
Parser = Callable[[list[SeriesSpec], RawCache], Parsed]
