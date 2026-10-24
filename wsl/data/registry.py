"""Load and validate the series registry (config/series.yaml)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from wsl.paths import SERIES_REGISTRY

FREQS = frozenset({"D", "M", "A"})
POLICIES = frozenset({"C", "D", "L"})
SPECIAL_OBS_TIMES = frozenset({"daily_rate", "intraday_average", "close_unknown"})
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


@dataclass(frozen=True)
class SeriesSpec:
    id: str
    source: str
    key: str
    freq: str
    units: str
    asset_class: str
    obs_time: str
    obs_time_verified: bool
    tz: str
    policy: str
    event_window_ok: bool
    name: str
    country: str | None = None
    group_key: str | None = None  # shared artifact key for expanded (multi-country) entries


def _expand(entry: dict[str, Any]) -> list[dict[str, Any]]:
    codes = entry.pop("expand", None)
    if codes is None:
        return [entry]
    group_key = entry["key"].replace("{countries}", "+".join(codes))
    out = []
    for code in codes:
        e = dict(entry)
        e["id"] = entry["id"].format(country=code)
        e["key"] = code
        e["country"] = code
        e["group_key"] = group_key
        out.append(e)
    return out


def load_registry(path: Path = SERIES_REGISTRY) -> list[SeriesSpec]:
    raw = yaml.safe_load(path.read_text())
    defaults: dict[str, Any] = raw.get("defaults", {})
    specs: list[SeriesSpec] = []
    for entry in raw["series"]:
        for e in _expand(dict(entry)):
            merged = {**defaults, **e}
            specs.append(SeriesSpec(**merged))
    validate_registry(specs)
    return specs


def validate_registry(specs: list[SeriesSpec]) -> None:
    ids = [s.id for s in specs]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate series ids: {dupes}")
    for s in specs:
        if s.freq not in FREQS:
            raise ValueError(f"{s.id}: bad freq {s.freq!r}")
        if s.policy not in POLICIES:
            raise ValueError(f"{s.id}: bad policy {s.policy!r}")
        if not (_HHMM.match(s.obs_time) or s.obs_time in SPECIAL_OBS_TIMES):
            raise ValueError(f"{s.id}: bad obs_time {s.obs_time!r}")
        if s.event_window_ok and s.obs_time in {"intraday_average", "daily_rate"}:
            raise ValueError(f"{s.id}: a {s.obs_time} series cannot be used for event windows")


DERIVED_OPS = frozenset({"spread", "ew_return_index"})


@dataclass(frozen=True)
class DerivedSpec:
    id: str
    op: str
    inputs: tuple[str, ...]
    freq: str
    units: str
    asset_class: str
    obs_time: str
    obs_time_verified: bool
    tz: str
    policy: str
    event_window_ok: bool
    name: str
    country: str | None = None
    min_inputs: int | None = None


def load_derived(path: Path = SERIES_REGISTRY) -> list[DerivedSpec]:
    raw = yaml.safe_load(path.read_text())
    defaults: dict[str, Any] = raw.get("defaults", {})
    out = []
    for entry in raw.get("derived", []):
        merged = {**defaults, **entry, "inputs": tuple(entry["inputs"])}
        spec = DerivedSpec(**merged)
        if spec.op not in DERIVED_OPS:
            raise ValueError(f"{spec.id}: unknown op {spec.op!r}")
        if spec.op == "spread" and len(spec.inputs) != 2:
            raise ValueError(f"{spec.id}: spread needs exactly two inputs")
        out.append(spec)
    base_ids = {s.id for s in load_registry(path)}
    clash = base_ids & {d.id for d in out}
    if clash:
        raise ValueError(f"derived ids clash with registry ids: {sorted(clash)}")
    return out


def by_source(specs: list[SeriesSpec]) -> dict[str, list[SeriesSpec]]:
    out: dict[str, list[SeriesSpec]] = {}
    for s in specs:
        out.setdefault(s.source, []).append(s)
    return out
