"""Download every artifact into the immutable raw cache (``wsl data fetch``)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import ModuleType

from dotenv import load_dotenv

from wsl.data.cache import RawCache
from wsl.data.http import FetchError, Http
from wsl.data.registry import SeriesSpec, by_source, load_registry
from wsl.data.sources import (
    fred,
    iacoviello,
    nasdaq,
    oecd,
    statcan,
    valet,
    worldbank,
    yahoo,
)
from wsl.data.sources.base import Artifact
from wsl.paths import ROOT
from wsl.shocks import sources as shock_sources

SERIES_SOURCES: dict[str, ModuleType] = {
    "fred": fred,
    "valet": valet,
    "yahoo": yahoo,
    "statcan": statcan,
    "oecd": oecd,
    "worldbank": worldbank,
    "iacoviello": iacoviello,
}
# Groups that are not series sources in the registry.
EXTRA_GROUPS = ("shocks", "nasdaq")

_EXPECT_TEXT = {"json", "csv"}


@dataclass(frozen=True)
class FetchOutcome:
    key: str
    status: str  # pinned_new | unchanged | repinned | changed_not_pinned | skipped | failed
    detail: str = ""


def artifacts_by_group(specs: list[SeriesSpec]) -> dict[str, list[Artifact]]:
    groups = {name: mod.artifacts(s) for name, s in by_source(specs).items()
              if (mod := SERIES_SOURCES.get(name)) is not None}
    groups["shocks"] = shock_sources.artifacts()
    groups["nasdaq"] = nasdaq.artifacts([s for s in specs if s.source == "yahoo"])
    return groups


def _looks_like_html(content: bytes) -> bool:
    head = content[:512].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def fetch(
    only: set[str] | None = None,
    repin: bool = False,
    cache: RawCache | None = None,
    http: Http | None = None,
) -> list[FetchOutcome]:
    load_dotenv(ROOT / ".env")
    cache = cache or RawCache()
    http = http or Http()
    specs = load_registry()
    outcomes: list[FetchOutcome] = []
    for group, arts in artifacts_by_group(specs).items():
        if only and group not in only:
            continue
        for art in arts:
            if art.requires_env and not os.environ.get(art.requires_env):
                outcomes.append(FetchOutcome(art.key, "skipped", f"{art.requires_env} not set"))
                continue
            params = dict(art.params)
            params.update({p: os.environ[env] for p, env in art.secret_params.items()})
            try:
                got = http.request(art.method, art.url, params=params or None,
                                   json_body=art.payload)
            except FetchError as exc:
                outcomes.append(FetchOutcome(art.key, "failed", str(exc)))
                continue
            if art.ext in _EXPECT_TEXT and _looks_like_html(got.content):
                outcomes.append(FetchOutcome(art.key, "failed",
                                             "got an HTML page instead of data (bot check?)"))
                continue
            res = cache.store(art.key, got.content, url=got.url, ext=art.ext,
                              content_type=got.content_type, repin=repin)
            new, old = res.record.sha256, res.previous_pin
            if old is None:
                outcomes.append(FetchOutcome(art.key, "pinned_new"))
            elif new == old:
                outcomes.append(FetchOutcome(art.key, "unchanged"))
            elif res.pinned:
                outcomes.append(FetchOutcome(art.key, "repinned", f"{old[:12]} -> {new[:12]}"))
            else:
                outcomes.append(FetchOutcome(
                    art.key, "changed_not_pinned",
                    f"source now serves {new[:12]}, pinned {old[:12]}; --repin adopts it"))
    return outcomes


def prune(cache: RawCache | None = None) -> list[str]:
    """Drop manifest entries that no current registry entry or shock source refers to."""
    cache = cache or RawCache()
    wanted = {a.key for arts in artifacts_by_group(load_registry()).values() for a in arts}
    return cache.drop(set(cache.keys) - wanted)
