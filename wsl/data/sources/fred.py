"""FRED (Federal Reserve Bank of St. Louis).

Two routes to the same observations:

* **API** (api.stlouisfed.org), which needs ``FRED_API_KEY`` in ``.env``. Per series we store
  three artifacts: observations, series metadata (title, units, notes, last update) and tags
  (FRED's copyright tags, see data/SOURCES.md §3.1). The pinned research cache is built from
  this route.
* **Public graph CSV** (fred.stlouisfed.org/graph/fredgraph.csv), which needs no key and
  serves observations only. ml-study-v1 reads FRED this way, and live mode does when no key
  is set. The CSV rows are wrapped in the JSON shape ``parse`` reads, so one parser serves
  both routes.
"""

from __future__ import annotations

import json
import math
import re

from wsl.data.cache import RawCache
from wsl.data.http import Http
from wsl.data.registry import SeriesSpec
from wsl.data.sources.base import Artifact, Parsed, concat_obs, obs_frame

API = "https://api.stlouisfed.org/fred"
KEY_ENV = "FRED_API_KEY"
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
# Written beside the URL wherever bytes from the CSV route are recorded (live forecast and
# score files), because the SHA-256 there is of the wrapped rows, not of the CSV as served.
CSV_NOTE = ("FRED public graph CSV, no API key; the SHA-256 is of the rows as stored "
            '({"observations": [...]} JSON), not of the CSV file')
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class FredCsvError(RuntimeError):
    """FRED's public CSV did not contain what was asked for."""


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


def observations_key(series_key: str) -> str:
    """The raw-cache key of a series' observations, whichever route fetched them."""
    return f"fred/observations/{series_key}"


def observations_from_csv(content: bytes, series_key: str) -> bytes:
    """Wrap FRED's graph CSV for one series in the JSON shape ``parse`` reads.

    Nothing is filled or dropped: a blank value becomes FRED's "." (missing), as the API
    reports it. Anything that is not the CSV for this series raises, so a challenge page, a
    wrong series or a half-written file can never be read as data.
    """
    text = content.decode("utf-8-sig")
    if text.lstrip().startswith("<"):
        raise FredCsvError(f"FRED returned HTML for {series_key}")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise FredCsvError(f"FRED returned an empty file for {series_key}")
    header = [c.strip().strip('"') for c in lines[0].split(",")]
    if len(header) != 2 or header[1].upper() != series_key.upper():
        raise FredCsvError(f"unexpected CSV header for {series_key}: {lines[0]!r}")
    obs = []
    found = 0
    for line in lines[1:]:
        date, comma, val = line.partition(",")
        date, val = date.strip(), val.strip().strip('"')
        missing = val in {"", "."}
        if not comma or not _DATE.fullmatch(date) or not (missing or _is_number(val)):
            raise FredCsvError(f"unexpected CSV row for {series_key}: {line!r}")
        found += not missing
        obs.append({"date": date, "value": "." if missing else val})
    if not found:
        raise FredCsvError(f"FRED returned no observations for {series_key}")
    return json.dumps({"observations": obs}).encode()


def _is_number(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def fetch_observations_csv(http: Http, series_key: str) -> tuple[bytes, str]:
    """One series from the public CSV route: (wrapped observations, the URL they came from)."""
    got = http.get(CSV_URL, params={"id": series_key})
    return observations_from_csv(got.content, series_key), f"{CSV_URL}?id={series_key}"


def parse(specs: list[SeriesSpec], cache: RawCache) -> Parsed:
    frames, artifact_of = [], {}
    for s in specs:
        key = observations_key(s.key)
        content, _ = cache.read(key)
        rows = json.loads(content)["observations"]
        # FRED marks missing observations (e.g. market holidays) with ".".
        values = [None if r["value"] == "." else r["value"] for r in rows]
        frames.append(obs_frame(s.id, [r["date"] for r in rows], values))
        artifact_of[s.id] = key
    return Parsed(obs=concat_obs(frames), artifact_of=artifact_of)
