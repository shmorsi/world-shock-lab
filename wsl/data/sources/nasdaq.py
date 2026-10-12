"""Nasdaq quote API: validation-only second source for US-listed ETF closes.

Closes are split-adjusted but not dividend-adjusted, like Yahoo's "close" (verified on the
2016-11-07 EWU/EWI/EWJ and 2025-12-05 XLE splits; see docs/DATA_QUALITY.md §7).

Unofficial endpoint, about 10 years of history. Never part of the build; used only by the data
quality report to spot-check Yahoo (data/SOURCES.md §3.4). TSX-listed tickers are not covered.
"""

from __future__ import annotations

import json

import pandas as pd

from wsl.data.registry import SeriesSpec
from wsl.data.sources.base import Artifact

API = "https://api.nasdaq.com/api/quote"
WINDOW = ("2016-01-01", "2026-09-30")  # fixed so the request is reproducible


def us_listed(specs: list[SeriesSpec]) -> list[SeriesSpec]:
    return [s for s in specs if s.tz == "America/New_York" and not s.key.startswith("^")]


def artifacts(specs: list[SeriesSpec]) -> list[Artifact]:
    return [
        Artifact(
            key=f"nasdaq/{s.key}",
            url=f"{API}/{s.key}/historical",
            ext="json",
            params={"assetclass": "etf", "fromdate": WINDOW[0], "todate": WINDOW[1],
                    "limit": 9999},
        )
        for s in us_listed(specs)
    ]


def parse_closes(content: bytes) -> pd.Series:
    rows = (json.loads(content).get("data") or {}).get("tradesTable", {}).get("rows") or []
    if not rows:
        return pd.Series(dtype="float64")
    dates = pd.to_datetime([r["date"] for r in rows], format="%m/%d/%Y")
    closes = pd.to_numeric([r["close"].replace("$", "").replace(",", "") for r in rows])
    return pd.Series(closes, index=dates, dtype="float64").sort_index()
