"""Yahoo Finance chart endpoint (unofficial; personal-use terms -> policy L, never published).

For each ticker we keep what Yahoo reports and derive one series ourselves:
  <id>.close     split-adjusted close (Yahoo "close")
  <id>.adjclose  Yahoo's split+dividend adjusted close (kept only to cross-check our TR)
  <id>.dividend  cash dividend per share on its ex-date (rows only on ex-dates)
  <id>.tr        total-return index computed by wsl.data.transforms.total_return_index

Dates are the exchange-local trading date (meta.exchangeTimezoneName).
"""

from __future__ import annotations

import json
from urllib.parse import quote

import pandas as pd

from wsl.data.cache import RawCache
from wsl.data.registry import SeriesSpec
from wsl.data.sources.base import STATUS_INVALID, Artifact, Parsed, concat_obs, obs_frame
from wsl.data.transforms import total_return_index

API = "https://query1.finance.yahoo.com/v8/finance/chart"
PERIOD2 = 4102444800  # 2100-01-01: fixed so the request URL is stable across runs


def artifacts(specs: list[SeriesSpec]) -> list[Artifact]:
    return [
        Artifact(
            key=f"yahoo/{s.key}",
            url=f"{API}/{quote(s.key)}",
            ext="json",
            params={"period1": 0, "period2": PERIOD2, "interval": "1d",
                    "events": "div,split", "includeAdjustedClose": "true"},
        )
        for s in specs
    ]


def parse_chart(content: bytes) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, str]]:
    """-> (daily frame [close, adjclose] by local date, dividends, splits, meta)."""
    result = json.loads(content)["chart"]["result"][0]
    meta = result["meta"]
    tz = meta["exchangeTimezoneName"]
    dates = (
        pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None)
        .normalize()
    )
    quote_ = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    daily = pd.DataFrame(
        {"close": pd.to_numeric(quote_["close"], errors="coerce"),
         "adjclose": pd.to_numeric(adj, errors="coerce") if adj is not None else float("nan")},
        index=dates,
    )
    if daily.index.has_duplicates:  # Yahoo occasionally emits an intraday bar for "today"
        daily = daily[~daily.index.duplicated(keep="last")]
    events = result.get("events", {})

    def _events(name: str, field: str) -> pd.Series:
        raw = events.get(name, {})
        if not raw:
            return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
        idx = (pd.to_datetime([v["date"] for v in raw.values()], unit="s", utc=True)
               .tz_convert(tz).tz_localize(None).normalize())
        vals = [v[field] if field != "ratio" else v["numerator"] / v["denominator"]
                for v in raw.values()]
        return pd.Series(vals, index=idx, dtype="float64").sort_index()

    info = {"exchange": meta.get("exchangeName", ""), "currency": meta.get("currency", ""),
            "tz": tz, "symbol": meta.get("symbol", "")}
    return daily.sort_index(), _events("dividends", "amount"), _events("splits", "ratio"), info


def parse(specs: list[SeriesSpec], cache: RawCache) -> Parsed:
    frames, artifact_of, transformations = [], {}, {}
    for s in specs:
        key = f"yahoo/{s.key}"
        content, _ = cache.read(key)
        daily, divs, _splits, _ = parse_chart(content)
        # A price of zero or below is impossible for these instruments; Yahoo emits such bars
        # on some exchange holidays (e.g. CM.TO on 2001-12-26). Flag them, never use them.
        invalid = daily["close"] <= 0
        close = daily["close"].mask(invalid)
        tr = total_return_index(close, divs)
        close_obs = obs_frame(f"{s.id}.close", daily.index, close)
        close_obs.loc[invalid.to_numpy(), "status"] = STATUS_INVALID
        frames += [
            close_obs,
            obs_frame(f"{s.id}.adjclose", daily.index, daily["adjclose"]),
            obs_frame(f"{s.id}.tr", tr.index, tr.values),
        ]
        suffixes = ["close", "adjclose", "tr"]
        if not divs.empty:  # price indices (^GSPTSE, ^VIX) pay no dividends
            frames.append(obs_frame(f"{s.id}.dividend", divs.index, divs.values))
            suffixes.append("dividend")
        for suffix in suffixes:
            artifact_of[f"{s.id}.{suffix}"] = key
        transformations[f"{s.id}.tr"] = [
            "total_return_index(close, dividend): "
            "TR_t = TR_p * (close_t + sum(div in (p,t])) / close_p",
            "base 100 at first available close",
        ]
    return Parsed(obs=concat_obs(frames), artifact_of=artifact_of, transformations=transformations)
