"""Geopolitical Risk and Trade Policy Uncertainty indices (matteoiacoviello.com).

Reduced-form indices of news attention, not identified shocks (docs/PLAN.md §1).
Registry keys look like "gpr_daily:GPRD" or "tpu:TPU_DAILY:TPUD_index" (file[:sheet]:column).
"""

from __future__ import annotations

import io

import pandas as pd

from wsl.data.cache import RawCache
from wsl.data.registry import SeriesSpec
from wsl.data.sources.base import Artifact, Parsed, concat_obs, obs_frame

BASE = "https://www.matteoiacoviello.com"
FILES = {
    "gpr_daily": (f"{BASE}/gpr_files/data_gpr_daily_recent.xls", "xls", "date"),
    "gpr_monthly": (f"{BASE}/gpr_files/data_gpr_export.xls", "xls", "month"),
    "tpu": (f"{BASE}/tpu_files/tpu_web_latest.xlsx", "xlsx", "DATE"),
}


def artifacts(specs: list[SeriesSpec]) -> list[Artifact]:
    needed = sorted({s.key.split(":")[0] for s in specs})
    return [Artifact(key=f"iacoviello/{f}", url=FILES[f][0], ext=FILES[f][1]) for f in needed]


def parse(specs: list[SeriesSpec], cache: RawCache) -> Parsed:
    frames, artifact_of = [], {}
    books: dict[str, dict[str, pd.DataFrame]] = {}
    for s in specs:
        parts = s.key.split(":")
        fname, column = parts[0], parts[-1]
        sheet = parts[1] if len(parts) == 3 else None
        key = f"iacoviello/{fname}"
        if key not in books:
            content, _ = cache.read(key)
            books[key] = pd.read_excel(io.BytesIO(content), sheet_name=None)
        book = books[key]
        df = book[sheet] if sheet else next(iter(book.values()))
        date_col = FILES[fname][2]
        sub = df[[date_col, column]].dropna(subset=[date_col]).sort_values(date_col)
        frames.append(obs_frame(s.id, sub[date_col], sub[column]))
        artifact_of[s.id] = key
    return Parsed(obs=concat_obs(frames), artifact_of=artifact_of)
