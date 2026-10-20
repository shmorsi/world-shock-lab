"""Event panel for the ML study.

One row per announcement × asset × horizon whose target exists. Features are the
M5 tabular features (raw surprises, pre-announcement state, family indicators,
the asset's own 20-day volatility). The path is the 20 daily changes of that
asset ending at the last observation strictly before the announcement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wsl.analysis.boc_proxy import ASSUMED
from wsl.ml.dataset import (
    ASSET_NAMES,
    ASSETS,
    HORIZONS,
    SHOCK_COLS,
    STATE_COLS,
    EventData,
    build,
    in_test_window,
)
from wsl.ml.models import FAMILY_DUMMIES, base_frame
from wsl.shocks.windows import WindowSeries, load_series
from wsl.timing import utc_ns

PATH_LEN = 20
CELL_NAMES = [f"{a}|h{h}" for a in ASSET_NAMES for h in HORIZONS]
SHOCK_NAMES = [c for cols in SHOCK_COLS.values() for c in cols]
TABULAR = SHOCK_NAMES + STATE_COLS + list(FAMILY_DUMMIES) + ["asset_vol"]


@dataclass
class Panel:
    """Rows are aligned. ``event_i`` increases with time and is shared by the rows of one event."""

    event_id: np.ndarray
    event_i: np.ndarray
    year: np.ndarray
    asset_i: np.ndarray
    horizon_i: np.ndarray
    cell: np.ndarray  # object, "US2Y|h1"
    X: np.ndarray
    path: np.ndarray  # (n, PATH_LEN), NaN where the history is too short
    y: np.ndarray
    feature_names: list[str]

    def mask(self, event_index: np.ndarray) -> np.ndarray:
        """Boolean row mask for the given event_i values."""
        return np.isin(self.event_i, np.asarray(event_index, dtype=int))


def pre_event_changes(s: WindowSeries, ts: pd.Series, length: int = PATH_LEN) -> np.ndarray:
    """``(n, length)`` daily changes ending at the last print strictly before each event.

    ``pos`` is that print. The last column is ``x[pos] - x[pos - 1]``. Nothing at or
    after the announcement enters the path. Rows with fewer than ``length`` earlier
    changes are NaN.
    """
    pos = np.searchsorted(utc_ns(s.obs_ts), utc_ns(ts), side="left") - 1
    diff = np.diff(s.x)
    out = np.full((len(ts), length), np.nan)
    for i, p in enumerate(pos):
        if p >= length:
            out[i] = diff[p - length:p]
    return out


def _paths(events: pd.DataFrame) -> dict[str, np.ndarray]:
    ts = events["ts"]
    out = {}
    for name in ASSET_NAMES:
        sid = ASSETS[name]
        series = load_series(sid, ASSUMED.get(sid))
        out[name] = pre_event_changes(series, ts, PATH_LEN)
    return out


def from_events(d: EventData | None = None) -> tuple[Panel, EventData]:
    """Primary sample only: BoC announcements on FOMC days are dropped, and so is anything
    after the test window (those belong to the live period)."""
    d = d or build()
    keep = ~d.events["boc_same_day_fomc"].to_numpy() & in_test_window(d.events["ts"]).to_numpy()
    events = d.events.loc[keep].sort_values("ts")
    targets = d.targets.loc[events.index]
    vol = d.asset_vol.loc[events.index]
    paths = _paths(events)
    event_ids = events.index.to_numpy()
    event_i = np.arange(len(events))
    year = events["year"].to_numpy(dtype=int)

    rows_id, rows_i, rows_year = [], [], []
    rows_a, rows_h, rows_cell = [], [], []
    xs, ps, ys = [], [], []
    base = base_frame(events, pd.Series(0.0, index=events.index))
    for a, name in enumerate(ASSET_NAMES):
        for h_i, h in enumerate(HORIZONS):
            col = f"{name}|h{h}"
            y = targets[col]
            ok = y.notna().to_numpy()
            if not ok.any():
                continue
            block = base.loc[ok, [c for c in TABULAR if c != "asset_vol"]].copy()
            block["asset_vol"] = vol.loc[ok, name].to_numpy()
            block = block[TABULAR]
            n = int(ok.sum())
            rows_id.append(event_ids[ok])
            rows_i.append(event_i[ok])
            rows_year.append(year[ok])
            rows_a.append(np.full(n, a, dtype=int))
            rows_h.append(np.full(n, h_i, dtype=int))
            rows_cell.append(np.full(n, col, dtype=object))
            xs.append(block.to_numpy(dtype=float))
            ps.append(paths[name][ok])
            ys.append(y.to_numpy()[ok])
    panel = Panel(
        event_id=np.concatenate(rows_id),
        event_i=np.concatenate(rows_i),
        year=np.concatenate(rows_year),
        asset_i=np.concatenate(rows_a),
        horizon_i=np.concatenate(rows_h),
        cell=np.concatenate(rows_cell),
        X=np.vstack(xs),
        path=np.vstack(ps),
        y=np.concatenate(ys).astype(float),
        feature_names=list(TABULAR),
    )
    return panel, d


def event_table(panel: Panel) -> pd.DataFrame:
    """One row per event, in time order, with year and event_i."""
    _, idx = np.unique(panel.event_i, return_index=True)
    return pd.DataFrame({
        "event_id": panel.event_id[idx],
        "event_i": panel.event_i[idx],
        "year": panel.year[idx],
    }).sort_values("event_i").reset_index(drop=True)
