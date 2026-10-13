"""Publish forecasts before decisions and score them afterwards (``wsl live ...``).

Files (committed; never edited once written):
  live/forecasts/<decision date>_<bank>.json
  live/scores/<decision date>_<bank>_score<k>.json   k = 1 (daily proxy), 2 (JK intraday, FOMC)
A score names the forecast file and its SHA-256, and uses the model parameters stored in
that file, so scoring cannot drift from what was published.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from wsl.data.build import file_sha256, git_sha
from wsl.data.cache import sha256_bytes
from wsl.live.data import LIVE_CONFIG, LiveData, calendar, config
from wsl.live.model import CellModel, event_frame, fit
from wsl.paths import ROOT

LIVE_DIR = ROOT / "live"
FORECASTS = LIVE_DIR / "forecasts"
SCORES = LIVE_DIR / "scores"
PROXY_TEXT = {
    "fomc": "announcement-day change in the US 2-year Treasury yield (bp)",
    "boc": "announcement-day change in the Government of Canada 2-year yield (bp)",
}
SCHEMA = 1


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(t: datetime | pd.Timestamp) -> str:
    return pd.Timestamp(t).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_new(path: Path, doc: dict[str, Any]) -> Path:
    """Write a file that must not exist yet: published files are never overwritten."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; published files are never edited")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path


def _mechanical(cfg: dict[str, Any], bank: str, outcome: str, h: int) -> bool:
    return [bank, outcome, h] in cfg["mechanical"]


def _history(data: LiveData, bank: str, cfg: dict[str, Any], before: pd.Timestamp
             ) -> pd.Series:
    hist = data.fomc_history if bank == "fomc" else data.boc_history
    start = pd.Timestamp(cfg["model"]["sample_start"], tz="UTC")
    ts = pd.to_datetime(list(hist), utc=True)
    return pd.Series(ts[(ts >= start) & (ts < before)])


def make_forecast(cfg: dict[str, Any], data: LiveData, bank: str, decision_date: str,
                  ts: pd.Timestamp, collision: bool, now: datetime) -> dict[str, Any]:
    proxy_id = cfg["proxy"][bank]
    outcomes = list(cfg["outcomes"])
    horizons = [int(h) for h in cfg["horizons"]]
    frame = event_frame(_history(data, bank, cfg, pd.Timestamp(now)), data.series[proxy_id],
                        {o: data.series[o] for o in outcomes}, horizons)
    models = fit(frame, outcomes, horizons, list(cfg["model"]["interval_levels"]),
                 int(cfg["model"]["min_events"]))
    grid = []
    for s in cfg["grid_bp"]:
        for m in models.values():
            grid.append({"surprise_bp": float(s), "outcome": m.outcome, "h": m.h,
                         **{k: round(v, 4) for k, v in m.predict(float(s)).items()},
                         "mechanical": _mechanical(cfg, bank, m.outcome, m.h)})
    other = "BoC (09:45 ET)" if bank == "fomc" else "FOMC (14:00 ET)"
    return {
        "schema": SCHEMA, "kind": "forecast", "label": "model_forecast",
        "event_id": f"{decision_date}_{bank}", "bank": bank, "decision_date": decision_date,
        "announcement_ts": pd.Timestamp(ts).isoformat(),
        "published_at": _iso(now),
        "hours_before": round((pd.Timestamp(ts) - pd.Timestamp(now)).total_seconds() / 3600, 1),
        "collision": bool(collision),
        "collision_note": (f"The {other} also announces on this day; daily data cannot separate "
                           "the two, so this forecast is scored but reported separately.")
        if collision else "",
        "proxy": proxy_id, "proxy_text": PROXY_TEXT[bank],
        "horizon_convention": "h = 1 is the first observation after the announcement, h = 5 "
                              "the fifth; changes from the last observation before it",
        "units": {"CA2Y": "bp", "USDCAD": "% (100 × log change, CAD per USD)",
                  "CA_BANKS_EW6": "% (100 × log change)"},
        "grid": grid,
        "model": {col: {"a": m.a, "b": m.b, "n": m.n, "half": m.half, "first": m.first,
                        "last": m.last} for col, m in models.items()},
        "data": {k: data.provenance[k] for k in (proxy_id, *outcomes)},
        "live_config_sha256": file_sha256(LIVE_CONFIG),
        "code_git_sha": git_sha(),
        "protocol": "v4",
    }


def _realized(data: LiveData, cfg: dict[str, Any], f: dict[str, Any]) -> pd.Series | None:
    """The proxy and every outcome cell for this announcement, or None if not all published."""
    ts = pd.Series([pd.Timestamp(f["announcement_ts"])])
    horizons = [int(h) for h in cfg["horizons"]]
    row = event_frame(ts, data.series[f["proxy"]],
                      {o: data.series[o] for o in cfg["outcomes"]}, horizons).iloc[0]
    if row.drop(labels=["ts"]).isna().any():
        return None
    return row


def make_score(cfg: dict[str, Any], data: LiveData, f: dict[str, Any], forecast_path: Path,
               k: int, surprise: float, source: str, row: pd.Series, now: datetime
               ) -> dict[str, Any]:
    public = set(cfg["publish_realized_policies"])
    surprise_public = k == 2 or data.policy[f["proxy"]] in public
    scores = []
    for col, p in f["model"].items():
        m = CellModel(col.split("|")[0], int(col.split("|h")[1]), p["a"], p["b"], p["n"],
                      p["half"], p["first"], p["last"])
        pred = m.predict(surprise)
        y = float(row[col])
        err = y - pred["mean"]
        cell: dict[str, Any] = {
            "outcome": m.outcome, "h": m.h,
            "mechanical": _mechanical(cfg, f["bank"], m.outcome, m.h),
            "inside68": bool(pred["lo68"] <= y <= pred["hi68"]),
            "inside90": bool(pred["lo90"] <= y <= pred["hi90"]),
            "beats_no_change": bool(abs(err) < abs(y)),
        }
        if surprise_public:
            cell.update({kk: round(v, 4) for kk, v in pred.items()})
            cell["forecast"] = cell.pop("mean")
        if surprise_public and data.policy[m.outcome] in public:
            cell.update({"realized": round(y, 4), "abs_error": round(abs(err), 4),
                         "no_change_abs_error": round(abs(y), 4)})
        scores.append(cell)
    return {
        "schema": SCHEMA, "kind": "score", "score": k, "label": "model_forecast",
        "event_id": f["event_id"], "bank": f["bank"], "decision_date": f["decision_date"],
        "collision": f["collision"], "proxy": f["proxy"],
        "forecast_file": str(forecast_path.relative_to(ROOT)),
        "forecast_sha256": sha256_bytes(forecast_path.read_bytes()),
        "scored_at": _iso(now),
        "surprise_source": source,
        "realized_proxy_bp": round(surprise, 4) if surprise_public else None,
        "restricted_note": "" if surprise_public else
        "The realized BoC proxy is a CA 2Y change (open item O-3), so only hit rates are "
        "published for this score.",
        "scores": scores,
        "data": {kk: data.provenance[kk] for kk in (f["proxy"], *cfg["outcomes"])}
        | ({"JK": data.provenance["JK"]} if k == 2 else {}),
        "live_config_sha256": file_sha256(LIVE_CONFIG),
        "code_git_sha": git_sha(),
    }


def forecast_due(cfg: dict[str, Any], data: LiveData, now: datetime,
                 days: int | None = None) -> list[Path]:
    """Publish forecasts for decisions within ``days`` (default: config) and at least the
    minimum notice away that have none yet."""
    ahead = days if days is not None else int(cfg["schedule"]["publish_days_before"])
    horizon = pd.Timedelta(days=ahead)
    notice = pd.Timedelta(hours=cfg["schedule"]["min_hours_before"])
    stamp = pd.Timestamp(now)
    out = []
    for r in calendar(cfg).to_dict("records"):
        bank, day, ts = str(r["bank"]), str(r["decision_date"]), pd.Timestamp(r["ts"])
        path = FORECASTS / f"{day}_{bank}.json"
        if path.exists() or not (notice <= ts - stamp <= horizon):
            continue
        out.append(_write_new(path, make_forecast(cfg, data, bank, day, ts,
                                                  bool(r["collision"]), now)))
    return out


def score_due(cfg: dict[str, Any], data: LiveData, now: datetime) -> list[Path]:
    out = []
    for path in sorted(FORECASTS.glob("*.json")):
        f = json.loads(path.read_text())
        if pd.Timestamp(f["announcement_ts"]) > pd.Timestamp(now):
            continue
        row = _realized(data, cfg, f)
        if row is None:
            continue
        s1 = SCORES / f"{f['event_id']}_score1.json"
        if not s1.exists():
            out.append(_write_new(s1, make_score(cfg, data, f, path, 1, float(row["proxy"]),
                                                 f"daily proxy: {PROXY_TEXT[f['bank']]}", row,
                                                 now)))
        s2 = SCORES / f"{f['event_id']}_score2.json"
        if f["bank"] == "fomc" and not s2.exists():
            jk = data.jk_surprises
            hit = jk[jk["ts"].dt.date.astype(str) == f["decision_date"]]
            if len(hit) == 1 and pd.notna(hit["TFUT02_bp"].iloc[0]):
                out.append(_write_new(s2, make_score(
                    cfg, data, f, path, 2, float(hit["TFUT02_bp"].iloc[0]),
                    "JK 30-minute change in the 2-year Treasury futures yield (TFUT02, bp)",
                    row, now)))
    return out


def status(cfg: dict[str, Any] | None = None, now: datetime | None = None) -> list[str]:
    cfg = cfg or config()
    now = now or _now()
    lines = []
    for r in calendar(cfg).to_dict("records"):
        if pd.Timestamp(r["ts"]) < pd.Timestamp(now) - pd.Timedelta(days=60):
            continue
        bank, day = str(r["bank"]), str(r["decision_date"])
        fid = f"{day}_{bank}"
        have = [p.name for p in SCORES.glob(f"{fid}_score*.json")]
        state = "forecast published" if (FORECASTS / f"{fid}.json").exists() else "no forecast"
        flag = " [collision]" if r["collision"] else ""
        lines.append(f"{day} {bank.upper():4}{flag}: {state}"
                     + (f"; {', '.join(sorted(have))}" if have else ""))
    return lines
