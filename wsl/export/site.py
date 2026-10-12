"""``wsl export site``: the JSON files the static web app reads (web/public/data/).

Licence filter (data/SOURCES.md):
  public (default, what is committed and deployed)
    * only derived statistics (responses, metrics), never raw series values;
    * nothing derived from VIX or Moody's spreads (open item O-9): no VIX-regime responses;
    * Event Replay shows realized and predicted moves only for public-domain series (policy C);
      other assets appear only in aggregate metrics.
  local: everything, for the author's own use (git-ignored).
Every file carries the provenance block of wsl.export.results.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from wsl.data.build import file_sha256, git_sha
from wsl.export.results import provenance
from wsl.ml.dataset import ASSETS
from wsl.ml.implied import implied_responses
from wsl.paths import CLEAN_DIR, EVALUATION_CONFIG, ROOT
from wsl.scenarios.catalog import build_catalog, build_regime_irfs

WEB_DATA = ROOT / "web" / "public" / "data"
LOCAL_DATA = ROOT / "web" / "public" / "data-local"
TABLES = ["events_fomc", "events_fomc_bs", "events_opec", "events_boc", "shocks_monthly"]
# US-listed iShares country ETFs and H.10 currencies by ISO 3166-1 numeric code (world-atlas).
COUNTRIES = {
    "CAN": {"iso_n3": "124", "name": "Canada", "equity": "EWC.tr", "fx": "USDCAD", "fx_sign": -1},
    "GBR": {"iso_n3": "826", "name": "United Kingdom", "equity": "EWU.tr", "fx": "GBPUSD",
            "fx_sign": 1},
    "DEU": {"iso_n3": "276", "name": "Germany", "equity": "EWG.tr", "fx": "EURUSD", "fx_sign": 1},
    "FRA": {"iso_n3": "250", "name": "France", "equity": "EWQ.tr", "fx": "EURUSD", "fx_sign": 1},
    "ITA": {"iso_n3": "380", "name": "Italy", "equity": "EWI.tr", "fx": "EURUSD", "fx_sign": 1},
    "JPN": {"iso_n3": "392", "name": "Japan", "equity": "EWJ.tr", "fx": "USDJPY", "fx_sign": -1},
    "KOR": {"iso_n3": "410", "name": "South Korea", "equity": "EWY.tr", "fx": "USDKRW",
            "fx_sign": -1},
    "MEX": {"iso_n3": "484", "name": "Mexico", "equity": "EWW.tr", "fx": "USDMXN", "fx_sign": -1},
    "NOR": {"iso_n3": "578", "name": "Norway", "equity": "ENOR.tr", "fx": "USDNOK",
            "fx_sign": -1},
    "AUS": {"iso_n3": "036", "name": "Australia", "equity": "EWA.tr", "fx": "AUDUSD",
            "fx_sign": 1},
    "BRA": {"iso_n3": "076", "name": "Brazil", "equity": "EWZ.tr", "fx": "USDBRL", "fx_sign": -1},
    "CHN": {"iso_n3": "156", "name": "China", "equity": "FXI.tr", "fx": "USDCNY", "fx_sign": -1},
    "IND": {"iso_n3": "356", "name": "India", "equity": "INDA.tr", "fx": "USDINR", "fx_sign": -1},
    "USA": {"iso_n3": "840", "name": "United States", "equity": "SPY.tr", "fx": "USD_BROAD",
            "fx_sign": 1},
}


def _round(x: Any, sig: int = 6) -> Any:
    """Round floats to ``sig`` significant digits (display precision; keeps files small)."""
    if isinstance(x, float):
        return float(f"{x:.{sig}g}") if x == x and abs(x) != float("inf") else None
    if isinstance(x, dict):
        return {k: _round(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, sig) for v in x]
    return x


def _write(folder: Path, name: str, payload: dict[str, Any], series: set[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    doc = _round({"provenance": provenance(series, TABLES), **payload})
    (folder / f"{name}.json").write_text(json.dumps(doc, separators=(",", ":"), default=float))


def _final_eval(public: bool) -> dict[str, Any]:
    runs = sorted((ROOT / "results" / "final_eval").glob("*.json"))
    fin = json.loads(runs[-1].read_text())
    meta = pd.read_parquet(CLEAN_DIR / "series.parquet").set_index("series_id")
    open_assets = {a for a, sid in ASSETS.items() if str(meta.loc[sid, "policy"]) == "C"}
    replay = []
    for r in fin["replay"]:
        cells = {k: v for k, v in r["cells"].items()
                 if not public or k.split("|")[0] in open_assets}
        replay.append({**r, "cells": cells})
    keep = ("phase", "generated_at", "protocol_tag", "git_sha", "candidates",
            "test_events_primary", "test_years", "pooled_scaled_mae", "per_cell",
            "difference_ml_minus_econ", "verdict", "static_refit", "subsamples", "attributions")
    out = {k: fin[k] for k in keep}
    out["replay"] = replay
    out["replay_assets"] = sorted(open_assets) if public else list(ASSETS)
    return out


def export(public: bool = True) -> Path:
    folder = WEB_DATA if public else LOCAL_DATA
    catalog = build_catalog()
    series = {o["id"] for o in catalog["outcomes"]}
    _write(folder, "catalog", catalog, series)
    _write(folder, "regimes", {"regimes": build_regime_irfs(include_vix=not public),
                               "note": "VIX regimes omitted from the public build (O-9)"
                               if public else ""}, series | {"VIX"})
    m4 = {name: json.loads((ROOT / "results" / "m4" / f"{name}.json").read_text())["results"]
          for name in ("var_vs_lp", "canada", "exporters", "robustness", "state_dependence")}
    if public:
        m4["state_dependence"] = [c for c in m4["state_dependence"] if c["regime"] != "high_vix"]
    _write(folder, "m4", m4, series)
    _write(folder, "ml", {"final": _final_eval(public), "implied": implied_responses()},
           set(ASSETS.values()))
    m3 = json.loads((ROOT / "results" / "m3" / "replication.json").read_text())
    _write(folder, "m3", {"claims": [{k: c[k] for k in ("id", "source", "claim", "test",
                                                         "expected_sign", "mechanical",
                                                         "verdict", "estimate", "band90", "n")}
                                     for c in m3["claims"]]}, series)
    live_dir = ROOT / "live"
    live = [json.loads(p.read_text()) for p in sorted(live_dir.rglob("*.json"))] \
        if live_dir.exists() else []
    _write(folder, "live", {"entries": live}, set())
    _write(folder, "meta", {
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_sha": git_sha(),
        "evaluation_config_sha256": file_sha256(EVALUATION_CONFIG), "public": public,
        "countries": COUNTRIES,
        "labels": {"identified_causal": "Identified causal estimate",
                   "reduced_form": "Reduced-form association",
                   "model_forecast": "Model forecast",
                   "hypothetical_scenario": "Hypothetical scenario"}}, set())
    return folder.relative_to(ROOT)
