"""M5 evaluation: validation-phase model selection and the single final test run
(docs/EVALUATION_PROTOCOL.md §3-§7).

    uv run wsl evaluate --validation   # train <= 2014, score 2015-2018, choose candidates
    uv run wsl evaluate --final        # once: refuses unless the tree is clean, the config
                                       # matches the latest protocol tag, and no earlier run
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from wsl.data.build import file_sha256, git_sha
from wsl.estimation.irf import z_value
from wsl.ml import models
from wsl.ml.dataset import ASSET_NAMES, CFG, HORIZONS, EventData, build, split_of
from wsl.ml.metrics import bootstrap_difference, cell_metrics, pooled_scaled_mae, verdict
from wsl.paths import EVALUATION_CONFIG, ROOT

RESULTS = ROOT / "results" / "ml"
FINAL_DIR = ROOT / "results" / "final_eval"
BOOT = CFG["decision_rule"]["bootstrap"]
LEVELS = (0.68, 0.90)
GAUSSIAN = ("lp", "state")  # econometric residual intervals; the rest use split conformal


class ProtocolGuardError(RuntimeError):
    pass


@dataclass
class Cell:
    name: str
    asset: str
    h: int


def cells() -> list[Cell]:
    return [Cell(f"{a}|h{h}", a, h) for a in ASSET_NAMES for h in HORIZONS]


def _frame(d: EventData, cell: Cell) -> tuple[pd.DataFrame, pd.Series]:
    y = d.targets[cell.name]
    X = models.base_frame(d.events, d.asset_vol[cell.asset])
    keep = y.notna()
    return X[keep], y[keep]


def _primary(d: EventData) -> pd.Series:
    return ~d.events["boc_same_day_fomc"]


def _conformal_half(abs_resid: np.ndarray, level: float) -> float:
    n = len(abs_resid)
    k = min(n, int(np.ceil((n + 1) * level)))
    return float(np.sort(abs_resid)[k - 1])


def _halves(name: str, y_train: np.ndarray, resid_train: np.ndarray,
            resid_val: np.ndarray) -> dict[str, float]:
    if name == "zero":
        return {f"{lv}": float(np.quantile(np.abs(y_train), lv)) for lv in LEVELS}
    if name in GAUSSIAN:
        sd = float(np.std(resid_train, ddof=1))
        return {f"{lv}": z_value(lv) * sd for lv in LEVELS}
    return {f"{lv}": _conformal_half(np.abs(resid_val), lv) for lv in LEVELS}


# ----------------------------------------------------------------------------- validation
def run_validation() -> dict[str, Any]:
    d = build()
    split = split_of(d.events["year"])
    prim = _primary(d)
    candidates: list[tuple[str, dict[str, Any] | None]] = [
        ("zero", None), ("lp", None), ("state", None), ("ridge", None)]
    candidates += [("gbm", p) for p in models.GBM.GRID] + [("mlp", p) for p in models.MLP.GRID]
    per: dict[str, dict[str, Any]] = {}
    scale: dict[str, float] = {}
    p25: dict[str, float] = {}
    abs_err: dict[str, dict[str, pd.Series]] = {}
    for cell in cells():
        X, y = _frame(d, cell)
        tr = (split[X.index] == "train") & prim[X.index]
        va = (split[X.index] == "validation") & prim[X.index]
        ytr, yva = y[tr].to_numpy(), y[va].to_numpy()
        scale[cell.name] = float(np.mean(np.abs(ytr)))
        p25[cell.name] = float(np.quantile(np.abs(ytr), 0.25))
        for name, params in candidates:
            key = name if params is None else f"{name}:{json.dumps(params, sort_keys=True)}"
            m = models.make(name, params).fit(X[tr], ytr)
            pv, pt = m.predict(X[va]), m.predict(X[tr])
            halves = _halves(name, ytr, ytr - pt, yva - pv)
            per.setdefault(key, {})[cell.name] = {
                **cell_metrics(yva, pv, np.full(len(yva), halves["0.68"]),
                               np.full(len(yva), halves["0.9"]), scale[cell.name],
                               p25[cell.name]),
                "half68": halves["0.68"], "half90": halves["0.9"]}
            abs_err.setdefault(key, {})[cell.name] = pd.Series(np.abs(yva - pv),
                                                               index=X.index[va])
    s = pd.Series(scale)
    pooled = {k: pooled_scaled_mae(pd.DataFrame(v), s) for k, v in abs_err.items()}
    best_gbm = min((k for k in pooled if k.startswith("gbm:")), key=pooled.__getitem__)
    best_mlp = min((k for k in pooled if k.startswith("mlp:")), key=pooled.__getitem__)
    econ = min(["lp", "state", "ridge", best_gbm], key=pooled.__getitem__)
    out = {
        "phase": "validation", "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(), "evaluation_config_sha256": file_sha256(EVALUATION_CONFIG),
        "pooled_scaled_mae": pooled, "per_cell": per, "scale": scale, "p25_abs": p25,
        "chosen": {"gbm": best_gbm, "mlp": best_mlp},
        "candidates": {"econometric": econ, "ml": best_mlp},
        "counts": {"train": int(((split == "train") & prim).sum()),
                   "validation": int(((split == "validation") & prim).sum())},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "validation.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


# ---------------------------------------------------------------------------- final guard
def _latest_protocol_tag() -> tuple[str, str]:
    tags = subprocess.run(["git", "tag", "-l", "protocol-v*"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.split()
    if not tags:
        raise ProtocolGuardError("no protocol tag found")
    tag = max(tags, key=lambda t: int(t.split("-v")[1]))
    msg = subprocess.run(["git", "tag", "-l", "--format=%(contents)", tag], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    m = re.search(r"\b([0-9a-f]{64})\b", msg)
    if not m:
        raise ProtocolGuardError(f"tag {tag} records no config hash")
    return tag, m.group(1)


def check_final_allowed(rerun_reason: str | None) -> str:
    tag, sha = _latest_protocol_tag()
    if file_sha256(EVALUATION_CONFIG) != sha:
        raise ProtocolGuardError(f"config/evaluation.yaml does not match {tag}")
    if git_sha().endswith("-dirty") or git_sha() == "unknown":
        raise ProtocolGuardError("uncommitted code or config changes; commit first")
    if not (RESULTS / "validation.json").exists():
        raise ProtocolGuardError("run the validation phase first")
    previous = sorted(FINAL_DIR.glob("*.json")) if FINAL_DIR.exists() else []
    if previous and not rerun_reason:
        raise ProtocolGuardError(
            f"a final run already exists ({previous[-1].name}); the test set is used once. "
            "A re-run needs --rerun-reason and is logged in the protocol change log.")
    return tag


# ------------------------------------------------------------------------------- final run
def _fit_predict(name: str, params: dict[str, Any] | None, X: pd.DataFrame, y: pd.Series,
                 train: pd.Series, test: pd.Series) -> np.ndarray:
    m = models.make(name, params).fit(X[train], y[train].to_numpy())
    return np.asarray(m.predict(X[test]))


def _parse(key: str) -> tuple[str, dict[str, Any] | None]:
    if ":" not in key:
        return key, None
    name, params = key.split(":", 1)
    return name, json.loads(params)


def run_final(rerun_reason: str | None = None) -> dict[str, Any]:
    tag = check_final_allowed(rerun_reason)
    val = json.loads((RESULTS / "validation.json").read_text())
    keys = ["zero", "lp", "state", "ridge", val["chosen"]["gbm"], val["chosen"]["mlp"]]
    econ_key, ml_key = val["candidates"]["econometric"], val["candidates"]["ml"]
    d = build()
    split = split_of(d.events["year"])
    prim = _primary(d)
    test_rows = split == "test"
    years = sorted(d.events.loc[test_rows, "year"].unique())
    scale = pd.Series(val["scale"])

    preds: dict[str, dict[str, pd.Series]] = {k: {} for k in keys}
    static: dict[str, dict[str, pd.Series]] = {k: {} for k in keys}
    actual: dict[str, pd.Series] = {}
    for cell in cells():
        X, y = _frame(d, cell)
        sp, pr = split[X.index], prim[X.index]
        actual[cell.name] = y[sp == "test"]
        for key in keys:
            name, params = _parse(key)
            out = pd.Series(np.nan, index=y[sp == "test"].index)
            for year in years:  # primary: expanding annual refit
                tr = pr & (d.events.loc[X.index, "year"] < year)
                te = (sp == "test") & (d.events.loc[X.index, "year"] == year)
                if te.any():
                    out[te[te].index] = _fit_predict(name, params, X, y, tr, te)
            preds[key][cell.name] = out
            tr_s = pr & sp.isin(["train", "validation"])
            static[key][cell.name] = pd.Series(
                _fit_predict(name, params, X, y, tr_s, sp == "test"), index=out.index)

    def table(p: dict[str, dict[str, pd.Series]], rows: pd.Index) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for key in keys:
            per_cell, errs = {}, {}
            for cell in cells():
                a = actual[cell.name]
                idx = a.index.intersection(rows)
                yy, pp = a[idx].to_numpy(), p[key][cell.name][idx].to_numpy()
                vc = val["per_cell"][key][cell.name]
                per_cell[cell.name] = cell_metrics(
                    yy, pp, np.full(len(yy), vc["half68"]), np.full(len(yy), vc["half90"]),
                    val["scale"][cell.name], val["p25_abs"][cell.name])
                errs[cell.name] = pd.Series(np.abs(yy - pp), index=idx)
            err_df = pd.DataFrame(errs)
            res[key] = {"per_cell": per_cell, "pooled_scaled_mae": pooled_scaled_mae(err_df, scale),
                        "_err": err_df}
        return res

    test_index = d.events.index[test_rows]
    primary_rows = test_index[prim[test_index].to_numpy()]
    main = table(preds, primary_rows)
    diff, ci, _ = bootstrap_difference(main[ml_key]["_err"], main[econ_key]["_err"], scale,
                                       int(BOOT["resamples"]), int(BOOT["seed"]),
                                       float(BOOT["ci"]))
    static_tab = table(static, primary_rows)
    s_diff, s_ci, _ = bootstrap_difference(static_tab[ml_key]["_err"],
                                           static_tab[econ_key]["_err"], scale,
                                           int(BOOT["resamples"]), int(BOOT["seed"]),
                                           float(BOOT["ci"]))
    ev = d.events.loc[test_index]
    march2020 = ev.index[(ev["ts"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
                          == "2020-03") & (ev["family"] == "fomc")]
    subsamples = {
        "excluding March 2020 FOMC events": primary_rows.difference(march2020),
        **{f"family: {f}": primary_rows[(ev.loc[primary_rows, "family"] == f).to_numpy()]
           for f in ("fomc", "boc", "opec")},
        "BoC events on FOMC days (excluded by v2)": test_index[~prim[test_index].to_numpy()],
    }
    sub_out = {}
    for name, rows in subsamples.items():
        t = table(preds, rows)
        sub_out[name] = {"events": len(rows),
                         "pooled_scaled_mae": {k: t[k]["pooled_scaled_mae"] for k in keys}}

    attributions = _permutation_importance(d, static, keys, val, primary_rows)
    replay = _replay(d, actual, preds, keys, val, test_index)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "phase": "final", "generated_at": stamp, "protocol_tag": tag,
        "evaluation_config_sha256": file_sha256(EVALUATION_CONFIG), "git_sha": git_sha(),
        "rerun_reason": rerun_reason, "candidates": {"econometric": econ_key, "ml": ml_key},
        "test_events_primary": len(primary_rows), "test_years": [int(y) for y in years],
        "pooled_scaled_mae": {k: main[k]["pooled_scaled_mae"] for k in keys},
        "per_cell": {k: main[k]["per_cell"] for k in keys},
        "difference_ml_minus_econ": {"point": diff, "ci90": ci},
        "verdict": verdict(ci),
        "static_refit": {"pooled_scaled_mae": {k: static_tab[k]["pooled_scaled_mae"]
                                               for k in keys},
                         "difference_ml_minus_econ": {"point": s_diff, "ci90": s_ci},
                         "verdict": verdict(s_ci)},
        "subsamples": sub_out, "attributions": attributions, "replay": replay,
    }
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / f"{stamp}.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    return result


def _feature_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    from wsl.ml.dataset import SHOCK_COLS, STATE_COLS
    groups = {"surprise components": [c for cols in SHOCK_COLS.values() for c in cols],
              "event family": list(models.FAMILY_DUMMIES), "asset volatility": ["asset_vol"]}
    groups.update({c: [c] for c in STATE_COLS})
    return {k: [c for c in v if c in X] for k, v in groups.items()}


def _permutation_importance(d: EventData, static: dict[str, dict[str, pd.Series]],
                            keys: list[str], val: dict[str, Any],
                            rows: pd.Index) -> dict[str, dict[str, float]]:
    """Increase in pooled scaled MAE on the test events when a feature group is shuffled,
    using the static models (fit on data up to 2018). Computed for gbm and mlp."""
    split = split_of(d.events["year"])
    prim = _primary(d)
    rng = np.random.default_rng(int(BOOT["seed"]))
    out: dict[str, dict[str, float]] = {}
    for key in (val["chosen"]["gbm"], val["chosen"]["mlp"], "ridge"):
        name, params = _parse(key)
        deltas: dict[str, list[float]] = {}
        for cell in cells():
            X, y = _frame(d, cell)
            tr = prim[X.index] & split[X.index].isin(["train", "validation"])
            te = X.index.intersection(rows)
            if len(te) < 5:
                continue
            m = models.make(name, params).fit(X[tr], y[tr].to_numpy())
            base = np.mean(np.abs(m.predict(X.loc[te]) - y[te].to_numpy()))
            for g, cols in _feature_groups(X).items():
                Xp = X.loc[te].copy()
                perm = rng.permutation(len(te))
                Xp[cols] = Xp[cols].to_numpy()[perm]
                err = np.mean(np.abs(m.predict(Xp) - y[te].to_numpy()))
                deltas.setdefault(g, []).append((err - base) / val["scale"][cell.name])
        out[key] = {g: float(np.mean(v)) for g, v in deltas.items()}
    return out


def _replay(d: EventData, actual: dict[str, pd.Series], preds: dict[str, dict[str, pd.Series]],
            keys: list[str], val: dict[str, Any], test_index: pd.Index) -> list[dict[str, Any]]:
    ev = d.events.loc[test_index]
    rows = []
    for eid_raw, e in ev.iterrows():
        eid = str(eid_raw)
        cells_out = {}
        for cell in cells():
            a = actual[cell.name]
            if eid not in a.index:
                continue
            cells_out[cell.name] = {
                "actual": float(a[eid]),
                "pred": {k: float(preds[k][cell.name][eid]) for k in keys},
                "half90": {k: float(val["per_cell"][k][cell.name]["half90"]) for k in keys}}
        rows.append({"event": str(eid), "family": e["family"],
                     "ts": e["ts"].isoformat(), "same_day_fomc": bool(e["boc_same_day_fomc"]),
                     "cells": cells_out})
    return rows
