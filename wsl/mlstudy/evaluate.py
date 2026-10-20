"""Validation (model selection) and the single test run for ml-study-v1.

    uv run wsl mlstudy --validation
    uv run wsl mlstudy --final
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
import yaml
from scipy.stats import norm

from wsl.data.build import file_sha256, git_sha
from wsl.estimation.irf import z_value
from wsl.ml.dataset import ASSET_NAMES, HORIZONS, EventData
from wsl.ml.dataset import CFG as EVAL_CFG
from wsl.ml.metrics import (
    bootstrap_difference,
    cell_metrics,
    interval_score,
    pooled_scaled_mae,
    verdict,
)
from wsl.ml.models import Linear, base_frame
from wsl.mlstudy.explain import embedding_agreement, group_index, grouped_shap
from wsl.mlstudy.metrics import fissler_ziegel, pinball, tail_verdict
from wsl.mlstudy.models import QUANTILES, QuantileGBM, TorchModel, TrainCfg
from wsl.mlstudy.panels import CELL_NAMES, SHOCK_NAMES, Panel, event_table, from_events
from wsl.mlstudy.prepare import INPUTS_PATH
from wsl.paths import ROOT

STUDY_CONFIG = ROOT / "config" / "ml_study.yaml"
RESULTS = ROOT / "results" / "mlstudy"
FINAL_PATH = RESULTS / "final.json"
VAL_PATH = RESULTS / "validation.json"
TAUS_CVAR = (0.01, 0.02, 0.03, 0.04, 0.05)


class ProtocolGuardError(RuntimeError):
    pass


@dataclass
class CellFrame:
    name: str
    asset: str
    h: int
    X: pd.DataFrame
    y: pd.Series
    year: pd.Series
    family: pd.Series


def load_cfg() -> dict[str, Any]:
    doc: dict[str, Any] = yaml.safe_load(STUDY_CONFIG.read_text())
    return doc


def _bounds() -> tuple[int, int, int, int]:
    sp = EVAL_CFG["splits"]
    return (int(sp["train"]["end"][:4]), int(sp["validation"]["start"][:4]),
            int(sp["validation"]["end"][:4]), int(sp["test"]["start"][:4]))


def _cells(d: EventData) -> list[CellFrame]:
    out = []
    keep_event = ~d.events["boc_same_day_fomc"]
    for asset in ASSET_NAMES:
        for h in HORIZONS:
            name = f"{asset}|h{h}"
            y = d.targets[name]
            X = base_frame(d.events, d.asset_vol[asset])
            ok = y.notna() & keep_event.reindex(y.index).fillna(False)
            out.append(CellFrame(
                name, asset, h, X.loc[ok], y.loc[ok],
                d.events.loc[ok, "year"].astype(int),
                d.events.loc[ok, "family"].astype(str),
            ))
    return out


def _key(name: str, params: dict[str, Any] | None) -> str:
    if not params:
        return name
    return f"{name}:{json.dumps(params, sort_keys=True)}"


def _parse(key: str) -> tuple[str, dict[str, Any]]:
    if ":" not in key:
        return key, {}
    name, raw = key.split(":", 1)
    parsed: dict[str, Any] = json.loads(raw)
    return name, parsed


def _scale_mae(frames: list[CellFrame], train_end: int) -> pd.Series:
    out = {}
    for f in frames:
        y = f.y[f.year <= train_end]
        out[f.name] = float(np.mean(np.abs(y.to_numpy()))) if len(y) else np.nan
    return pd.Series(out)


def _scale_pinball(frames: list[CellFrame], train_end: int, tau: float) -> pd.Series:
    out = {}
    for f in frames:
        y = f.y[f.year <= train_end].to_numpy()
        if len(y) == 0:
            out[f.name] = np.nan
            continue
        q = float(np.quantile(y, tau))
        out[f.name] = float(np.mean(pinball(y, np.full(len(y), q), tau)))
    return pd.Series(out)


def _zero_on(frames: list[CellFrame], test_pred) -> dict[str, pd.Series]:  # type: ignore[no-untyped-def]
    """A zero forecast on every cell's own events. Cells do not share an index."""
    out = {}
    for f in frames:
        te = test_pred(f.year).fillna(False).to_numpy()
        out[f.name] = pd.Series(0.0, index=f.y.index[te])
    return out


def _as_frame(preds: dict[str, pd.Series], index: pd.Index) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    for name in CELL_NAMES:
        s = preds.get(name)
        df[name] = s.reindex(index) if s is not None else np.nan
    return df


def _abs_err(pred: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    return (pred - actual).abs()


def _events_where(frames: list[CellFrame], pred) -> pd.Index:  # type: ignore[no-untyped-def]
    """Union of event ids whose year matches, across every asset × horizon cell."""
    ids: list[pd.Index] = []
    for f in frames:
        m = pred(f.year).fillna(False).to_numpy()
        ids.append(f.y.index[m])
    return ids[0].append(ids[1:]).unique()


def _actual(frames: list[CellFrame], index: pd.Index) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    for f in frames:
        df[f.name] = f.y.reindex(index)
    return df


def _family(frames: list[CellFrame], index: pd.Index) -> pd.Series:
    parts = []
    for f in frames:
        parts.append(f.family)
    fam = pd.concat(parts)
    fam = fam[~fam.index.duplicated(keep="first")]
    return pd.Series(fam.reindex(index))


def _torch(kind: str, params: dict[str, Any], cfg: TrainCfg) -> TorchModel:
    return TorchModel(kind, cfg, **params)


def _gbm(params: dict[str, Any], cfg: dict[str, Any], quantiles: tuple[float, ...]) -> QuantileGBM:
    fixed = cfg["qgbm_fixed"]
    return QuantileGBM(
        TrainCfg(), max_depth=int(params["max_depth"]),
        learning_rate=float(fixed["learning_rate"]), max_iter=int(fixed["max_iter"]),
        min_samples_leaf=int(fixed["min_samples_leaf"]), quantiles=quantiles,
    )


def _panel_mask_years(panel: Panel, years_ok: np.ndarray) -> np.ndarray:
    return np.isin(panel.year, years_ok)


def _unpack(panel: Panel, mask: np.ndarray, values: np.ndarray) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    cells = panel.cell[mask]
    eids = panel.event_id[mask]
    for c in np.unique(cells):
        m = cells == c
        out[str(c)] = pd.Series(np.asarray(values[m], dtype=float), index=eids[m])
    return out


def _fit_torch_predict(kind: str, params: dict[str, Any], cfg: TrainCfg, panel: Panel,
                       train_years: np.ndarray, test_years: np.ndarray,
                       ) -> dict[str, pd.Series]:
    pred, _halves = _fit_torch_both(kind, params, cfg, panel, train_years, test_years)
    return pred


def _fit_torch_both(kind: str, params: dict[str, Any], cfg: TrainCfg, panel: Panel,
                    train_years: np.ndarray, test_years: np.ndarray,
                    ) -> tuple[dict[str, pd.Series], dict[str, dict[str, float]]]:
    """Predictions, and split-conformal half-widths from this fit's early-stopping slice."""
    train_mask = _panel_mask_years(panel, train_years)
    model = _torch(kind, params, cfg).fit(panel, train_mask)
    te = _panel_mask_years(panel, test_years)
    pred = _unpack(panel, te, model.predict(panel, te))
    ev = np.unique(panel.event_i[train_mask])
    cut = max(1, int(len(ev) * (1.0 - cfg.early_stop_frac)))
    if len(ev) > 1:
        cut = min(cut, len(ev) - 1)
    hold = np.isin(panel.event_i, ev[cut:]) & train_mask
    hold_pred = model.predict(panel, hold)
    y = panel.y[hold]
    cells = panel.cell[hold]
    halves: dict[str, dict[str, float]] = {}
    for c in np.unique(cells):
        r = np.abs(y[cells == c] - hold_pred[cells == c])
        n = len(r)
        one: dict[str, float] = {}
        for level, name in ((0.68, "68"), (0.90, "90")):
            k = min(n, int(np.ceil((n + 1) * level)))
            one[name] = float(np.sort(r)[k - 1]) if n else float("nan")
        halves[str(c)] = one
    return pred, halves


def _mean_table(pred: dict[str, pd.Series], actual: pd.DataFrame, scale: pd.Series,
                halves: dict[str, dict[str, float]] | None, p25: pd.Series,
                ) -> dict[str, Any]:
    err = _abs_err(_as_frame(pred, actual.index), actual)
    per = {}
    for name in CELL_NAMES:
        yy = actual[name].to_numpy()
        pp = _as_frame(pred, actual.index)[name].to_numpy()
        ok = ~np.isnan(yy) & ~np.isnan(pp)
        if ok.sum() == 0:
            continue
        h68 = halves[name]["68"] if halves and name in halves else np.nan
        h90 = halves[name]["90"] if halves and name in halves else np.nan
        per[name] = cell_metrics(
            yy[ok], pp[ok], np.full(int(ok.sum()), h68), np.full(int(ok.sum()), h90),
            float(scale[name]), float(p25[name]),
        )
    return {"pooled_scaled_mae": pooled_scaled_mae(err, scale), "per_cell": per, "_err": err}


def _scaled(block: dict[str, Any]) -> dict[str, float]:
    return {k: float(v["scaled_mae"]) for k, v in block["per_cell"].items()}


def _pinball_frame(pred: dict[str, pd.Series], actual: pd.DataFrame, tau: float) -> pd.DataFrame:
    wide = _as_frame(pred, actual.index)
    out = pd.DataFrame(index=actual.index)
    for name in CELL_NAMES:
        y = actual[name].to_numpy()
        q = wide[name].to_numpy()
        score = np.full(len(y), np.nan)
        ok = ~np.isnan(y) & ~np.isnan(q)
        score[ok] = pinball(y[ok], q[ok], tau)
        out[name] = score
    return out


def _p25(frames: list[CellFrame], train_end: int) -> pd.Series:
    return pd.Series({
        f.name: float(np.quantile(np.abs(f.y[f.year <= train_end].to_numpy()), 0.25))
        for f in frames
    })


def _log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------------------- validation
def run_validation() -> dict[str, Any]:
    cfg = load_cfg()
    train_cfg = TrainCfg.from_yaml(cfg)
    train_end, val_start, val_end, _test_start = _bounds()
    panel, data = from_events()
    frames = _cells(data)
    events = event_table(panel)
    tr_years = events.loc[events["year"] <= train_end, "year"].unique()
    va_years = events.loc[events["year"].between(val_start, val_end), "year"].unique()
    scale = _scale_mae(frames, train_end)
    pscale = _scale_pinball(frames, train_end, float(cfg["questions"]["tail"]["tau"]))
    p25 = _p25(frames, train_end)

    def _val_year(y: pd.Series) -> pd.Series:
        return (y >= val_start) & (y <= val_end)

    actual_val = _actual(frames, _events_where(frames, _val_year))
    pooled: dict[str, float] = {}
    pin_pooled: dict[str, float] = {}

    def mae_of(pred: dict[str, pd.Series]) -> float:
        err = _abs_err(_as_frame(pred, actual_val.index), actual_val)
        return pooled_scaled_mae(err, scale)

    def pin_of(pred: dict[str, pd.Series]) -> float:
        return pooled_scaled_mae(_pinball_frame(pred, actual_val, tau), pscale)

    _log("validation: zero and linear projection")
    zero = _zero_on(frames, _val_year)
    lp = _lp_on(frames, lambda y: y <= train_end, _val_year)
    pooled["zero"] = mae_of(zero)
    pooled["lp"] = mae_of(lp)

    tau = float(cfg["questions"]["tail"]["tau"])
    hist = _hist_on(frames, lambda y: y <= train_end, _val_year, tau)
    gauss = _gauss_on(frames, lambda y: y <= train_end, _val_year, tau)
    pin_pooled["hist"] = pin_of(hist)
    pin_pooled["gauss_lp"] = pin_of(gauss)
    _log(f"  lp scaled MAE {pooled['lp']:.4f}  gauss pinball {pin_pooled['gauss_lp']:.4f}")

    for params in cfg["grids"]["mt_mlp"]:
        key = _key("mt_mlp", params)
        _log(f"validation: {key}")
        pred = _fit_torch_predict("mt_mlp", params, train_cfg, panel, tr_years, va_years)
        pooled[key] = mae_of(pred)
        _log(f"  scaled MAE {pooled[key]:.4f}")
    for params in cfg["grids"]["tcn"]:
        key = _key("tcn", params)
        _log(f"validation: {key}")
        pred = _fit_torch_predict("tcn", params, train_cfg, panel, tr_years, va_years)
        pooled[key] = mae_of(pred)
        _log(f"  scaled MAE {pooled[key]:.4f}")
    for params in cfg["grids"]["qgbm"]:
        key = _key("qgbm", params)
        _log(f"validation: {key}")
        pred = _gbm_on(panel, params, cfg, tr_years, va_years, (tau,))
        pin_pooled[key] = pin_of(pred)
        _log(f"  scaled pinball {pin_pooled[key]:.4f}")
    for params in cfg["grids"]["qnet"]:
        key = _key("qnet", params)
        _log(f"validation: {key}")
        pred = _qnet_on(panel, params, train_cfg, tr_years, va_years, tau)
        pin_pooled[key] = pin_of(pred)
        _log(f"  scaled pinball {pin_pooled[key]:.4f}")

    mean_cands = [k for k in pooled if k.startswith(("mt_mlp:", "tcn:"))]
    tail_cands = [k for k in pin_pooled if k.startswith(("qgbm:", "qnet:"))]
    chosen_mean = min(mean_cands, key=pooled.__getitem__)
    chosen_tail = min(tail_cands, key=pin_pooled.__getitem__)
    out = {
        "phase": "validation",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config_sha256": file_sha256(STUDY_CONFIG),
        "evaluation_config_sha256": file_sha256(ROOT / "config" / "evaluation.yaml"),
        "counts": {
            "train_events": int((events["year"] <= train_end).sum()),
            "validation_events": int(events["year"].between(val_start, val_end).sum()),
        },
        "pooled_scaled_mae": pooled,
        "pooled_scaled_pinball": pin_pooled,
        "scale_mae": scale.to_dict(),
        "scale_pinball": pscale.to_dict(),
        "p25_abs": p25.to_dict(),
        "chosen": {"mean": chosen_mean, "tail": chosen_tail},
        "inputs": _inputs(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    VAL_PATH.write_text(json.dumps(_clean(out), indent=1) + "\n")
    _log(f"chosen mean {chosen_mean}  tail {chosen_tail}")
    return out


def _lp_on(frames: list[CellFrame], train_pred, test_pred) -> dict[str, pd.Series]:  # type: ignore[no-untyped-def]
    out: dict[str, pd.Series] = {}
    for f in frames:
        tr = train_pred(f.year).fillna(False).to_numpy()
        te = test_pred(f.year).fillna(False).to_numpy()
        if tr.sum() < 20 or te.sum() == 0:
            continue
        model = Linear(with_state=False).fit(f.X.loc[tr], f.y.to_numpy()[tr])
        out[f.name] = pd.Series(model.predict(f.X.loc[te]), index=f.y.index[te])
    return out


def _hist_on(frames: list[CellFrame], train_pred, test_pred, tau: float  # type: ignore[no-untyped-def]
             ) -> dict[str, pd.Series]:
    out = {}
    for f in frames:
        tr = train_pred(f.year).fillna(False).to_numpy()
        te = test_pred(f.year).fillna(False).to_numpy()
        if tr.sum() < 20 or te.sum() == 0:
            continue
        q = float(np.quantile(f.y.to_numpy()[tr], tau))
        out[f.name] = pd.Series(q, index=f.y.index[te])
    return out


def _gauss_on(frames: list[CellFrame], train_pred, test_pred, tau: float  # type: ignore[no-untyped-def]
              ) -> dict[str, pd.Series]:
    """Linear projection plus a Gaussian residual quantile. Also stashes residual sd."""
    out = {}
    z = float(norm.ppf(tau))
    for f in frames:
        tr = train_pred(f.year).fillna(False).to_numpy()
        te = test_pred(f.year).fillna(False).to_numpy()
        if tr.sum() < 20 or te.sum() == 0:
            continue
        model = Linear(with_state=False).fit(f.X.loc[tr], f.y.to_numpy()[tr])
        mu = model.predict(f.X.loc[te])
        sd = float(np.std(f.y.to_numpy()[tr] - model.predict(f.X.loc[tr]), ddof=1))
        out[f.name] = pd.Series(mu + z * sd, index=f.y.index[te])
    return out


def _gauss_band(frames: list[CellFrame], train_pred, test_pred  # type: ignore[no-untyped-def]
                ) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, pd.Series]]:
    """Median (the LP mean), 5% and 95% Gaussian quantiles."""
    q05, q50, q95 = {}, {}, {}
    for f in frames:
        tr = train_pred(f.year).fillna(False).to_numpy()
        te = test_pred(f.year).fillna(False).to_numpy()
        if tr.sum() < 20 or te.sum() == 0:
            continue
        model = Linear(with_state=False).fit(f.X.loc[tr], f.y.to_numpy()[tr])
        mu_te = model.predict(f.X.loc[te])
        sd = float(np.std(f.y.to_numpy()[tr] - model.predict(f.X.loc[tr]), ddof=1))
        idx = f.y.index[te]
        q50[f.name] = pd.Series(mu_te, index=idx)
        q05[f.name] = pd.Series(mu_te + norm.ppf(0.05) * sd, index=idx)
        q95[f.name] = pd.Series(mu_te + norm.ppf(0.95) * sd, index=idx)
    return q05, q50, q95


def _gbm_on(panel: Panel, params: dict[str, Any], cfg: dict[str, Any],
            train_years: np.ndarray, test_years: np.ndarray, quantiles: tuple[float, ...],
            tau: float = 0.05) -> dict[str, pd.Series]:
    model = _gbm(params, cfg, quantiles).fit(panel, _panel_mask_years(panel, train_years))
    te = _panel_mask_years(panel, test_years)
    return _unpack(panel, te, model.predict(panel, te, tau))


def _qnet_on(panel: Panel, params: dict[str, Any], cfg: TrainCfg,
             train_years: np.ndarray, test_years: np.ndarray, tau: float) -> dict[str, pd.Series]:
    model = _torch("qnet", params, cfg).fit(panel, _panel_mask_years(panel, train_years))
    te = _panel_mask_years(panel, test_years)
    q = model.predict(panel, te)
    col = QUANTILES.index(tau)
    return _unpack(panel, te, q[:, col])


def _inputs() -> dict[str, Any]:
    if INPUTS_PATH.exists():
        doc: dict[str, Any] = json.loads(INPUTS_PATH.read_text())
        return {"matched_pin": doc.get("matched_pin"), "revised": doc.get("revised"),
                "inputs": doc.get("inputs", [])}
    return {}


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, pd.Series):
        return _clean(obj.to_dict())
    return obj


# ------------------------------------------------------------------------------------- final
def _latest_tag() -> tuple[str, str]:
    tags = subprocess.run(["git", "tag", "-l", "ml-study-v*"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.split()
    if not tags:
        raise ProtocolGuardError("no ml-study-v* tag; tag the protocol before the test run")
    tag = max(tags, key=lambda t: int(t.rsplit("v", 1)[1]))
    msg = subprocess.run(["git", "tag", "-l", "--format=%(contents)", tag], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    found = re.search(r"\b([0-9a-f]{64})\b", msg)
    if not found:
        raise ProtocolGuardError(f"tag {tag} records no config hash")
    return tag, found.group(1)


def check_final_allowed(rerun_reason: str | None) -> str:
    tag, sha = _latest_tag()
    if file_sha256(STUDY_CONFIG) != sha:
        raise ProtocolGuardError(f"config/ml_study.yaml does not match {tag}")
    sha_now = git_sha()
    if sha_now.endswith("-dirty") or sha_now == "unknown":
        raise ProtocolGuardError("uncommitted code or config changes; commit first")
    if not VAL_PATH.exists():
        raise ProtocolGuardError("run the validation phase first")
    val = json.loads(VAL_PATH.read_text())
    if val.get("config_sha256") != sha:
        raise ProtocolGuardError("validation.json was produced under a different config")
    if FINAL_PATH.exists() and not rerun_reason:
        raise ProtocolGuardError(
            "results/mlstudy/final.json already exists. The test set is scored once. "
            "A re-run needs --rerun-reason and is logged in the protocol change log.")
    return tag


def run_final(rerun_reason: str | None = None) -> dict[str, Any]:
    tag = check_final_allowed(rerun_reason)
    cfg = load_cfg()
    train_cfg = TrainCfg.from_yaml(cfg)
    val = json.loads(VAL_PATH.read_text())
    _train_end, _vs, val_end, test_start = _bounds()
    panel, data = from_events()
    frames = _cells(data)
    events = event_table(panel)
    test_years = np.sort(events.loc[events["year"] >= test_start, "year"].unique())
    scale = pd.Series(val["scale_mae"])
    pscale = pd.Series(val["scale_pinball"])
    p25 = pd.Series(val["p25_abs"])
    tau = float(cfg["questions"]["tail"]["tau"])
    mean_key, tail_key = val["chosen"]["mean"], val["chosen"]["tail"]
    mean_name, mean_params = _parse(mean_key)
    tail_name, tail_params = _parse(tail_key)

    def cat(parts: list[dict[str, pd.Series]]) -> dict[str, pd.Series]:
        out: dict[str, list[pd.Series]] = {}
        for part in parts:
            for k, s in part.items():
                out.setdefault(k, []).append(s)
        return {k: pd.concat(v) for k, v in out.items()}

    _log(f"final: walk-forward {list(map(int, test_years))}  mean={mean_key} tail={tail_key}")
    zero_p, lp_p, mean_p = [], [], []
    mean_halves: list[dict[str, dict[str, float]]] = []
    lp_halves_y: list[dict[str, dict[str, float]]] = []
    hist_p, gauss_p, tail_p = [], [], []
    gauss50, gauss95 = [], []
    tail_q: dict[float, list[dict[str, pd.Series]]] = {t: [] for t in QUANTILES}
    for year in test_years:
        _log(f"  year {int(year)}")
        tr = np.sort(events.loc[events["year"] < year, "year"].unique())
        te = np.array([year])
        zero_p.append(_zero_on(frames, lambda y, cut=year: y == cut))
        lp_p.append(_lp_on(frames, lambda y, cut=year: y < cut, lambda y, cut=year: y == cut))
        pred_y, halves_y = _fit_torch_both(
            mean_name, mean_params, train_cfg, panel, tr, te)
        mean_p.append(pred_y)
        mean_halves.append(halves_y)
        lp_halves_y.append(_lp_gaussian_halves(frames, lambda y, cut=year: y < cut))
        hist_p.append(_hist_on(
            frames, lambda y, cut=year: y < cut, lambda y, cut=year: y == cut, tau))
        g05, g50, g95 = _gauss_band(
            frames, lambda y, cut=year: y < cut, lambda y, cut=year: y == cut)
        gauss50.append(g50)
        gauss95.append(g95)
        gauss_p.append(g05)
        if tail_name == "qgbm":
            model = _gbm(tail_params, cfg, QUANTILES).fit(panel, _panel_mask_years(panel, tr))
            te_m = _panel_mask_years(panel, te)
            for t in QUANTILES:
                tail_q[t].append(_unpack(panel, te_m, model.predict(panel, te_m, t)))
            tail_p.append(tail_q[tau][-1])
        else:
            train_m = _panel_mask_years(panel, tr)
            model_q = _torch("qnet", tail_params, train_cfg).fit(panel, train_m)
            te_m = _panel_mask_years(panel, te)
            q = model_q.predict(panel, te_m)
            for i, t in enumerate(QUANTILES):
                tail_q[t].append(_unpack(panel, te_m, q[:, i]))
            tail_p.append(tail_q[tau][-1])

    zero_c, lp_c, mean_c = cat(zero_p), cat(lp_p), cat(mean_p)
    hist_c, gauss_c, tail_c = cat(hist_p), cat(gauss_p), cat(tail_p)
    g95_c, g50_c = cat(gauss95), cat(gauss50)
    tail_all = {t: cat(parts) for t, parts in tail_q.items()}

    test_index = _events_where(frames, lambda y: y >= test_start)
    actual = _actual(frames, test_index)
    fam = _family(frames, test_index)

    # Each test year keeps the half-widths of the model that scored it. Coverage is
    # secondary and does not enter either verdict.
    _log("final: static refit")
    through = np.sort(events.loc[events["year"] <= val_end, "year"].unique())
    ml_widths = _widths(mean_p, mean_halves)
    lp_widths = _widths(lp_p, lp_halves_y)

    def mean_block(pred: dict[str, pd.Series],
                   half: dict[str, dict[str, float]] | None) -> dict[str, Any]:
        tab = _mean_table(pred, actual, scale, half, p25)
        tab.pop("_err")
        return tab

    err_mean = _abs_err(_as_frame(mean_c, actual.index), actual)
    err_lp = _abs_err(_as_frame(lp_c, actual.index), actual)
    err_zero = _abs_err(_as_frame(zero_c, actual.index), actual)
    diff, ci, _ = bootstrap_difference(
        err_mean, err_lp, scale, int(cfg["decision_rule"]["bootstrap"]["resamples"]),
        int(cfg["decision_rule"]["bootstrap"]["seed"]),
        float(cfg["decision_rule"]["bootstrap"]["ci"]))
    boot = cfg["decision_rule"]["bootstrap"]

    # Static refit, secondary.
    static_mean = _fit_torch_predict(
        mean_name, mean_params, train_cfg, panel, through,
        np.sort(events.loc[events["year"] >= test_start, "year"].unique()))
    static_lp = _lp_on(frames, lambda y: y <= val_end, lambda y: y >= test_start)
    s_diff, s_ci, _ = bootstrap_difference(
        _abs_err(_as_frame(static_mean, actual.index), actual),
        _abs_err(_as_frame(static_lp, actual.index), actual),
        scale, int(boot["resamples"]), int(boot["seed"]), float(boot["ci"]))

    pin_tail = _pinball_frame(tail_c, actual, tau)
    pin_gauss = _pinball_frame(gauss_c, actual, tau)
    pin_hist = _pinball_frame(hist_c, actual, tau)
    t_diff, t_ci, _ = bootstrap_difference(
        pin_tail, pin_gauss, pscale, int(boot["resamples"]), int(boot["seed"]), float(boot["ci"]))

    def hit(pred: dict[str, pd.Series], actual_df: pd.DataFrame) -> float:
        wide = _as_frame(pred, actual_df.index)
        rates = []
        for name in CELL_NAMES:
            y = actual_df[name].to_numpy()
            q = wide[name].to_numpy()
            ok = ~np.isnan(y) & ~np.isnan(q)
            if ok.any():
                rates.append(float(np.mean(y[ok] <= q[ok])))
        return float(np.mean(rates)) if rates else float("nan")

    def cover(lo: dict[str, pd.Series], hi: dict[str, pd.Series]) -> float:
        a = _as_frame(lo, actual.index)
        b = _as_frame(hi, actual.index)
        rates = []
        for name in CELL_NAMES:
            y = actual[name].to_numpy()
            ok = ~np.isnan(y) & ~np.isnan(a[name].to_numpy()) & ~np.isnan(b[name].to_numpy())
            if ok.any():
                rates.append(float(np.mean((y[ok] >= a[name].to_numpy()[ok])
                                           & (y[ok] <= b[name].to_numpy()[ok]))))
        return float(np.mean(rates)) if rates else float("nan")

    def fz(var: dict[str, pd.Series], es: dict[str, pd.Series]) -> float:
        v = _as_frame(var, actual.index)
        e = _as_frame(es, actual.index)
        scores = []
        for name in CELL_NAMES:
            y = actual[name].to_numpy()
            ok = ~np.isnan(y) & np.isfinite(v[name].to_numpy()) & np.isfinite(e[name].to_numpy())
            if ok.any():
                scores.append(float(np.mean(fissler_ziegel(
                    y[ok], v[name].to_numpy()[ok], e[name].to_numpy()[ok], tau))))
        return float(np.mean(scores)) if scores else float("nan")

    def cvar_from_quantiles(qs: dict[float, dict[str, pd.Series]]) -> dict[str, pd.Series]:
        acc: dict[str, list[pd.Series]] = {}
        for t in TAUS_CVAR:
            for name, s in qs[t].items():
                acc.setdefault(name, []).append(s)
        out: dict[str, pd.Series] = {}
        for name, parts in acc.items():
            total = parts[0]
            for part in parts[1:]:
                total = total + part
            out[name] = total / len(parts)
        return out

    tail_cvar = cvar_from_quantiles(tail_all)
    hist_cvar = _expanding_hist_cvar(frames, test_years, tau)
    gauss_cvar = _gauss_cvar_from(g50_c, frames, test_years)

    subs = _family_subs(err_mean, err_lp, err_zero, scale, fam, pooled_scaled_mae)

    _log("final: embeddings, Shapley values, portfolio, stress")
    extras = _extras(cfg, train_cfg, panel, frames, data, events, mean_name, mean_params,
                     mean_key, tail_name, tail_params, tail_key, through, test_years, tau,
                     val_end, test_start)

    result = {
        "phase": "final", "generated_at": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "protocol_tag": tag, "config_sha256": file_sha256(STUDY_CONFIG),
        "git_sha": git_sha(), "rerun_reason": rerun_reason,
        "parent_final_eval": "results/final_eval/20261012T233454Z.json",
        "n_test": len(test_index), "test_years": [int(y) for y in test_years],
        "counts": val["counts"],
        "chosen": {"mean": mean_key, "tail": tail_key},
        "mean": {
            "baseline": "lp",
            "pooled_scaled_mae": {
                "zero": pooled_scaled_mae(err_zero, scale),
                "lp": pooled_scaled_mae(err_lp, scale),
                mean_key: pooled_scaled_mae(err_mean, scale),
            },
            "difference_ml_minus_lp": {"point": diff, "ci90": list(ci)},
            "verdict": verdict(ci),
            "per_cell_scaled_mae": {
                "zero": _scaled(mean_block(zero_c, None)),
                "lp": _scaled(mean_block(lp_c, None)),
                mean_key: _scaled(mean_block(mean_c, None)),
            },
            "coverage90": {
                "lp": _coverage_widths(lp_c, actual, lp_widths),
                mean_key: _coverage_widths(mean_c, actual, ml_widths),
            },
            "static_refit": {
                "pooled_scaled_mae": {
                    "lp": pooled_scaled_mae(
                        _abs_err(_as_frame(static_lp, actual.index), actual), scale),
                    mean_key: pooled_scaled_mae(
                        _abs_err(_as_frame(static_mean, actual.index), actual), scale),
                },
                "difference_ml_minus_lp": {"point": s_diff, "ci90": list(s_ci)},
                "verdict": verdict(s_ci),
            },
            "subsamples": subs,
        },
        "tail": {
            "baseline": "gauss_lp", "tau": tau,
            "pooled_scaled_pinball": {
                "hist": pooled_scaled_mae(pin_hist, pscale),
                "gauss_lp": pooled_scaled_mae(pin_gauss, pscale),
                tail_key: pooled_scaled_mae(pin_tail, pscale),
            },
            "difference_ml_minus_gauss": {"point": t_diff, "ci90": list(t_ci)},
            "verdict": tail_verdict(t_ci),
            "hit_rate_05": {"hist": hit(hist_c, actual), "gauss_lp": hit(gauss_c, actual),
                            tail_key: hit(tail_c, actual)},
            "coverage_90": {
                "gauss_lp": cover(gauss_c, g95_c),
                tail_key: cover(tail_all[0.05], tail_all[0.95]),
            },
            "interval_score90_scaled": {
                "gauss_lp": _interval_score(actual, gauss_c, g95_c, scale),
                tail_key: _interval_score(actual, tail_all[0.05], tail_all[0.95], scale),
            },
            "fissler_ziegel": {
                "hist": fz(hist_c, hist_cvar),
                "gauss_lp": fz(gauss_c, gauss_cvar),
                tail_key: fz(tail_c, tail_cvar),
            },
        },
        "validation_pooled_scaled_mae": val["pooled_scaled_mae"],
        "validation_pooled_scaled_pinball": val["pooled_scaled_pinball"],
        "inputs": val.get("inputs", _inputs()),
        **extras,
    }
    if FINAL_PATH.exists() and rerun_reason:
        stamp = json.loads(FINAL_PATH.read_text()).get("generated_at", "previous")
        FINAL_PATH.replace(RESULTS / f"final_rerun_{stamp}.json")
    FINAL_PATH.write_text(json.dumps(_clean(result), indent=1) + "\n")
    _log(f"verdict mean: {result['mean']['verdict']}")
    _log(f"verdict tail: {result['tail']['verdict']}")
    return result


def _lp_gaussian_halves(frames: list[CellFrame], train_pred) -> dict[str, dict[str, float]]:  # type: ignore[no-untyped-def]
    out = {}
    for f in frames:
        tr = train_pred(f.year).fillna(False).to_numpy()
        if tr.sum() < 20:
            continue
        model = Linear(with_state=False).fit(f.X.loc[tr], f.y.to_numpy()[tr])
        sd = float(np.std(f.y.to_numpy()[tr] - model.predict(f.X.loc[tr]), ddof=1))
        out[f.name] = {"68": z_value(0.68) * sd, "90": z_value(0.90) * sd}
    return out


def _widths(parts: list[dict[str, pd.Series]], halves: list[dict[str, dict[str, float]]],
            level: str = "90") -> dict[str, pd.Series]:
    """Broadcast each year's half-width onto that year's events."""
    out: dict[str, list[pd.Series]] = {}
    for pred, half in zip(parts, halves, strict=True):
        for name, s in pred.items():
            w = half.get(name, {}).get(level, np.nan)
            out.setdefault(name, []).append(pd.Series(float(w), index=s.index))
    return {k: pd.concat(v) for k, v in out.items()}


def _coverage_widths(pred: dict[str, pd.Series], actual: pd.DataFrame,
                     widths: dict[str, pd.Series]) -> float:
    rates = []
    wide = _as_frame(pred, actual.index)
    w = _as_frame(widths, actual.index)
    for name in CELL_NAMES:
        y = actual[name].to_numpy()
        p = wide[name].to_numpy()
        h = w[name].to_numpy()
        ok = ~np.isnan(y) & ~np.isnan(p) & ~np.isnan(h)
        if ok.any():
            rates.append(float(np.mean(np.abs(y[ok] - p[ok]) <= h[ok])))
    return float(np.mean(rates)) if rates else float("nan")


def _interval_score(actual: pd.DataFrame, lo: dict[str, pd.Series], hi: dict[str, pd.Series],
                    scale: pd.Series) -> float:
    a = _as_frame(lo, actual.index)
    b = _as_frame(hi, actual.index)
    scores = []
    for name in CELL_NAMES:
        y = actual[name].to_numpy()
        ok = ~np.isnan(y) & np.isfinite(a[name].to_numpy()) & np.isfinite(b[name].to_numpy())
        if ok.any() and scale[name]:
            scores.append(float(np.mean(interval_score(
                y[ok], a[name].to_numpy()[ok], b[name].to_numpy()[ok], 0.10)) / float(scale[name])))
    return float(np.mean(scores)) if scores else float("nan")


def _family_subs(err_ml: pd.DataFrame, err_lp: pd.DataFrame, err_zero: pd.DataFrame,
                 scale: pd.Series, fam: pd.Series, pool: Any) -> dict[str, Any]:
    out = {}
    for name, m in (("fomc", fam == "fomc"), ("boc", fam == "boc"), ("opec", fam == "opec")):
        idx = fam.index[m.fillna(False).to_numpy()]
        out[name] = {
            "events": len(idx),
            "zero": pool(err_zero.loc[idx], scale),
            "lp": pool(err_lp.loc[idx], scale),
            "ml": pool(err_ml.loc[idx], scale),
        }
    return out


def _expanding_hist_cvar(frames: list[CellFrame], test_years: np.ndarray, tau: float
                         ) -> dict[str, pd.Series]:
    parts: dict[str, list[pd.Series]] = {}
    for year in test_years:
        for f in frames:
            tr = (f.year < year).to_numpy()
            te = (f.year == year).to_numpy()
            if tr.sum() < 20 or te.sum() == 0:
                continue
            y = f.y.to_numpy()[tr]
            q = float(np.quantile(y, tau))
            tail = y[y <= q]
            es = float(tail.mean()) if len(tail) else float(np.min(y))
            parts.setdefault(f.name, []).append(pd.Series(es, index=f.y.index[te]))
    return {k: pd.concat(v) for k, v in parts.items()}


def _gauss_cvar_from(median: dict[str, pd.Series], frames: list[CellFrame],
                     test_years: np.ndarray) -> dict[str, pd.Series]:
    """CVaR = μ − σ φ(z)/α, with μ the LP prediction and σ the training residual sd."""
    z = float(norm.ppf(0.05))
    phi = float(norm.pdf(z))
    parts: dict[str, list[pd.Series]] = {}
    for year in test_years:
        for f in frames:
            tr = (f.year < year).to_numpy()
            te = (f.year == year).to_numpy()
            if f.name not in median or tr.sum() < 20 or te.sum() == 0:
                continue
            model = Linear(with_state=False).fit(f.X.loc[tr], f.y.to_numpy()[tr])
            sd = float(np.std(f.y.to_numpy()[tr] - model.predict(f.X.loc[tr]), ddof=1))
            mu = median[f.name].reindex(f.y.index[te])
            parts.setdefault(f.name, []).append(mu - sd * phi / 0.05)
    return {k: pd.concat(v) for k, v in parts.items()}


def _extras(cfg: dict[str, Any], train_cfg: TrainCfg, panel: Panel, frames: list[CellFrame],
            data: EventData, events: pd.DataFrame, mean_name: str, mean_params: dict[str, Any],
            mean_key: str, tail_name: str, tail_params: dict[str, Any], tail_key: str,
            through: np.ndarray, test_years: np.ndarray, tau: float, val_end: int,
            test_start: int) -> dict[str, Any]:
    """Embeddings, Shapley, the equity book and the stress row. All secondary."""
    model = _torch(mean_name, mean_params, train_cfg).fit(panel, _panel_mask_years(panel, through))
    vectors = np.mean([model.embedding(i) for i in range(len(train_cfg.seeds))], axis=0)
    n_perm = int(cfg["embeddings"]["permutations"])
    emb = embedding_agreement(
        vectors, cfg["embeddings"]["groups"], n_perm, int(cfg["embeddings"]["seed"]))
    emb["model"] = mean_key
    seed_aris = []
    for i in range(len(train_cfg.seeds)):
        one = embedding_agreement(model.embedding(i), cfg["embeddings"]["groups"],
                                  200, int(cfg["embeddings"]["seed"]) + i)
        seed_aris.append(one["ari"])
    emb["ari_by_seed"] = seed_aris
    # The permutation test above is expensive if called 4 times at 2000. The mean
    # embedding uses the full 2000; per-seed ARI uses 200 shuffles only as a stability note.
    te_h1 = _panel_mask_years(panel, test_years) & (panel.horizon_i == 0)
    groups = group_index(panel.feature_names, use_path=mean_name == "tcn")
    bx, bp = model.asset_baseline(panel.asset_i[te_h1], panel.X.shape[1])

    def predict(xm: np.ndarray, pm: np.ndarray) -> np.ndarray:
        return model.predict_rows(xm, pm, panel.asset_i[te_h1], panel.horizon_i[te_h1],
                                  panel.cell[te_h1])

    shap, gap = grouped_shap(predict, panel.X[te_h1], panel.path[te_h1], bx, bp, groups)
    mean_abs = np.mean(np.abs(shap), axis=0)
    total = float(mean_abs.sum()) or 1.0
    shap_doc = {
        "model": mean_key, "horizon": 1, "n_rows": int(te_h1.sum()),
        "mean_abs": {groups[j][0]: float(mean_abs[j]) for j in range(len(groups))},
        "share": {groups[j][0]: float(mean_abs[j] / total) for j in range(len(groups))},
        "efficiency_gap": gap,
    }

    book = _portfolio(cfg, train_cfg, panel, frames, data, events, tail_name, tail_params,
                      tail_key, test_years, tau, through)
    stress = _stress(cfg, train_cfg, panel, frames, data, events, tail_name, tail_params,
                     through, tau)
    return {"embeddings": emb, "shap": shap_doc, "portfolio": book, "stress": stress}


def _portfolio(cfg: dict[str, Any], train_cfg: TrainCfg, panel: Panel, frames: list[CellFrame],
               data: EventData, events: pd.DataFrame, tail_name: str, tail_params: dict[str, Any],
               tail_key: str, test_years: np.ndarray, tau: float, through: np.ndarray,
               ) -> dict[str, Any]:
    spec = cfg["portfolio"]
    weights = dict(zip(spec["assets"], spec["weights"], strict=True))
    y = sum(weights[a] * data.targets[f"{a}|h{spec['horizon']}"] for a in spec["assets"])
    # One-asset panel: SPY's features at h=1, target replaced by the book.
    spy = ASSET_NAMES.index("SPY")
    h = HORIZONS.index(int(spec["horizon"]))
    m = (panel.asset_i == spy) & (panel.horizon_i == h)
    book_y = pd.Series(y.to_numpy(), index=data.targets.index).reindex(panel.event_id[m])
    ok = book_y.notna().to_numpy()
    sub = Panel(
        event_id=panel.event_id[m][ok], event_i=panel.event_i[m][ok], year=panel.year[m][ok],
        asset_i=np.zeros(int(ok.sum()), dtype=int), horizon_i=np.zeros(int(ok.sum()), dtype=int),
        cell=np.full(int(ok.sum()), "PORT|h1", dtype=object),
        X=panel.X[m][ok], path=panel.path[m][ok], y=book_y.to_numpy()[ok].astype(float),
        feature_names=panel.feature_names,
    )
    # Scales from the training split.
    train_end = _bounds()[0]
    y_tr = sub.y[sub.year <= train_end]
    q_hist = float(np.quantile(y_tr, tau))
    pscale = float(np.mean(pinball(y_tr, np.full(len(y_tr), q_hist), tau)))
    parts_hist, parts_gauss, parts_ml = [], [], []
    for year in test_years:
        tr = np.sort(events.loc[events["year"] < year, "year"].unique())
        te = np.array([year])
        tr_m = np.isin(sub.year, tr)
        te_m = np.isin(sub.year, te)
        q = float(np.quantile(sub.y[tr_m], tau))
        parts_hist.append(pd.Series(q, index=sub.event_id[te_m]))
        # Gaussian around a linear fit on the same rows, via a one-cell frame built below.
        if tail_name == "qnet":
            model = _torch("qnet", tail_params, train_cfg).fit(sub, tr_m)
            pred = model.predict(sub, te_m)[:, QUANTILES.index(tau)]
        else:
            model_g = _gbm(tail_params, cfg, (tau,)).fit(sub, tr_m)
            pred = model_g.predict(sub, te_m, tau)
        parts_ml.append(pd.Series(pred, index=sub.event_id[te_m]))
        g = _book_gauss(sub, tr_m, te_m, tau)
        parts_gauss.append(g)
    actual = pd.Series(sub.y[sub.year >= int(test_years.min())],
                       index=sub.event_id[sub.year >= int(test_years.min())])

    def frame(parts: list[pd.Series]) -> pd.DataFrame:
        s = pd.concat(parts)
        s = s[~s.index.duplicated(keep="first")]
        score = pd.Series(np.nan, index=actual.index)
        aligned = s.reindex(actual.index)
        ok = actual.notna() & aligned.notna()
        score.loc[ok] = pinball(actual[ok].to_numpy(), aligned[ok].to_numpy(), tau)
        return pd.DataFrame({"PORT|h1": score})

    scale = pd.Series({"PORT|h1": pscale})
    fh, fg, fm = frame(parts_hist), frame(parts_gauss), frame(parts_ml)
    boot = cfg["decision_rule"]["bootstrap"]
    diff, ci, _ = bootstrap_difference(
        fm, fg, scale, int(boot["resamples"]), int(boot["seed"]), float(boot["ci"]))
    return {
        "name": spec["name"], "weights": weights, "horizon": int(spec["horizon"]),
        "vol_feature": spec["vol_feature"], "secondary": True,
        "pooled_scaled_pinball": {
            "hist": pooled_scaled_mae(fh, scale),
            "gauss_lp": pooled_scaled_mae(fg, scale),
            tail_key: pooled_scaled_mae(fm, scale),
        },
        "difference_ml_minus_gauss": {"point": diff, "ci90": list(ci)},
        "note": "Secondary. Does not change either verdict.",
    }


def _book_gauss(sub: Panel, tr_m: np.ndarray, te_m: np.ndarray, tau: float) -> pd.Series:
    """The same linear projection as the asset-level baseline, plus a Gaussian residual quantile."""
    xtr = pd.DataFrame(sub.X[tr_m], columns=sub.feature_names)
    xte = pd.DataFrame(sub.X[te_m], columns=sub.feature_names)
    model = Linear(with_state=False).fit(xtr, sub.y[tr_m])
    mu = model.predict(xte)
    sd = float(np.std(sub.y[tr_m] - model.predict(xtr), ddof=1))
    return pd.Series(mu + float(norm.ppf(tau)) * sd, index=sub.event_id[te_m])


def _stress(cfg: dict[str, Any], train_cfg: TrainCfg, panel: Panel, frames: list[CellFrame],
            data: EventData, events: pd.DataFrame, tail_name: str, tail_params: dict[str, Any],
            through: np.ndarray, tau: float) -> dict[str, Any]:
    """Hypothetical +1 training-SD FOMC surprise versus a zero surprise, state at the median."""
    del frames, data
    names = panel.feature_names
    tr = _panel_mask_years(panel, through)
    Xtr = panel.X[tr]
    main = names.index(cfg["stress"]["main_surprise"])
    fomc = Xtr[:, names.index("is_fomc")] > 0.5
    mvals = Xtr[fomc, main]
    mvals = mvals[np.isfinite(mvals)]
    sd = float(np.std(mvals, ddof=1))
    base = np.nanmedian(Xtr, axis=0)
    base = np.where(np.isnan(base), 0.0, base)
    zero = base.copy()
    shock = base.copy()
    for c in SHOCK_NAMES:
        zero[names.index(c)] = 0.0
        shock[names.index(c)] = 0.0
    for c, val in (("is_fomc", 1.0), ("is_boc", 0.0), ("is_opec", 0.0)):
        zero[names.index(c)] = val
        shock[names.index(c)] = val
    shock[main] = sd
    mm = Xtr[fomc, main]
    main_name = cfg["stress"]["main_surprise"]
    others = [n for n in SHOCK_NAMES if n.startswith("fomc_") and n != main_name]
    for c in others:
        j = names.index(c)
        col = Xtr[fomc, j]
        ok = np.isfinite(col) & np.isfinite(mm)
        if int(ok.sum()) < 10:
            continue
        x, yy = mm[ok], col[ok]
        xc = x - x.mean()
        denom = float(np.dot(xc, xc))
        if denom < 1e-8:
            continue
        beta = float(np.dot(xc, yy - yy.mean()) / denom)
        alpha = float(yy.mean() - beta * x.mean())
        shock[j] = alpha + beta * sd
    # Score SPY, TSX, XEG, WTI at horizon 1 with the model fit through 2018.
    want = ["SPY", "TSX", "XEG", "WTI"]
    rows_x, rows_a, rows_p = [], [], []
    for asset in want:
        a = ASSET_NAMES.index(asset)
        vol_j = names.index("asset_vol")
        asset_rows = tr & (panel.asset_i == a)
        vol_med = float(np.nanmedian(panel.X[asset_rows, vol_j]))
        path_med = np.nanmean(panel.path[asset_rows], axis=0)
        path_med = np.where(np.isnan(path_med), 0.0, path_med)
        for row in (zero, shock):
            r = row.copy()
            r[vol_j] = 0.0 if np.isnan(vol_med) else vol_med
            rows_x.append(r)
            rows_a.append(a)
            rows_p.append(path_med)
    X = np.vstack(rows_x)
    path = np.vstack(rows_p)
    asset_ix = np.asarray(rows_a, dtype=int)
    horizon = np.zeros(len(asset_ix), dtype=int)
    cell = np.array([f"{ASSET_NAMES[i]}|h1" for i in asset_ix], dtype=object)
    if tail_name == "qnet":
        model = _torch("qnet", tail_params, train_cfg).fit(panel, tr)
        q = model.predict_rows(X, path, asset_ix, horizon, cell)
        q05 = q[:, QUANTILES.index(0.05)]
        q50 = q[:, QUANTILES.index(0.50)]
    else:
        model_g = _gbm(tail_params, cfg, (0.05, 0.50)).fit(panel, tr)
        q05 = model_g.predict_rows(X, cell, 0.05)
        q50 = model_g.predict_rows(X, cell, 0.50)
    out_assets = {}
    for i, asset_name in enumerate(want):
        out_assets[asset_name] = {
            "q50_zero": float(q50[2 * i]), "q50_shock": float(q50[2 * i + 1]),
            "q05_zero": float(q05[2 * i]), "q05_shock": float(q05[2 * i + 1]),
        }
    return {
        "label": "hypothetical_scenario",
        "family": "fomc",
        "main_surprise": cfg["stress"]["main_surprise"],
        "size_sd": sd,
        "fit": f"through_{val_end_label(through)}",
        "assets": out_assets,
        "note": "Not a test-set score. The state is the training median and the other "
                "futures surprises move with their training co-movement.",
    }


def val_end_label(through: np.ndarray) -> int:
    return int(np.max(through)) if len(through) else 0
