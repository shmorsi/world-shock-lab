"""Benchmarks and the ML model for M5 (protocol §5). All share fit(X, y) / predict(X), with
preprocessing (median imputation + missing indicators + scaling) fitted on training rows only.

  zero   B0  always 0
  lp     B1  OLS on family intercepts and the raw surprise components (linear in the shock)
  state  B2  B1 + state features + (family shock index × state) interactions
  ridge  B3  B2's features plus every surprise component × state interaction, penalty chosen
             by blocked time-ordered cross-validation inside the training data
  gbm    B4  histogram gradient boosting on the raw features
  mlp    ML  small MLP (PyTorch), L1 penalty on the first layer for sparsity, weight decay,
             early stopping on the last 20% of the training window; 5-seed ensemble
The family shock index is the first principal component of a family's surprise components,
estimated on the training rows only (the published full-sample PCs are never used).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from wsl.ml.dataset import SHOCK_COLS, STATE_COLS

FAMILY_DUMMIES = ["is_fomc", "is_boc", "is_opec"]


class Model(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Model: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def base_frame(events: pd.DataFrame, vol: pd.Series) -> pd.DataFrame:
    X = events[[c for cols in SHOCK_COLS.values() for c in cols] + STATE_COLS].copy()
    for fam, d in zip(("fomc", "boc", "opec"), FAMILY_DUMMIES, strict=True):
        X[d] = (events["family"] == fam).astype(float)
    X["asset_vol"] = vol.to_numpy()
    return X


@dataclass
class Prep:
    """Median imputation with missing indicators, then standardization (train stats)."""

    scale: bool = True
    medians: pd.Series = field(default_factory=pd.Series)
    flag_cols: list[str] = field(default_factory=list)
    mean: pd.Series = field(default_factory=pd.Series)
    std: pd.Series = field(default_factory=pd.Series)

    def fit(self, X: pd.DataFrame) -> Prep:
        self.medians = X.median()
        self.flag_cols = [c for c in X.columns if X[c].isna().any()]
        Z = self._impute(X)
        self.mean, self.std = Z.mean(), Z.std().replace(0.0, 1.0)
        return self

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = X.copy()
        for c in self.flag_cols:
            Z[f"{c}__missing"] = X[c].isna().astype(float)
        return Z.fillna(self.medians).fillna(0.0)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self._impute(X)
        return (Z - self.mean) / self.std if self.scale else Z


class ShockIndex:
    """Per-family first principal component of the surprise components (train rows only)."""

    def fit(self, X: pd.DataFrame) -> ShockIndex:
        self.load: dict[str, tuple[list[str], np.ndarray, np.ndarray, np.ndarray]] = {}
        for fam, cols in SHOCK_COLS.items():
            rows = X[X[f"is_{fam}"] == 1.0][cols].fillna(X[cols].median())
            if len(rows) < 3:  # family absent from this training window
                self.load[fam] = (cols, np.zeros(len(cols)), np.ones(len(cols)),
                                  np.zeros(len(cols)))
                continue
            mu, sd = rows.mean().to_numpy(), rows.std().fillna(1.0).replace(0, 1).to_numpy()
            Z = (rows.to_numpy() - mu) / sd
            if len(cols) == 1:
                w = np.ones(1)
            else:
                _, _, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
                w = vt[0] * np.sign(vt[0].sum())
            self.load[fam] = (cols, mu, sd, w)
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        out = np.zeros(len(X))
        for fam, (cols, mu, sd, w) in self.load.items():
            m = (X[f"is_{fam}"] == 1.0).to_numpy()
            vals = X.loc[m, cols].fillna(pd.Series(mu, index=cols)).to_numpy()
            out[m] = ((vals - mu) / sd) @ w
        return pd.Series(out, index=X.index)


class Zero:
    name = "zero"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Zero:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class Linear:
    """B1 (with_state=False) and B2 (with_state=True), OLS."""

    def __init__(self, with_state: bool) -> None:
        self.with_state = with_state
        self.name = "state" if with_state else "lp"

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        shocks = [c for cols in SHOCK_COLS.values() for c in cols]
        Z = self.prep.transform(X)
        cols = FAMILY_DUMMIES + shocks
        D = Z[[c for c in cols if c in Z]].copy()
        D[FAMILY_DUMMIES] = X[FAMILY_DUMMIES].to_numpy()  # keep 0/1 intercepts unscaled
        if self.with_state:
            idx = self.index.transform(X)
            idx = (idx - self.idx_mu) / self.idx_sd
            state = [c for c in Z.columns if c in STATE_COLS or c == "asset_vol"
                     or c.endswith("__missing")]
            for c in state:
                D[c] = Z[c]
                D[f"idx×{c}"] = idx.to_numpy() * Z[c].to_numpy()
        return D

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Linear:
        self.prep = Prep().fit(X)
        self.index = ShockIndex().fit(X)
        raw = self.index.transform(X)
        self.idx_mu, self.idx_sd = float(raw.mean()), float(raw.std() or 1.0)
        D = self._design(X).to_numpy()
        self.beta, *_ = np.linalg.lstsq(D, y, rcond=None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._design(X).to_numpy() @ self.beta, dtype=float)


class RidgeModel(Linear):
    name = "ridge"
    ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)

    def __init__(self) -> None:
        super().__init__(with_state=True)
        self.name = "ridge"

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        D = super()._design(X)
        Z = self.prep.transform(X)
        shocks = [c for cols in SHOCK_COLS.values() for c in cols]
        for s in shocks:
            for c in STATE_COLS:
                D[f"{s}×{c}"] = Z[s].to_numpy() * Z[c].to_numpy()
        return D

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RidgeModel:
        self.prep = Prep().fit(X)
        self.index = ShockIndex().fit(X)
        raw = self.index.transform(X)
        self.idx_mu, self.idx_sd = float(raw.mean()), float(raw.std() or 1.0)
        D = self._design(X).to_numpy()
        self.alpha = blocked_cv_alpha(D, y, self.ALPHAS)
        self.model = Ridge(alpha=self.alpha).fit(D, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self._design(X).to_numpy()))


def blocked_cv_alpha(D: np.ndarray, y: np.ndarray, alphas: Sequence[float],
                     folds: int = 5) -> float:
    """Expanding-window blocked CV in time order: fit on the first k blocks, score MAE on
    block k+1."""
    n = len(y)
    edges = np.linspace(n // 3, n, folds + 1).astype(int)
    best, best_err = alphas[0], np.inf
    for a in alphas:
        errs = []
        for i in range(folds):
            tr, te = slice(0, edges[i]), slice(edges[i], edges[i + 1])
            if edges[i + 1] <= edges[i]:
                continue
            m = Ridge(alpha=a).fit(D[tr], y[tr])
            errs.append(np.mean(np.abs(m.predict(D[te]) - y[te])))
        err = float(np.mean(errs)) if errs else np.inf
        if err < best_err:
            best, best_err = a, err
    return float(best)


class GBM:
    name = "gbm"
    GRID: tuple[dict[str, Any], ...] = tuple(
        {"max_depth": d, "max_iter": it, "learning_rate": 0.05, "min_samples_leaf": 20}
        for d in (2, 3) for it in (100, 300))

    def __init__(self, **params: Any) -> None:
        self.params = params or dict(self.GRID[0])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> GBM:
        self.prep = Prep(scale=False).fit(X)
        self.model = HistGradientBoostingRegressor(loss="absolute_error", random_state=0,
                                                   **self.params)
        self.model.fit(self.prep.transform(X).to_numpy(), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self.prep.transform(X).to_numpy()))


class _Net(torch.nn.Module):
    def __init__(self, d: int, width: int) -> None:
        super().__init__()
        self.first = torch.nn.Linear(d, width)
        self.body = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(width, width),
                                        torch.nn.ReLU(), torch.nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.body(self.first(x)).squeeze(-1)
        return out


class MLP:
    name = "mlp"
    GRID: tuple[dict[str, Any], ...] = tuple(
        {"width": w, "weight_decay": wd, "l1": l1}
        for w in (16, 32) for wd in (1e-3, 1e-2) for l1 in (0.0, 1e-3))
    SEEDS = (0, 1, 2, 3, 4)
    MAX_EPOCHS = 400
    PATIENCE = 30

    def __init__(self, **params: Any) -> None:
        self.params = params or dict(self.GRID[0])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> MLP:
        self.prep = Prep().fit(X)
        Z = self.prep.transform(X).to_numpy(dtype=np.float32)
        self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y) or 1.0)
        t = ((y - self.y_mu) / self.y_sd).astype(np.float32)
        n_fit = int(len(t) * 0.8)  # time-ordered: early-stop on the last 20% of the window
        self.nets = []
        for seed in self.SEEDS:
            torch.manual_seed(seed)
            net = _Net(Z.shape[1], int(self.params["width"]))
            opt = torch.optim.Adam(net.parameters(), lr=1e-2,
                                   weight_decay=float(self.params["weight_decay"]))
            xf, yf = torch.from_numpy(Z[:n_fit]), torch.from_numpy(t[:n_fit])
            xv, yv = torch.from_numpy(Z[n_fit:]), torch.from_numpy(t[n_fit:])
            best, best_state, wait = np.inf, None, 0
            for _ in range(self.MAX_EPOCHS):
                net.train()
                opt.zero_grad()
                loss = torch.nn.functional.huber_loss(net(xf), yf)
                loss = loss + float(self.params["l1"]) * net.first.weight.abs().sum()
                loss.backward()  # type: ignore[no-untyped-call]
                opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(torch.nn.functional.huber_loss(net(xv), yv))
                if v < best - 1e-6:
                    best, wait = v, 0
                    best_state = {k: p.clone() for k, p in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.PATIENCE:
                        break
            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self.nets.append(net)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = torch.from_numpy(self.prep.transform(X).to_numpy(dtype=np.float32))
        with torch.no_grad():
            preds = np.mean([n(Z).numpy() for n in self.nets], axis=0)
        return np.asarray(preds * self.y_sd + self.y_mu, dtype=float)


def make(name: str, params: dict[str, Any] | None = None) -> Any:
    if name == "zero":
        return Zero()
    if name == "lp":
        return Linear(with_state=False)
    if name == "state":
        return Linear(with_state=True)
    if name == "ridge":
        return RidgeModel()
    if name == "gbm":
        return GBM(**(params or {}))
    if name == "mlp":
        return MLP(**(params or {}))
    raise ValueError(name)


ECONOMETRIC = ("lp", "state", "ridge", "gbm")
ML = ("mlp",)
ALL = ("zero", *ECONOMETRIC, *ML)
