"""Models for ml-study-v1. Fit on a Panel; nothing here touches the test split.

Mean models predict one number. Quantile models predict an ordered set of quantiles
(the first head is free, later heads add a softplus increment, so the quantiles
cannot cross). Every scaler is fit on the training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from wsl.mlstudy.panels import PATH_LEN, Panel

QUANTILES = (0.01, 0.02, 0.03, 0.04, 0.05, 0.50, 0.95)


@dataclass
class TrainCfg:
    lr: float = 1e-3
    max_epochs: int = 200
    patience: int = 20
    grad_clip: float = 1.0
    dropout: float = 0.1
    emb_dim: int = 4
    seeds: tuple[int, ...] = (0, 1, 2)
    early_stop_frac: float = 0.2
    kernel: int = 3

    @classmethod
    def from_yaml(cls, doc: dict[str, Any]) -> TrainCfg:
        t = doc["training"]
        return cls(
            lr=float(t["lr"]), max_epochs=int(t["max_epochs"]), patience=int(t["patience"]),
            grad_clip=float(t["grad_clip"]), dropout=float(t["dropout"]),
            emb_dim=int(t["emb_dim"]), seeds=tuple(int(s) for s in t["seeds"]),
            early_stop_frac=float(t["early_stop_frac"]), kernel=int(doc["tcn_kernel"]),
        )


def _impute(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    def fill(a: np.ndarray) -> np.ndarray:
        out = a.copy()
        inds = np.where(np.isnan(out))
        out[inds] = np.take(med, inds[1])
        return out
    return fill(train), fill(other), med


def _scale(train: np.ndarray, other: np.ndarray,
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (other - mu) / sd, mu, sd


class _MeanNet(torch.nn.Module):
    def __init__(self, n_feat: int, n_assets: int, n_horizons: int, width: int, emb: int,
                 dropout: float, channels: int, kernel: int, use_path: bool) -> None:
        super().__init__()
        self.use_path = use_path
        self.asset = torch.nn.Embedding(n_assets, emb)
        self.horizon = torch.nn.Embedding(n_horizons, 4)
        seq_dim = 0
        if use_path:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(1, channels, kernel, padding=kernel // 2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(channels, channels, kernel, padding=kernel // 2),
                torch.nn.ReLU(),
            )
            seq_dim = channels
        d = n_feat + emb + 4 + seq_dim
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(d, width),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor, path: torch.Tensor, asset: torch.Tensor,
                horizon: torch.Tensor) -> torch.Tensor:
        parts = [x, self.asset(asset), self.horizon(horizon)]
        if self.use_path:
            h = self.conv(path.unsqueeze(1))
            parts.append(h.mean(dim=2))
        out: torch.Tensor = self.trunk(torch.cat(parts, dim=1)).squeeze(1)
        return out


class _QuantileNet(torch.nn.Module):
    def __init__(self, n_feat: int, n_assets: int, n_horizons: int, width: int, emb: int,
                 dropout: float, n_q: int) -> None:
        super().__init__()
        self.asset = torch.nn.Embedding(n_assets, emb)
        self.horizon = torch.nn.Embedding(n_horizons, 4)
        d = n_feat + emb + 4
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(d, width),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, n_q),
        )

    def forward(self, x: torch.Tensor, asset: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        raw = self.trunk(torch.cat([x, self.asset(asset), self.horizon(horizon)], dim=1))
        # Non-crossing: q_0 free, each later quantile sits above the previous one.
        inc = torch.nn.functional.softplus(raw[:, 1:])
        q = torch.cumsum(torch.cat([raw[:, :1], inc], dim=1), dim=1)
        return q


def _pinball_loss(y: torch.Tensor, q: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    err = y.unsqueeze(1) - q
    loss = torch.maximum(taus * err, (taus - 1.0) * err)
    return loss.mean()


def _weights(asset: np.ndarray, horizon: np.ndarray) -> np.ndarray:
    """Each asset × horizon cell contributes the same total weight."""
    key = asset.astype(np.int64) * 100 + horizon.astype(np.int64)
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w * (len(w) / w.sum())


def _split_events(event_i: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.unique(event_i)
    cut = max(1, int(len(order) * (1.0 - frac)))
    cut = min(cut, len(order) - 1) if len(order) > 1 else 1
    return order[:cut], order[cut:]


class TorchModel:
    """Shared training loop. ``kind`` is 'mean', 'tcn' or 'qnet'."""

    def __init__(self, kind: str, cfg: TrainCfg, **params: Any) -> None:
        self.kind = kind
        self.cfg = cfg
        self.params = params
        self.nets: list[torch.nn.Module] = []
        self.name = kind

    def _arrays(self, panel: Panel, mask: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "X": panel.X[mask], "path": panel.path[mask], "y": panel.y[mask],
            "asset": panel.asset_i[mask], "horizon": panel.horizon_i[mask],
            "event": panel.event_i[mask], "cell": panel.cell[mask],
        }

    def fit(self, panel: Panel, mask: np.ndarray) -> TorchModel:
        tr = self._arrays(panel, mask)
        fit_ev, val_ev = _split_events(tr["event"], self.cfg.early_stop_frac)
        fit_m = np.isin(tr["event"], fit_ev)
        val_m = np.isin(tr["event"], val_ev)
        x_fit, x_val, self.x_med = _impute(tr["X"][fit_m], tr["X"][val_m])
        # Scale using the gradient rows only (the early-stopping slice must not set the scale).
        x_fit, x_val, self.x_mu, self.x_sd = _scale(x_fit, x_val)
        p_fit, p_val, self.p_med = _impute(tr["path"][fit_m], tr["path"][val_m])
        p_fit, p_val, self.p_mu, self.p_sd = _scale(p_fit, p_val)
        self.y_mu, self.y_sd = self._cell_scale(tr["cell"][fit_m], tr["y"][fit_m])
        y_fit = self._standardize(tr["cell"][fit_m], tr["y"][fit_m])
        y_val = self._standardize(tr["cell"][val_m], tr["y"][val_m])
        w_fit = _weights(tr["asset"][fit_m], tr["horizon"][fit_m]).astype(np.float32)
        n_assets = int(panel.asset_i.max()) + 1
        n_horizons = int(panel.horizon_i.max()) + 1
        self.nets = []
        for seed in self.cfg.seeds:
            torch.manual_seed(seed)
            net = self._new(x_fit.shape[1], n_assets, n_horizons)
            self._run(net, x_fit, p_fit, tr["asset"][fit_m], tr["horizon"][fit_m], y_fit, w_fit,
                      x_val, p_val, tr["asset"][val_m], tr["horizon"][val_m], y_val)
            self.nets.append(net)
        # Keep the training-row medians of the *unscaled* features for attribution baselines.
        self._train_X = tr["X"]
        self._train_path = tr["path"]
        self._train_asset = tr["asset"]
        return self

    def _cell_scale(self, cell: np.ndarray, y: np.ndarray,
                    ) -> tuple[dict[str, float], dict[str, float]]:
        mu, sd = {}, {}
        for c in np.unique(cell):
            v = y[cell == c]
            mu[str(c)] = float(np.mean(v))
            s = float(np.std(v))
            sd[str(c)] = s if s > 1e-8 else 1.0
        return mu, sd

    def _standardize(self, cell: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.empty_like(y, dtype=np.float32)
        for i, c in enumerate(cell):
            out[i] = (y[i] - self.y_mu[str(c)]) / self.y_sd[str(c)]
        return out

    def _new(self, n_feat: int, n_assets: int, n_horizons: int) -> torch.nn.Module:
        if self.kind == "qnet":
            return _QuantileNet(n_feat, n_assets, n_horizons, int(self.params["width"]),
                                self.cfg.emb_dim, self.cfg.dropout, len(QUANTILES))
        # The temporal CNN searches the number of channels. Its fully connected width stays 32.
        width = int(self.params["width"]) if self.kind == "mt_mlp" else 32
        return _MeanNet(
            n_feat, n_assets, n_horizons, width, self.cfg.emb_dim, self.cfg.dropout,
            int(self.params.get("channels", 8)), self.cfg.kernel, use_path=self.kind == "tcn",
        )

    def _run(self, net: torch.nn.Module, x_fit: np.ndarray, p_fit: np.ndarray, a_fit: np.ndarray,
             h_fit: np.ndarray, y_fit: np.ndarray, w_fit: np.ndarray, x_val: np.ndarray,
             p_val: np.ndarray, a_val: np.ndarray, h_val: np.ndarray, y_val: np.ndarray) -> None:
        opt = torch.optim.Adam(net.parameters(), lr=self.cfg.lr,
                               weight_decay=float(self.params["weight_decay"]))
        xf = torch.from_numpy(x_fit.astype(np.float32))
        pf = torch.from_numpy(p_fit.astype(np.float32))
        af = torch.from_numpy(a_fit.astype(np.int64))
        hf = torch.from_numpy(h_fit.astype(np.int64))
        yf = torch.from_numpy(y_fit.astype(np.float32))
        wf = torch.from_numpy(w_fit)
        xv = torch.from_numpy(x_val.astype(np.float32))
        pv = torch.from_numpy(p_val.astype(np.float32))
        av = torch.from_numpy(a_val.astype(np.int64))
        hv = torch.from_numpy(h_val.astype(np.int64))
        yv = torch.from_numpy(y_val.astype(np.float32))
        taus = torch.tensor(QUANTILES, dtype=torch.float32)
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.cfg.max_epochs):
            net.train()
            opt.zero_grad()
            if self.kind == "qnet":
                pred = net(xf, af, hf)
                err = yf.unsqueeze(1) - pred
                loss_row = torch.maximum(taus * err, (taus - 1.0) * err).mean(dim=1)
                loss = (loss_row * wf).mean()
            else:
                pred = net(xf, pf, af, hf)
                loss = (torch.nn.functional.huber_loss(pred, yf, reduction="none") * wf).mean()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip)
            opt.step()
            net.eval()
            with torch.no_grad():
                if self.kind == "qnet":
                    v = float(_pinball_loss(yv, net(xv, av, hv), taus))
                else:
                    v = float(torch.nn.functional.huber_loss(net(xv, pv, av, hv), yv))
            if v < best - 1e-6:
                best, wait = v, 0
                best_state = {k: p.clone() for k, p in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.cfg.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()

    def _prep(self, X: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        def fill(a: np.ndarray, med: np.ndarray) -> np.ndarray:
            out = a.copy()
            inds = np.where(np.isnan(out))
            out[inds] = np.take(med, inds[1])
            return out
        Xs = (fill(X, self.x_med) - self.x_mu) / self.x_sd
        Ps = (fill(path, self.p_med) - self.p_mu) / self.p_sd
        return Xs.astype(np.float32), Ps.astype(np.float32)

    def predict(self, panel: Panel, mask: np.ndarray) -> np.ndarray:
        return self.predict_rows(panel.X[mask], panel.path[mask], panel.asset_i[mask],
                                 panel.horizon_i[mask], panel.cell[mask])

    def predict_rows(self, X: np.ndarray, path: np.ndarray, asset: np.ndarray,
                     horizon: np.ndarray, cell: np.ndarray) -> np.ndarray:
        Xs, Ps = self._prep(X, path)
        xt = torch.from_numpy(Xs)
        pt = torch.from_numpy(Ps)
        at = torch.from_numpy(asset.astype(np.int64))
        ht = torch.from_numpy(horizon.astype(np.int64))
        acc = []
        with torch.no_grad():
            for net in self.nets:
                net.eval()
                if self.kind == "qnet":
                    acc.append(net(xt, at, ht).detach().numpy())
                else:
                    acc.append(net(xt, pt, at, ht).detach().numpy())
        pred = np.mean(acc, axis=0)
        sd = np.array([self.y_sd[str(c)] for c in cell])
        mu = np.array([self.y_mu[str(c)] for c in cell])
        if self.kind == "qnet":
            return np.asarray(pred * sd[:, None] + mu[:, None], dtype=float)
        return np.asarray(pred * sd + mu, dtype=float)

    def embedding(self, seed_index: int = 0) -> np.ndarray:
        net = self.nets[seed_index]
        emb = cast(torch.nn.Embedding, net.asset)
        return np.asarray(emb.weight.detach().numpy(), dtype=float)

    def asset_baseline(self, asset: np.ndarray, n_feat: int) -> tuple[np.ndarray, np.ndarray]:
        """Training mean of the raw features and of the path, within each asset."""
        bx = np.zeros((len(asset), n_feat))
        bp = np.zeros((len(asset), PATH_LEN))
        for a in np.unique(self._train_asset):
            rows = self._train_asset == a
            mx = np.nanmean(self._train_X[rows], axis=0)
            mp = np.nanmean(self._train_path[rows], axis=0)
            mx = np.where(np.isnan(mx), 0.0, mx)
            mp = np.where(np.isnan(mp), 0.0, mp)
            take = asset == a
            bx[take] = mx
            bp[take] = mp
        return bx, bp


class QuantileGBM:
    """One histogram gradient-boosting model per cell and quantile. Tabular features only."""

    name = "qgbm"

    def __init__(self, cfg: TrainCfg, max_depth: int, learning_rate: float, max_iter: int,
                 min_samples_leaf: int, quantiles: tuple[float, ...] = QUANTILES) -> None:
        self.cfg = cfg
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.quantiles = quantiles
        self.models: dict[tuple[str, float], Any] = {}
        self.medians: dict[str, np.ndarray] = {}

    def fit(self, panel: Panel, mask: np.ndarray) -> QuantileGBM:
        X, y, cell = panel.X[mask], panel.y[mask], panel.cell[mask]
        self.models, self.medians = {}, {}
        for c in np.unique(cell):
            rows = cell == c
            med = np.nanmedian(X[rows], axis=0)
            med = np.where(np.isnan(med), 0.0, med)
            self.medians[str(c)] = med
            Z = self._fill(X[rows], med)
            for tau in self.quantiles:
                m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=float(tau), max_depth=self.max_depth,
                    learning_rate=self.learning_rate, max_iter=self.max_iter,
                    min_samples_leaf=self.min_samples_leaf, random_state=0,
                )
                m.fit(Z, y[rows])
                self.models[(str(c), float(tau))] = m
        return self

    @staticmethod
    def _fill(X: np.ndarray, med: np.ndarray) -> np.ndarray:
        out = X.copy()
        inds = np.where(np.isnan(out))
        out[inds] = np.take(med, inds[1])
        return out

    def predict(self, panel: Panel, mask: np.ndarray, tau: float = 0.05) -> np.ndarray:
        return self.predict_rows(panel.X[mask], panel.cell[mask], tau)

    def predict_rows(self, X: np.ndarray, cell: np.ndarray, tau: float) -> np.ndarray:
        out = np.empty(len(cell), dtype=float)
        for c in np.unique(cell):
            rows = cell == c
            m = self.models[(str(c), float(tau))]
            out[rows] = m.predict(self._fill(X[rows], self.medians[str(c)]))
        return out
