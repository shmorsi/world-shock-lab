"""ml-study-v1 on synthetic data: scores, leakage, models, attribution, the final-run guard."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from wsl.mlstudy import evaluate as ev
from wsl.mlstudy.explain import embedding_agreement, grouped_shap
from wsl.mlstudy.metrics import fissler_ziegel, pinball, tail_verdict
from wsl.mlstudy.models import QUANTILES, QuantileGBM, TorchModel, TrainCfg
from wsl.mlstudy.panels import PATH_LEN, Panel, pre_event_changes
from wsl.shocks.windows import WindowSeries

NY = "America/New_York"


def test_pinball_known_values() -> None:
    y = np.array([1.0, -1.0])
    q = np.array([0.0, 0.0])
    np.testing.assert_allclose(pinball(y, q, 0.2), [0.2, 0.8])


def test_tail_verdict_wording() -> None:
    assert tail_verdict((-0.2, -0.01)) == "ML adds value on the 5% tail"
    assert tail_verdict((0.01, 0.2)) == "ML is worse on the 5% tail"
    assert tail_verdict((-0.1, 0.1)) == "No evidence ML adds value on the 5% tail"


def test_fissler_ziegel_prefers_the_true_normal() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=20_000)
    z = float(norm.ppf(0.05))
    var = np.full(len(y), z)
    es = np.full(len(y), -float(norm.pdf(z)) / 0.05)
    good = float(np.mean(fissler_ziegel(y, var, es, 0.05)))
    bad = float(np.mean(fissler_ziegel(y, np.zeros(len(y)), np.zeros(len(y)), 0.05)))
    shifted = float(np.mean(fissler_ziegel(y, var + 0.4, es + 0.4, 0.05)))
    assert good < bad
    assert good < shifted


def test_pre_event_path_excludes_the_announcement_print() -> None:
    dates = pd.bdate_range("2024-01-01", periods=40)
    ts = pd.DatetimeIndex([pd.Timestamp(d, tz=NY) + pd.Timedelta(hours=16) for d in dates])
    x = np.arange(40, dtype=float) ** 2
    series = WindowSeries("X", pd.DatetimeIndex(dates), x, ts, "bp", "16:00", None)
    before = pd.Series([pd.Timestamp(dates[10], tz=NY) + pd.Timedelta(hours=14)])
    after = pd.Series([pd.Timestamp(dates[10], tz=NY) + pd.Timedelta(hours=18)])
    # diff[k] = x[k+1] - x[k] = 2k + 1. A path ending at day 9 has last change diff[8] = 17.
    assert pre_event_changes(series, before, 5)[0, -1] == 17
    assert pre_event_changes(series, after, 5)[0, -1] == 19
    short = pd.Series([pd.Timestamp(dates[3], tz=NY) + pd.Timedelta(hours=14)])
    assert np.isnan(pre_event_changes(series, short, 5)).all()


def test_events_where_keeps_cells_the_first_asset_does_not_have() -> None:
    def frame(name: str, ids: list[str], years: list[int]) -> ev.CellFrame:
        idx = pd.Index(ids)
        return ev.CellFrame(
            name, name.split("|")[0], 1, pd.DataFrame(index=idx), pd.Series(1.0, index=idx),
            pd.Series(years, index=idx), pd.Series(["fomc"] * len(ids), index=idx),
        )

    frames = [frame("US2Y|h1", ["e1"], [2019]), frame("SPY|h1", ["e1", "e2"], [2019, 2019])]
    assert set(ev._events_where(frames, lambda y: y >= 2019)) == {"e1", "e2"}


def _panel(n_events: int = 80, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    n_assets, n_h = 2, 2
    event_i = np.repeat(np.arange(n_events), n_assets * n_h)
    year = np.repeat(np.full(n_events, 2010), n_assets * n_h)
    asset = np.tile(np.repeat(np.arange(n_assets), n_h), n_events)
    horizon = np.tile(np.arange(n_h), n_events * n_assets)
    shock = rng.normal(size=n_events)
    rows = n_events * n_assets * n_h
    x = np.zeros((rows, 4))
    x[:, 0] = np.repeat(shock, n_assets * n_h)
    y = (3.0 - asset) * x[:, 0] + rng.normal(scale=0.2, size=rows)
    path = rng.normal(scale=0.01, size=(rows, PATH_LEN))
    path[:, -1] = x[:, 0]
    cell = np.array([f"A{a}|h{h}" for a, h in zip(asset, horizon, strict=True)], dtype=object)
    return Panel(
        event_id=np.array([f"e{i}" for i in event_i], dtype=object),
        event_i=event_i, year=year, asset_i=asset, horizon_i=horizon, cell=cell,
        X=x, path=path, y=y, feature_names=["shock", "s1", "s2", "s3"],
    )


def _cfg(**over: object) -> TrainCfg:
    base = dict(lr=1e-2, max_epochs=60, patience=12, grad_clip=1.0, dropout=0.0, emb_dim=4,
                seeds=(0,), early_stop_frac=0.25, kernel=3)
    base.update(over)
    return TrainCfg(**base)  # type: ignore[arg-type]


def test_networks_beat_zero_on_a_linear_shock_and_quantiles_do_not_cross() -> None:
    panel = _panel()
    train = panel.event_i < 60
    test = panel.event_i >= 60
    zero = float(np.mean(np.abs(panel.y[test])))
    for kind, params in (("mt_mlp", {"width": 32, "weight_decay": 1e-3}),
                         ("tcn", {"channels": 8, "weight_decay": 1e-3})):
        pred = TorchModel(kind, _cfg(), **params).fit(panel, train).predict(panel, test)
        assert np.mean(np.abs(pred - panel.y[test])) < 0.6 * zero, kind
    qnet = TorchModel("qnet", _cfg(), width=32, weight_decay=1e-3).fit(panel, train)
    q = qnet.predict(panel, test)
    assert q.shape[1] == len(QUANTILES)
    assert np.all(np.diff(q, axis=1) >= -1e-5)


def test_scaler_ignores_the_early_stopping_slice() -> None:
    panel = _panel(40, seed=1)
    late = panel.event_i >= 30
    panel.X[:, 3] = np.nan
    panel.X[late, 3] = 1e9
    model = TorchModel("mt_mlp", _cfg(max_epochs=2, patience=1), width=8, weight_decay=0.0)
    model.fit(panel, np.ones(len(panel.y), dtype=bool))
    assert model.x_med[3] == 0.0


def test_quantile_gbm_runs_on_a_small_panel() -> None:
    panel = _panel(40, seed=2)
    train = panel.event_i < 30
    test = panel.event_i >= 30
    model = QuantileGBM(_cfg(), max_depth=2, learning_rate=0.05, max_iter=20,
                        min_samples_leaf=8, quantiles=(0.05,)).fit(panel, train)
    pred = model.predict(panel, test, 0.05)
    assert np.isfinite(pred).all()
    # The truth is symmetric around zero, so a 5% forecast should sit below the outcomes.
    assert float(np.mean(panel.y[test] <= pred)) < 0.5


def test_grouped_shap_is_efficient() -> None:
    x = np.array([[1.0, 3.0, 0.0], [0.5, -1.0, 4.0]])
    path = np.zeros((2, 2))
    base_x = np.zeros_like(x)
    base_p = np.zeros_like(path)

    def predict(xm: np.ndarray, pm: np.ndarray) -> np.ndarray:
        return xm[:, 0] + 2.0 * xm[:, 1] + pm[:, 1]

    groups = [
        ("surprise", "cols", np.array([0])),
        ("macro_state", "cols", np.array([1])),
        ("return_path", "path", np.array([], dtype=int)),
    ]
    values, gap = grouped_shap(predict, x, path, base_x, base_p, groups)
    assert gap < 1e-9
    np.testing.assert_allclose(values[0], [1.0, 6.0, 0.0])


def test_embedding_ari_of_separated_groups_is_one() -> None:
    groups = {
        "yields": ["US2Y", "CA2Y"],
        "fx": ["USDCAD"],
        "equity": ["SPY", "TSX", "CA_BANKS_EW6", "XEG"],
        "oil": ["WTI"],
    }
    vectors = np.array([
        [5.0, 0, 0, 0], [5.1, 0, 0, 0],
        [0, 5.0, 0, 0],
        [0, 0, 5.0, 0], [0, 0, 5.1, 0], [0, 0, 4.9, 0], [0, 0, 5.2, 0],
        [0, 0, 0, 5.0],
    ], dtype=float)
    out = embedding_agreement(vectors, groups, permutations=300, seed=1)
    assert out["ari"] == 1.0
    assert float(out["ari_permutation_p"]) < 0.05  # type: ignore[arg-type]


def test_study_note_matches_the_committed_results() -> None:
    from wsl.mlstudy.report import CARD, DOC, WEB, _load, _md, _web, model_card

    doc = _load()
    assert DOC.read_text() == _md(doc)
    assert json.loads(WEB.read_text()) == _web(doc)
    assert CARD.read_text() == model_card(doc)


def test_final_run_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    monkeypatch.setattr(ev, "_latest_tag", lambda: ("ml-study-v1", "0" * 64))
    monkeypatch.setattr(ev, "file_sha256", lambda _p: "f" * 64)
    with pytest.raises(ev.ProtocolGuardError, match="does not match"):
        ev.check_final_allowed(None)
    monkeypatch.setattr(ev, "file_sha256", lambda _p: "0" * 64)
    monkeypatch.setattr(ev, "git_sha", lambda: "abc-dirty")
    with pytest.raises(ev.ProtocolGuardError, match="uncommitted"):
        ev.check_final_allowed(None)
    monkeypatch.setattr(ev, "git_sha", lambda: "abc")
    monkeypatch.setattr(ev, "VAL_PATH", tmp_path / "missing.json")
    with pytest.raises(ev.ProtocolGuardError, match="validation"):
        ev.check_final_allowed(None)
