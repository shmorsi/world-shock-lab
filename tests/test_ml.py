"""M5 machinery on synthetic data: metrics, leakage guards, models, the final-run guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wsl.ml import evaluate as ev
from wsl.ml import models
from wsl.ml.dataset import SHOCK_COLS, STATE_COLS, _value_before
from wsl.ml.metrics import (
    bootstrap_difference,
    cell_metrics,
    interval_score,
    pooled_scaled_mae,
    verdict,
)
from wsl.shocks.windows import make_series

NY = "America/New_York"


def test_interval_score_known_values() -> None:
    y = np.array([0.0, 2.0, -3.0])
    s = interval_score(y, np.full(3, -1.0), np.full(3, 1.0), alpha=0.1)
    np.testing.assert_allclose(s, [2.0, 2.0 + 20 * 1.0, 2.0 + 20 * 2.0])


def test_cell_metrics() -> None:
    y = np.array([1.0, -2.0, 3.0, 0.1])
    pred = np.array([1.0, 2.0, 2.0, 0.0])
    m = cell_metrics(y, pred, np.full(4, 0.5), np.full(4, 1.5), zero_mae_train=2.0,
                     p25_abs_train=0.5)
    assert m["mae"] == pytest.approx((0 + 4 + 1 + 0.1) / 4)
    assert m["scaled_mae"] == pytest.approx(m["mae"] / 2.0)
    assert m["directional_accuracy"] == pytest.approx(2 / 3)  # |y| > 0.5: three events
    assert m["coverage90"] == pytest.approx(3 / 4)
    assert m["oos_r2_vs_zero"] == pytest.approx(1 - (0 + 16 + 1 + 0.01) / (1 + 4 + 9 + 0.01))


def test_pooled_metric_weights_cells_equally() -> None:
    err = pd.DataFrame({"a": [1.0, 1.0, np.nan], "b": [4.0, 4.0, 4.0]})
    assert pooled_scaled_mae(err, pd.Series({"a": 1.0, "b": 2.0})) == pytest.approx(1.5)


def test_bootstrap_difference_is_deterministic_and_centred() -> None:
    rng = np.random.default_rng(0)
    a = pd.DataFrame(rng.exponential(1.0, (200, 3)), columns=list("xyz"))
    b = a * 1.1
    scale = pd.Series(1.0, index=list("xyz"))
    p1, ci1, _ = bootstrap_difference(a, b, scale, 500, 7)
    p2, ci2, _ = bootstrap_difference(a, b, scale, 500, 7)
    assert (p1, ci1) == (p2, ci2)
    assert ci1[1] < 0 and verdict(ci1) == "ML adds value"
    assert verdict((-0.1, 0.1)) == "No evidence ML adds value"
    assert verdict((0.05, 0.2)) == "ML is worse"


def test_state_features_use_only_data_before_the_announcement() -> None:
    dates = pd.bdate_range("2024-01-01", periods=30)
    s = make_series("X", pd.Series(np.arange(30, dtype=float), index=dates), "16:00", NY, "bp")
    t = pd.Series([pd.Timestamp(dates[10], tz=NY) + pd.Timedelta(hours=14)])  # day 10, 14:00
    assert _value_before(s, t)[0] == 100 * 9  # day 9's close, not day 10's 16:00 close
    assert _value_before(s, t, lag=3)[0] == 100 * 6


def _synthetic(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    fam = rng.choice(["fomc", "boc", "opec"], n)
    ev_ = pd.DataFrame({"family": fam})
    for f, cols in SHOCK_COLS.items():
        for c in cols:
            ev_[c] = np.where(fam == f, rng.normal(size=n), 0.0)
    for c in STATE_COLS:
        ev_[c] = rng.normal(size=n)
    X = models.base_frame(ev_, pd.Series(rng.uniform(0.5, 2, n)))
    y = 2.0 * X["fomc_FF4"] - 1.0 * X["boc_d_ca2y"] + 0.5 * X["opec_Front"] \
        + 0.3 * rng.normal(size=n)
    return X, y.to_numpy()


def test_linear_models_recover_a_linear_truth() -> None:
    X, y = _synthetic()
    for name in ("lp", "state", "ridge"):
        m = models.make(name).fit(X.iloc[:300], y[:300])
        err = np.mean(np.abs(m.predict(X.iloc[300:]) - y[300:]))
        assert err < 0.35, name  # noise MAE is about 0.24


def test_flexible_models_beat_zero() -> None:
    X, y = _synthetic(seed=1)
    zero_err = np.mean(np.abs(y[300:]))
    for name, params in (("gbm", None), ("mlp", {"width": 16, "weight_decay": 1e-3, "l1": 0.0})):
        m = models.make(name, params).fit(X.iloc[:300], y[:300])
        assert np.mean(np.abs(m.predict(X.iloc[300:]) - y[300:])) < 0.7 * zero_err, name


def test_preprocessing_uses_training_rows_only() -> None:
    X, _ = _synthetic(50)
    X.loc[X.index[:5], "vix"] = np.nan
    prep = models.Prep().fit(X.iloc[:40])
    test = X.iloc[40:].copy()
    test["vix"] = 1e6  # extreme test values must not move the training statistics
    before = prep.mean.copy()
    prep.transform(test)
    pd.testing.assert_series_equal(prep.mean, before)
    assert "vix__missing" in prep.transform(X.iloc[:5]).columns


def test_mlp_is_reproducible() -> None:
    X, y = _synthetic(200, seed=2)
    p = {"width": 16, "weight_decay": 1e-3, "l1": 1e-3}
    a = models.make("mlp", p).fit(X, y).predict(X.iloc[:10])
    b = models.make("mlp", p).fit(X, y).predict(X.iloc[:10])
    np.testing.assert_allclose(a, b)


def test_final_run_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
                         ) -> None:
    monkeypatch.setattr(ev, "_latest_protocol_tag", lambda: ("protocol-v9", "0" * 64))
    with pytest.raises(ev.ProtocolGuardError, match="does not match"):
        ev.check_final_allowed(None)
    monkeypatch.setattr(ev, "file_sha256", lambda _p: "0" * 64)
    monkeypatch.setattr(ev, "git_sha", lambda: "abc-dirty")
    with pytest.raises(ev.ProtocolGuardError, match="uncommitted"):
        ev.check_final_allowed(None)
    monkeypatch.setattr(ev, "git_sha", lambda: "abc")
    monkeypatch.setattr(ev, "RESULTS", tmp_path)  # type: ignore[arg-type]
    with pytest.raises(ev.ProtocolGuardError, match="validation phase"):
        ev.check_final_allowed(None)
    (tmp_path / "validation.json").write_text("{}")  # type: ignore[operator]
    final_dir = tmp_path / "final"  # type: ignore[operator]
    final_dir.mkdir()
    (final_dir / "20260101T000000Z.json").write_text("{}")
    monkeypatch.setattr(ev, "FINAL_DIR", final_dir)
    with pytest.raises(ev.ProtocolGuardError, match="used once"):
        ev.check_final_allowed(None)
    assert ev.check_final_allowed("documented reason") == "protocol-v9"
