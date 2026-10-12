"""OLS and robust covariances: cross-checks against statsmodels, plus property tests."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from hypothesis import given, settings
from hypothesis import strategies as st

from wsl.estimation.ols import newey_west_meat, ols


def _data(n: int = 300, k: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    e = rng.standard_normal(n) * (1 + np.abs(X[:, 1]))  # heteroskedastic
    e = np.convolve(e, [1.0, 0.6, 0.3], mode="same")  # and autocorrelated
    return X @ np.arange(1, k + 1) + e, X


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC3"])
def test_hc_matches_statsmodels(cov_type: str) -> None:
    y, X = _data()
    ours = ols(y, X, cov_type=cov_type)  # type: ignore[arg-type]
    ref = sm.OLS(y, X).fit(cov_type=cov_type)
    np.testing.assert_allclose(ours.params, ref.params, rtol=1e-10)
    np.testing.assert_allclose(ours.cov, ref.cov_params(), rtol=1e-8)


@pytest.mark.parametrize("lags", [0, 1, 5, 12])
@pytest.mark.parametrize("correction", [False, True])
def test_newey_west_matches_statsmodels(lags: int, correction: bool) -> None:
    y, X = _data()
    ours = ols(y, X, cov_type="NW", nw_lags=lags, df_correction=correction)
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags,
                                                     "use_correction": correction})
    np.testing.assert_allclose(ours.cov, ref.cov_params(), rtol=1e-8)


def test_newey_west_zero_lags_equals_hc0() -> None:
    y, X = _data()
    np.testing.assert_allclose(ols(y, X, cov_type="NW", nw_lags=0).cov,
                               ols(y, X, cov_type="HC0").cov)


def test_missing_rows_are_dropped_listwise_and_counted() -> None:
    y, X = _data(n=100)
    y[3] = np.nan
    X[10, 1] = np.nan
    r = ols(y, X)
    assert r.nobs == 98
    keep = np.ones(100, bool)
    keep[[3, 10]] = False
    np.testing.assert_allclose(r.params, ols(y[keep], X[keep]).params)


def test_needs_more_observations_than_regressors() -> None:
    with pytest.raises(ValueError):
        ols(np.ones(3), np.ones((3, 3)))


@settings(max_examples=40, deadline=None)
@given(st.floats(min_value=-50, max_value=50).filter(lambda c: abs(c) > 1e-3),
       st.integers(min_value=0, max_value=6))
def test_scaling_y_scales_coef_and_se(c: float, lags: int) -> None:
    y, X = _data(n=120, seed=1)
    a = ols(y, X, cov_type="NW", nw_lags=lags)
    b = ols(c * y, X, cov_type="NW", nw_lags=lags)
    np.testing.assert_allclose(b.params, c * a.params, rtol=1e-9)
    np.testing.assert_allclose(b.se, abs(c) * a.se, rtol=1e-9)


@settings(max_examples=40, deadline=None)
@given(st.floats(min_value=0.01, max_value=100))
def test_scaling_a_regressor_rescales_its_coefficient(c: float) -> None:
    y, X = _data(n=120, seed=2)
    X2 = X.copy()
    X2[:, 1] *= c
    a, b = ols(y, X, cov_type="HC3"), ols(y, X2, cov_type="HC3")
    assert b.params[1] == pytest.approx(a.params[1] / c, rel=1e-9)
    assert b.se[1] == pytest.approx(a.se[1] / c, rel=1e-9)


@settings(max_examples=30, deadline=None)
@given(st.integers(min_value=0, max_value=30), st.integers(min_value=0, max_value=10**6))
def test_newey_west_meat_is_positive_semidefinite(lags: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((80, 4)).cumsum(axis=0) * 0.1  # persistent scores
    eig = np.linalg.eigvalsh(newey_west_meat(scores, lags))
    assert eig.min() >= -1e-9 * max(1.0, abs(eig.max()))
