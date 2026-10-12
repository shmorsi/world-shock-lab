"""LP-IV pieces: 2SLS against the textbook formula, Anderson-Rubin sets against brute force."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wsl.estimation.irf import z_value
from wsl.estimation.lp import event_lp, lp
from wsl.estimation.lpiv import ar_set, iv_just_identified


def _iv_data(n: int = 400, strength: float = 1.0, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    W = np.column_stack([np.ones(n), rng.standard_normal(n)])
    z = rng.standard_normal(n)
    common = rng.standard_normal(n)  # endogeneity: enters both x and y
    x = strength * z + 0.5 * W[:, 1] + common + rng.standard_normal(n)
    y = 2.0 * x - 1.0 * W[:, 1] + 2.0 * common + rng.standard_normal(n)
    return y, x, z, W


def test_iv_matches_textbook_2sls() -> None:
    y, x, z, W = _iv_data()
    beta, *_ = iv_just_identified(y, x, z, W, cov_type="HC0")
    Z, X = np.column_stack([z, W]), np.column_stack([x, W])
    b2sls = np.linalg.solve(Z.T @ X, Z.T @ y)
    assert beta == pytest.approx(b2sls[0], rel=1e-10)
    # and OLS is biased upward here (endogeneity), IV is not far from the truth of 2.0
    assert abs(beta - 2.0) < 0.3


def test_iv_beta_equals_reduced_form_over_first_stage() -> None:
    y, x, z, W = _iv_data(seed=3)
    beta, _, _, pi, _, _, a = iv_just_identified(y, x, z, W)
    assert beta == pytest.approx(a / pi, rel=1e-10)


def test_effective_f_is_robust_first_stage_wald() -> None:
    _, x, z, W = _iv_data(seed=4)
    *_, f_eff, _joint, _ = iv_just_identified(np.zeros_like(x), x, z, W, cov_type="HC0")
    from wsl.estimation.ols import ols
    fs = ols(x, np.column_stack([z, W]), cov_type="HC0")
    assert f_eff == pytest.approx((fs.params[0] / fs.se[0]) ** 2, rel=1e-8)


@settings(max_examples=200, deadline=None)
@given(st.floats(-3, 3), st.floats(-3, 3), st.floats(0.01, 2), st.floats(0.01, 2),
       st.floats(-0.9, 0.9), st.sampled_from([0.68, 0.90]))
def test_ar_set_matches_grid_inversion(a: float, b: float, sa: float, sb: float, rho: float,
                                       level: float) -> None:
    V = np.array([[sa**2, rho * sa * sb], [rho * sa * sb, sb**2]])
    s = ar_set(a, b, V, level)
    c = z_value(level) ** 2
    for b0 in np.linspace(-20, 20, 401):
        stat = (a - b0 * b) ** 2 / (V[0, 0] - 2 * b0 * V[0, 1] + b0**2 * V[1, 1])
        if abs(stat - c) > 1e-6:  # skip points on the boundary
            assert s.contains(float(b0)) == (stat <= c), (s, b0, stat, c)


def test_ar_set_shapes() -> None:
    V = np.diag([0.01, 0.01])
    strong = ar_set(2.0, 1.0, V, 0.90)
    assert strong.kind == "interval" and strong.contains(2.0)
    weak = ar_set(0.5, 0.01, V, 0.90)
    assert weak.kind in ("two_rays", "real_line")


def test_lp_change_and_level_give_same_shock_coefficient() -> None:
    rng = np.random.default_rng(5)
    idx = pd.RangeIndex(300)
    s = pd.Series(rng.standard_normal(300), index=idx)
    y = pd.Series(np.cumsum(rng.standard_normal(300)) + 0.5 * s, index=idx)
    a = lp(y, s, range(4), dependent="change", cov_type="HC1")
    b = lp(y, s, range(4), dependent="level", cov_type="HC1")
    np.testing.assert_allclose(a.coef, b.coef, rtol=1e-9)  # y_{t-1} is among the controls


def test_event_lp_reports_one_regression_per_horizon() -> None:
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"s": rng.standard_normal(80)})
    df["d0"] = 2 * df["s"] + rng.standard_normal(80) * 0.1
    df["d1"] = 3 * df["s"] + rng.standard_normal(80) * 0.1
    irf = event_lp(df, {0: "d0", 1: "d1"}, "s")
    assert irf.horizons == (0, 1)
    np.testing.assert_allclose(irf.coef, [2, 3], atol=0.05)
    frame = irf.to_frame()
    assert {"lo68", "hi68", "lo90", "hi90"} <= set(frame.columns)
    assert (frame["lo90"] <= frame["lo68"]).all()
