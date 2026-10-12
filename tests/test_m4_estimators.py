"""M4 estimators on simulated data with known answers."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from wsl.estimation import bvar
from wsl.estimation.multiple import bh_qvalues
from wsl.estimation.ols import ols
from wsl.estimation.simulate import textbook_var
from wsl.estimation.state import state_lp


def test_bh_matches_statsmodels() -> None:
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(size=40), rng.uniform(0, 0.01, size=10)])
    ref = multipletests(p, alpha=0.10, method="fdr_bh")[1]
    np.testing.assert_allclose(bh_qvalues(p), ref)


def test_cluster_covariance_matches_statsmodels() -> None:
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(60), 8)
    X = np.column_stack([np.ones(480), rng.normal(size=480)])
    y = X @ [1.0, 2.0] + rng.normal(size=60)[g] + rng.normal(size=480)
    ours = ols(y, X, cov_type="CL", groups=g)
    ref = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    np.testing.assert_allclose(ours.cov, ref.cov_params(), rtol=1e-8)


def _state_data(rng: np.random.Generator, n: int, H: int, c: float):  # type: ignore[no-untyped-def]
    s = rng.standard_t(4, n)
    R = (rng.random(n) < 0.4).astype(float)[:, None]
    noise = np.cumsum(rng.normal(size=(n, H + 1)), axis=1)  # cumulative changes: correlated
    effect = (1.0 + c * R[:, 0])[:, None] * s[:, None] * np.linspace(1, 1.5, H + 1)
    return noise + effect, s, R


def test_state_test_has_correct_size_under_the_null() -> None:
    rng = np.random.default_rng(2)
    rejects_joint = rejects_head = 0
    R_mc = 120
    for _ in range(R_mc):
        Y, s, R = _state_data(rng, 250, 10, c=0.0)
        res = state_lp(Y, s, R, ["high"], None, range(11), headline=5, boot=199,
                       seed=int(rng.integers(1 << 31)))
        rejects_joint += res.p_joint < 0.10
        rejects_head += res.p_headline < 0.10
    assert 0.03 <= rejects_joint / R_mc <= 0.19
    assert 0.03 <= rejects_head / R_mc <= 0.19


def test_state_test_detects_a_real_difference() -> None:
    rng = np.random.default_rng(3)
    Y, s, R = _state_data(rng, 400, 10, c=1.0)  # effect doubles in the "high" regime
    res = state_lp(Y, s, R, ["high"], None, range(11), headline=5, boot=299)
    assert res.p_joint < 0.01 and res.p_headline < 0.01
    np.testing.assert_allclose(res.interaction[:, 0], np.linspace(1, 1.5, 11), atol=0.35)
    assert res.n_by_regime["high"] + res.n_by_regime["baseline"] == res.n_events


def test_bvar_with_loose_prior_matches_ols_var() -> None:
    var = textbook_var()
    Y, _ = var.simulate(800, np.random.default_rng(4))
    m = bvar.fit(Y, 2, lam=1e4)
    X = bvar.lag_matrix(Y, 2)
    B_ols = np.linalg.lstsq(X, Y[2:], rcond=None)[0]
    np.testing.assert_allclose(m.B, B_ols, atol=1e-4)


def test_proxy_identification_recovers_true_relative_irf_in_large_sample() -> None:
    var = textbook_var()
    rng = np.random.default_rng(5)
    Y, eps = var.simulate(60000, rng)  # sampling error ~0.02 at this size
    z = eps[:, 0] + rng.normal(size=len(eps))
    m = bvar.fit(Y, 2, lam=1e4)
    pid = bvar.identify(m.U, z[2:])
    est = bvar.irf(m, pid.impact, 12)
    truth = var.irf(12)[:, :, 0] / var.irf(12)[0, 0, 0]
    np.testing.assert_allclose(est, truth, atol=0.03)
    assert pid.first_stage_f > 1000


def test_proxy_bvar_bootstrap_coverage_matches_documented_behaviour() -> None:
    """Moving-block bootstrap bands on a shrunk, over-parameterized BVAR (p = 4 on a VAR(2),
    the pre-registered tightness 0.2). Documented in docs/SIMULATION.md: percentile bands for
    persistent VARs under-cover (Kilian 1998), and shrinkage biases the normalized variable's
    own path toward zero, so its bands under-cover more. Responses of the other variables,
    which are the ones compared with LPs, stay close to the M2 standard. This test pins that
    behaviour so a regression is caught; it does not claim nominal coverage."""
    from wsl.estimation import montecarlo as mc

    res = mc.proxy_bvar_mc(R=40, draws=150)
    assert res["coverage90_other_variables"] >= 0.80
    assert res["coverage90_own_path"] >= 0.60
    assert res["max_abs_bias_other_over_peak"] <= 0.10


@pytest.mark.parametrize("T", [100, 400, 1600])
def test_block_length_rule(T: int) -> None:
    assert bvar.default_block_length(T) == round(5.03 * T ** 0.25)
