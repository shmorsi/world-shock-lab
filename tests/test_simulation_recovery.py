"""Simulation recovery (M2 acceptance criteria, docs/PLAN.md §6).

Data are drawn from a structural VAR with known impulse responses. The estimators must recover
them: bias within a fixed tolerance, and 90% bands covering the truth 85-95% of the time on
average across horizons (no single horizon below 80%). Seeds are fixed, so results are
deterministic; ``uv run wsl validate simulation`` writes the same numbers to docs/SIMULATION.md.

Known, documented finite-sample effects (not hidden by the tolerances):
* a small downward bias of the policy variable's response to its own shock (persistent
  dynamics, T = 500), under 2% of the peak response;
* time-series LPs with zero-filled sparse, fat-tailed shocks under-cover (about 86% for a
  nominal 90%); the pre-registered event-sample LP with HC3 does not. See
  test_zero_filled_daily_lp_undercovers_relative_to_event_sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from wsl.estimation import montecarlo as mc

TOL_PEAK = 0.05  # bias tolerance: 5% of the peak true response ...
TOL_MCSE = 4.0  # ... plus 4 Monte Carlo standard errors


def assert_recovers(res: mc.MCResult) -> None:
    peak = float(np.max(np.abs(res.truth))) or 1.0
    allowed = TOL_PEAK * peak + TOL_MCSE * res.mc_se
    assert np.all(np.abs(res.bias) <= allowed), (res.name, res.bias, allowed)
    assert 0.85 <= res.coverage90.mean() <= 0.95, (res.name, res.coverage90)
    assert res.coverage90.min() >= 0.80, (res.name, res.coverage90)
    assert 0.60 <= res.coverage68.mean() <= 0.76, (res.name, res.coverage68)


@pytest.fixture(scope="module")
def dense_nw() -> mc.MCResult:
    return mc.dense("NW")


@pytest.fixture(scope="module")
def sparse_event() -> mc.MCResult:
    return mc.sparse("event_HC3")


def test_dense_lp_newey_west_recovers_irf(dense_nw: mc.MCResult) -> None:
    assert_recovers(dense_nw)


def test_dense_lp_lag_augmented_hc_recovers_irf() -> None:
    # Montiel Olea & Plagborg-Møller (2021): with lag augmentation, HC SEs suffice.
    assert_recovers(mc.dense("HC1"))


@pytest.mark.parametrize("outcome", [0, 2])
def test_dense_lp_other_outcomes(outcome: int) -> None:
    assert_recovers(mc.dense("NW", outcome=outcome, R=150))


def test_sparse_event_sample_lp_recovers_irf(sparse_event: mc.MCResult) -> None:
    assert_recovers(sparse_event)
    assert 200 <= sparse_event.extra["events per draw (mean)"] <= 280


def test_zero_filled_daily_lp_undercovers_relative_to_event_sample(
        sparse_event: mc.MCResult) -> None:
    """Why the pre-registered primary daily spec is the event-sample LP (protocol §9)."""
    zero_filled = mc.sparse("zerofill_NW")
    assert np.all(np.abs(zero_filled.bias) <= TOL_PEAK * np.max(np.abs(zero_filled.truth))
                  + TOL_MCSE * zero_filled.mc_se)  # still unbiased...
    assert zero_filled.coverage90.mean() < sparse_event.coverage90.mean()  # ...but under-covers


def test_lpiv_strong_instrument_recovers_relative_irf() -> None:
    res = mc.lpiv(strength=1.0)
    assert_recovers(res)
    assert res.extra["median effective F"] > 100
    assert res.extra["Anderson-Rubin coverage 90 (mean)"] >= 0.85


def test_lpiv_weak_instrument_is_flagged_and_anderson_rubin_stays_valid() -> None:
    res = mc.lpiv(strength=0.05, R=200)
    assert res.extra["median effective F"] < 5  # the diagnostic flags it
    assert res.extra["Anderson-Rubin coverage 90 (mean)"] >= 0.85
    assert res.extra["Anderson-Rubin coverage 90 (min h)"] >= 0.80


def test_placebo_shock_gives_zero_response_with_correct_size() -> None:
    res = mc.placebo()
    assert_recovers(res)  # truth is zero: bias ~ 0 and the 90% band contains 0 ~ 90% of draws


def test_estimation_error_shrinks_with_sample_size() -> None:
    small, large = mc.consistency()
    assert large < 0.5 * small
