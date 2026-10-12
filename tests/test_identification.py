"""M3 identification tools on synthetic data with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wsl.analysis.hetero import control_times, rigobon_sack, variance_ratio
from wsl.analysis.placebo import placebo_test
from wsl.analysis.replication import CLAIMS, Claim, ClaimResult
from wsl.data.transforms import ew_return_index
from wsl.estimation.irf import IRF
from wsl.shocks.windows import make_series

NY = "America/New_York"


def test_variance_ratio_recovers_known_ratio() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    control = rng.normal(0, 1.0, n)
    event = rng.normal(0, 1.0, n) + rng.normal(0, 1.5, n)  # announcement adds variance 2.25
    vr = variance_ratio(event, control, boot=500)
    assert vr.ci90[0] < 3.25 < vr.ci90[1]
    assert vr.signal_share == pytest.approx(1 - 1 / vr.ratio)


def test_variance_ratio_ci_contains_one_without_announcement_effect() -> None:
    rng = np.random.default_rng(2)
    vr = variance_ratio(rng.normal(size=300), rng.normal(size=300), boot=1000)
    assert vr.ci90[0] < 1.0 < vr.ci90[1]


def test_rigobon_sack_beats_ols_under_simultaneity() -> None:
    """i = m + u, y = 2 m + 3 u + e: OLS on event days is biased by the common news u;
    Rigobon-Sack uses the extra policy variance on event days and recovers 2."""
    rng = np.random.default_rng(3)
    n = 3000

    def draw(policy_sd: float) -> tuple[np.ndarray, np.ndarray]:
        m, u, e = rng.normal(0, policy_sd, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
        return m + u, 2 * m + 3 * u + e

    i_e, y_e = draw(2.0)
    i_c, y_c = draw(0.0)
    rs = rigobon_sack(i_e, y_e, i_c, y_c, boot=400)
    assert rs.ci90[0] < 2.0 < rs.ci90[1]
    assert abs(rs.ols_event_days - 2.0) > 0.1  # OLS is biased toward 3


def test_control_times_drop_controls_near_other_announcements() -> None:
    ev = pd.Series(pd.to_datetime(["2024-03-13 10:00", "2024-04-17 10:00"]).tz_localize(NY))
    others = pd.Series(pd.to_datetime(["2024-03-06 14:00"]).tz_localize(NY))  # a week before #1
    ctrl = control_times(ev, others)
    assert pd.isna(ctrl.iloc[0])
    assert ctrl.iloc[1] == pd.Timestamp("2024-04-10 10:00", tz=NY)


def _random_walk(n: int, seed: int, daily_sd: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, daily_sd, n))), index=dates)


def _events(series: pd.Series, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(np.arange(10, len(series) - 30), size=k, replace=False))
    ts = pd.Series(series.index[pos]).dt.tz_localize(NY) + pd.Timedelta(hours=10)
    return pd.DataFrame({"ts": ts.to_numpy(), "s": rng.normal(size=k)}, index=pos)


def test_placebo_does_not_reject_an_unrelated_shock() -> None:
    price = _random_walk(3000, 4)
    s = make_series("P", price, "16:00", NY, "logpct")
    events = _events(price, 120, 5)
    from wsl.analysis.eventstudy import event_lp  # same estimator the real runs use
    from wsl.shocks.windows import window_changes
    w = window_changes(s, events["ts"], 5).join(events[["s"]])
    irf = event_lp(w, {0: "d0", 5: "d5"}, "s", [f"pre{i}" for i in range(1, 6)])
    res = placebo_test(s, events, ["s"], {0: irf.coef[0], 5: irf.coef[1]}, draws=300)
    assert all(r.p_value > 0.05 for r in res)
    assert all(abs(r.placebo_mean) < 3 * r.placebo_sd / np.sqrt(300) + 0.05 for r in res)


def test_placebo_rejects_a_real_effect() -> None:
    price = _random_walk(3000, 6, daily_sd=0.002)  # quiet market, so the effect is sharp
    events = _events(price, 120, 7)
    logp = np.log(price.to_numpy())
    idx = price.index.get_indexer(pd.DatetimeIndex(events["ts"].dt.tz_localize(None).dt.normalize()))
    for j, sh in zip(idx, events["s"], strict=True):  # each event moves the price by 0.5% per unit
        logp[j:] += 0.005 * sh
    moved = pd.Series(np.exp(logp), index=price.index)
    s = make_series("P", moved, "16:00", NY, "logpct")
    from wsl.estimation.lp import event_lp
    from wsl.shocks.windows import window_changes
    w = window_changes(s, events["ts"], 0).join(events[["s"]])
    irf = event_lp(w, {0: "d0"}, "s", [f"pre{i}" for i in range(1, 6)])
    assert irf.coef[0] == pytest.approx(0.5, abs=0.1)
    res = placebo_test(s, events, ["s"], {0: irf.coef[0]}, draws=300)
    assert res[0].p_value < 0.01


def test_ew_return_index() -> None:
    idx = pd.bdate_range("2024-01-01", periods=5)
    a = pd.Series([100, 110, 121, np.nan, 133.1], index=idx)
    b = pd.Series([50, 55, 60.5, 66.55, 73.205], index=idx)
    out = ew_return_index(pd.DataFrame({"a": a, "b": b}), min_inputs=2)
    np.testing.assert_allclose(out.iloc[:3], [100, 110, 121])
    assert np.isnan(out.iloc[3])  # only one member that day
    assert out.iloc[4] == pytest.approx(121 * 1.1)  # a: 121 -> 133.1 (+10%), b: +10%


def _fake_result(coef: float, se: float, expected: int) -> ClaimResult:
    c = Claim("X", "src", "claim", "fomc", "s", "y", 0, expected)
    irf = IRF((0,), np.array([coef]), np.array([se]), (100,), "event_lp", "HC3",
              "identified_causal")
    return ClaimResult(c, irf)


@pytest.mark.parametrize(("coef", "se", "expected", "verdict"), [
    (1.0, 0.1, +1, "replicated"), (1.0, 1.0, +1, "same sign"),
    (-1.0, 0.1, +1, "opposite"), (-1.0, 1.0, +1, "opposite (ns)"),
    (-1.0, 0.1, -1, "replicated"),
])
def test_verdict_rules(coef: float, se: float, expected: int, verdict: str) -> None:
    assert _fake_result(coef, se, expected).verdict == verdict


def test_claims_are_well_formed() -> None:
    ids = [c.id for c in CLAIMS]
    assert len(ids) == len(set(ids))
    assert all(c.expected in (-1, 1) for c in CLAIMS)
    for c in CLAIMS:
        if c.id.endswith("r"):  # reference rows must pair with a main claim
            assert c.id[:-1] in ids
