"""Scenario engine: arithmetic, uncertainty, labels and warnings on a small known catalog."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wsl.scenarios.engine import LINEARITY, Z90, compose

FIXTURES = Path(__file__).parent / "fixtures"


def catalog() -> dict:  # type: ignore[type-arg]
    def shock(i: str, label: str, per: float, unit: str, lo: float, hi: float) -> dict:  # type: ignore[type-arg]
        return {"id": i, "name": i.upper(), "label": label, "per": per, "unit": unit,
                "distribution": {"min": lo, "max": hi}}

    return {
        "horizons": [0, 1],
        "shocks": [shock("fed_mp", "identified_causal", 10, "bp", -40, 17),
                   shock("fed_cbi", "identified_causal", 10, "bp", -20, 20),
                   shock("opec", "identified_causal", 1, "%", -5, 5),
                   shock("gpr", "reduced_form", 1, "sd", -6, 9)],
        "outcomes": [{"id": "Y"}],
        "irfs": {
            "fed_mp": {"Y": {"coef": [2.0, 3.0], "se": [0.5, 1.0], "cov_partner": [0.1, 0.2]}},
            "fed_cbi": {"Y": {"coef": [1.0, 1.0], "se": [0.4, 0.4], "cov_partner": [0.1, 0.2]}},
            "opec": {"Y": {"coef": [-1.0, 0.0], "se": [0.3, 0.3], "cov_partner": [0.0, 0.0]}},
            "gpr": {"Y": {"coef": [0.5, 0.5], "se": [0.1, 0.1], "cov_partner": [0.0, 0.0]}},
        },
    }


def test_single_shock_scales_linearly() -> None:
    r = compose(catalog(), {"fed_mp": 15}, 0)  # inside the observed range (-40 to 17)
    y = r.outcomes[0]
    assert y.mean == pytest.approx(3.0) and y.se == pytest.approx(0.75)
    assert y.hi90 - y.mean == pytest.approx(Z90 * 0.75)
    assert r.evidence == "identified_causal" and r.warnings == [LINEARITY]


def test_independent_families_add_variances() -> None:
    y = compose(catalog(), {"fed_mp": 10, "opec": 2}, 0).outcomes[0]
    assert y.mean == pytest.approx(2.0 - 2.0)
    assert y.se == pytest.approx(np.sqrt(0.5**2 + (2 * 0.3) ** 2))


def test_fed_components_use_their_covariance() -> None:
    y = compose(catalog(), {"fed_mp": 10, "fed_cbi": 10}, 1).outcomes[0]
    assert y.se == pytest.approx(np.sqrt(1.0 + 0.16 + 2 * 0.2))


def test_weakest_label_and_range_warning() -> None:
    r = compose(catalog(), {"fed_mp": 25, "gpr": 1}, 0)
    assert r.evidence == "reduced_form"
    assert any("outside the range" in w for w in r.warnings)


def test_empty_scenario() -> None:
    assert compose(catalog(), {}, 0).outcomes == []


def test_write_fixtures_for_the_typescript_engine() -> None:
    """The web app's engine must reproduce these results exactly (web/src/engine)."""
    cases = [({"fed_mp": 15}, 0), ({"fed_mp": 10, "fed_cbi": 10}, 1),
             ({"fed_mp": 25, "gpr": 1}, 0), ({"opec": -3, "gpr": 2}, 1)]
    out = []
    for sizes, h in cases:
        r = compose(catalog(), sizes, h)
        out.append({"sizes": sizes, "horizon": h, "evidence": r.evidence,
                    "n_warnings": len(r.warnings),
                    "outcomes": [o.__dict__ for o in r.outcomes]})
    FIXTURES.mkdir(exist_ok=True)
    (FIXTURES / "scenario_cases.json").write_text(
        json.dumps({"catalog": catalog(), "cases": out}, indent=1) + "\n")
