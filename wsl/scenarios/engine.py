"""Scenario composition (reference implementation; the web app's TypeScript engine is tested
against fixtures produced here, web/src/engine/engine.test.ts).

A scenario is a set of shock sizes in each shock's natural unit (bp, %, or sd). The combined
response is the sum of size/per × response. Its variance treats different shock families as
independent (they are estimated on disjoint announcement samples) and uses the estimated
covariance for the two Fed components, which come from one regression.

Every result carries:
  * the weakest evidence label among the shocks used (identified_causal > reduced_form);
  * a linearity warning (responses are added, so interactions are ignored);
  * an out-of-range warning for any size beyond what was ever observed for that shock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

LABEL_ORDER = ("identified_causal", "reduced_form", "model_forecast", "hypothetical_scenario")
Z68, Z90 = 0.9944579, 1.6448536
PARTNERS = {("fed_mp", "fed_cbi"), ("fed_cbi", "fed_mp")}
LINEARITY = ("Combined by adding each shock's estimated response. This assumes effects are "
             "linear and do not interact; large or simultaneous shocks may behave differently.")


@dataclass
class OutcomeResult:
    id: str
    mean: float
    se: float
    lo68: float
    hi68: float
    lo90: float
    hi90: float


@dataclass
class ScenarioResult:
    horizon: int
    evidence: str
    warnings: list[str] = field(default_factory=list)
    outcomes: list[OutcomeResult] = field(default_factory=list)


def compose(catalog: dict[str, Any], sizes: dict[str, float], horizon: int) -> ScenarioResult:
    shocks = {s["id"]: s for s in catalog["shocks"]}
    used = {k: v for k, v in sizes.items() if v != 0}
    if not used:
        return ScenarioResult(horizon, "hypothetical_scenario", [], [])
    labels = [shocks[k]["label"] for k in used]
    evidence = max(labels, key=LABEL_ORDER.index)
    warnings = [LINEARITY]
    for k, v in used.items():
        d = shocks[k]["distribution"]
        lo, hi = d["min"], d["max"]
        if v > hi or v < lo:
            warnings.append(f"{shocks[k]['name']}: a size of {v:g} {shocks[k]['unit']} is "
                            f"outside the range ever observed ({lo:.1f} to {hi:.1f} "
                            f"{shocks[k]['unit']}); the estimate is an extrapolation.")
    out = []
    for o in catalog["outcomes"]:
        mean, var = 0.0, 0.0
        for k, v in used.items():
            r = catalog["irfs"][k][o["id"]]
            w = v / shocks[k]["per"]
            mean += w * r["coef"][horizon]
            var += (w * r["se"][horizon]) ** 2
        for a, b in PARTNERS:
            if a in used and b in used and a < b:
                wa, wb = used[a] / shocks[a]["per"], used[b] / shocks[b]["per"]
                var += 2 * wa * wb * catalog["irfs"][a][o["id"]]["cov_partner"][horizon]
        se = float(np.sqrt(max(var, 0.0)))
        out.append(OutcomeResult(o["id"], mean, se, mean - Z68 * se, mean + Z68 * se,
                                 mean - Z90 * se, mean + Z90 * se))
    return ScenarioResult(horizon, evidence, warnings, out)
