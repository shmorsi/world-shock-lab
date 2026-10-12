"""Model-implied responses, for comparing ML with the local projections in the explorer.

For each primary asset, horizon and event family, the implied response to a typical surprise
is the average, over that family's events, of prediction(surprise added) minus prediction
(no surprise), holding each event's own state. A typical surprise is:
  FOMC  the component vector that goes with a +10 bp JK pc1 surprise (each raw component
        regressed on pc1);
  BoC   +10 bp on the day;
  OPEC  the component vector that goes with a +1% front-month surprise.
Models are fit on all primary events (full sample) and labelled model_forecast: this is a
description of what the fitted models do, not a held-out test (that is docs/ML_BENCHMARK.md).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from wsl.ml import models
from wsl.ml.dataset import ASSET_NAMES, HORIZONS, SHOCK_COLS, build
from wsl.ml.evaluate import RESULTS, _parse
from wsl.shocks import events as ev


def _typical(d_events: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    f = d_events[d_events["family"] == "fomc"]
    pc1 = ev.fomc()["pc1_bp"]
    pc1.index = "fomc:" + pc1.index
    common = f.index.intersection(pc1.index)
    vec = {}
    for c in SHOCK_COLS["fomc"]:
        y, x = f.loc[common, c], pc1[common]
        ok = y.notna() & x.notna()
        vec[c] = float(np.polyfit(x[ok], y[ok], 1)[0] * 10.0)
    out["fomc"] = pd.Series(vec)
    out["boc"] = pd.Series({"boc_d_ca2y": 10.0})
    o = d_events[d_events["family"] == "opec"]
    vec = {}
    for c in SHOCK_COLS["opec"]:
        ok = o[c].notna() & o["opec_Front"].notna()
        vec[c] = float(np.polyfit(o.loc[ok, "opec_Front"], o.loc[ok, c], 1)[0] * 1.0)
    out["opec"] = pd.Series(vec)
    return out


def implied_responses() -> dict[str, Any]:
    val = json.loads((RESULTS / "validation.json").read_text())
    keys = {"lp": "lp", "gbm": val["chosen"]["gbm"], "mlp": val["chosen"]["mlp"]}
    d = build()
    prim = ~d.events["boc_same_day_fomc"]
    typical = _typical(d.events)
    out: dict[str, Any] = {"typical_surprise": {k: v.to_dict() for k, v in typical.items()},
                           "label": "model_forecast", "responses": {}}
    for asset in ASSET_NAMES:
        for h in HORIZONS:
            cell = f"{asset}|h{h}"
            y = d.targets[cell]
            X = models.base_frame(d.events, d.asset_vol[asset])
            keep = y.notna() & prim
            for mname, key in keys.items():
                name, params = _parse(key)
                m = models.make(name, params).fit(X[keep], y[keep].to_numpy())
                for fam, vec in typical.items():
                    rows = X[keep & (d.events["family"] == fam)]
                    zero = rows.copy()
                    zero[SHOCK_COLS[fam]] = 0.0
                    shocked = zero.copy()
                    for c, v in vec.items():
                        shocked[c] = v
                    eff = m.predict(shocked) - m.predict(zero)
                    out["responses"].setdefault(fam, {}).setdefault(asset, {}).setdefault(
                        f"h{h}", {})[mname] = {"mean": float(np.mean(eff)),
                                               "sd_across_states": float(np.std(eff))}
    return out
