"""``wsl export preview``: the milestone preview page (web/preview/index.html).

The page shows only what exists: the M2 estimators on a simulated economy with a known
answer, the real shock calendar, and pipeline-health numbers from the build. Its data are
produced here from the repo's own code; the page template is web/preview/template.html.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wsl.data.build import git_sha
from wsl.estimation import montecarlo as mc
from wsl.estimation.lp import event_lp, lp
from wsl.estimation.simulate import sparse_shock, textbook_var
from wsl.paths import CLEAN_DIR, MANIFEST, ROOT
from wsl.shocks.boc_fad import load_target_rate, validate

TEMPLATE = ROOT / "web" / "preview" / "template.html"
OUT = ROOT / "web" / "preview" / "index.html"
DRAW_SEED = 777
N_DRAWS = 6


def _r(x: Any) -> list[float]:
    return [round(float(v), 5) for v in x]


def _explorer() -> dict[str, Any]:
    var = textbook_var()
    H = mc.H
    truth = var.irf(H)[:, :, 0]
    draws: dict[str, list[Any]] = {k: [] for k in ("dense_NW", "dense_HC1", "event_HC3",
                                                   "zerofill_NW")}
    rng = np.random.default_rng(DRAW_SEED)
    for _ in range(N_DRAWS):
        Y, eps = var.simulate(500, rng)
        idx = pd.RangeIndex(500)
        for key, cov in (("dense_NW", "NW"), ("dense_HC1", "HC1")):
            per = []
            for j in range(3):
                irf = lp(pd.Series(Y[:, j], index=idx), pd.Series(eps[:, 0], index=idx),
                         range(H + 1), cov_type=cov)  # type: ignore[arg-type]
                per.append({"coef": _r(irf.coef), "se": _r(irf.se), "nobs": int(irf.nobs[0])})
            draws[key].append(per)
        T = 8000
        s0 = sparse_shock(T, rng)
        Y, eps = var.simulate(T, rng, shock0=s0)
        y = Y[:, 1]
        idx = pd.RangeIndex(T)
        z = lp(pd.Series(y, index=idx), pd.Series(eps[:, 0], index=idx), range(H + 1),
               cov_type="NW")
        ev = np.flatnonzero(s0 != 0)
        ev = ev[(ev > 2) & (ev < T - H - 1)]
        df = pd.DataFrame({"s": eps[ev, 0]}, index=ev)
        for h in range(H + 1):
            df[f"d{h}"] = y[ev + h] - y[ev - 1]
        for lag in (1, 2):
            df[f"pre{lag}"] = y[ev - lag] - y[ev - lag - 1]
        e = event_lp(df, {h: f"d{h}" for h in range(H + 1)}, "s", ["pre1", "pre2"],
                     cov_type="HC3")
        draws["event_HC3"].append([None, {"coef": _r(e.coef), "se": _r(e.se),
                                          "nobs": int(e.nobs[0])}, None])
        draws["zerofill_NW"].append([None, {"coef": _r(z.coef), "se": _r(z.se),
                                            "nobs": len(ev)}, None])
    return {"horizons": list(range(H + 1)), "truth": [_r(truth[:, j]) for j in range(3)],
            "outcomes": ["Policy rate", "Activity", "Prices"], "draws": draws}


def _coverage() -> dict[str, Any]:
    def pack(x: mc.MCResult) -> dict[str, Any]:
        return {"cov90": _r(x.coverage90), "cov68": _r(x.coverage68),
                "mean90": round(float(x.coverage90.mean()), 3), "R": x.R}

    cov: dict[str, Any] = {
        "dense_NW": [pack(mc.dense("NW", outcome=j)) for j in range(3)],
        "dense_HC1": [pack(mc.dense("HC1", outcome=j)) for j in range(3)],
        "event_HC3": [None, pack(mc.sparse("event_HC3")), None],
        "zerofill_NW": [None, pack(mc.sparse("zerofill_NW")), None],
    }
    return cov


def _calendar_and_health() -> dict[str, Any]:
    f = pd.read_parquet(CLEAN_DIR / "events_fomc.parquet")
    b = pd.read_parquet(CLEAN_DIR / "events_boc.parquet")
    fdates = set(f["date"].dt.normalize())

    def opt(v: Any, scale: float = 1.0, nd: int = 2) -> float | None:
        return None if pd.isna(v) else round(float(v) * scale, nd)

    fomc = [{"d": t.strftime("%Y-%m-%d"), "t": t.strftime("%H:%M"), "bp": opt(p, 100),
             "mp": opt(m, 100), "cbi": opt(c, 100), "ev": str(e)[:90]}
            for t, p, m, c, e in zip(f["ts"], f["pc1"], f["MP_median"], f["CBI_median"],
                                     f["event"], strict=True)]
    boc = [{"d": d.strftime("%Y-%m-%d"), "dec": dec if isinstance(dec, str) else "n/a",
            "bp": None if pd.isna(c) else int(c), "rate": opt(rt, 1.0, 4),
            "sched": bool(sc), "fomc": d in fdates}
           for d, dec, c, rt, sc in zip(b["date"], b["decision"], b["change_bp"],
                                        b["rate_after_pct"], b["scheduled"], strict=True)]
    obs = pd.read_parquet(CLEAN_DIR / "obs.parquet", columns=["series_id", "status"])
    series = pd.read_parquet(CLEAN_DIR / "series.parquet")
    info = pd.read_parquet(CLEAN_DIR / "build_info.parquet").iloc[0]
    manifest = json.loads(MANIFEST.read_text())["artifacts"]
    health = {
        "series": int(series["parent_id"].nunique()), "derived_series": len(series),
        "observations": len(obs), "artifacts": len(manifest),
        "sources": len({k.split("/")[0] for k in manifest}),
        "missing_flagged": int((obs["status"] == "missing_at_source").sum()),
        "invalid_flagged": int((obs["status"] == "invalid_at_source").sum()),
        "boc_events": len(b), "boc_problems": len(validate(b, load_target_rate())),
        "fomc_events": len(f), "collisions": int(b["date"].isin(fdates).sum()),
        "built_at": str(info["built_at"]), "git": git_sha()[:7],
    }
    return {"fomc": fomc, "boc": boc, "health": health}


def write(path: Path = OUT) -> Path:
    data = {"explorer": _explorer(), "coverage": _coverage(), **_calendar_and_health()}
    page = TEMPLATE.read_text().replace("/*WSL_DATA*/null",
                                        json.dumps(data, separators=(",", ":")))
    path.write_text(page)
    return path.relative_to(ROOT)
