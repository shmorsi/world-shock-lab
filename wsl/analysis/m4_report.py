"""``wsl analyze``: run M4 and write docs/RESULTS.md plus results/m4/*.json.

Every number in the document is computed here; NOTES holds text only, written after reading
the results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wsl.analysis import m4
from wsl.export.results import write
from wsl.paths import DOCS_DIR, ROOT

OUT = DOCS_DIR / "RESULTS.md"
# Written after reading the first M4 results (2026-10-12); text only, no numbers.
NOTES: dict[str, str] = {
    "Brief's state-dependence goal": "Not met. None of the 80 pre-registered regime tests "
    "survives the false-discovery-rate control, and the raw count of small p-values is only "
    "modestly above what chance alone would produce. The minimum detectable effects show why: "
    "with these event counts only large regime differences are visible. The most suggestive "
    "cells before correction (the Fed's effect on the US 2-year at one week being smaller when "
    "VIX is high; OPEC news moving the TSX less when inflation is high) are hypotheses for the "
    "live data to test, not findings. No further specifications were searched.",
    "Canada lens": "A Fed move and a BoC move of the same size in their own 2-year yields move "
    "the Canadian 10-year yield by similar amounts, and the Canadian dollar by similar amounts "
    "in opposite directions. Canadian equities are where they differ: the Fed's effect on the "
    "TSX, the Big-6 banks and energy is clearly negative and robust, while the BoC's is small "
    "in the event LP. The Rigobon-Sack estimates narrow that gap, so the size of the Fed's "
    "advantage depends on the estimator; its direction does not.",
    "BoC proxy contamination": "The network finds the BoC proxy 'moving' US yields. A BoC "
    "decision should barely move US Treasuries, so this is evidence that the daily proxy still "
    "carries common North American rate news despite passing its M3 validation. It is the "
    "reason Rigobon-Sack estimates are shown next to every BoC result, and why BoC equity "
    "results are classed as fragile.",
    "VAR vs LP": "Monthly identification is weak for both instruments, so these are the least "
    "reliable results in the project. The BVAR's prior dominates (many coefficients per "
    "equation on a short sample) and pulls most responses toward zero with narrow bands; "
    "simulation shows those bands are too narrow. The LP-IV with the orthogonalized "
    "Bauer-Swanson surprise shows the textbook contraction, and for US industrial production "
    "at one year and US and Canadian prices at two years even the weak-instrument-robust "
    "Anderson-Rubin sets, though very wide, exclude zero. With the JK surprise the signs are "
    "puzzling and nothing is significant. Daily event studies, where identification is "
    "strong, carry the project's main causal claims.",
    "Oil exporters vs importers": "Every country ETF rises on negative oil supply news, and the "
    "ordering matches oil exposure (Norway, Canada and Australia most; India least), but the "
    "exporter-importer difference is not significant. The ETFs are priced in US dollars and "
    "OPEC news may coincide with global demand news, so the common rise is not by itself "
    "evidence about exporters.",
}


def _f(v: float, d: int = 2) -> str:
    return "" if v is None or np.isnan(v) else f"{v:+.{d}f}".replace("-", "−")


def _band(c: float, se: float) -> str:
    return f"{_f(c)} [{_f(c - m4.Z90 * se)}, {_f(c + m4.Z90 * se)}]"


def _ar(kind: str, lo: float, hi: float) -> str:
    if kind == "interval":
        return f"[{_f(lo)}, {_f(hi)}]"
    if kind == "two_rays":
        return f"(−∞, {_f(lo)}] ∪ [{_f(hi)}, ∞)"
    return "(−∞, ∞)" if kind == "real_line" else "empty"


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- sections
def var_section(results: list[m4.VarLpResult]) -> tuple[str, dict[str, Any]]:
    parts, payload = [], {}
    for r in results:
        rows = []
        for i, n in enumerate(r.variables):
            for h in (0, 6, 12, 18, 24):
                lp = r.lp[n]
                rows.append({
                    "variable": n, "h (months)": h,
                    "proxy BVAR [90%]": f"{_f(r.var.irf[h, i])} [{_f(r.var.lo90[h, i])}, "
                                        f"{_f(r.var.hi90[h, i])}]",
                    "LP-IV [90%]": _band(float(lp.coef[h]), float(lp.se[h])),
                    "LP-IV Anderson-Rubin 90% set": _ar(*r.ar90[n][h]),
                    "BVAR band width": f"{r.var.hi90[h, i] - r.var.lo90[h, i]:.2f}",
                    "LP band width": f"{2 * m4.Z90 * float(lp.se[h]):.2f}",
                })
        df = pd.DataFrame(rows)
        ratio = float(np.median(
            [(r.var.hi90[h, i] - r.var.lo90[h, i]) / (2 * m4.Z90 * float(r.lp[n].se[h]))
             for i, n in enumerate(r.variables) for h in range(1, 25)]))
        weak = max(r.var.first_stage_f, r.lp_first_stage_f) < 10
        parts.append(
            f"### Instrument: {r.instrument}\n\n"
            + ("**Weak instrument (first-stage F below 10): the Wald bands below are not "
               "reliable; the Anderson-Rubin sets are. Read these monthly estimates as "
               "suggestive.**\n\n" if weak else "")
            + f"Sample {r.sample[0]} to {r.sample[1]} ({r.T} months), 7 variables, 12 lags. "
            f"First-stage F of the instrument on the 1-year yield's VAR residual: "
            f"{r.var.first_stage_f:.1f} (bootstrap median {r.var.boot_f_median:.1f}); LP-IV "
            f"effective F at h = 0: {r.lp_first_stage_f:.1f}. Bootstrap: {r.var.draws} draws, "
            f"block length {r.var.block_length}. Median ratio of BVAR to LP band width over "
            f"h = 1..24: {ratio:.2f}.\n\n" + _md(df))
        payload[r.instrument] = {
            "variables": r.variables, "sample": r.sample, "T": r.T,
            "first_stage_f": r.var.first_stage_f, "lp_first_stage_f": r.lp_first_stage_f,
            "bvar": {"irf": r.var.irf.tolist(), "lo90": r.var.lo90.tolist(),
                     "hi90": r.var.hi90.tolist(), "lo68": r.var.lo68.tolist(),
                     "hi68": r.var.hi68.tolist()},
            "lp_iv": {n: r.lp[n].to_dict() for n in r.variables},
            "band_width_ratio_median": ratio}
    head = ("## 1. Two estimators of the same thing: proxy BVAR vs LP-IV (monthly)\n\n"
            "Response to a US monetary policy shock that raises the 1-year yield by 25 bp on "
            "impact. Units: yields in percentage points, other variables in % (100 × log). "
            "Both estimators use the same variables, instrument and lags. The BVAR imposes VAR "
            "dynamics and Minnesota shrinkage (lower variance, possible bias); the LP-IV does "
            "not (less bias, more variance). docs/SIMULATION.md shows the BVAR's bootstrap bands "
            "under-cover in simulation, so where the two disagree, read the LP bands as the "
            "honest measure of uncertainty.\n\n")
    return head + "\n".join(parts), payload


def state_section(cells: list[m4.StateCell]) -> tuple[str, list[dict[str, Any]]]:
    rows, payload = [], []
    hl = m4.HEADLINE
    for c in cells:
        r = c.result
        hi = r.horizons.index(hl)
        for k, name in enumerate(r.regimes):
            rows.append({
                "shock": c.family, "outcome": c.outcome, "regime": name,
                "events (regime / baseline)":
                    f"{r.n_by_regime[name]} / {r.n_by_regime['baseline']}",
                f"effect in baseline, h={hl}": _f(float(r.base_effect[hi])),
                f"difference in regime, h={hl} [90%]": _band(float(r.interaction[hi, k]),
                                                            float(r.interaction_se[hi, k])),
                "MDE": f"{r.mde_headline[k]:.2f}",
                "p joint": f"{r.p_joint:.3f}", "q joint": f"{c.q_joint:.3f}",
                f"p h={hl}": f"{r.p_headline:.3f}", f"q h={hl}": f"{c.q_headline:.3f}",
                "survives FDR": "yes" if min(c.q_joint, c.q_headline) <= m4.Q else "",
            })
        payload.append({"family": c.family, "outcome": c.outcome, "regime": c.regime,
                        "regimes": list(r.regimes), "n_by_regime": r.n_by_regime,
                        "base_effect": r.base_effect.tolist(),
                        "interaction": r.interaction.tolist(),
                        "interaction_se": r.interaction_se.tolist(), "p_joint": r.p_joint,
                        "p_headline": r.p_headline, "q_joint": c.q_joint,
                        "q_headline": c.q_headline})
    n_tests = 2 * len(cells)
    survivors = sum(1 for c in cells for q in (c.q_joint, c.q_headline) if q <= m4.Q)
    raw = sum(1 for c in cells for p in (c.result.p_joint, c.result.p_headline) if p <= 0.10)
    text = ("## 2. Does the same shock matter more in some regimes?\n\n"
            f"{len(cells)} pre-registered cells (protocol v3), each tested jointly over h = "
            f"0..{m4.H} and at the headline horizon h = {hl}: {n_tests} tests in one "
            f"Benjamini–Hochberg family at q = {m4.Q:.2f}. p-values come from a wild cluster "
            "bootstrap (999 draws). Regimes are measured before each announcement: high VIX is "
            "VIX above its trailing 252-day median; hiking and cutting are measured relative to "
            "holding, from the 6-month change in the policy rate; high inflation is CPI above 3%. "
            "MDE is the smallest difference this sample could detect (80% power, 10% test). The "
            "hiking and cutting rows of a cell share one test (both differences jointly).\n\n"
            f"**{raw} of {n_tests} tests have p ≤ 0.10 before correction; {survivors} survive "
            "the false-discovery-rate control.** Without the correction, about "
            f"{0.10 * n_tests:.0f} would be expected by chance alone.\n\n"
            + _md(pd.DataFrame(rows)))
    return text, payload


def canada_section(rows: list[m4.CanadaRow]) -> tuple[str, list[dict[str, Any]]]:
    hl = m4.HEADLINE
    table, payload = [], []
    for r in rows:
        for h in (0, hl):
            fc, fs, fn = r.fed[h]
            bc, bs, bn = r.boc[h]
            rb, rlo, rhi = r.boc_rs[h]
            table.append({
                "Canadian outcome": r.outcome, "units": r.units, "h": h,
                "Fed: US 2y +10 bp [90%]": _band(fc, fs), "N (Fed)": fn,
                "BoC: CA 2y +10 bp, event LP [90%]": _band(bc, bs),
                "BoC, Rigobon-Sack [90%]": f"{_f(rb)} [{_f(rlo)}, {_f(rhi)}]", "N (BoC)": bn,
            })
        payload.append({"outcome": r.outcome, "units": r.units, "fed": r.fed, "boc": r.boc,
                        "boc_rigobon_sack": r.boc_rs, "fed_first_stage_f": r.fed_first_stage_f})
    f_min = min(r.fed_first_stage_f for r in rows)
    text = ("## 3. The Canada lens: Fed vs Bank of Canada transmission\n\n"
            "The same Canadian assets, hit by a Fed move and by a BoC move of the same size: each "
            "raises its own country's 2-year yield by 10 bp on the announcement day.\n\n"
            "- **Fed:** the US 2-year change is instrumented by the JK policy (MP) shock, "
            "controlling for the information (CBI) shock. Effective first-stage F at day 0 is at "
            f"least {f_min:.0f}.\n"
            "- **BoC:** the announcement-day proxy, validated in M3, with the Rigobon–Sack "
            "estimate alongside it.\n\n"
            "Units are bp for yields and % for prices and FX. USD/CAD is CAD per USD, so a rise "
            "means a weaker Canadian dollar.\n\n" + _md(pd.DataFrame(table)))
    return text, payload


def panel_section(results: list[m4.PanelResult], per_country: pd.DataFrame
                  ) -> tuple[str, dict[str, Any]]:
    rows = [{"h": r.horizon, "importers [90%]": _band(*r.importer),
             "exporters [90%]": _band(*r.exporter),
             "exporter − importer [90%]": _band(r.difference[0], r.difference[1]),
             "p (difference)": f"{r.difference[2]:.3f}", "country-event rows": r.n_rows,
             "events": r.n_events} for r in results]
    hl = m4.HEADLINE
    pc = per_country.copy()
    pc_rows = [{"country": row["country"], "ETF": row["etf"],
                "classified exporter (share of events)": f"{row['exporter_share']:.0%}",
                "h=0 [90%]": _band(row["h0"][0], row["h0"][1]),
                f"h={hl} [90%]": _band(row[f"h{hl}"][0], row[f"h{hl}"][1]), "N": row["h0"][2]}
               for _, row in pc.sort_values("exporter_share", ascending=False).iterrows()]
    text = ("## 4. Oil supply news: exporters vs importers\n\n"
            "Response of US-listed country equity ETFs (total return in USD, so equity and "
            "currency moves combined) to a 1% OPEC oil supply surprise. A country is an exporter "
            "in a given year if its WDI net energy imports were negative in the latest earlier "
            "year, so classification can change over time (the US flips). Panel with country "
            "fixed effects and errors clustered by announcement.\n\n" + _md(pd.DataFrame(rows))
            + "\n### By country\n\n" + _md(pd.DataFrame(pc_rows)))
    payload = {"panel": [vars(r) for r in results],
               "per_country": per_country.to_dict(orient="records")}
    return text, payload


def network_section(df: pd.DataFrame) -> tuple[str, list[dict[str, Any]]]:
    edges = df[df["edge"]]
    summary = (df.groupby(["shock", "h"])
               .agg(tested=("edge", "size"), edges=("edge", "sum")).reset_index())
    top = (edges.assign(abs_t=(edges["coef"] / edges["se"]).abs())
           .sort_values("abs_t", ascending=False).groupby(["shock", "h"]).head(8))
    top_rows = [{"shock": sh, "h": h, "outcome": o, "units": u,
                 "effect [90%]": _band(float(c), float(se)), "q": f"{q:.4f}"}
                for sh, h, o, u, c, se, q in zip(top["shock"], top["h"], top["outcome"],
                                                  top["units"], top["coef"], top["se"],
                                                  top["q"], strict=True)]
    text = ("## 5. Spillover network\n\n"
            "Edges are shock → asset responses at h ∈ {1, 5} (protocol §9) that survive "
            f"Benjamini–Hochberg at q = {m4.Q:.2f} across all {len(df)} tested edges. Shocks are "
            "scaled as in §2. The full edge list (including non-significant ones, which the "
            "web app will draw faintly) is in results/m4/network.json.\n\n"
            + _md(summary) + "\n### Strongest edges by shock and horizon\n\n"
            + _md(pd.DataFrame(top_rows)))
    records: list[dict[str, Any]] = [{str(k): v for k, v in rec.items()}
                                     for rec in df.to_dict(orient="records")]
    return text, records


def robustness_section(rows: list[m4.RobustRow]) -> tuple[str, list[dict[str, Any]]]:
    table, payload = [], []
    for r in rows:
        sig = sum(lo > 0 or hi < 0 for _, lo, hi in r.variants.values())
        signs = sorted({"+" if c > 0 else "−" for c, _, _ in r.variants.values()})
        table.append({"shock": r.family, "outcome": r.outcome, "h": r.h,
                      "baseline [90%]": "{} [{}, {}]".format(*map(_f, r.variants["baseline"])),
                      "variants": len(r.variants), "signs": "/".join(signs),
                      "significant in": f"{sig}/{len(r.variants)}", "verdict": r.verdict})
        payload.append({"family": r.family, "outcome": r.outcome, "h": r.h,
                        "variants": r.variants, "verdict": r.verdict})
    n_rob = sum(1 for r in rows if r.verdict == "robust")
    text = ("## 6. Robustness matrix\n\n"
            "Each headline response (the outcomes of the state-dependence cells, at h = 0 and the "
            "headline horizon) is re-estimated under every applicable variant from protocol v3: "
            "no controls, ten pre-event lags, pre-2008, post-2008, excluding 2020, a zero-filled "
            "daily LP, the alternative Fed shocks (JK pc1, BS MPS_ORTH), and Rigobon–Sack for "
            "BoC. The rule was fixed in advance: **robust** = the same sign in every variant and "
            "90%-significant in at least two thirds of them.\n\n"
            f"**{n_rob} of {len(rows)} results are robust.** Per-variant estimates are in "
            "results/m4/robustness.json.\n\n" + _md(pd.DataFrame(table)))
    return text, payload


def write_report(path: Path = OUT, draws: int | None = None, boot: int = 999) -> Path:
    var_results = [m4.run_var_vs_lp(i, draws) for i in
                   (m4.SPEC["var"]["instrument"]["primary"],
                    m4.SPEC["var"]["instrument"]["robustness"])]
    cells = m4.run_state_dependence(boot=boot)
    canada = m4.run_canada()
    panel, per_country = m4.run_exporter_panel()
    network = m4.run_network()
    robust = m4.run_robustness()

    sections = [var_section(var_results), state_section(cells), canada_section(canada),
                panel_section(panel, per_country), network_section(network),
                robustness_section(robust)]
    names = ["var_vs_lp", "state_dependence", "canada", "exporters", "network", "robustness"]
    series = set(m4.network_outcomes()) | {"US2Y", "CA2Y", "VIX", "US_CPI_NSA", "CA_CPI_NSA",
                                          "BOC_TARGET", "FF_TARGET", "FF_TARGET_UPPER"}
    tables = ["events_fomc", "events_fomc_bs", "events_opec", "events_boc", "shocks_monthly"]
    for name, (_, payload) in zip(names, sections, strict=True):
        write(f"m4/{name}", {"results": payload}, series, tables)

    parts = ["# Results (M4): VAR vs LP, regimes, the Canada lens, spillovers, robustness\n",
             "Generated by `uv run wsl analyze` (code: `wsl/analysis/m4.py`), following "
             "protocol v3 (fixed before any of these estimates; `git tag protocol-v3`). Do not "
             "edit by hand. All results are in-sample causal or reduced-form estimates on the "
             "full sample, as the protocol allows; nothing here touches the held-out test "
             "evaluation of M5. Results with provenance are in results/m4/.\n"]
    parts += [text for text, _ in sections]
    if NOTES:
        parts.append("## 7. Notes\n\n" + "\n".join(f"- **{k}.** {v}" for k, v in NOTES.items())
                     + "\n")
    path.write_text("\n".join(parts))
    return path.relative_to(ROOT)
