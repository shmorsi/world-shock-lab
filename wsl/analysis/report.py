"""``wsl replicate``: run M3 (replication, placebo, BoC proxy validation, timing check) and
write docs/REPLICATION.md plus results/m3/*.json. Every number in the document is computed
here; the hand-written notes contain no numbers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from wsl.analysis import boc_proxy
from wsl.analysis.eventstudy import event_irf, monthly_shock
from wsl.analysis.placebo import placebo_test
from wsl.analysis.replication import ClaimResult, run_all, snapshot
from wsl.export.results import write
from wsl.paths import DOCS_DIR, ROOT
from wsl.shocks import events as ev
from wsl.shocks.windows import load_series, window_changes

OUT = DOCS_DIR / "REPLICATION.md"
PLACEBO_CLAIMS = ("JK1", "JK2", "FX2", "KZ1", "KZ2")
SNAP_DAILY = (0, 1, 5, 10, 20)
SNAP_MONTHLY = (0, 6, 12, 18, 24)
TIMING_CLOSES = ("16:00", "13:30")

# Written after reading the first results (2026-10-12); text only, no numbers. Numbers live in
# the tables above, so they can never drift from the code.
NOTES: dict[str, str] = {
    "JK1-JK4, BS1, KZ1": "Marked mechanical: the surprise is measured inside the daily window "
    "(or, for KZ1, in oil futures that move with spot WTI). They confirm the plumbing "
    "(timestamps, windows, units) rather than test economics.",
    "JK4": "The information shock lifts stocks by construction in the 30-minute window, but the "
    "effect is not significant over the whole day, while its effect on rates (JK3) is strong. "
    "The equity part of information effects looks short-lived or too noisy to see at daily "
    "frequency. This is reported, not explained away.",
    "BS3 vs BS3r": "With the raw surprise, industrial production rises after a tightening: the "
    "classic puzzle. With the orthogonalized surprise it falls, significantly. Bauer and "
    "Swanson attribute the puzzle to the Fed responding to news markets had not yet priced; "
    "removing the predictable part of the surprise fixes it. This reproduces the paper's "
    "central point.",
    "KZ5": "Activity falls, as the paper reports, but the monthly LP is imprecise at month 12 and "
    "only significant at the longest horizon (horizon table). The paper uses a VAR, which is "
    "more efficient at long horizons than an LP; M4 compares the two directly.",
    "KZ3": "The paper's dollar result comes from a monthly VAR; our test asks a narrower daily "
    "question (does the dollar weaken in the week after OPEC news). The sign agrees and the "
    "effect is small.",
    "FX1, FX2": "Expected signs taken from the project brief, not verified paper claims; they "
    "are included as stylized facts.",
    "BoC proxy": "For the 10-year yield and equities, the event LP and the Rigobon-Sack estimate "
    "differ. The event LP assumes the proxy contains only BoC news; Rigobon-Sack does not. "
    "Other same-day news that moves Canadian yields and stocks together would bias the event "
    "LP for those outcomes, so Rigobon-Sack is the more credible estimate there. M4 reports "
    "both for every Canadian outcome.",
}


def _f(v: float, d: int = 2) -> str:
    return f"{v:+.{d}f}".replace("-", "−")


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def _units(r: ClaimResult) -> str:
    return str(r.irf.meta.get("units", ""))


def claims_table(results: list[ClaimResult]) -> str:
    rows = []
    for r in results:
        c = r.claim
        lo, hi = r.band90
        rows.append({
            "ID": c.id, "Source": c.source,
            "Claim": c.claim + (" *(mechanical)*" if c.mechanical else ""),
            "Our test": c.test, "Expected": "+" if c.expected > 0 else "−",
            "Estimate (90% band)": f"{_f(r.est)} [{_f(lo)}, {_f(hi)}] {_units(r)}",
            "N": r.n, "Verdict": r.verdict,
        })
    return _md(pd.DataFrame(rows))


def snapshots_table(results: list[ClaimResult]) -> str:
    rows = []
    for r in results:
        hs = SNAP_MONTHLY if r.claim.family == "monthly" else SNAP_DAILY
        cells = {f"h={s['h']}": f"{_f(s['coef'])} [{_f(s['lo90'])}, {_f(s['hi90'])}]"
                 for s in snapshot(r.irf, hs)}
        rows.append({"ID": r.claim.id, "units": _units(r), **cells})
    return _md(pd.DataFrame(rows).fillna(""))


def placebo_section(results: list[ClaimResult]) -> tuple[str, dict[str, Any]]:
    by_id = {r.claim.id: r for r in results}
    others = {"fomc": [ev.boc(False, False)["ts"], ev.opec()["ts"]],
              "opec": [ev.fomc()["ts"], ev.boc(False, False)["ts"]]}
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    for cid in PLACEBO_CLAIMS:
        r = by_id[cid]
        c = r.claim
        events = (ev.fomc().dropna(subset=["mp_bp", "cbi_bp"]) if c.family == "fomc"
                  else ev.opec())
        s = load_series(c.outcome)
        w = window_changes(s, events["ts"], 0)
        used = events[w["d0"].notna() & events[c.shock].notna()]
        res = placebo_test(s, used, [c.shock, *c.controls], {0: r.irf.coef[0], 5: r.irf.coef[5]},
                           per=c.per, other_families=others[c.family])
        for p in res:
            rows.append({"ID": cid, "h": p.horizon, "real estimate": _f(p.actual),
                         "placebo mean": _f(p.placebo_mean), "placebo s.d.": f"{p.placebo_sd:.2f}",
                         "permutation p": f"{p.p_value:.3f}", "draws": len(p.placebo)})
            payload[f"{cid}_h{p.horizon}"] = {"actual": p.actual, "placebo_mean": p.placebo_mean,
                                              "placebo_sd": p.placebo_sd, "p_value": p.p_value}
    text = ("## 3. Placebo tests\n\nThe real surprises are reassigned to random non-announcement "
            "trading days (1,000 draws, seed 20261012; days within 5 trading days of the same "
            "family's announcements and other families' announcement days excluded), and the "
            "same regression is re-run. The placebo mean should be near zero; the permutation p "
            "is the share of placebo draws at least as large in absolute value as the real "
            "estimate.\n\n" + _md(pd.DataFrame(rows)))
    return text, payload


def proxy_section(v: boc_proxy.ProxyValidation) -> str:
    vr_rows = [{"sample": "all events", "pairs": v.vr.n_pairs, "ratio": f"{v.vr.ratio:.2f}",
                "90% CI": f"[{v.vr.ci90[0]:.2f}, {v.vr.ci90[1]:.2f}]",
                "share of variance from the announcement": f"{v.vr.signal_share:.0%}"}]
    for name, x in v.vr_by_period.items():
        vr_rows.append({"sample": name, "pairs": x.n_pairs, "ratio": f"{x.ratio:.2f}",
                        "90% CI": f"[{x.ci90[0]:.2f}, {x.ci90[1]:.2f}]",
                        "share of variance from the announcement": f"{x.signal_share:.0%}"})
    vr_rows.append({"sample": "US-purged proxy, all events", "pairs": v.vr_purged.n_pairs,
                    "ratio": f"{v.vr_purged.ratio:.2f}",
                    "90% CI": f"[{v.vr_purged.ci90[0]:.2f}, {v.vr_purged.ci90[1]:.2f}]",
                    "share of variance from the announcement": f"{v.vr_purged.signal_share:.0%}"})
    dec = v.by_decision.reset_index().rename(columns={"decision": "decision"})
    dec["mean_bp"] = dec["mean_bp"].map(lambda x: _f(x))
    dec["se_bp"] = dec["se_bp"].map(lambda x: f"{x:.2f}")
    out_rows = []
    for o, irf in v.irfs.items():
        lo, hi = irf.band(0.90)
        rs = v.rs[o]
        scale = boc_proxy.PER
        p0 = next(p for p in v.placebo[o] if p.horizon == 0)
        out_rows.append({
            "outcome": o, "units": irf.meta["units"],
            "event LP, day 0 (90% band)": f"{_f(irf.coef[0])} [{_f(lo[0])}, {_f(hi[0])}]",
            "event LP, day 5": f"{_f(irf.coef[5])} [{_f(lo[5])}, {_f(hi[5])}]",
            "placebo p (day 0)": f"{p0.p_value:.3f}",
            "Rigobon-Sack, day 0 (90% CI)":
                f"{_f(rs.beta * scale)} [{_f(rs.ci90[0] * scale)}, {_f(rs.ci90[1] * scale)}]",
            "N": irf.nobs[0],
        })
    crit = "\n".join(f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in v.criteria.items())
    coef, se, r2 = v.slope_on_decision
    return (
        "## 4. Bank of Canada announcement-day proxy\n\n"
        "Proxy = change in the GoC 2-year benchmark yield from the close before a scheduled "
        "BoC announcement to the close on the day (bp). Events follow protocol v2 (scheduled, "
        f"known time, no FOMC on the same day): {v.n_events} events, {v.first} to {v.last}; "
        f"standard deviation {v.proxy_sd:.2f} bp. Control days are the same weekday and time one "
        "week earlier, dropped if within a day of any announcement.\n\n"
        "### Pass criteria (fixed in `wsl/analysis/boc_proxy.py` before the first run)\n\n"
        f"{crit}\n\n**Label: `{v.label}`.**\n\n"
        "### Is the announcement day special? (variance ratio, Rigobon-Sack)\n\n"
        + _md(pd.DataFrame(vr_rows)) +
        f"\nThe US-purged proxy removes the same-day US 2-year move, with its loading "
        f"({v.purge_beta:.2f}) estimated on control days only; its correlation with the raw "
        f"proxy is {v.corr_raw_purged:.2f}.\n\n"
        "### Does the proxy line up with the decision?\n\n" + _md(dec) +
        f"\nRegressing the proxy on the size of the rate change gives a slope of "
        f"{_f(coef, 3)} (s.e. {se:.3f}, R² {r2:.2f}): a fully expected decision would not move "
        "the 2-year yield, so a low R² is expected and not a failure.\n\n"
        "### Responses to a 10 bp hawkish proxy move\n\n"
        "Event LP on announcement days (HC3, five pre-event changes as controls) next to the "
        "Rigobon-Sack estimate, which does not assume the proxy is free of other news.\n\n"
        + _md(pd.DataFrame(out_rows))
    )


def timing_section() -> tuple[str, dict[str, Any]]:
    fomc = ev.fomc().dropna(subset=["mp_bp", "cbi_bp"])
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    fits: dict[str, tuple[float, float, float]] = {}
    for outcome in ("CA2Y", "CA10Y"):
        for close in TIMING_CLOSES:
            irf = event_irf(fomc, "mp_bp", outcome, other_shocks=("cbi_bp",), per=10,
                            assumed_time=close)
            lo, hi = irf.band(0.90)
            rows.append({"outcome": outcome, "assumed close (ET)": close,
                         "day 0": f"{_f(irf.coef[0])} [{_f(lo[0])}, {_f(hi[0])}]",
                         "day 1": f"{_f(irf.coef[1])} [{_f(lo[1])}, {_f(hi[1])}]",
                         "N": irf.nobs[0]})
            payload[f"{outcome}_{close}"] = irf.to_dict()
            if outcome == "CA2Y":
                fits[close] = (float(irf.coef[0]), float(lo[0]), float(hi[0]))
    late, early = fits["16:00"], fits["13:30"]
    after = late[1] > 0 and late[0] > early[2]
    verdict = ("**Conclusion: the data indicate the Valet closing yields are taken after "
               "14:00 ET.** The 2-year response is significant in the 16:00-assumed window "
               "and its estimate lies above the 13:30-assumed band. 16:00 stays the working "
               "assumption, and open "
               "item O-1 is resolved empirically." if after else
               "**Conclusion: the data do not settle the closing time;** O-1 stays open and FOMC "
               "responses of Canadian yields are reported under both assumptions.")
    payload["conclusion_close_after_14h"] = after
    text = ("## 5. Timing check: when do Canadian bond yields close? (open item O-1)\n\n"
            "The BoC does not document the time of its closing yields. BoC announcements are "
            "in the morning, so this only matters for 14:00 ET FOMC announcements. If the "
            "close is after 14:00, the FOMC day is day 0; if before, the next day is. The "
            "response of Canadian yields to a 10 bp Fed policy (MP) shock under both "
            "assumptions:\n\n" + _md(pd.DataFrame(rows)) +
            "\nTwo hypotheses make opposite predictions. If the close is after 14:00, the "
            "16:00-assumed day-0 window contains the announcement and should respond strongly, "
            "while the 13:30-assumed day 0 (the next day) holds only later news. If the close "
            "is before 14:00, the reverse holds.\n\n" + verdict + "\n")
    return text, payload


def write_report(path: Path = OUT) -> Path:
    results = run_all()
    placebo_text, placebo_payload = placebo_section(results)
    v = boc_proxy.validate()
    timing_text, timing_payload = timing_section()
    news_sd = float(monthly_shock("KAENZIG_OIL_NEWS").std(ddof=1))

    series_ids = {r.claim.outcome for r in results} | set(boc_proxy.OUTCOMES) | {
        "CA2Y", "US2Y", "CA10Y"}
    tables = ["events_fomc", "events_fomc_bs", "events_opec", "events_boc", "shocks_monthly"]
    write("m3/replication", {
        "label_note": "identified_causal in-sample estimates; see docs/REPLICATION.md",
        "claims": [{"id": r.claim.id, "source": r.claim.source, "claim": r.claim.claim,
                    "test": r.claim.test, "expected_sign": r.claim.expected,
                    "mechanical": r.claim.mechanical, "verdict": r.verdict,
                    "estimate": r.est, "band90": r.band90, "n": r.n, "irf": r.irf.to_dict()}
                   for r in results],
        "placebo": placebo_payload}, series_ids, tables)
    write("m3/boc_proxy", {
        "label": v.label, "criteria": v.criteria, "n_events": v.n_events,
        "variance_ratio": vars(v.vr) | {"signal_share": v.vr.signal_share},
        "variance_ratio_purged": vars(v.vr_purged), "purge_beta": v.purge_beta,
        "irfs": {o: i.to_dict() for o, i in v.irfs.items()},
        "rigobon_sack_per_bp": {o: vars(r) for o, r in v.rs.items()}}, series_ids, tables)
    write("m3/timing_o1", {"irfs": timing_payload}, {"CA2Y", "CA10Y"}, ["events_fomc"])

    verdicts = pd.Series([r.verdict for r in results if not r.claim.id.endswith("r")])
    econ = [r for r in results if not r.claim.mechanical and not r.claim.id.endswith("r")]
    parts = [
        "# Replication: paper vs ours\n",
        "Generated by `uv run wsl replicate` (code: `wsl/analysis/`). Do not edit by hand. "
        "Results with full provenance are in `results/m3/`.\n",
        "## 1. How to read this\n\n"
        "- **Daily claims:** event-sample local projections, the pre-registered primary daily "
        "specification (protocol §9). Each outcome's change runs from the last observation "
        "before the announcement to the h-th observation after it, using that series' own "
        "observation time. Controls are the five pre-event daily changes, plus the other JK "
        "shock where relevant. Standard errors are HC3, and bands are 90%.\n"
        "- **Monthly claims:** local projections on monthly data, with 12 lags of the outcome "
        "and the shock and Newey–West errors (h+1 lags). The Känzig news shock has standard "
        f"deviation {news_sd:.2f}.\n"
        "- **Units:** yields in bp; prices, FX and indices in % (100 × log change). Fed "
        "surprises are scaled per 10 bp; OPEC surprises per 1% move in oil futures.\n"
        "- **Verdict:**\n"
        "  - *replicated*: the sign matches and the 90% band excludes zero.\n"
        "  - *same sign*: the sign matches but the band includes zero.\n"
        "  - *opposite*: the sign is wrong.\n"
        "- **Rows marked *(mechanical)*** are close to true by construction. For example, the "
        "30-minute Fed surprise sits inside the daily window. They check the plumbing, not "
        "the economics.\n"
        "- **Rows ending in `r`** are reference runs for comparisons (Bauer–Swanson raw vs "
        "orthogonalized).\n",
        f"**Summary:** of {len(verdicts)} claims (excluding reference rows), "
        f"{int((verdicts == 'replicated').sum())} replicated, "
        f"{int((verdicts == 'same sign').sum())} have the expected sign without significance, "
        f"{int(verdicts.str.startswith('opposite').sum())} have the opposite sign. Among the "
        f"{len(econ)} non-mechanical claims, "
        f"{sum(r.verdict == 'replicated' for r in econ)} replicated.\n",
        "## 2. Paper vs ours\n", claims_table(results),
        "\n### Responses over the horizon (estimate [90% band])\n", snapshots_table(results),
        placebo_text, proxy_section(v), timing_text,
    ]
    if NOTES:
        parts.append("## 6. Notes on disagreements and caveats\n\n" + "\n".join(
            f"- **{cid}.** {txt}" for cid, txt in NOTES.items()) + "\n")
    path.write_text("\n".join(parts))
    return path.relative_to(ROOT)

