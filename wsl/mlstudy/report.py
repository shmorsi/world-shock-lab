"""docs/ML_STUDY.md, its figure, and the site payload. Numbers are read from the committed JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from wsl.mlstudy.evaluate import FINAL_PATH, VAL_PATH
from wsl.paths import ROOT
from wsl.report.figures import GRID, INK, INK2, Svg, palette

DOC = ROOT / "docs" / "ML_STUDY.md"
FIG = ROOT / "docs" / "figures" / "fig_ml_study.svg"
WEB = ROOT / "web" / "public" / "data" / "mlstudy.json"


def _load() -> dict[str, Any]:
    if not FINAL_PATH.exists():
        raise FileNotFoundError(
            "results/mlstudy/final.json is missing. The report is generated from that file.")
    doc: dict[str, Any] = json.loads(FINAL_PATH.read_text())
    if VAL_PATH.exists():
        doc["validation_file"] = json.loads(VAL_PATH.read_text())
    return doc


def _n(v: float, d: int = 3) -> str:
    return f"{float(v):.{d}f}"


def _ci(pair: list[float], d: int = 3) -> str:
    return f"{_n(pair[0], d)} to {_n(pair[1], d)}"


def _short(key: str) -> str:
    name = key.split(":", 1)[0]
    return {"mt_mlp": "Multi-task network", "tcn": "Temporal CNN", "qgbm": "Quantile boosting",
            "qnet": "Quantile network", "lp": "Linear projection", "zero": "Predict zero",
            "hist": "Historical quantile", "gauss_lp": "Gaussian linear projection"}.get(name, name)


def _bars(title: str, rows: list[tuple[str, float, str]], y: int, svg: Svg, *,
          ink: str = INK, ink2: str = INK2, grid: str = GRID) -> int:
    svg.text(16, y, title, 13, ink, weight=600)
    y += 22
    vals = [v for _nme, v, _c in rows]
    lo, hi = min(vals) - 0.02, max(vals) + 0.04
    left, right = 210, 70

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (svg.w - left - right)

    for name, val, col in rows:
        svg.text(16, y + 4, name, 12, ink2)
        svg.line(left, y, x(val), y, grid, 1)
        svg.raw(f'<circle cx="{x(val):.1f}" cy="{y:.1f}" r="5.5" fill="{col}"/>')
        svg.text(x(val) + 10, y + 4, _n(val), 11, ink2)
        y += 26
    return y + 8


def _figure(doc: dict[str, Any], path: Path, *, dark: bool = False) -> Path:
    """Selected-model chart. ``dark=True`` is the README raster; the print SVG stays light."""
    p = palette(dark)
    mean = doc["mean"]["pooled_scaled_mae"]
    tail = doc["tail"]["pooled_scaled_pinball"]
    mean_key = doc["chosen"]["mean"]
    tail_key = doc["chosen"]["tail"]
    svg = Svg(720, 340, "Held-out scaled error of the selected models and their baselines",
              font=p["font"], bg=p["bg"])
    svg.text(16, 22, "Held-out test set: selected model against the pre-specified baselines", 14,
             p["ink"], weight=600)
    svg.text(16, 40, "Lower is better. Scales are the training-split loss of a naive forecast.",
             11.5, p["muted"])
    y = _bars("Mean response, pooled scaled MAE", [
        ("Predict zero", mean["zero"], p["muted"]),
        ("Linear projection", mean["lp"], p["blue"]),
        (_short(mean_key), mean[mean_key], p["orange"]),
    ], 64, svg, ink=p["ink"], ink2=p["ink2"], grid=p["grid"])
    _bars("5% tail, pooled scaled pinball", [
        ("Historical quantile", tail["hist"], p["muted"]),
        ("Gaussian linear projection", tail["gauss_lp"], p["blue"]),
        (_short(tail_key), tail[tail_key], p["orange"]),
    ], y, svg, ink=p["ink"], ink2=p["ink2"], grid=p["grid"])
    return svg.save(path)


def _md(doc: dict[str, Any]) -> str:
    mean, tail = doc["mean"], doc["tail"]
    mk, tk = doc["chosen"]["mean"], doc["chosen"]["tail"]
    md, td = mean["difference_ml_minus_lp"], tail["difference_ml_minus_gauss"]
    inputs = doc.get("inputs", {})
    lines = [
        "# Machine-learning study",
        "",
        "Generated from `results/mlstudy/final.json`. "
        f"Protocol tag `{doc['protocol_tag']}`, config `{doc['config_sha256'][:12]}`. "
        "This study does not replace the M5 benchmark in [ML_BENCHMARK.md](ML_BENCHMARK.md).",
        "",
        f"**Mean.** {_short(mk)} minus the linear projection: {_n(md['point'])} "
        f"(90% CI {_ci(md['ci90'])}). Verdict: {mean['verdict']}.",
        "",
        f"**Tail.** {_short(tk)} minus the Gaussian linear projection, 5% pinball: "
        f"{_n(td['point'])} (90% CI {_ci(td['ci90'])}). Verdict: {tail['verdict']}.",
        "",
        "| Question | Model | Score | Baseline | Difference (90% CI) | Verdict |",
        "|---|---|---:|---:|---|---|",
        f"| Mean, pooled scaled MAE | {_short(mk)} | {_n(mean['pooled_scaled_mae'][mk])} "
        f"| linear {_n(mean['pooled_scaled_mae']['lp'])} | {_n(md['point'])} ({_ci(md['ci90'])}) "
        f"| {mean['verdict']} |",
        f"| Mean, predict zero | — | {_n(mean['pooled_scaled_mae']['zero'])} | — | — | reported |",
        f"| 5% pinball | {_short(tk)} | {_n(tail['pooled_scaled_pinball'][tk])} "
        f"| Gaussian {_n(tail['pooled_scaled_pinball']['gauss_lp'])} | {_n(td['point'])} "
        f"({_ci(td['ci90'])}) | {tail['verdict']} |",
        f"| 5% pinball, historical | — | {_n(tail['pooled_scaled_pinball']['hist'])} | — | — | reported |",
        "",
        f"Test announcements: {doc['n_test']}. "
        f"Years {doc['test_years'][0]}–{doc['test_years'][-1]}. "
        f"Walk-forward refit. Static-refit mean difference "
        f"{_n(mean['static_refit']['difference_ml_minus_lp']['point'])} "
        f"(90% CI {_ci(mean['static_refit']['difference_ml_minus_lp']['ci90'])}), "
        "which cannot change the verdict.",
        "",
        "Validation scores, which chose the models and were not the test:",
        "",
    ]
    val_mae = doc["validation_pooled_scaled_mae"]
    val_pin = doc["validation_pooled_scaled_pinball"]
    lines.append("| Candidate | Validation score |")
    lines.append("|---|---:|")
    for key, val in val_mae.items():
        lines.append(f"| `{key}` | {_n(float(val), 4)} scaled MAE |")
    for key, val in val_pin.items():
        lines.append(f"| `{key}` | {_n(float(val), 4)} scaled pinball |")
    lines += [
        "",
        f"90% interval coverage on the test set: linear {_n(mean['coverage90']['lp'], 2)}, "
        f"selected mean model {_n(mean['coverage90'][mk], 2)}. "
        f"5% hit rate: historical {_n(tail['hit_rate_05']['hist'], 2)}, "
        f"Gaussian {_n(tail['hit_rate_05']['gauss_lp'], 2)}, "
        f"selected tail model {_n(tail['hit_rate_05'][tk], 2)}.",
        "",
        f"Fissler–Ziegel (secondary, lower is better): historical {_n(tail['fissler_ziegel']['hist'])}, "
        f"Gaussian {_n(tail['fissler_ziegel']['gauss_lp'])}, "
        f"selected {_n(tail['fissler_ziegel'][tk])}.",
        "",
        "## Why the comparison looks like this",
        "",
        f"The mean-model point estimate is higher error than the linear projection "
        f"({_n(md['point'])}), and the 90% interval includes zero, so the pre-registered rule "
        f"is \"{mean['verdict']}\". Validation had already ranked every network behind the linear "
        "model; the test did not reverse that. The static refit, which cannot change the verdict, "
        f"is \"{mean['static_refit']['verdict']}\" "
        f"(difference {_n(mean['static_refit']['difference_ml_minus_lp']['point'])}, "
        f"90% CI {_ci(mean['static_refit']['difference_ml_minus_lp']['ci90'])}). "
        "Pooling the assets and showing the network the pre-event path did not beat a linear "
        "function of the surprise.",
        "",
        f"The tail question is separate. The selected model's pinball difference is "
        f"{_n(td['point'])}, and the whole interval is below zero: \"{tail['verdict']}\". "
        f"The same 5% quantile is crossed "
        f"{_n(tail['hit_rate_05'][tk], 2)} of the time, against "
        f"{_n(tail['hit_rate_05']['gauss_lp'], 2)} for the Gaussian model, and the central "
        f"interval covers {_n(tail['coverage_90'][tk], 2)} rather than 0.90. "
        "The gain is the scoring rule that was named in advance, not a claim of calibration.",
        "",
        "The Fissler–Ziegel average is reported below and is not a decision. Yields are in "
        "basis points and equities in percent, and the exponential score lets one cell dominate, "
        "so a very negative Gaussian score is not evidence the Gaussian model is better.",
        "",
        "## Embeddings, attribution, portfolio, stress",
        "",
    ]
    emb = doc["embeddings"]
    lines.append(
        f"Asset-embedding adjusted Rand index against the pre-specified groups: {_n(emb['ari'], 2)} "
        f"(permutation p {_n(float(emb['ari_permutation_p']), 3)}). Descriptive.")
    shap = doc["shap"]
    shares = ", ".join(f"{k} {_n(float(v), 2)}" for k, v in shap["share"].items())
    lines.append(f"Grouped Shapley shares of mean absolute contribution, horizon 1: {shares}. "
                 f"Efficiency gap {_n(float(shap['efficiency_gap']), 6)}.")
    book = doc["portfolio"]
    bd = book["difference_ml_minus_gauss"]
    lines.append(
        f"Equity book ({book['name']}) scaled 5% pinball: selected {_n(book['pooled_scaled_pinball'][tk])}, "
        f"Gaussian {_n(book['pooled_scaled_pinball']['gauss_lp'])}, "
        f"difference {_n(bd['point'])} (90% CI {_ci(bd['ci90'])}). Secondary.")
    stress = doc["stress"]
    lines.append(
        f"Hypothetical FOMC surprise of {_n(float(stress['size_sd']), 2)} training standard "
        "deviations in the two-year futures surprise, model fit through 2018. "
        "Median and 5% quantile versus a zero-surprise row (not a test score):")
    lines.append("")
    lines.append("| Asset | Median, zero | Median, shock | 5%, zero | 5%, shock |")
    lines.append("|---|---:|---:|---:|---:|")
    for asset, row in stress["assets"].items():
        lines.append(
            f"| {asset} | {_n(row['q50_zero'], 2)} | {_n(row['q50_shock'], 2)} | "
            f"{_n(row['q05_zero'], 2)} | {_n(row['q05_shock'], 2)} |")
    matched = inputs.get("matched_pin")
    revised = inputs.get("revised")
    lines += [
        "",
        f"Data vintage: {matched} artifacts matched the committed pins and {revised} were revised "
        "by the source after the pin date. Shock files are the matched ones when that count is "
        "the shock releases. The comparison is internal to this vintage. The M5 verdict stays "
        "the one in `results/final_eval/`.",
        "",
        "![Selected models against their baselines](figures/fig_ml_study.svg)",
        "",
    ]
    return "\n".join(lines)


def _web(doc: dict[str, Any]) -> dict[str, Any]:
    """Aggregates only. No prices, no raw surprises."""
    mk, tk = doc["chosen"]["mean"], doc["chosen"]["tail"]
    return {
        "protocol_tag": doc["protocol_tag"],
        "generated_at": doc["generated_at"],
        "n_test": doc["n_test"],
        "test_years": doc["test_years"],
        "chosen": {"mean": _short(mk), "tail": _short(tk), "mean_key": mk, "tail_key": tk},
        "mean": {
            "scores": doc["mean"]["pooled_scaled_mae"],
            "difference": doc["mean"]["difference_ml_minus_lp"],
            "verdict": doc["mean"]["verdict"],
            "coverage90": doc["mean"]["coverage90"],
            "static": doc["mean"]["static_refit"]["difference_ml_minus_lp"],
        },
        "tail": {
            "scores": doc["tail"]["pooled_scaled_pinball"],
            "difference": doc["tail"]["difference_ml_minus_gauss"],
            "verdict": doc["tail"]["verdict"],
            "hit_rate_05": doc["tail"]["hit_rate_05"],
            "coverage_90": doc["tail"]["coverage_90"],
            "fissler_ziegel": doc["tail"]["fissler_ziegel"],
        },
        "shap_share": doc["shap"]["share"],
        "embeddings": {"ari": doc["embeddings"]["ari"],
                       "p": doc["embeddings"]["ari_permutation_p"],
                       "assets": doc["embeddings"]["assets"],
                       "clusters": doc["embeddings"]["clusters"]},
        "portfolio": doc["portfolio"],
        "stress": doc["stress"],
        "inputs": {"matched_pin": doc.get("inputs", {}).get("matched_pin"),
                   "revised": doc.get("inputs", {}).get("revised")},
    }


CARD = ROOT / "docs" / "MODEL_CARD.md"


def model_card(doc: dict[str, Any]) -> str:
    """Reviewer-facing card. Every score is read from the committed study file."""
    mean, tail = doc["mean"], doc["tail"]
    mk, tk = doc["chosen"]["mean"], doc["chosen"]["tail"]
    md, td = mean["difference_ml_minus_lp"], tail["difference_ml_minus_gauss"]
    ev = yaml.safe_load((ROOT / "config" / "evaluation.yaml").read_text())
    cfg = yaml.safe_load((ROOT / "config" / "ml_study.yaml").read_text())
    splits, rule = ev["splits"], cfg["decision_rule"]["bootstrap"]
    counts = doc["counts"]
    q_mean, q_tail = cfg["questions"]["mean"], cfg["questions"]["tail"]
    return "\n".join([
        "# Model card",
        "",
        "Generated from `results/mlstudy/final.json` by `wsl.mlstudy.report.model_card`. "
        "Do not edit by hand. This is the second study. It does not replace the per-asset "
        "benchmark in [ML_BENCHMARK.md](ML_BENCHMARK.md).",
        "",
        "## What was compared",
        "",
        f"Protocol `{doc['protocol_tag']}`, frozen {cfg['frozen_on']}. The calendar split is "
        f"the parent protocol's: train through {splits['train']['end']}, validation "
        f"{splits['validation']['start']} to {splits['validation']['end']}, test "
        f"{splits['test']['start']} to {splits['test']['end']}. This file counts "
        f"{counts['train_events']} training announcements, {counts['validation_events']} "
        f"validation, and {doc['n_test']} test ({doc['test_years'][0]}–{doc['test_years'][-1]}).",
        "",
        "Validation chooses the model. The test set is scored once. A model adds value only "
        f"when the {rule['ci']:.0%} event-block bootstrap interval of selected minus baseline "
        f"lies entirely below zero ({rule['resamples']} resamples, seed {rule['seed']}). "
        "Negative means lower loss.",
        "",
        f"- **Mean move.** Metric `{q_mean['metric']}`. Baseline `{q_mean['baseline']}`, the "
        f"linear projection, not re-selected. Candidates: {', '.join(f'`{c}`' for c in q_mean['candidates'])}. "
        f"Selected `{mk}` ({_short(mk)}). Refit `{q_mean['refit']}`.",
        f"- **Left tail.** Metric `{q_tail['metric']}` at tau {tail['tau']}. Baseline "
        f"`{q_tail['baseline']}`, a Gaussian residual around the linear projection. Candidates: "
        f"{', '.join(f'`{c}`' for c in q_tail['candidates'])}. Selected `{tk}` ({_short(tk)}).",
        "",
        "Training settings are fixed in `config/ml_study.yaml` (learning rate, early stopping, "
        f"{len(cfg['training']['seeds'])} seeds averaged, Huber loss for the mean, pinball loss "
        "for the quantiles). The only choice made on validation is the small grid in that file.",
        "",
        "## Scores on the test set",
        "",
        "| Question | Selected model | Score | Baseline | Difference (90% CI) | Verdict |",
        "|---|---|---:|---:|---|---|",
        f"| Mean, pooled scaled MAE | {_short(mk)} | {_n(mean['pooled_scaled_mae'][mk])} "
        f"| linear {_n(mean['pooled_scaled_mae']['lp'])} | {_n(md['point'])} ({_ci(md['ci90'])}) "
        f"| {mean['verdict']} |",
        f"| Mean, predict zero | — | {_n(mean['pooled_scaled_mae']['zero'])} | — | — | reported |",
        f"| 5% pinball | {_short(tk)} | {_n(tail['pooled_scaled_pinball'][tk])} "
        f"| Gaussian {_n(tail['pooled_scaled_pinball']['gauss_lp'])} | {_n(td['point'])} "
        f"({_ci(td['ci90'])}) | {tail['verdict']} |",
        f"| 5% pinball, historical | — | {_n(tail['pooled_scaled_pinball']['hist'])} | — | — | reported |",
        "",
        f"Static refit of the mean model, which cannot change the verdict: "
        f"{_n(mean['static_refit']['difference_ml_minus_lp']['point'])} "
        f"(90% CI {_ci(mean['static_refit']['difference_ml_minus_lp']['ci90'])}), "
        f"{mean['static_refit']['verdict']}.",
        "",
        "## Calibration is reported and is not the decision",
        "",
        f"Mean-model 90% interval coverage: linear {_n(mean['coverage90']['lp'], 3)}, "
        f"selected {_n(mean['coverage90'][mk], 3)}. "
        f"Tail 5% hit rate: historical {_n(tail['hit_rate_05']['hist'], 3)}, "
        f"Gaussian {_n(tail['hit_rate_05']['gauss_lp'], 3)}, "
        f"selected {_n(tail['hit_rate_05'][tk], 3)}. "
        f"Tail central-interval coverage: Gaussian {_n(tail['coverage_90']['gauss_lp'], 3)}, "
        f"selected {_n(tail['coverage_90'][tk], 3)}. "
        "The tail win is the pinball loss named in advance. A hit rate above the nominal "
        "5% level is a calibration miss, and it is published beside the win.",
        "",
        "## Intended use",
        "",
        "A research comparison of flexible models with a linear local projection on "
        "announcement-window returns, using public data and a split that was tagged before "
        "fitting. The scenario tool and the live forecasts are separate code paths.",
        "",
        "## Out of scope",
        "",
        "Not a forecast of what markets will do, and not investment advice. Statement text, "
        "options, positioning and intraday paths were not features. Nothing here should be "
        "re-scored on the test set without a logged reason: `wsl mlstudy --final` refuses a "
        "second run. Limitations that apply to the whole project are in "
        "[LIMITATIONS.md](LIMITATIONS.md).",
        "",
        "## Provenance",
        "",
        f"- Result file: `results/mlstudy/final.json`, generated {doc['generated_at']}.",
        f"- Git commit recorded in that file: `{doc['git_sha']}`.",
        f"- Config sha256: `{doc['config_sha256']}`.",
        f"- Parent evaluation: `{doc['parent_final_eval']}`.",
        "- Inputs are pinned. The study note records how many artifacts still matched those "
        "pins. Sources and licences: [data/SOURCES.md](../data/SOURCES.md). "
        "How a hash gets into a result file: [PROVENANCE.md](PROVENANCE.md).",
        "",
        "The longer write-up, including validation scores, attribution and the secondary "
        "checks, is [ML_STUDY.md](ML_STUDY.md).",
        "",
    ])


def write() -> list[Path]:
    doc = _load()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    WEB.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(_md(doc))
    _figure(doc, FIG)
    WEB.write_text(json.dumps(_web(doc), indent=1) + "\n")
    CARD.write_text(model_card(doc))
    return [DOC, FIG, WEB, CARD]
