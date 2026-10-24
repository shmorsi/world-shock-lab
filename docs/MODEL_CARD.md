# Model card

Generated from `results/mlstudy/final.json` by `wsl.mlstudy.report.model_card`. Do not edit by hand. This is the second study. It does not replace the per-asset benchmark in [ML_BENCHMARK.md](ML_BENCHMARK.md).

## What was compared

Protocol `ml-study-v1`, frozen 2026-10-20. The calendar split is the parent protocol's: train through 2014-12-31, validation 2015-01-01 to 2018-12-31, test 2019-01-01 to 2026-09-30. This file counts 456 training announcements, 74 validation, and 168 test (2019–2026).

Validation chooses the model. The test set is scored once. A model adds value only when the 90% event-block bootstrap interval of selected minus baseline lies entirely below zero (10000 resamples, seed 20261020). Negative means lower loss.

- **Mean move.** Metric `pooled_scaled_mae`. Baseline `lp`, the linear projection, not re-selected. Candidates: `mt_mlp`, `tcn`. Selected `tcn:{"channels": 16, "weight_decay": 0.001}` (Temporal CNN). Refit `expanding_annual`.
- **Left tail.** Metric `pooled_scaled_pinball` at tau 0.05. Baseline `gauss_lp`, a Gaussian residual around the linear projection. Candidates: `qgbm`, `qnet`. Selected `qgbm:{"max_depth": 2}` (Quantile boosting).

Training settings are fixed in `config/ml_study.yaml` (learning rate, early stopping, 3 seeds averaged, Huber loss for the mean, pinball loss for the quantiles). The only choice made on validation is the small grid in that file.

## Scores on the test set

| Question | Selected model | Score | Baseline | Difference (90% CI) | Verdict |
|---|---|---:|---:|---|---|
| Mean, pooled scaled MAE | Temporal CNN | 1.157 | linear 1.142 | 0.015 (-0.006 to 0.037) | No evidence ML adds value |
| Mean, predict zero | — | 1.195 | — | — | reported |
| 5% pinball | Quantile boosting | 1.446 | Gaussian 1.558 | -0.112 (-0.228 to -0.010) | ML adds value on the 5% tail |
| 5% pinball, historical | — | 1.595 | — | — | reported |

Static refit of the mean model, which cannot change the verdict: 0.026 (90% CI 0.006 to 0.046), ML is worse.

## Calibration is reported and is not the decision

Mean-model 90% interval coverage: linear 0.887, selected 0.871. Tail 5% hit rate: historical 0.068, Gaussian 0.063, selected 0.090. Tail central-interval coverage: Gaussian 0.887, selected 0.835. The tail win is the pinball loss named in advance. A hit rate above the nominal 5% level is a calibration miss, and it is published beside the win.

## Intended use

A research comparison of flexible models with a linear local projection on announcement-window returns, using public data and a split that was tagged before fitting. The scenario tool and the live forecasts are separate code paths.

## Out of scope

Not a forecast of what markets will do, and not investment advice. Statement text, options, positioning and intraday paths were not features. Nothing here should be re-scored on the test set without a logged reason: `wsl mlstudy --final` refuses a second run. Limitations that apply to the whole project are in [LIMITATIONS.md](LIMITATIONS.md).

## Provenance

- Result file: `results/mlstudy/final.json`, generated 20261020T183639Z.
- Git commit recorded in that file: `d0dc395c530d37d225381b3bd7391ee54f83c223`.
- Config sha256: `ff092f530809698de3e788d0411e7bc8e29ae8da70b11cb0cf05f1d1d886b2dd`.
- Parent evaluation: `results/final_eval/20261012T233454Z.json`.
- Inputs are pinned. The study note records how many artifacts still matched those pins. Sources and licences: [data/SOURCES.md](../data/SOURCES.md). How a hash gets into a result file: [PROVENANCE.md](PROVENANCE.md).

The longer write-up, including validation scores, attribution and the secondary checks, is [ML_STUDY.md](ML_STUDY.md).
