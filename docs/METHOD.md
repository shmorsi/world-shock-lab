# Methodology

A short account of how an estimate is made. The numbers live in the generated reports, not here, so this page cannot drift from them. The site's Methodology view says the same thing beside the replication table.

## Identification

Markets move for many reasons at once. A Fed surprise is the move in interest-rate futures inside a short window around a scheduled announcement (Jarociński and Karadi), so almost nothing else is in it. Oil supply news is the oil-futures move on OPEC announcement days (Känzig). There is no free intraday record for Canada, so the Bank of Canada surprise is the announcement-day change in the 2-year Government of Canada yield. That proxy was validated before it was used, and every equity result is also shown with the Rigobon–Sack estimator. Geopolitical risk and trade-policy uncertainty are news-intensity indexes. They are labelled reduced form, not shocks.

## Estimation

For each horizon, one local projection asks how much an asset moved from just before the announcement to that many days after, per unit of surprise. Each series uses its own observation clock, so a 14:00 statement is not in a noon exchange-rate fix until the next print. Bands are 68% and 90%. The same estimators were checked on simulated data with a known answer before they were used on the announcements ([SIMULATION.md](SIMULATION.md)).

Where a shock is weak, the note reports an Anderson–Rubin set instead of treating a point estimate as precise. Monthly macro responses are the weak case. Event-window responses are the strong case.

## Labels

Every number on the site and in the notes carries one of four labels.

- **Identified causal.** A high-frequency surprise, or the validated Bank of Canada proxy.
- **Reduced form.** An association with a news index.
- **Model forecast.** A machine-learning score or a live forecast.
- **Hypothetical scenario.** Responses added together in the scenario tool, with the linearity assumption stated.

## What was fixed before looking

Splits, metrics and the rule for "adds value" are in [config/evaluation.yaml](../config/evaluation.yaml) and [config/ml_study.yaml](../config/ml_study.yaml), tagged `protocol-v1` through `protocol-v4` and `ml-study-v1` before the fitting those tags govern. Each test set was scored once. The change log is [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). The model card for the second study is [MODEL_CARD.md](MODEL_CARD.md).

## Where the code is

Timing is `wsl/timing.py`. Local projections, LP-IV and the bootstrap are `wsl/estimation/`. The Canada comparison, replication and robustness are `wsl/analysis/`. The two engines that build a scenario, Python and TypeScript, share fixtures under `wsl/scenarios/` and `web/src/engine/`.
