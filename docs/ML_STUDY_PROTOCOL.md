# Machine-learning study protocol (ml-study-v1)

Frozen on 2026-10-20, **before any model in this study was fit**. The machine-readable copy is `config/ml_study.yaml`. If this page and the YAML disagree, the YAML wins.

This study does **not** amend `config/evaluation.yaml` or the M5 verdict in `results/final_eval/`. Protocol v3 was already scored once ("no evidence ML adds value"). Re-opening that comparison would be moving the goalposts. The questions below are new, on the same announcements and the same calendar splits.

## Why a second study

M5 fit a separate ridge, gradient-boosting and small MLP model on each asset and horizon. With a few hundred announcements, those models had little to learn beyond the linear surprise, and the published permutation importances say so. This study asks whether two designs that M5 did not try change that, and whether the *tail* of the move (a risk question, not a mean question) is where a more flexible model helps.

Both expectations are written down before fitting. A gain is not required. A result that loses to the linear model is reported with the same confidence interval as a win.

## Questions

**Q1, mean response.** Pooled scaled MAE, the M5 metric: each asset × horizon's mean absolute error, divided by that cell's mean absolute move on the training split (through 2014), then averaged with equal weights over 8 assets × 2 horizons.

- Baseline, pre-specified: the unconditional linear local projection (M5's validation winner). A forecast of zero is reported too.
- Candidates, one chosen on the 2015–2018 validation window: a multi-task MLP (shared trunk, learned asset embedding, both horizons) and a temporal CNN that adds the asset's 20 pre-event daily moves.
- Primary scores: expanding walk-forward. For each test year Y the model is refit on every earlier announcement.
- A single fit through 2018, frozen for the whole test window, is reported and cannot change the verdict.

**Q2, shock-conditional 5% VaR.** Pooled scaled pinball loss at τ = 0.05. The scale is the training-split pinball loss of the historical (unconditional) 5% quantile.

- Baseline, pre-specified: the linear projection plus a Gaussian residual. The historical quantile is reported too.
- Candidates: histogram gradient boosting with quantile loss, per asset and horizon, and a multi-task network with non-crossing quantile heads.
- The lower-tail expectation (CVaR) is the average of the predicted quantiles at 0.01, 0.02, 0.03, 0.04 and 0.05. Gaussian and historical CVaR use their closed forms.
- The decision uses the 5% pinball loss only.

## Decision rule

On validation, pick the candidate with the best primary metric. On the test set, compute selected minus baseline. A 90% interval comes from an event-block bootstrap: resample announcements with replacement, keep every asset and horizon of an announcement together, 10,000 draws, seed 20261020.

- The whole interval below zero: the selected model **adds value** (on the mean, or on the 5% tail).
- The whole interval above zero: it **is worse**.
- Otherwise: **no evidence it adds value**.

Family splits, interval coverage, the Fissler–Ziegel score, the equity-book portfolio, the embedding clusters and the Shapley values are descriptive. They cannot change either verdict. The Fissler–Ziegel score is the strictly consistent form with g(x) = x and φ(e) = exp(e) (Fissler and Ziegel 2016), so the comparison stays valid when the shortfall is negative.

## What the models may see

Same leakage rules as protocol v3. Features are the raw surprise components (never a full-sample principal component), the pre-announcement state, and, for the temporal CNN only, the 20 daily changes of that asset ending at the last observation strictly before the announcement. Targets are the move from that same observation to 1 and 5 observations after it. Bank of Canada announcements that fall on FOMC days stay out of the primary sample.

Hyperparameters that are searched are only the lists in the YAML. Learning rate, epochs, seeds, dropout, embedding size and early-stopping fraction are fixed. Early stopping uses the last 20% of the training window, in time order. Imputation and scaling use only the earlier 80% (the rows that receive gradients). That same early-stopping slice calibrates the mean model's split-conformal intervals, and the walk-forward coverage uses each test year's own slice rather than one width for the whole window. The linear projection's intervals are Gaussian, using that year's training residual standard deviation. Quantile models use their predicted 5% and 95% quantiles as the interval, with no further adjustment. Coverage is a result, not a target that gets tuned. Gradient boosting fits one model per quantile; model selection fits only the 5% quantile, and the chosen depth is refit at the other reported quantiles for the test scores. Those other quantiles do not enter the decision.

## Embeddings, attribution, portfolio, stress

- **Embeddings.** The selected mean model's asset embedding, fit on data through 2018. k-means with k = 4 against the pre-specified groups yields / FX / equity / oil. An adjusted Rand index, and a permutation p-value over 2,000 shuffles of those labels. Descriptive.
- **Attribution.** Exact Shapley values over feature groups (surprise, event family, macro state, the asset's own volatility, and the return path when the model reads it), on the test announcements, horizon 1, from the model fit through 2018. The baseline for a feature is its training mean within the asset.
- **Portfolio.** A fixed book, 40% SPY, 30% TSX, 20% the big-six banks, 10% XEG, one-day move. The same tail models, scored with pinball loss. Secondary.
- **Stress.** A hypothetical FOMC surprise of one training-sample standard deviation in the two-year futures surprise, other futures components set to their training co-movement, state at the training median. The model fit through 2018 reports the median and the 5% quantile against an otherwise identical zero-surprise row. Labelled hypothetical.

## Test-set access

`wsl mlstudy --final` refuses to run unless the working tree is clean of code and config changes, `sha256(config/ml_study.yaml)` matches the `ml-study-v1` tag, a validation file exists and was produced under that same hash, and no final file exists yet. The command writes `results/mlstudy/final.json`.

## Change log

| Date | Version | Change | Seen this study's test results? |
|---|---|---|---|
| 2026-10-20 | ml-study-v1 | Initial protocol. Protocol v3 and its verdict are unchanged. | No |
