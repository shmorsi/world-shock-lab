# Evaluation protocol (pre-registered, v1)

- **Frozen:** 2026-10-12, at M0, **before any estimation, model fitting or look at outcome data around events.**
- **Machine-readable version:** `config/evaluation.yaml`. Its SHA-256 is written into every results file. If this document and the YAML disagree, the YAML wins and the disagreement is a bug.
- **Rule:** nothing below changes after test-set results exist. Any change is logged in the change log at the bottom, with the reason, and results under both versions are reported.

## 1. What this protocol governs

| Claim type | Governed by | Label |
|---|---|---|
| Out-of-sample prediction of asset moves given a shock (M5 ML benchmark) | §2–§7 | `model_forecast` |
| Event Replay errors shown in the web app (M7) | Same test events and models as §2–§7. No other events are scored. | `model_forecast` |
| Live pre-registered forecasts (M8) | §8 | `model_forecast` |
| In-sample causal estimates (M2–M4) | §9 fixes the *primary specification*, to limit forking paths. They are not "evaluated" out of sample. | `identified_causal` / `reduced_form` |

## 2. Event sets and time splits

All bounds are inclusive and use the announcement timestamp in the event's local time zone.

| Split | Window | FOMC (JK) | OPEC (Känzig daily) | BoC FAD (proxy) |
|---|---|---|---|---|
| Train | earliest → 2014-12-31 | 234 (from 1990-02-08) | 111 (from 1983-07-19) | TBD in M1 (from the first FAD with 2Y data, 2001) |
| Validation | 2015-01-01 → 2018-12-31 | 32 | 10 | TBD (about 32 expected) |
| Test | 2019-01-01 → 2026-09-30 | 68 | 48 (data end 2025-12-01) | TBD (about 62 expected) |
| Live | 2026-10-01 → | forward | — | forward |

The FOMC and OPEC counts were computed on 2026-10-12 from the pinned source files (see `data/SOURCES.md`). They count events, not outcomes. The BoC counts will be filled in from `data/events/boc_fad.csv` in M1 and logged in the change log as a factual completion, not a protocol change.

**Why these cut points**
- The test window holds about 8 years of events, including the 2020 emergency cuts, the 2022–23 hiking cycle and the 2024–26 easing. The validation window holds 4 years.
- The cut points were chosen from event counts and calendar years only, before any outcome data were loaded.
- Validation for OPEC is thin (10 events), and we say so in the write-up.

**Look-ahead controls**
- **Features** use only information available before the announcement: state variables at t−1, the latest *first-release* macro data, and raw surprise components.
- **No full-sample shock series as forecasting features.** JK `pc1` and its rotations, Känzig's `PC` and BS `MPS_ORTH` are estimated on the full sample (`docs/PLAN.md` P8). Any principal component used in forecasting is re-estimated inside each training window.
- **Targets** are computed with timestamp-aware windows: day 0 is the first observation of that series strictly after the announcement timestamp (`data/SOURCES.md` §5).

## 3. Refit policy

- **Primary: expanding annual refit.** For each test year Y, models are fit on all events up to Dec 31 of Y−1, using hyperparameters chosen on validation and then frozen.
- **Secondary: static.** Fit once on all data up to 2018-12-31 and frozen through the whole test window.
- **Hyperparameter search:** done only on the validation split, with blocked time-ordered cross-validation inside train where a model needs it (ridge penalty, boosting depth and learning rate, MLP width, weight decay and early stopping).

## 4. Targets

- **Horizons:** h ∈ {1, 5} trading days. The target is the cumulative change from the last observation before the announcement to the h-th observation after it.
- **Units:**
  - Basis points for yields and spreads.
  - 100 × log change for prices and FX. Prices are total-return, computed by us from raw closes and dividends.
- **Primary assets (8):**
  - Yields: US 2Y, CA 2Y
  - FX: USD/CAD
  - Equities: SPY, S&P/TSX Composite, Big-6 Canadian banks equal-weight, XEG.TO
  - Oil: WTI
- **Secondary assets:** everything else, reported but not part of the decision rule.

## 5. Models

| Code | Model | Notes |
|---|---|---|
| B0 | Zero forecast | Sanity floor |
| B1 | Unconditional LP | Linear in the shock, fitted on training data only |
| B2 | State-dependent LP | Pre-specified interactions with the §2 state features |
| B3 | Ridge | Shock components × state features |
| B4 | Gradient boosting | scikit-learn HistGradientBoosting |
| ML | Small MLP (PyTorch) | 5-seed ensemble. Optional monotonicity in the shock. |

**Sequence models** (TFT etc.) are allowed only if an event family has at least 2,000 training events. No family does, so none will be fitted.

## 6. Metrics and intervals

- **Primary metric:** pooled scaled MAE, i.e. the MAE for each asset × horizon divided by that cell's training-sample MAE of the zero forecast, then averaged with equal weights over 8 assets × 2 horizons.
- **Secondary metrics:**
  - RMSE
  - Out-of-sample R² against the zero forecast
  - Directional accuracy, computed on events where |realized move| is above the asset's training-sample 25th percentile of |move|; an all-events version is also reported
  - Coverage of the 68% and 90% intervals
  - Interval score (Gneiting & Raftery 2007)
- **Intervals:** econometric models use HAC-based residual intervals. ML models use split-conformal intervals calibrated on validation residuals.

## 7. Decision rule ("does ML add anything?")

1. On **validation**, pick one ML candidate (best primary metric among ML) and one econometric candidate (best among B1–B4). B0 is always reported.
2. On **test**, compute the difference in the primary metric, ML minus econometric.
3. Build a 90% CI for the difference with an **event-block bootstrap**: resample events with replacement and keep every asset and horizon of an event together. Use 10,000 resamples with seed 20261012.
4. Verdict:
   - **"ML adds value"** only if the whole CI is below zero.
   - **"ML is worse"** if the whole CI is above zero.
   - **"No evidence ML adds value"** otherwise.
   - The verdict is published as computed, whatever it is.
5. **Secondary splits** are descriptive only and cannot overturn the verdict:
   - excluding the March 2020 unscheduled events
   - by event family
   - excluding FOMC/BoC same-day collisions

**Test-set access.**
- Test events are loaded only by `wsl evaluate --final`.
- That command refuses to run unless the git working tree is clean and `sha256(config/evaluation.yaml)` matches the hash recorded at the `protocol-v1` tag.
- It writes `results/final_eval/<timestamp>.json`, and that file is committed.
- Every further run is logged in the change log with its reason.

## 8. Live pre-registered mode (M8)

- **Publish early:** a forecast is published at least 24 h before each FOMC and BoC decision, committed by a GitHub Action, and **never edited**. Corrections are new files that reference the old one.
- **Forecast content:** a conditional grid of surprise ∈ {−10, 0, +10} bp, in **proxy units**:
  - FOMC: the daily change in the US 2Y.
  - BoC: the daily change in the CA 2Y on the FAD.
  - Each cell gives the expected 1-day and 5-day move in CA 2Y, USD/CAD and TSX banks, with 68% and 90% intervals.
- **Scoring:**
  - Score 1 runs the day after the decision, using the realized proxy.
  - Score 2 runs when JK publishes the intraday surprise, with the grid interpolated linearly. Both scores are kept.
- **Collision days:** FOMC + BoC on the same day (2026-10-28 and 2026-12-09) are flagged as `collision` in the forecast file before the decision, and reported separately.

## 9. Primary in-sample specifications (M2–M4)

These are fixed now so robustness checks are clearly labelled as robustness.

- **Daily LP:**
  - Primary: event-sample regression of Δ_h y on the shock, for h = 0…20 trading days, with 5 lags of daily outcome changes and the pre-event state as controls, and HC3 standard errors.
  - Robustness: full daily series with zero-filled shocks and Newey–West with at least h+1 lags.
  - Bands: 68% and 90%.
- **Monthly LP:** h = 0…24, 12 lags of controls, Newey–West with at least h+1 lags.
- **LP-IV and proxy SVAR:** always report the effective first-stage F (Montiel Olea–Pflueger) and Anderson–Rubin intervals.
- **Regimes** (all use information at t−1; thresholds fixed in YAML):
  - High VIX: VIX(t−1) above its trailing 252-day median. Robustness: fixed thresholds of 20 and 25.
  - Cycle: sign of the 6-month change in the policy rate (hiking, cutting or hold).
  - High inflation: latest-release CPI YoY above 3.0%.
  - Oil exporter: WDI `EG.IMP.CONS.ZS` below 0 in the latest vintage available before the event year.
- **Multiple testing:** Benjamini–Hochberg at q = 0.10 within each family (spillover edges; state-dependence tests). The spillover network uses h ∈ {1, 5}.
- **Placebo:** 1,000 random non-event trading days (seed 20261012). The 5 days around any event are excluded.

## Change log

| Date | Version | Change | Reason | Seen test results? |
|---|---|---|---|---|
| 2026-10-12 | v1 | Initial protocol | — | No |
