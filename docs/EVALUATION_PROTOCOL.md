# Evaluation protocol (pre-registered, v3)

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
| Train | earliest → 2014-12-31 | 234 (from 1990-02-08) | 111 (from 1983-07-19) | 112 scheduled (from 2001-01-23); 111 after v2 |
| Validation | 2015-01-01 → 2018-12-31 | 32 | 10 | 32 scheduled; 32 after v2 |
| Test | 2019-01-01 → 2026-09-30 | 68 | 48 (data end 2025-12-01) | 62 scheduled; 52 after v2 (10 on FOMC days) |
| Live | 2026-10-01 → | forward | — | forward |

The FOMC and OPEC counts were computed on 2026-10-12 from the pinned source files (see `data/SOURCES.md`). They count events, not outcomes. The BoC counts come from `data/events/boc_fad.csv`, built and validated in M1, and are logged in the change log as a factual completion, not a protocol change.

FOMC/BoC same-day announcements fall very unevenly across the splits: 1 in train, 0 in validation, **10 of 62 BoC test events**.

**v2 amendment (2026-10-12, before any estimation):** BoC events on FOMC days are excluded from the BoC event family, leaving 111 / 32 / 52 scheduled events.
- **Why exclude them from the BoC family:** on those days the daily CA 2Y change, which is our BoC proxy, also contains the effect of the 14:00 ET Fed announcement. That makes it a contaminated measure of the BoC surprise, with an error correlated with a shock that also moves the outcomes.
- **Why keep them in the FOMC family:** the JK surprise is a 30-minute window and stays clean. The BoC announced earlier the same morning (09:45–10:00 ET), so its decision cannot respond to that day's Fed surprise; on those days it only adds noise to Canadian outcomes, not bias.
- **Transparency:** the excluded BoC events are still reported as a secondary subsample.

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
   - the BoC events dropped by the v2 amendment (FOMC same day), reported separately

**Test-set access.**
- Test events are loaded only by `wsl evaluate --final`.
- That command refuses to run unless the git working tree is clean and `sha256(config/evaluation.yaml)` matches the hash recorded at the latest `protocol-v*` tag (currently `protocol-v3`).
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

### v4 addendum: live details fixed before the first live forecast

§8 did not fix the forecasting model, the horizon convention, or what the committed files may contain. These are fixed in `config/live.yaml` (version 1) before any live forecast was generated:

- **Calendar:** FOMC and BoC dates for 2026-10-28 through 2027-12-08, verified on 2026-10-12 against the Federal Reserve and Bank of Canada schedule pages. Collisions (both banks on one day): 2026-10-28, 2026-12-09, 2027-01-27, 2027-04-28, 2027-10-27 and 2027-12-08.
- **Model:** for each bank, outcome and horizon, OLS of the outcome change on the proxy over the bank's past announcements since 2000 with complete data, refit at each forecast on data published before it. BoC history follows v2 (scheduled, no FOMC same-day decisions); FOMC history is every JK announcement.
- **Intervals:** 68% and 90% prediction intervals from the absolute leave-one-out residuals (jackknife), so they describe the realized outcome rather than the mean response.
- **Horizons:** h = 1 is the first observation after the announcement and h = 5 the fifth, both measured from the last observation before it (the M5 convention). On BoC days, CA 2Y at h = 1 *is* the proxy; it is shown but not scored.
- **Score 1 timing:** §8 said "the day after the decision". Score 1 needs every h = 5 value, and USD/CAD comes from the weekly H.10 release, so Score 1 runs once all of them are published (typically 6–10 days after). The forecast at the realized proxy is the linear model, which equals linear interpolation of the grid inside it and extends it linearly outside.
- **Score 2 (FOMC only):** JK's 30-minute change in the 2-year Treasury futures yield (`TFUT02`, in bp) is the intraday counterpart of the proxy.
- **Benchmark:** every score is also computed for a forecast of zero (no change).
- **Collision days:** both banks' forecasts are published and scored, flagged `collision`, and summarized separately from clean days.
- **Licensing:** committed files contain realized values only for public-domain series ([C]). For CA yields (O-3) and Yahoo prices (O-4), a score records only whether the outcome fell inside each band and whether the model beat the benchmark. The realized BoC proxy (a CA 2Y change) is not published for the same reason.

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

### v3 additions: M4 details fixed before any M4 estimation

§9 promised a test "at the pre-specified headline horizon" but never named it, and the full list of state-dependence tests was not written down. These are fixed here, and in `config/evaluation.yaml`, before any M4 estimate exists.

- **Headline horizons:** 5 trading days (daily) and 12 months (monthly).
- **State-dependence family:** every shock × outcome × regime cell listed in the YAML. That gives 18 FOMC, 12 BoC and 10 OPEC cells, each tested two ways: jointly across all horizons, and at the headline horizon. All 80 tests form one Benjamini–Hochberg family at q = 0.10.
  - Interaction coefficients are estimated by a stacked event-sample LP with errors clustered by event, so the joint test across horizons uses their covariance.
  - Regime CPI is the year-on-year rate of the month two months before the event, which is always published by then.
- **Canada comparison:** effects of a Fed move and a BoC move on Canadian assets, both normalized as a 10 bp move in the own country's 2-year yield on day 0. For the Fed, the US 2-year change is instrumented by the JK MP shock; for the BoC, it is the proxy itself.
- **Oil exporters vs importers:** a panel of US-listed country ETFs on the OPEC surprise, with an exporter interaction (classification as in §9) and errors clustered by event.
- **Robustness:** the variants listed in the YAML. A result is *robust* only if it has the same sign in every applicable variant and is 90%-significant in at least two thirds of them; otherwise it is *fragile*.
- **Proxy BVAR:**
  - Data: monthly, 7 variables, 12 lags.
  - Prior: Minnesota, implemented with dummy observations; overall tightness 0.2.
  - Instrument: primary BS `MPS_ORTH`, robustness JK MP.
  - Normalization: the US 1-year yield rises 25 bp on impact.
  - Bands: Jentsch–Lunsford moving-block bootstrap, 500 draws.
  - Compared with a monthly LP-IV on the same variables and instrument.
- **Disclosure:** the primary VAR instrument (BS `MPS_ORTH`) was chosen after M3 showed the raw surprise produces the output puzzle. That is also the paper's own recommendation. JK MP is reported alongside it.

## Change log

| Date | Version | Change | Reason | Seen test results? |
|---|---|---|---|---|
| 2026-10-12 | v1 | Initial protocol | — | No |
| 2026-10-12 | v1 (no change) | BoC event counts filled in (112 / 32 / 62); collision imbalance noted | Factual completion promised in §2; `config/evaluation.yaml` unchanged | No |
| 2026-10-12 | v2 | BoC family excludes FOMC same-day events (primary); they are reported as a secondary subsample | The daily BoC proxy on those days also contains the Fed announcement (contaminated shock measure). Found while building the BoC list in M1. Approved by the project owner. | No: no estimation of any kind had been run |
| 2026-10-12 | v3 | M4 details: headline horizons, the full state-dependence test family, the Canada normalization, the exporter panel, robustness variants and rule, the proxy-BVAR spec | §9 left the headline horizon and test family unspecified; fixed before any M4 estimate. The out-of-sample (ML) protocol is unchanged. | No: M3 in-sample results existed; no M4 estimate and no test-set evaluation had been run |
| 2026-10-12 | v3 (no change) | **Final test evaluation, run 1** (20261012T233454Z, commit ad8c4e5). Verdict as computed: "No evidence ML adds value". Results: `results/final_eval/20261012T233454Z.json`, report: docs/ML_BENCHMARK.md | The single test-set run the protocol allows | Yes: this is that run; no model, feature or setting was changed afterwards |
| 2026-10-12 | v4 | Live-mode details (`config/live.yaml` v1): calendar, forecasting model, jackknife intervals, horizon convention, Score 1 timing, Score 2 surprise measure, benchmark, licensing of committed scores | §8 left the live model and these details unspecified; fixed before the first live forecast. `config/evaluation.yaml` is unchanged. | Yes: the single test-set run (v3) had been made. The live model is a simple proxy regression, not one of the M5 candidates, and was not tuned on test results. |
| 2026-10-13 | v4 (no change) | Git history re-authored to the owner's account before first publication. File trees, messages and dates unchanged; commit hashes changed. Old hashes cited in this log and in result files map to published ones in [COMMIT_MAP.md](COMMIT_MAP.md) | Attribution | No new evaluation run |
