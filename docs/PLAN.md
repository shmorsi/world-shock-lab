# World Shock Lab: M0 plan (scope, design decisions, risks)

**Status (final):** all milestones M0–M8 are built and committed; see "Where the project ended up" below. What follows it is the original M0 plan, written 2026-10-12 before any estimation and kept unchanged as the record of what was planned. Counts in it describe the input data, not results.

## Where the project ended up

Results live in the generated reports (README findings, docs/RESEARCH_NOTE.md, docs/RESULTS.md, docs/ML_BENCHMARK.md); this table records only what was delivered against §6 and where it fell short.

| Milestone | Delivered | Short of the plan, or changed |
|---|---|---|
| M1 Data | Hashed raw cache, schemas, data-quality report, BoC announcement list validated against the target rate | O-8 (OECD CPI gaps for some countries) is still open; it affects only cross-country monthly macro, which the final scope does not use for headline results |
| M2 Estimation | LP, LP-IV, HAC/HC errors, Monte Carlo recovery tests in CI | The zero-filled daily LP under-covers in simulation, so it is a robustness check only (as planned in P12) |
| M3 Identification | Timestamp-aware shocks, replication table, placebo, BoC proxy validation, O-1 resolved by testing | The BoC proxy passed validation but still carries common North American news (found in M4); Rigobon–Sack is shown alongside every BoC result |
| M4 VAR, regimes, Canada | Proxy BVAR vs LP-IV, state-dependence family with FDR, Canada lens, exporter panel, spillover network, robustness matrix | **One acceptance criterion not met:** the proxy-BVAR bootstrap bands cover below the [0.85, 0.95] window in simulation. Documented rather than tuned; LP bands are the reference wherever the two disagree. No regime effect survives FDR control |
| M5 ML benchmark | Run exactly per the protocol; test set scored once; verdict written as computed | — |
| M6 Scenario engine | Catalog of single-shock event LPs (Fed policy and information responses from one joint regression, with their covariance); shocks from different event families are added; label propagation; out-of-range and linearity warnings; JSON with provenance; Python and TypeScript engines tied by fixtures | A fully joint LP across families is not estimable because their events rarely coincide; the additivity assumption is stated in every scenario |
| M7 Web app | Six views plus overview and track record; static JSON; keyboard access and a table alternative for every chart; brief "no number not in JSON" test | Load time is measured with a scripted Playwright profile equal to Lighthouse's slow-4G and 4x-CPU settings rather than with Lighthouse itself; accessibility with axe-core |
| M8 Live mode and write-up | `config/live.yaml` (protocol v4), daily workflow, first forecasts for 2026-10-28, research note and PDF, README, limitations, resume bullets | Publishing steps (repository, secrets, Pages, licensing decisions O-3/O-4/O-9) are for the owner: docs/SHIP_CHECKLIST.md |

---

**Original M0 status line:** M0, written 2026-10-12. No estimation has been run. All counts below describe the input data (number of events, date coverage) and were computed while verifying the sources. None of them is a result.

Companion documents: `data/SOURCES.md` (every source, verified) and `docs/EVALUATION_PROTOCOL.md` + `config/evaluation.yaml` (the pre-registered splits).

---

## 1. What changed from the brief after checking the data

| Brief | Finding | Decision |
|---|---|---|
| Bauer–Swanson (BS) as a US policy shock | BS ends 2023-12-13. Jarociński–Karadi (JK) runs to **2026-09-16** (334 events) and includes the raw 30-min futures surprises. JK is CC BY 4.0. BS states no licence. | **JK is the primary Fed shock.** BS is used for the replication table and as a robustness instrument. Because JK is current, the held-out test can use real intraday surprises up to Sep 2026. |
| ETF prices from "a reliable free source" + a second source | Yahoo works but is unofficial and for personal use only. **Stooq now serves a JS proof-of-work bot check, which we will not get around.** Nasdaq's quote API gives about 10 years of unadjusted closes for US-listed ETFs. **No free second source for TSX-listed prices.** | Yahoo is primary. Nasdaq spot-checks US-listed ETFs. For the TSX we rely on internal consistency checks and state the gap as a limitation. Prices are never committed or shown. |
| S&P 500 from FRED | FRED SP500 has only 10 years and needs S&P pre-approval to republish. | Use SPY (total return computed from raw close + dividends). |
| Credit spread (Moody's Baa–10Y) | BAA10Y carries a Moody's notice: it may not be redistributed or stored. | Local-only **[L]**. We publish derived statistics only. |
| USD/CAD from FRED | DEXCAUS is the **noon** NY fix, but FOMC statements come at 14:00 ET. BoC's own FXUSDCAD is a *daily average* and starts in 2017. | Use **timestamp-aware event windows**: each series' day 0 is its first observation strictly after the announcement time. For FOMC events, FX day 0 runs from the announcement-day noon fix to the next day's noon fix. |
| Trade-policy shocks | TPU and GPR are newspaper-based *uncertainty and attention* indices, not exogenous shocks. The only identified tariff series we found (Den Besten–Barnichon–Känzig–Singh 2026) is **annual**, 1840–2024. | Every TPU and GPR output is labelled **reduced-form association**. A "trade-policy spike" in a scenario can never be labelled causal. |
| BoC announcement-day proxy | No official list of past announcement dates exists. Valet has no such group, and the business-daily policy-rate series starts in 2009. **The closing time of GoC benchmark yields is not documented.** Valet T-bills and FX are intraday averages. | We build `boc_fad.csv` from press releases with a URL per row and validate it against policy-rate changes. Open item O-1 (closing time) must be resolved before M3. |
| Live mode on upcoming FOMC and BoC dates | **FOMC and the BoC both announce on 2026-10-28 and on 2026-12-09.** | Both dates are flagged as *collisions*. The BoC daily proxy cannot separate the two announcements on those days. See P11. |

---

## 2. Where I'm pushing back (statistical)

**P1. "+25 bp Fed surprise" is well outside the data.**
- The JK `pc1` surprise has a standard deviation of **6.5 bp** across 334 events.
- Only **5 events** were 25 bp or larger, all of them *cuts* before 2009.
- The largest *hawkish* surprise is **+16.6 bp**.
- A +25 bp hawkish scenario is therefore an extrapolation, and the linear model behind it has never been tested at that size.
- **Decision:** the Scenario Composer sizes shocks in standard deviations, shows each size's percentile in the historical distribution, and shows a hard warning beyond the observed range. The live grid of −10 / 0 / +10 bp (about ±1.5 s.d.) is reasonable.

**P2. Combining shocks needs more than a "linearity" warning.**
- Adding responses assumes (a) linearity, (b) no interaction between shocks, (c) comparable units, and (d) a joint sampling distribution. Summing marginal bands ignores the covariance between estimates.
- **Decision:**
  - Estimate a *joint* multi-shock local projection on the common sample, so the bands of a combined scenario come from one covariance matrix (or a joint bootstrap).
  - A combined scenario gets the **weakest label** among its parts, so causal plus reduced-form is reduced-form.

**P3. Spillover network where "edges = significant responses" is a multiple-testing trap.**
- About 30 nodes × several shocks × 21 horizons would produce many false edges by chance.
- **Decision:** pre-specify the horizons (cumulative h = 1 and h = 5 days) and apply Benjamini–Hochberg at q = 0.10. Edge weight is the point estimate with its interval. Non-significant edges are drawn faint rather than removed, so the graph doesn't overstate certainty.

**P4. State dependence needs more than visual differences.**
- **Regimes from data before the event.** Every regime is defined from information available at t−1.
- **Report sample sizes.** Show the number of events per regime.
- **Test the difference.** Use a joint HAC Wald test across horizons, plus a test at the pre-specified headline horizon.
- **Fix thresholds in advance.** Thresholds are set in `config/evaluation.yaml`.
- **Caveat on hiking vs cutting.** Surprise distributions differ by cycle, so a regime difference can partly reflect different surprise sizes. We will show surprise distributions by regime.
- **Caveat on power.** High-VIX regimes contain few FOMC events, so we report minimum detectable effects alongside the tests.

**P5. "Oil exporter vs importer" is cross-sectional, small-N and changes over time.**
- With roughly 12–15 economies, power is low.
- Status changes over time. In the World Bank data the US has net energy imports of −9.8% of energy use in 2024, making it a net *exporter* today.
- **Decision:**
  - Panel LP with Driscoll–Kraay standard errors.
  - A time-varying classification from WDI `EG.IMP.CONS.ZS`, with the rule fixed in config.
  - We expect wide intervals and will say so.

**P6. Proxy SVAR/BVAR belongs at monthly frequency.**
- Daily VARs with external instruments are non-standard, and the literature this replicates (Gertler–Karadi, Känzig) is monthly.
- **Decision:**
  - Daily data: local projections (LP) only.
  - Monthly data: both LP and the proxy BVAR on the *same* variables and instrument, so the LP-vs-VAR comparison is like for like.
  - Always report the first-stage effective F (Montiel Olea–Pflueger) and weak-IV-robust (Anderson–Rubin) intervals. Monthly-aggregated surprises are known to be weak instruments in some samples.

**P7. A Temporal Fusion Transformer is not justified.**
- Labelled events: FOMC 234 train / 32 validation / 68 test, OPEC 111 / 10 / 48. BoC is expected to be about 8 a year from 2001.
- **Decision:** ML is limited to ridge, gradient boosting and a small MLP (seed ensemble, optional monotonicity in the shock). A sequence model would only be considered if each event family had at least 2,000 training events. That rule is fixed in config, and none of our families meets it.
- **Expectation, stated in advance:** realized moves are dominated by non-shock news, so every model's out-of-sample R² against a zero forecast will be small. The question is only whether state-conditioning reduces error, measured with event-clustered bootstrap intervals.

**P8. The published shock series contain look-ahead.**
- JK `pc1` and its rotations, Känzig's `PC`, and BS `MPS_ORTH` are all estimated on the full sample. For example, the 1983-07-19 Känzig `PC` differs between the full and pre-Covid sheets.
- **Decision:** forecasting (M5, M8) uses **raw surprise components** only, plus PCs re-estimated within each training window. The full-sample series are used only for in-sample causal estimates.

**P9. Validating the BoC proxy without free survey expectations.**
- Bloomberg and Reuters polls and BAX/OIS quotes are not free. Planned tests:
  1. **Heteroskedasticity check.** Compare the variance of 2Y changes on FADs with the variance on matched non-FAD days (Rigobon–Sack). We also report a heteroskedasticity-based IV estimate as an alternative identification.
  2. **Placebo days.**
  3. **Survey expectations.** The BoC's *Market Participants Survey*, published on BoC pages since about 2014, covers only some FADs. Its format is to be checked in M3.
  4. **US-purged variant.** Remove the same-day US 2Y change. This also removes legitimate co-movement, so it is robustness only.
  5. **Drop collision days** with FOMC.
- If the variance ratio is close to 1, the proxy is weak. **We will report that as a negative result** and lean on Fed→Canada transmission, which is cleanly identified, for the Canada lens.

**P10. Tariff events in Event Replay** (e.g. 2025) can only be replayed with the reduced-form TPU/GPR model, and are labelled that way.

**P11. Live mode design.**
- **Score against a proxy that is available the same day.** That means the daily change in the US 2Y Treasury on FOMC days and the CA 2Y on FAD days. The conditional forecast grid must be stated **in those same proxy units**, so the forecast and the score measure the same thing.
- **Score a second time** once JK publishes the intraday surprise (the 2026-09-16 meeting appeared by 2026-10-11, about 4 weeks later).
- **Flag the 2026-10-28 and 2026-12-09 collisions** in the pre-registration.
- **Don't start early.** Pre-registering forecasts before the M3 validation would mean publishing forecasts from an unvalidated model. The first live date we could realistically use is after M4.

**P12. Daily LP specification.**
- **Primary specification:** event-sample regressions of the cumulative change from the last pre-announcement observation to h days after, on the surprise, with pre-event controls and robust standard errors.
- **Robustness:** a full daily LP with zero-filled shocks and Newey–West with at least h+1 lags.
- Scheduled meetings are about 6 weeks (roughly 30 trading days) apart, so 20-day windows rarely overlap. Unscheduled meetings are handled by the HAC correction.

---

## 3. Final proposed scope

**In scope**
- **Shocks:**
  - JK Fed (pc1, MP, CBI) and raw components
  - BS (replication)
  - JK ECB
  - Känzig oil supply (daily OPEC surprises + monthly news shock)
  - Baumeister–Hamilton oil supply and demand (monthly)
  - BoC FAD proxy (constructed)
  - GPR and TPU (reduced-form)
- **Daily outcomes** (core, about 35 series):
  - US 1Y / 2Y / 10Y and CA 2Y / 5Y / 10Y yields, CORRA
  - USD/CAD and 9 other H.10 FX pairs
  - WTI, Brent, Henry Hub
  - VIX, Baa–10Y [L], 5Y/10Y breakevens
  - SPY, S&P/TSX Composite, Big-6 Canadian banks (equal weight), XEG.TO (Canadian energy)
  - About 12 US-listed iShares country ETFs
- **Monthly outcomes:** CPI, IP (Canada: monthly GDP) and unemployment for the US, Canada and 6–8 OECD economies, chosen to mix oil exporters and importers.
- **Methods:** A–E as in the brief, with the changes in §2.

**Out of scope or deferred**
- TFT and other sequence models (P7).
- Intraday data, which is not free (we use the published intraday surprises).
- Daily foreign government yields beyond the US and Canada. ECB and BoE daily curves could come later, but they are not yet verified.
- Local-currency equity indices with non-synchronous closes. US-listed country ETFs close at 16:00 ET with the US market, but they are USD-denominated, so they mix the equity and FX responses, which we state.

**Fixed labels** (every output carries exactly one):
- `identified_causal`: JK, BS, Känzig, Baumeister–Hamilton and the BoC proxy, after M3 validation.
- `reduced_form`: GPR, TPU, and any regression without an identified shock.
- `model_forecast`: ML outputs and live forecasts.
- `hypothetical_scenario`: Composer outputs. These also carry the weakest label among their parts.

---

## 4. Architecture

```
downloaders (wsl/data/sources/*.py)
  └─> data/raw/<source>/<retrieved_at>/<file>    immutable; SHA-256 in data/manifest.json (committed)
validate + normalize (pandera schemas, explicit missing-data rules, Hypothesis tests)
  └─> data/clean/*.parquet  (long format: series_id, ts_utc, value, obs_time_local, provenance_id)
DuckDB views (data/wsl.duckdb, rebuilt from parquet, never edited by hand)
estimation (wsl/estimation: lp.py, lpiv.py, svar.py, bvar.py, state_dependent.py)
  └─> results/*.json   each carries input hashes, git SHA, config hash, label
web/ (static, reads results JSON only)
```

- **Python:** Python 3.12 via `uv`. `uv python install 3.12` is a small download; only 3.14 is installed today.
- **Libraries:** pandas, pyarrow, duckdb, numpy/scipy, statsmodels (HAC kernels, cross-checks), pandera, typer (the `wsl` command), scikit-learn (ridge, HistGradientBoosting, so no LightGBM dependency), torch (CPU), pytest and hypothesis, ruff, mypy.
- **Own code:** LP, LP-IV and the Minnesota BVAR are our own implementations, verified against simulated data with known IRFs and cross-checked against statsmodels where an equivalent exists.
- **Missing data:**
  - Holidays: US and Canadian calendars differ (Canada Day, Victoria Day, Canadian Thanksgiving). Never forward-fill into an event window. A missing day-0 observation means the event is dropped for that series and logged.
  - **Negative WTI (April 2020):** prices are kept as levels. Log returns are undefined for that print, so WTI responses use the level change ($/bbl) as a robustness check and drop the event from the log-return sample, with the drop logged.
  - **Benchmark switches** in GoC yields are flagged and excluded from event windows.
  - Discontinued series are flagged.

---

## 5. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Yahoo endpoint changes or blocks | Medium | High | Immutable raw cache. Downloader fails loudly. Never bypass blocks. Fallback is to stop and ask you. |
| R2 | BoC proxy turns out weak | Medium | Medium | Variance-ratio test first. Report a negative result. Canada lens leans on Fed→Canada transmission. |
| R3 | Weak instruments at monthly frequency | Medium | High | Effective F, Anderson–Rubin intervals, report honestly |
| R4 | Licensing (Moody's, Yahoo, CanDeal, OECD) | Medium | Medium | [L] and [D] policies. Publish derived statistics only. Open items O-3 to O-5. |
| R5 | Forking paths in regimes, spillovers and the ML comparison | High | High | Pre-registered config, BH-FDR, a single primary metric and decision rule |
| R6 | Small test set (e.g. 68 FOMC events) | Certain | Medium | Event-clustered bootstrap CIs. No "winner" without a CI. |
| R7 | Sources revised in place (JK, Känzig, Yahoo adjusted closes) | High | Medium | Pinned git SHAs, SHA-256 manifest, diff report on refresh |
| R8 | Scope creep (six web views + live mode) | High | Medium | Milestone gates. The web app consumes JSON only. |
| R9 | Building the BoC FAD list (about 210 events) takes a long time and is error-prone | Medium | Medium | Script the press-release scrape with manual review. Validate against policy-rate changes. |
| R10 | GoC yield closing time unknown, so the BoC proxy window is uncertain | Medium | Medium | O-1 must be resolved before M3. Otherwise label the window as uncertain and test sensitivity. |

---

## 6. Milestone acceptance criteria

- **M1 Data layer**
  - `uv run wsl data fetch` and `wsl data build` run end to end.
  - Every raw file is hashed in the manifest.
  - pandera schemas pass.
  - The data-quality report lists gaps, holiday handling, the negative-WTI handling, benchmark switches and the Yahoo-vs-Nasdaq spot check.
  - `boc_fad.csv` exists and passes the policy-rate-change test.
  - O-2, O-5 and O-6 are resolved.
- **M2 Estimation core**
  - LP, LP-IV and HAC code.
  - pytest simulation recovery: data drawn from a known VAR, with LP and LP-IV IRFs inside a fixed tolerance across 200 Monte Carlo draws, and coverage of the 90% bands within [0.85, 0.95].
- **M3 Identification and replication**
  - Shocks loaded with timestamp-aware alignment.
  - BoC proxy validated (tests 1–5 in P9).
  - `docs/REPLICATION.md` paper-vs-ours table.
  - Placebo distribution from 1,000 random non-event days.
- **M4 Proxy SVAR/BVAR, state dependence, Canada spillovers**
  - BVAR + proxy SVAR pass the simulation test.
  - State-dependence tests as pre-specified.
  - FDR-controlled network.
  - Robustness matrix (controls, samples, estimators).
- **M5 ML benchmark:** run exactly per `EVALUATION_PROTOCOL.md`, with the test set touched once and the verdict written whatever it is.
- **M6 Scenario engine:** joint multi-shock LP, JSON export with full provenance, the label-propagation rule, and the out-of-support warning.
- **M7 Web app**
  - Six views built from static JSON.
  - Accessibility: keyboard navigation, alt text, and a table alternative for every chart.
  - Load time under 3 s on a throttled Lighthouse run, and works on a phone.
  - Strategist Brief "no number not in JSON" test.
- **M8 Live mode:** pre-registered forecasts, research note, README, resume bullets.

---

## 7. Decisions needed from you

1. **FRED API key.** Put it in `.env` as `FRED_API_KEY` (free key: https://fred.stlouisfed.org/docs/api/api_key.html).
2. **Accept or reject the push-backs P1–P12**, especially P6 (BVAR monthly only), P7 (no TFT) and P11 (live mode scored in proxy units, starting after M4).
3. **Accept the TSX second-source gap**, or approve a free-key provider (e.g. Tiingo or Alpha Vantage, which need credentials and so require your approval).
4. **Commit the M0 docs now** so the git timestamp proves the evaluation protocol came before any results. I recommend a commit plus a `protocol-v1` tag.
