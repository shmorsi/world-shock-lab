# Data sources

Every series used in World Shock Lab is listed here, with where it comes from, what was checked, its licence and what we may commit to the repo. If a source is not listed here, the code must not use it.

- **Verification date:** 2026-10-12 (M0). The person or process that verified each source is named in the commit that adds it.
- **What "verified" means:** the URL returned HTTP 200 from this machine, the file parsed, and its coverage was read from the data, not from documentation. The coverage figures below come from that check. They are **not** results.
- **Redistribution policy:**
  - **[C] commit allowed.** The licence permits redistribution (CC BY 4.0 or an open-government licence). We pin the exact vintage, by git commit SHA or by file hash.
  - **[D] downloader only.** The licence is unclear or restrictive. We commit the download script and the expected SHA-256, never the file.
  - **[L] local only, restrictive.** The licence forbids redistribution. Raw values never leave `data/raw/`. Only derived statistics (impulse responses, test statistics) are published, never the series and never realized levels.
- **Default:** `data/raw/` is git-ignored for every source. Even for [C] sources the downloader is the canonical path. Small [C] files may also be vendored under `data/vendor/` so the repo reproduces offline.

---

## 1. Shock series (identification backbone)

### 1.1 Jarociński–Karadi (JK) Fed shocks: **primary US monetary policy shock**

| Field | Value |
|---|---|
| Provider | Marek Jarociński & Peter Karadi (ECB), GitHub |
| URL | https://github.com/marekjarocinski/jkshocks_update_fed |
| Pinned commit | `f788aef7d573f87f28b70ffc3763e73c4886a53f` (2026-10-11, "update Fed shocks until September 2026") |
| Files | `shocks_fed_jk_t.csv` (per announcement, date-time stamped), `shocks_fed_jk_m.csv` (monthly), `source_data/fomc_surprises_jk.csv` (raw 30-min surprises) |
| Coverage (verified) | 334 announcements, 1990-02-08 11:30 → 2026-09-16 14:00. Raw surprises from 1987-11-04. |
| Variables | `pc1` (1st principal component of MP1, FF4, ED2–ED4, scaled to the ED4 s.d., in % p.a.), `SP500`, `MP_pm`, `CBI_pm`, `MP_median`, `CBI_median`. Raw file: FF1–FF4, MP1, ED1–ED4 (SOFR futures from Jan 2022), TFUT02/05/10/30, SP500, SP500FUT, EUR. |
| Licence | Data **CC BY 4.0**, code BSD-3 (stated in README) → **[C]** |
| Citation | Jarociński, M. and Karadi, P. (2020). Deconstructing Monetary Policy Surprises – The Role of Information Shocks. *AEJ: Macroeconomics* 12(2). doi:10.1257/mac.20180090 |
| Retrieved | 2026-10-12 |

Known issues:
1. **Full-sample estimation.** `pc1` loadings and scaling, and the `*_median` rotation, are estimated on the full sample, which is look-ahead for out-of-sample work. ML and forecast evaluation (M5, M8) must use the **raw** surprise columns, or a PC re-estimated on the training window only. See `docs/EVALUATION_PROTOCOL.md`.
2. **Non-FOMC events are included.** The sample includes events that are not FOMC meetings, plus off-hours timestamps. Examples: 2020-03-15 17:00 (a Sunday) and 2022-06-13 15:15. Daily alignment must use the timestamp, not the date.
3. **The repo is updated in place.** Always fetch by pinned SHA.

### 1.2 Bauer–Swanson (BS) monetary policy surprises: **replication benchmark**

| Field | Value |
|---|---|
| Provider | Federal Reserve Bank of San Francisco (Michael Bauer) |
| Page | https://www.frbsf.org/research-and-insights/data-and-indicators/monetary-policy-surprises/ |
| File | https://www.frbsf.org/wp-content/uploads/monetary-policy-surprises-data.xlsx (≈295 KB, xlsx) |
| Coverage (verified) | Sheet `FOMC (update 2023)`: 361 announcements, 1988-02-04 → 2023-12-13. Also `Monthly (update 2023)` 1988-02 → 2023-12, plus the paper's `FOMC (original)` and `Monthly (original)` sheets. |
| Variables | `MPS` (raw, 1st PC of futures changes), `MPS_ORTH` (orthogonalized to 6 pre-announcement predictors), the underlying futures/Treasury/S&P changes, and the predictors. |
| Licence | None stated on the page → **[D]** |
| Citation | Bauer, M. D. and Swanson, E. T. (2023). A Reassessment of Monetary Policy Surprises and High-Frequency Identification. *NBER Macroeconomics Annual* 37. |
| Retrieved | 2026-10-12 |

Known issues:
1. **Ends December 2023.** We use BS only for the replication table and as a robustness instrument. JK is the primary series because it runs to September 2026.
2. **`MPS_ORTH` uses a full-sample regression.** Same look-ahead caveat as JK.

### 1.3 Jarociński–Karadi ECB shocks: **euro-area monetary policy shock**

| Field | Value |
|---|---|
| URL | https://github.com/marekjarocinski/jkshocks_update_ecb |
| Pinned commit | `f7ffc821b0ade71dd38539e4044e25368dfb4dc1` (2026-01-07) |
| Files | `shocks_ecb_mpd_me_d.csv` (per Governing Council meeting), `shocks_ecb_mpd_me_m.csv` |
| Coverage (verified) | 312 meetings, 1999-01-07 → 2025-10-30 |
| Underlying data | Altavilla et al. (2019) EA-MPD, https://www.ecb.europa.eu/pub/pdf/annex/Dataset_EA-MPD.xlsx (HTTP 200, 377 KB on 2026-10-12). Optional; the ECB's reuse terms will be checked before we use it directly. |
| Licence | Data **CC BY 4.0** → **[C]** |
| Citation | Jarociński & Karadi (2020), as above. Altavilla, Brugnolini, Gürkaynak, Motto, Ragusa (2019), *JME*, doi:10.1016/j.jmoneco.2019.08.016 |

### 1.4 Känzig oil supply news shocks: **primary oil supply shock**

| Field | Value |
|---|---|
| URL | https://github.com/dkaenzig/oilsupplynews |
| Pinned commit | `3496b6e848ba760d4be25289fc68c1e989998790` (2026-06-09, "Corrected error in World IP data") |
| File | `oilSupplyNewsShocks_2025M12.xlsx`. Vintages from 2017M12 onward are kept in the repo. |
| Coverage (verified) | Sheet `Daily`: 169 OPEC announcement surprises, 1983-07-19 → 2025-12-01 (front-month to 12-month WTI futures changes + `PC`). Sheet `Monthly`: surprise series + VAR-based news shock, 1975M01 → 2025M12. `pre-Covid` sheets also available. |
| Licence | **CC BY 4.0** (stated in README) → **[C]** |
| Citation | Känzig, D. R. (2021). The Macroeconomic Effects of Oil Supply News: Evidence from OPEC Announcements. *AER* 111(4): 1092–1125. doi:10.1257/aer.20190964 |
| Update cadence | About every 6 months, with a 4–5 month lag (README) |

Known issues:
1. **28 of the 169 daily events fall on a weekend.** They must be mapped to the next trading day for each market.
2. **The daily `PC` depends on the sample.** The 1983-07-19 value differs between the `Daily` and `Daily (pre-Covid)` sheets, which is direct evidence of full-sample estimation. Out-of-sample work uses the raw futures columns.
3. **The monthly news shock comes from a full-sample VAR.** Use it in-sample only.

### 1.5 Baumeister–Hamilton structural oil shocks: **second oil identification**

| Field | Value |
|---|---|
| Page | https://sites.google.com/site/cjsbaumeister/datasets |
| Files | Supply: `BH2_supply_shocks.xlsx` via https://drive.google.com/uc?export=download&id=1OsA8btgm2rmDucUFngiLkwv4uywTDmya ; Demand: `BH2_demand_shocks.xlsx` via https://drive.google.com/uc?export=download&id=1neFXLrIvGwggebQRwjmtrWK-dfQZ9NH8 |
| Coverage (verified) | Monthly 1975M2 → 2026M6. Values are posterior medians. The demand file has 3 shock columns, whose labels will be confirmed in M1. The page says the next update is 2026-12-20. |
| Licence | None stated → **[D]** |
| Citation | Baumeister, C. and Hamilton, J. D. (2019). Structural Interpretation of Vector Autoregressions with Incomplete Identification: Revisiting the Role of Oil Supply and Demand Shocks. *AER* 109(5): 1873–1910. |

Known issue: these are full-sample posterior medians, so they are for in-sample monthly analysis only. The Google Drive links can change, so the downloader must verify the SHA-256 and fail loudly if it differs.

### 1.6 Geopolitical Risk (GPR) index: **reduced-form, not an identified shock**

| Field | Value |
|---|---|
| Page | https://www.matteoiacoviello.com/gpr.htm |
| Files | Daily: https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls ; Monthly + country indices: https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls |
| Coverage (verified) | Daily `GPRD`, `GPRD_ACT`, `GPRD_THREAT`: 1985-01-01 → 2026-10-05 (15,253 rows). Monthly: 1,521 rows × 115 columns, including country-specific `GPRC_*` (e.g. `GPRC_CAN`). |
| Licence | None stated → **[D]** |
| Citation | Caldara, D. and Iacoviello, M. (2022). Measuring Geopolitical Risk. *American Economic Review* 112(4): 1194–1225. Verified against the authors' page. |

### 1.7 Trade Policy Uncertainty (TPU) index: **reduced-form, not an identified shock**

| Field | Value |
|---|---|
| Page | https://www.matteoiacoviello.com/tpu.htm |
| File | https://www.matteoiacoviello.com/tpu_files/tpu_web_latest.xlsx |
| Coverage (verified) | `TPU_MONTHLY` 1960-01 → 2026-09. `TPU_DAILY` 2015-01-01 → 2026-10-05. `TPU_QUARTERLY` 1960Q1 → 2026Q2. |
| Licence | None stated → **[D]** |
| Citation | Caldara, D., Iacoviello, M., Molligo, P., Prestipino, A. and Raffo, A. (2020). The Economic Effects of Trade Policy Uncertainty. *Journal of Monetary Economics*. The authors' page gives the authors, year, title and journal. Volume and pages will be confirmed from `TPU_PAPER.pdf` in M1. |

Known issues:
1. **The daily TPU is built from very few articles** (around 8–12 a day in Oct 2026), so it is very noisy.
2. **The `ReadMe` sheet says monthly coverage is 1985–2020, but the data run 1960–2026.** We trust the data and note the discrepancy.

### 1.8 Bank of Canada announcement-day proxy: **constructed by us**

- **Construction:** the change in the 2-year GoC benchmark yield (`BD.CDN.2YR.DQ.YLD`, §3.2) from the close on the day before a fixed announcement date (FAD) to the close on the FAD.
- **Event list:** constructed from official BoC press releases (§2.2).
- **Label:** daily-window proxy. It is weaker than an intraday futures surprise and is validated in M3 (see `docs/SCOPE.md` §4).
- **Licence:** the inputs are [D] (the yields come from a third party, see §3.2). The derived proxy series is our own work. Whether we may publish it is **open item O-3**.

### 1.9 Considered but out of scope

| Source | Why not |
|---|---|
| Narrative tariff shocks, Den Besten, Barnichon, Känzig & Singh (2026), NBER WP 35102. https://github.com/dkaenzig/narrative-tariff-shocks, commit `69547f33…`, CC BY 4.0 | Verified: **annual**, US-only, 1840–2024. Far too coarse for daily or monthly cross-asset responses. It is the only *identified* tariff shock we found. We cite it in LIMITATIONS to explain why our trade-policy results are reduced-form. |

---

## 2. Event calendars

### 2.1 FOMC

- **Historical timestamps:** taken from JK `shocks_fed_jk_t.csv` (§1.1).
- **Future meetings:** https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (verified 2026-10-12). The page lists 2026 meetings Jan 27–28, Mar 17–18*, Apr 28–29, Jun 16–17*, Jul 28–29, Sep 15–16*, Oct 27–28, Dec 8–9* (* = SEP), and 2027 meetings Jan 26–27, Mar 16–17*, Apr 27–28, Jun 8–9*, Jul 27–28, Sep 14–15*, Oct 26–27, Dec 7–8*. The page notes that each date is tentative until confirmed at the preceding meeting.
- **Licence:** US government work, public domain.

### 2.2 Bank of Canada FADs

- **Upcoming:** https://www.bankofcanada.ca/core-functions/monetary-policy/key-interest-rate/ (verified 2026-10-12). Remaining 2026 dates are **Oct 28** and **Dec 9**. The 2027 dates are Jan 27, Mar 3, Apr 28, Jun 2, Jul 21, Sep 8, Oct 27, Dec 8.
- **History:** there is **no single official list**. Valet has no FAD group. The policy-rate series `V39079` only starts 2009-04-21 on Valet and only reveals changes, not holds. The fixed-date system was introduced after the BoC's September 2000 consultation (https://www.bankofcanada.ca/2000/09/informationconsultation-paper/).
- **Built in M1:** `data/events/boc_fad.csv`, produced by `uv run wsl events boc-fad` (code in `wsl/shocks/boc_fad.py`).
  - **Inputs:** the BoC press-release listing (826 entries), monthly archives for months the listing misses, and the yearly schedule releases. Every page is hashed in `data/manifest_boc_web.json`.
  - **Result:** 211 events, 2000-12-05 → 2026-09-02. 207 are scheduled. 4 are unscheduled: 2001-09-17, 2008-10-08, 2020-03-13 and 2020-03-27. Each row has its release URL.
  - **Decisions:** the change in bp comes from the daily target-rate series (Valet `CBC20210`), not from the release text.
  - **Validation (all pass, 0 problems):**
    - (a) every target-rate change since 2001 matches a listed event;
    - (b) each title's verb agrees with the sign of the change;
    - (c) each release's "next scheduled date" is the next listed event;
    - (d) 8 scheduled events in every year 2001–2025.
  - **Effective-date change:** from 2021 the BoC applies a new target "on the business day following each rate announcement" (2021 schedule release, 2020-07-30). Before that, decisions applied immediately. The series shows this from the first change after the switch (2022-03-02 → 2022-03-03), so changes are matched to the announcement by window, not by date.
  - **Announcement times** come from the yearly schedule releases (§5).
- **Licence:** BoC terms (https://www.bankofcanada.ca/terms/) allow use, copying and distribution with attribution, for BoC-owned content.

### 2.3 OPEC

Dates are taken from the Känzig daily sheet (§1.4).

### 2.4 Same-day FOMC/BoC announcements

On these days a daily-window BoC proxy also contains the 14:00 ET Fed news.
- **Ahead:** 2026-10-28 and 2026-12-09.
- **History (counted in M1):** 12 dates: 2008-01-22, 2008-10-08, 2019-10-30, 2022-01-26, 2025-01-29, 2025-07-30, 2025-09-17, 2025-10-29, 2025-12-10, 2026-01-28, 2026-03-18 and 2026-04-29.
- **By split (scheduled FADs only):** 10 of the 62 BoC events in the pre-registered test window fall on such days, against 1 in train and 0 in validation (`docs/EVALUATION_PROTOCOL.md` §2).

---

## 3. Daily outcomes

### 3.1 FRED (Federal Reserve Bank of St. Louis)

- **API:** https://api.stlouisfed.org/fred/ . It is reachable from this machine (it answers with HTTP 400 when no key is sent) and needs a free key in `.env` as `FRED_API_KEY`, which is **not yet set**. The website (fred.stlouisfed.org) times out from this machine when called with curl, so the pipeline uses the API only.
- **Terms:** FRED API terms of use (https://fred.stlouisfed.org/docs/api/terms_of_use.html, read 2026-10-12).
  - Third-party copyrighted series need the owner's permission for any use beyond personal use. FRED cannot grant it.
  - Products must say they use the FRED® API but are not endorsed or certified by the Federal Reserve Bank of St. Louis (the notice is in README.md).
- **Copyright tags:** read from the API (`fred/series/tags`) in M1, which resolves O-6:

| Series | What | Copyright tag (API) | Policy |
|---|---|---|---|
| DGS1, DGS2, DGS5, DGS10, DFII5, DFII10 | US Treasury nominal and TIPS constant-maturity yields (H.15) | public domain: citation requested | [C] |
| DFF, DFEDTAR, DFEDTARU, DFEDTARL | Effective fed funds rate and target (range) | public domain: citation requested | [C] |
| DEXCAUS and 11 other H.10 FX series | FX, **noon buying rates in New York** | public domain: citation requested | [C] |
| DCOILWTICO, DCOILBRENTEU, DHHNGSP | WTI, Brent, Henry Hub spot (EIA) | public domain: citation requested | [C] |
| T5YIE, T10YIE | Breakeven inflation | **copyrighted: citation required** | [D]. For anything published, we compute breakevens ourselves as DGS5 − DFII5 and DGS10 − DFII10 (public-domain inputs, same definition). T5YIE and T10YIE serve only as a cross-check. |
| VIXCLS | VIX close (CBOE) | **copyrighted: citation required** | [D], never published (O-9) |
| BAA10Y | Moody's Baa – 10Y spread | **copyrighted**. The notes say it may not be copied, redistributed or stored without Moody's consent. | **[L]** (O-9) |
| SP500 | S&P 500 | "Copyrighted: Pre-Approval Required", and only 10 years of history | **Not used.** We use SPY (§3.3). |
| CPIAUCNS, CPIAUCSL, INDPRO, UNRATE, PCEPILFE | US monthly macro | public domain: citation requested | [C] |

### 3.2 Bank of Canada Valet API

- **API:** https://www.bankofcanada.ca/valet/ . No key needed. Verified 2026-10-12.
- **Terms:** https://www.bankofcanada.ca/terms/ . BoC-owned content may be reused with attribution. **Third-party content is excluded.** The benchmark bond yields are sourced from CanDeal DNA, according to the BoC's notes on bond yields (https://www.bankofcanada.ca/rates/interest-rates/lookup-bond-yields/notes-on-canadian-bond-yields/), and the FX rates come from LSEG. The terms also forbid getting around Valet's rate limits, so the downloader is rate-limited and cached.

| Series | What | Coverage (verified) | Notes | Policy |
|---|---|---|---|---|
| BD.CDN.2YR.DQ.YLD (also 3YR, 5YR, 7YR, 10YR, LONG) | GoC benchmark yields | 2YR: 6,443 obs, 2001-01-02 → 2026-10-08 | "Mid-market closing yields". Benchmark bond switches can create artificial jumps (BoC notes); these must be flagged. Time of the close is **not documented** (open item O-1). | [D] |
| AVG.INTWO | CORRA | 7,285 obs, 1997-08-12 → 2026-10-08 | | [C]? (O-3) |
| CBC20210 | Target overnight rate, daily | 2001-01-01 → 2026-10-09 (6,712 obs) | Resolves O-2. Records changes on the announcement day until 2020 and on the next business day from 2021 (§2.2). | [C] |
| V39079 | Target overnight rate, business daily | 2009-04-21 → 2026-10-09 | Superseded by CBC20210 | [C] |
| TB.CDN.90D.MID, TB.CDN.1Y.MID | T-bill yields | 2001-01-02 → 2026-10-09 | **"Average of sample secondary market yields taken throughout the business day"**. Not a close, so not used for event windows. | [D] |
| FXUSDCAD | USD/CAD | 2017-01-03 → 2026-10-09 only | **"Daily average exchange rate"**. Not a close, so not used for event windows. | [D] |

### 3.3 Equity and ETF prices: Yahoo Finance (primary, unofficial)

- **Endpoint:** `https://query1.finance.yahoo.com/v8/finance/chart/{symbol}`. It is unofficial and undocumented, and it is the same endpoint the `yfinance` package uses. It returned HTTP 200 on 2026-10-12.
- **Terms:** Yahoo's terms allow personal, non-commercial use only. Policy **[L]**: prices are never committed or shown. The site shows only derived statistics. Event Replay shows realized *returns* on a handful of dates. Whether that counts as acceptable derived use is **open item O-4**.
- **Verified history depth** (first bar → last bar, as of 2026-10-12):

| Symbol | Role | First bar | Exchange / currency / time zone |
|---|---|---|---|
| ^GSPTSE | S&P/TSX Composite | 1979-06-29 | TSI / CAD / America/Toronto |
| RY.TO (also TD, BNS, BMO, CM, NA .TO: *pending*) | Canadian banks; an equal-weight Big-6 portfolio gives a longer sample than ZEB.TO | 1995-01-12 | TOR / CAD |
| ZEB.TO | BMO equal-weight banks ETF | 2010-04-20 | TOR / CAD |
| XEG.TO | iShares S&P/TSX Capped Energy | 2001-03-23 | TOR / CAD |
| SPY | S&P 500 ETF | 1993-01-29 | PCX / USD / New York |
| EWC, EWJ (other iShares country ETFs *pending*) | Country equity, USD-listed, synchronous with the NY close | 1996-03-18 | PCX / USD |
| ^VIX | VIX (cross-check for VIXCLS) | 1990-01-02 | |
| CAD=X | USD/CAD | 2003-09-16 | **Time zone Europe/London, so the closing time is ambiguous. Not used for event windows.** |

- **Known problems:**
  1. Adjusted closes are recomputed backward after every dividend or split, so the adjusted history changes between downloads. We store **raw close and the dividend/split events** and compute total returns ourselves.
  2. History can be revised silently. A SHA-256 per download makes any revision visible.
  3. Delisted tickers are absent, which is a survivorship issue (minor here, because the Big-6 banks all survive).
  4. **Found in M1: impossible values on exchange holidays.** For example, CM.TO has a close of 0 on 2001-12-26 (Boxing Day). These are flagged `invalid_at_source` and never used.
  5. **Found in M1: our total-return index vs Yahoo's adjclose.** They agree to a median of about 0.002 bp a day, with a few larger daily gaps, mostly on year-end distribution dates (e.g. 2000-12-20 for several iShares funds). See `docs/DATA_QUALITY.md` §6.

### 3.4 Nasdaq quote API: **validation-only second source**

- **Endpoint:** `https://api.nasdaq.com/api/quote/{SYM}/historical?assetclass=etf&...`. It is unofficial. Verified 2026-10-12: about 10 years of unadjusted daily OHLCV for EWC (2,514 rows, 2016-10-10 → 2026-10-09).
- **Use:** spot-check Yahoo closes for US-listed ETFs only. The result is stored in the validation report, never as data. Policy [L].
- **What the data are:** split-adjusted, not dividend-adjusted, the same as Yahoo's "close". This was established in M1 from the 2016-11-07 EWU, EWI and EWJ splits and the 2025-12-05 XLE split.
- **Result (M1):** across all 16 US-listed ETFs, 0 of 2,507 overlapping days differ by more than 0.5%. The largest difference in the sampled dates is 0.2%.
- **Gap:** **no free second source was found for TSX-listed prices.** Planned mitigations:
  - Check internal consistency: ZEB.TO against the equal-weight portfolio of the six bank stocks.
  - Compare ^GSPTSE in USD terms with EWC.
  - Optionally, compare against issuer NAV on sample dates (iShares and BMO pages load, but the format is not yet checked).

### 3.5 Rejected

| Source | Reason |
|---|---|
| Stooq (`stooq.com/q/d/l/`) | Verified 2026-10-12: it now serves a JavaScript proof-of-work bot check instead of CSV. **We will not try to get around bot detection.** |
| FRED SP500 | Only 10 years of history, and republishing needs S&P pre-approval. |
| Valet FXUSDCAD / T-bills for event windows | Daily averages, not closes. |

---

## 4. Monthly macro

| Source | Series / table | Coverage (verified) | Licence | Policy |
|---|---|---|---|---|
| Statistics Canada WDS API (`https://www150.statcan.gc.ca/t1/wds/rest/`) | 18-10-0004-01 CPI, monthly NSA | 1914-01 → 2026-08 | **Statistics Canada Open Licence** (https://www.statcan.gc.ca/en/terms-conditions/open-licence): redistribution allowed, including commercially. Attribution: "Source: Statistics Canada, <product>, <reference date>." | [C] |
| | 14-10-0287-01 Labour force characteristics, SA | 1976-01 → 2026-09 | same | [C] |
| | 36-10-0434-01 GDP by industry, monthly. Canada has no official IP index, so this is our real-activity measure. | 1997-01 → 2026-07 | same | [C] |
| OECD SDMX API (`https://sdmx.oecd.org/public/rest/`) | e.g. `OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0` CPI. A CSV query was verified for CAN/USA/JPN. IP and unemployment dataflows will be pinned in M1. | per query | OECD Terms & Conditions (https://www.oecd.org/termsandconditions/ returns HTTP 403 to scripted requests, so a human must read it). The July 2024 open-access change applies CC BY 4.0, but how far it covers *data* is unconfirmed (open item O-5). The API is rate-limited. | [D] until O-5 is resolved |
| World Bank WDI API (`https://api.worldbank.org/v2/`) | EG.IMP.CONS.ZS (net energy imports, % of energy use; latest 2024), TX.VAL.FUEL.ZS.UN (fuel exports, % of merchandise exports; latest 2025). Used to classify oil exporters vs importers. | verified for 11 countries | World Bank datasets default to CC BY 4.0 (https://datacatalog.worldbank.org/public-licenses); per-indicator licence to be checked | [C] |
| FRED | US CPI, IP, unemployment (§3.1) | | | |

Known issue: exporter or importer status **changes over time.** The US shows **net energy imports of −9.8% of energy use in 2024**, which makes it a net exporter, whereas it was a large importer in the 2000s. The classification must therefore be time-varying and fixed in the config before estimation.

---

## 5. Observation timing (needed to align event windows correctly)

The day-0 window for a series must contain the announcement. The table records when each series is observed. **"Unverified"** means it must be confirmed in M1 before any event-window code is written.

| Series | Observation time | Status |
|---|---|---|
| H.10 FX (DEXCAUS etc.) | 12:00 ET (noon buying rate) | Verified (FRED notes) |
| FOMC statements | Usually 14:00 ET. JK timestamps show exceptions (e.g. 2013-01-30 14:15, 2020-03-15 17:00 Sunday). | Verified (JK data) |
| Treasury CMT (DGS*) | Quotes "at or near 3:30 PM" ET (Treasury yield-curve methodology page) | Verified (M1) |
| GoC benchmark yields (Valet) | "Closing" mid-market, time not documented; the data show it is after 14:00 ET | Resolved empirically (M3) |
| BoC FAD announcements | 09:00 ET 2002–2012, 10:00 ET 2014–2020 and 2022–2024, 09:45 ET 2025–2026, as stated in the yearly schedule releases. 2001 and 2013 are inferred from the next year's "will continue to be made at" wording; 2021 is inferred from the same time being stated for 2020 and 2022. All are before the 12:00 H.10 fix. | Verified (`boc_fad.csv` `time_basis`) |
| US equities and ETFs (Yahoo) | 16:00 ET close | Standard. To be confirmed against Nasdaq on sample dates. |
| TSX (Yahoo) | 16:00 ET close | Standard |
| WTI spot (EIA) | NYMEX settlement, believed to be 14:30 ET | **Unverified** |
| VIX | 16:15 ET | **Unverified** |

**Consequence:** an FOMC statement at 14:00 ET comes after the H.10 noon fix. **Day 0 for FX must therefore be the move from the noon fix on the announcement day to the noon fix on the next day,** not the move into the announcement day. For BoC announcements, which come before noon, the same-day fix contains the event. The pipeline will implement this as `first observation strictly after the announcement timestamp`, per series.

---

## 6. Open items (must be resolved before the step that needs them)

| ID | Item | Needed by |
|---|---|---|
| ~~O-1~~ | **Resolved empirically in M3** (docs/REPLICATION.md §5): Canadian yields respond to 14:00 ET FOMC news within the announcement-day window, which is only possible if the Valet close is after 14:00. 16:00 ET stays the working assumption. BoC announcements (09:00–10:00 ET) are unaffected either way. | — |
| ~~O-2~~ | **Resolved in M1:** Valet `CBC20210` has the daily target from 2001-01-01. | — |
| O-3 | Whether CanDeal-sourced Valet yields, and statistics derived from them (our BoC proxy), may be published. **2026-10-13:** derived responses are published; the final-evaluation record in the repository includes per-event CA 2-year moves (see O-4); the web app and live scores still withhold per-event CA yield values. | M6 / M7 |
| O-4 | Whether showing realized returns on specific dates (Event Replay) is acceptable under Yahoo's terms. The fallback is to show only model error metrics, not levels or returns. **Decision 2026-10-13 (owner):** the repository is published with the final-evaluation record (`results/final_eval/`) complete and unaltered, including per-event 1- and 5-day moves for all eight assets, because it is the pre-registered evidence. The web app keeps the conservative display: per-event values only for public-domain series. | M7 |
| O-5 | Whether the OECD's CC BY 4.0 covers SDMX data | M1 |
| ~~O-6~~ | **Resolved in M1:** tags read from the FRED API, terms URL recorded (§3.1). | — |
| O-7 | Pre-2017 USD/CAD closing rate (BoC legacy noon and closing archive). Not required, because H.10 with a timing-aware window covers it. | Optional |
| O-8 | **Found in M1: OECD CPI is incomplete for some countries.** Checked in both `DF_PRICES_ALL` and `DF_KEI` on 2026-10-12: CPI ends 2025-12 for FRA, ITA and NOR, 2024-07 for MEX, and 2021-06 for JPN (and JPN has no CPI in `DF_KEI`). Candidate replacements are IMF data and national agencies; some agencies need an API key, which would need your approval. US and Canada are unaffected (FRED, StatCan). | M4 (cross-country panel) |
| O-10 | **BoC Market Participants Survey** as an expectations benchmark for the BoC proxy. Checked in M3: it is published as quarterly web pages from late 2022, not as data, which gives at most about 15 usable announcements, all in the test window. Deferred. The proxy was validated by the variance-ratio, US-purged and placebo tests instead. | Optional |
| O-9 | **VIX (CBOE) and Moody's Baa via FRED:** FRED's terms require the owner's permission for non-personal use. These series are used only for local research computations, and nothing derived from them is published until this is settled. Options: ask CBOE and Moody's, or for regimes use realized volatility of public-domain Treasury yields instead of VIX. | M7 (public site) |

## 7. Live mode and site assets (M7–M8)

| Item | Source | Verified | Licence and use |
|---|---|---|---|
| FOMC meeting dates 2026–2027 | https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm | 2026-10-12: dates parsed from the page and written to `config/live.yaml` (statement on the second meeting day, 14:00 ET) | Public facts. Dates only. |
| BoC announcement dates 2026–2027 | https://www.bankofcanada.ca/core-functions/monetary-policy/key-interest-rate/ ("Schedule for 2026", "Schedule for 2027") | 2026-10-12: dates written to `config/live.yaml` (09:45 ET) | Public facts. Dates only. |
| JK surprises, current version | https://raw.githubusercontent.com/marekjarocinski/jkshocks_update_fed/main/source_data/fomc_surprises_jk.csv | 2026-10-12: HTTP 200. Same bytes as the pinned research copy on that date | CC BY 4.0, as §1. Live mode only (event history and score 2). It deliberately follows `main` rather than a pinned commit; each forecast and score records the SHA-256 it used. |
| Live market data | FRED (DGS2, DEXCAUS), Valet (BD.CDN.2YR.DQ.YLD), Yahoo (RY, TD, BNS, BMO, CM, NA on the TSX) | Same adapters as §3; fresh downloads into the git-ignored `data/live_cache/` | Policies as §3. Committed scores contain realized values only for [C] series (`config/live.yaml`). |
| World map | `world-atlas` 2.0.2 (npm, ISC licence), a TopoJSON redistribution of Natural Earth 1:110m Admin 0 boundaries | 2026-10-12: package metadata and README | Natural Earth data are public domain. Bundled with the site. |
| Fonts | Site: `@fontsource/space-grotesk` 5.2.8, `@fontsource/inter` 5.2.8, `@fontsource/ibm-plex-mono` 5.1.0. Research-note PDF and figure labels: `@fontsource/ibm-plex-sans` 5.2.8 | 2026-10-23: package metadata | SIL Open Font License 1.1. Bundled with the site or the PDF script, so no request goes to a font host. |
