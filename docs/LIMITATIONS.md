# Limitations

What the results in this repository cannot tell you, and why. Numbers are not repeated here. They are in the generated reports this page links to, so this page cannot drift from them.

## Identification

- **The Bank of Canada surprise is a daily proxy.** There is no free intraday Canadian futures data, so the BoC surprise is the announcement-day change in the 2-year Government of Canada yield. It passed its pre-set validation ([REPLICATION.md](REPLICATION.md) §4). However, the spillover network finds it "moving" US Treasury yields, which a BoC decision should barely do ([RESULTS.md](RESULTS.md) §7), so it still carries some common North American rate news. Every BoC result is therefore shown next to a Rigobon–Sack estimate. The two disagree for equities, and the BoC equity results are classed as fragile.
- **BoC decisions on FOMC days cannot be studied** with daily data, because the window contains both announcements (protocol v2). Many scheduled BoC dates in 2026–2027 fall on FOMC days, which matters for the live mode (below).
- **The Fed's policy/information split** (Jarociński and Karadi) rests on sign restrictions: a policy surprise moves rates and stocks in opposite directions, an information surprise in the same direction. The split is model-based, and the median rotation used here is one of several the authors provide. Effects of the press conference that fall outside the 30-minute window are not measured.
- **OPEC surprises** come from daily oil-futures moves on announcement days. They may contain some same-day demand news. The weak response of exporters relative to importers is consistent with that ([RESULTS.md](RESULTS.md) §4).
- **Geopolitical and trade-policy risk are not shocks.** The GPR and TPU indices count news coverage, which responds to events and to markets. Every result that uses them is labelled *reduced form* and is not causal. The only identified tariff shock we found (Den Besten, Barnichon, Känzig and Singh, 2026, NBER WP 35102) is annual and US-only, far too coarse for market responses ([data/SOURCES.md](../data/SOURCES.md)).
- **Monthly macro effects rest on weak instruments.** The first-stage F statistics are below conventional thresholds for both instruments ([RESULTS.md](RESULTS.md) §1). Anderson–Rubin sets are reported, but these are the least reliable results in the project.

## Measurement

- **Country equity responses use US-listed ETFs in US dollars.** They mix the local stock market and the currency, and they trade on New York hours even when the home market is closed.
- **Exchange rates are the H.10 noon fix in New York.** A 14:00 FOMC statement therefore appears in the next day's rate, so FX responses are measured over a longer window that contains more unrelated news.
- **The closing time of the Canadian yield data is not documented.** It was tested rather than assumed (open item O-1, resolved in [REPLICATION.md](REPLICATION.md) §5), and the event windows use a 16:00 close.
- **Yahoo prices may be revised** (dividends, splits). Total-return indices are computed from closes and dividends, and every download is hashed, so a revision changes a hash and is never silently adopted.
- **Inflation breakevens** are computed as nominal minus TIPS yields from public-domain series. They start when the TIPS series do, so their samples are shorter than those for nominal yields.

## Statistics

- **Responses are averages over past announcements.** A given announcement can differ. Scenarios add responses linearly and assume shocks do not interact; the tool says so and warns when a size lies outside the historical range.
- **State dependence has low power.** With a few hundred announcements split into regimes, only large differences are detectable. The minimum detectable effect of every test is reported ([RESULTS.md](RESULTS.md) §2). "No regime effect survives" is not evidence that regimes do not matter.
- **Multiple testing is controlled within the pre-registered families** (regime tests; spillover edges), not across the whole project. The Canada comparison and the robustness matrix are reported in full rather than corrected.
- **Simulation evidence is conditional on the simulated process.** The event-sample LP's coverage was validated on a VAR with sparse, fat-tailed shocks ([SIMULATION.md](SIMULATION.md)). Real announcement data could be harder.
- **The BVAR bands are too narrow** in simulation (Kilian's small-sample problem plus shrinkage bias). Where VAR and LP disagree, read the LP bands.

## The machine-learning benchmark

- **"No evidence ML adds value" is not "evidence of no value".** The test period has a modest number of announcements and includes the 2020 turmoil, and the confidence interval for the difference is narrow but not zero-width ([ML_BENCHMARK.md](ML_BENCHMARK.md)).
- **The feature set is deliberately limited** to information available before each announcement plus its own surprise, using public data only. Intraday data, options, positioning and text were out of scope.
- **The test set has been used once.** No model, feature or setting can now be changed and re-scored on it without that being logged as a new run.

## Live mode

- **Live forecasts are not causal.** They condition on the day's 2-year yield move, which mixes policy and information news; that is why their bank-stock slopes are near zero while the identified Fed effect is clearly negative.
- **Collision days** (both banks announce) cannot be separated with daily data. They are scored but summarized separately, and they include the first two live decision dates.
- **Scoring depends on data access.** The scheduled job downloads from FRED, the BoC and Yahoo. If Yahoo blocks the CI runner, scores for Canadian banks wait until the job is run locally (`uv run wsl live run`).
- **A track record takes years.** A handful of scored decisions says little: with ten decisions, a 90% range would be expected to miss several cells by chance alone.

## Licensing and what is withheld

- **CA yields** (CanDeal, via the BoC's Valet; open item O-3): derived responses are published, but per-event realized values are not, and the live scores report only whether outcomes fell inside the forecast ranges.
- **Yahoo prices** (open item O-4): only derived statistics are published. Event Replay shows realized per-event values only for public-domain series. The pre-registered final-evaluation record in `results/final_eval/` is kept complete and unaltered, so it does contain per-event moves for every asset.
- **VIX and Moody's spreads** (open item O-9): used for local research only. Nothing derived from them is published, so the public site has no VIX-regime responses.

## Scope

This is a research instrument, not investment advice. It models no positions, transaction costs or risk limits. A scenario's output is a conditional, hypothetical response with estimation uncertainty, not a forecast of what markets will do.
