# World Shock Lab

How the world, and Canada in particular, reacts to identified economic shocks: monetary policy surprises, oil supply news, and trade-policy and geopolitical risk.

**Status: M2 (estimation core) complete. The estimators are validated on simulated data; no estimates on real data yet. Every number in this repo is produced by code in it.**

## Quick start

```bash
make setup          # Python 3.12 environment via uv (pinned in uv.lock)
cp .env.example .env   # add a free FRED API key
make data           # fetch -> BoC event list -> build -> data-quality report
make check          # ruff, mypy, unit + property + simulation-recovery tests
make simulation     # regenerate docs/SIMULATION.md
make test-data      # checks on the built data
```

`make data` downloads about 50 MB into `data/raw/`. That folder is git-ignored and every file in it is read-only and SHA-256 hashed. The pinned hashes are committed in `data/manifest.json`. If a source has revised a file since it was pinned, the build fails loudly. `uv run wsl data fetch --repin` adopts the new version, and the change then shows up in the git diff.

## What is here

| Path | What it is |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | Scope, push-backs on the original brief, architecture, risks, milestones |
| [data/SOURCES.md](data/SOURCES.md) | Every data source: verified URL, coverage, licence, redistribution policy, open items |
| [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md), [config/evaluation.yaml](config/evaluation.yaml) | Pre-registered out-of-sample protocol, frozen before any estimation (git tag `protocol-v1`) |
| [docs/DATA_QUALITY.md](docs/DATA_QUALITY.md) | Generated data-quality report: coverage, gaps, invalid values, second-source checks |
| [docs/SIMULATION.md](docs/SIMULATION.md) | Generated Monte Carlo report: do the estimators recover known impulse responses, with correct coverage? |
| [config/series.yaml](config/series.yaml) | Registry of every ingested series, with units, observation time and licence policy |
| [data/events/boc_fad.csv](data/events/boc_fad.csv) | Bank of Canada rate announcements 2000–2026, built from official press releases and validated against the target-rate series |
| `wsl/data/` | Downloaders, immutable raw cache, parsers, pandera schemas, build to parquet and DuckDB |
| `wsl/shocks/` | Published shock series (Jarociński–Karadi, Bauer–Swanson, Känzig, Baumeister–Hamilton) and the BoC announcement list |
| `wsl/timing.py` | Timestamp-aware event windows (e.g. a 14:00 FOMC statement comes after the 12:00 FX fix) |
| `wsl/estimation/` | OLS with HC/Newey–West errors, local projections (time-series and event-sample), LP-IV with effective F and Anderson–Rubin sets, Monte Carlo harness |

## Licence

Code: MIT. Data: each provider's terms, as listed in [data/SOURCES.md](data/SOURCES.md). Raw data are never committed. Yahoo prices are only ever published as derived statistics. Nothing derived from VIX or Moody's spreads is published until their permission questions are settled (open item O-9).

This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
