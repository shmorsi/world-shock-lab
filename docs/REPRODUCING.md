# Reproducing World Shock Lab

## One command

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. `make setup` fails with the install link if `uv` is not on `PATH`. uv reads `.python-version` and creates the Python 3.12 environment from `uv.lock` (`uv sync --locked` in CI). The site needs Node 22, the version in `web/.nvmrc` and in the GitHub Actions workflows.

```bash
make setup                 # Python 3.12 + pinned packages via uv (uv.lock)
cp .env.example .env       # add a free FRED API key
make all                   # everything below, in order
```

`make all` runs:

| Step | Target | Produces |
|---|---|---|
| Data | `make data` | `data/raw/` (downloads, hashed), `data/events/boc_fad.csv`, `data/clean/` (parquet, DuckDB), `docs/DATA_QUALITY.md` |
| Checks | `make check`, `make test-data` | lint, types, unit/property/simulation tests, checks on the built data |
| Simulation | `make simulation` | `docs/SIMULATION.md`, `results/simulation.json` |
| Replication | `make replicate` | `docs/REPLICATION.md`, `results/m3/` |
| Analysis | `make analyze` | `docs/RESULTS.md`, `results/m4/` |
| ML report | `make evaluate-report` | `docs/ML_BENCHMARK.md` from the committed validation and final-evaluation files |
| Site data | `make site` | `web/public/data/` (public licence filter) |
| Documents | `make docs` | README findings, `docs/RESEARCH_NOTE.md`, figures |

Then `make web-check` builds the site and runs its tests, the accessibility check and the load-time check, and `make pdf` writes `docs/RESEARCH_NOTE.pdf`. Both need Node 22 (`web/.nvmrc`) and run `npx playwright install chromium` once (about 150 MB).

`make all` does not fit a model. The second study's note is regenerated from the committed file by `make reproduce` (`uv run wsl mlstudy --report`), which also rewrites `docs/ML_STUDY.md`, its figure, and `web/public/data/mlstudy.json`.

## What is deliberately not re-run

- **The held-out ML evaluation** (`uv run wsl evaluate --final`). The protocol allows one run. Its result is committed in `results/final_eval/`, and the command refuses to run on a dirty tree or a changed protocol. Re-running it is possible (`--rerun-reason`), but every run is logged in the protocol change log.
- **The second study's test run** (`uv run wsl mlstudy --final`). Same rule, under the tag `ml-study-v1`. The record is `results/mlstudy/final.json`.
- **Live forecasts.** They are made before each decision and never regenerated.

The lockfile installs the CPU torch wheel. The result files do not record how long the original fits took, so this guide does not state a duration. Do not change either protocol so that a refit fits a smaller machine.

## When sources have changed since the pins

Every download is stored once, read-only, and its SHA-256 is pinned in `data/manifest.json`. On a fresh clone, `make data` downloads the sources again.

- **Unchanged sources** (the published shock files at fixed commits, most historical series) reproduce the pinned bytes, and the build proceeds.
- **Sources that have since added observations or revised history** (FRED, the BoC's Valet, Yahoo) produce different bytes. The fetch then reports `changed_not_pinned`, and the build refuses to use them rather than silently producing different numbers.
- To adopt the new data, run `uv run wsl data fetch --repin`. The manifest diff then shows exactly which sources changed. Results will differ slightly where history was revised and gain new observations at the end. The pre-registered splits are fixed by date, so the test window does not move.

The raw bytes behind the committed results are not redistributable (see `data/SOURCES.md`), so exact reproduction of every number needs the same raw files. Everything else, including the logic that turns raw files into numbers, is in the repository, and each result file records the hashes of its inputs.

## Platform notes

- Python 3.12 on macOS or Linux. On Linux, uv installs CPU-only PyTorch (see `pyproject.toml`). The neural network was trained on macOS; tiny numerical differences across platforms do not change any reported verdict, but the committed final-evaluation file is the authoritative record.
- All randomness is seeded: simulation, placebo draws, bootstraps and model training.
