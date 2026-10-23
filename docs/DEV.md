# Developing World Shock Lab

## Setup

[uv](https://docs.astral.sh/uv/getting-started/installation/) installs the pinned Python. The site needs Node 22 (`web/.nvmrc`).

```bash
make setup
cd web && npm ci && npx playwright install chromium
```

`make setup` prints the uv install link and stops if `uv` is not on `PATH`. Copy `.env.example` to `.env` only when a command needs a FRED key. The key is never committed. In Actions it is the `FRED_API_KEY` secret.

## Checks

```bash
make check
cd web && npm run typecheck && npm test && npm run build
```

`make check` is ruff, mypy and pytest. `make reproduce` runs those checks and then regenerates the documents from the committed results. It does not refit a model or download data.

With the site preview running, `cd web && npm run a11y` checks WCAG. Do not pass `--write`. That flag overwrites `results/web_a11y.json`. The same applies to `npm run measure -- --write` and `results/web_perf.json`.

## Where a number is allowed to come from

`uv run wsl report docs` is how a number enters the marked regions of `README.md`, `docs/RESEARCH_NOTE.md` and `docs/RESUME_BULLETS.md`. Change the wording in `wsl/report/docs.py`, then run the generator. `tests/test_docs.py` fails if a marked region contains a number the formatter did not emit, or if a generated file differs from a fresh render.

The README hero is a raster of `docs/figures/fig1_canada_lens.svg` (`docs/figures/fig1_canada_lens.png`). Re-export the PNG when that SVG is regenerated. The SVG is the figure the generator writes.

## Files a change should leave alone

- `results/`, including scored evaluations, simulation output, and the accessibility and load-time records
- published forecasts in `live/forecasts/` and scores in `live/scores/` (write-once)
- `config/evaluation.yaml`, `config/live.yaml`, and `config/ml_study.yaml` once the protocol tag that freezes them exists
- pins in `data/manifest.json`, unless the change is an intentional `wsl data fetch --repin`
- provenance fields already recorded in `web/public/data/` (see [COMMIT_MAP.md](COMMIT_MAP.md))

`wsl evaluate --final` and `wsl mlstudy --final` refuse a second run unless a reason is logged. Do not hand-edit their outputs.
