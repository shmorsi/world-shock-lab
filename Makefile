.PHONY: setup fetch events build report data test test-data lint typecheck check all

setup:            ## create the pinned environment (Python 3.12 via uv)
	uv sync

fetch:            ## download every source into data/raw (hashes -> data/manifest.json)
	uv run wsl data fetch

events:           ## rebuild data/events/boc_fad.csv from cached BoC pages (add CRAWL=1 to re-download)
	uv run wsl events boc-fad $(if $(CRAWL),--crawl,)

build:            ## pinned raw -> validated parquet -> DuckDB
	uv run wsl data build

report:           ## docs/DATA_QUALITY.md
	uv run wsl data report

data: fetch events build report

test:             ## unit and property tests (no data needed)
	uv run pytest

test-data:        ## checks on the built data
	uv run pytest -m data

lint:
	uv run ruff check .

typecheck:
	uv run mypy wsl

check: lint typecheck test

all: setup data check test-data
