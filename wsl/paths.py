"""Filesystem layout. Every path the pipeline touches is defined here."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT / "config"
SERIES_REGISTRY = CONFIG_DIR / "series.yaml"
EVALUATION_CONFIG = CONFIG_DIR / "evaluation.yaml"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # immutable download cache (git-ignored)
CLEAN_DIR = DATA_DIR / "clean"  # validated parquet (git-ignored, rebuilt from raw)
EVENTS_DIR = DATA_DIR / "events"  # hand-curated / constructed event lists (committed)
MANIFEST = DATA_DIR / "manifest.json"  # hashes of every raw artifact (committed)
DUCKDB_PATH = DATA_DIR / "wsl.duckdb"

DOCS_DIR = ROOT / "docs"
DATA_QUALITY_REPORT = DOCS_DIR / "DATA_QUALITY.md"
