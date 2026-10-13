"""live.json for the web app: every published forecast and score, verbatim."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from wsl.data.build import file_sha256, git_sha
from wsl.live.data import LIVE_CONFIG
from wsl.paths import ROOT

WEB_DATA = ROOT / "web" / "public" / "data"


def export_live_json(folder: Path = WEB_DATA) -> Path:
    """Each forecast and score file carries its own provenance, so this needs no research
    data: the live workflow runs it on its own."""
    live_dir = ROOT / "live"
    entries = [json.loads(p.read_text()) for p in sorted(live_dir.rglob("*.json"))] \
        if live_dir.exists() else []
    doc = {"provenance": {"generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "git_sha": git_sha(), "live_config_sha256": file_sha256(LIVE_CONFIG)},
           "entries": entries}
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "live.json"
    path.write_text(json.dumps(doc, separators=(",", ":")))
    return path
