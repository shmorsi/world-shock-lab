"""The live workflow reads FRED_API_KEY from secrets and can publish a score commit.

The scheduled job has never had the secret, so the commit step has never run. These checks
lock the two failure modes that would drop the 2026-10-28 scores: a missing key that exits
0, and a push from the detached HEAD that actions/checkout leaves behind.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / ".github" / "workflows" / "live.yml"


def _workflow() -> dict:
    # PyYAML's default resolver reads the GitHub key "on" as the boolean true.
    loaded = yaml.safe_load(LIVE.read_text())
    if True in loaded and "on" not in loaded:
        loaded["on"] = loaded.pop(True)
    return loaded


def _step(doc: dict, name: str) -> dict:
    steps = doc["jobs"]["live"]["steps"]
    return next(s for s in steps if s.get("name") == name)


def test_live_schedule_and_secret() -> None:
    text = LIVE.read_text()
    doc = _workflow()
    assert doc["on"]["schedule"][0]["cron"] == "30 22 * * 1-5"
    assert doc["jobs"]["live"]["env"]["FRED_API_KEY"] == "${{ secrets.FRED_API_KEY }}"
    assert "FRED_API_KEY:" in text and "sk-" not in text
    assert "api_key=" not in text
    assert doc["jobs"]["live"]["if"] == "github.ref == 'refs/heads/main'"


def test_missing_fred_key_fails_the_job() -> None:
    step = _step(_workflow(), "Fail if the FRED_API_KEY secret is missing")
    assert step["if"] == "success() && env.FRED_API_KEY == ''"
    assert "exit 1" in step["run"]
    assert "FRED_API_KEY secret is not set" in step["run"]
    assert "wsl.live.pending --no-key" in step["run"]


def test_score_commit_pushes_the_detached_head_to_main() -> None:
    doc = _workflow()
    commit = _step(doc, "Commit new forecasts and scores")
    assert commit["if"] == "success() && env.FRED_API_KEY != ''"
    script = commit["run"]
    assert 'git push origin "HEAD:${GITHUB_REF_NAME}"' in script
    assert "\ngit push\n" not in script
    # A timestamp-only rewrite of live.json is not a new score. Add it after live/ changed.
    live_add = script.index("git add live\n")
    quiet = script.index("git diff --cached --quiet")
    site_add = script.index("git add web/public/data/live.json")
    assert live_add < quiet < site_add
    owed = _step(doc, "Report what is still owed")
    assert owed["if"] == "always() && env.FRED_API_KEY != ''"
