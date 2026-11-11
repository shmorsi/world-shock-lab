"""The live workflow needs no FRED key and can publish a score commit.

The scheduled job never had the FRED_API_KEY secret, and until 2026-11-11 it stopped without
it, so it published and scored nothing. These checks lock the failure modes that would drop
a forecast or the 2026-10-28 scores again: a step that waits for the secret, a run step that
can be skipped, and a push from the detached HEAD that actions/checkout leaves behind.
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
    # The key is optional, and when it exists the secret is the only place it is read.
    assert doc["jobs"]["live"]["env"]["FRED_API_KEY"] == "${{ secrets.FRED_API_KEY }}"
    assert "FRED_API_KEY:" in text and "sk-" not in text
    assert "api_key=" not in text
    assert doc["jobs"]["live"]["if"] == "github.ref == 'refs/heads/main'"


def test_a_missing_fred_key_does_not_stop_the_job() -> None:
    doc = _workflow()
    steps = doc["jobs"]["live"]["steps"]
    # No step waits for the secret or fails because it is missing.
    assert not any("FRED_API_KEY" in str(s.get("if", "")) for s in steps)
    assert not any("exit 1" in str(s.get("run", "")) for s in steps)
    # The route is chosen from the secret alone and passed on; the run itself always happens.
    route = _step(doc, "Choose the FRED route")
    assert route["id"] == "fred" and "if" not in route
    assert 'if [ -n "$FRED_API_KEY" ]; then route=api; else route=csv; fi' in route["run"]
    assert 'echo "route=$route" >> "$GITHUB_OUTPUT"' in route["run"]
    run = _step(doc, "Score finished decisions and publish due forecasts")
    assert "if" not in run and run["run"] == "uv run --no-dev wsl live run"
    assert "if" not in _step(doc, "Install")
    names = [s.get("name") for s in steps]
    assert names.index("Choose the FRED route") < names.index(
        "Score finished decisions and publish due forecasts") < names.index(
        "Regenerate the documents that quote the live results") < names.index(
        "Commit new forecasts and scores") < names.index("Report what is still owed")


def test_a_new_score_is_committed_with_the_documents_that_count_it() -> None:
    """A score changes the README's "Live forecasts" line. If the job committed the score
    alone, tests/test_docs.py would fail on the next push to main."""
    doc = _workflow()
    regen = _step(doc, "Regenerate the documents that quote the live results")
    assert "if" not in regen and regen["run"] == "uv run --no-dev wsl report docs"
    script = _step(doc, "Commit new forecasts and scores")["run"]
    quiet = script.index("git diff --cached --quiet")
    docs_add = script.index(
        "git add web/public/data/live.json README.md docs/RESEARCH_NOTE.md docs/figures")
    assert quiet < docs_add  # documents are added only when live/ itself changed


def test_score_commit_pushes_the_detached_head_to_main() -> None:
    doc = _workflow()
    commit = _step(doc, "Commit new forecasts and scores")
    assert "if" not in commit  # runs whenever the live run succeeded, with or without a key
    script = commit["run"]
    assert 'git push origin "HEAD:${GITHUB_REF_NAME}"' in script
    assert "\ngit push\n" not in script
    # A timestamp-only rewrite of live.json is not a new score. Add it after live/ changed.
    live_add = script.index("git add live\n")
    quiet = script.index("git diff --cached --quiet")
    site_add = script.index("git add web/public/data/live.json")
    assert live_add < quiet < site_add
    # The report runs even after a failed step, says which route was used, and fails the job
    # when a forecast that is due was not written.
    owed = _step(doc, "Report what is still owed")
    assert owed["if"] == "always()"
    assert 'wsl.live.pending --route "${{ steps.fred.outputs.route }}"' in owed["run"]
