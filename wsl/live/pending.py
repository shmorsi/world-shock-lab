"""What live mode owes right now, read from the committed files alone (no data, no network).

The live workflow runs this on every scheduled run and writes the result to the log and the
job summary. Without the FRED_API_KEY secret nothing can be forecast or scored, so this is how
a skipped run says what it skipped. A forecast that is due fails the run: one that is not
published at least ``min_hours_before`` the announcement can never count, while scores can
always be computed later.

    python -m wsl.live.pending [--no-key] [--now 2026-11-30T22:30:00Z]

Imports only the standard library and PyYAML, so the workflow can run it without installing
the project.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from wsl.paths import ROOT

LIVE_CONFIG = ROOT / "config" / "live.yaml"
FORECASTS = ROOT / "live" / "forecasts"
SCORES = ROOT / "live" / "scores"
LOOKBACK = timedelta(days=60)
# A decision whose forecast window opened less than this long ago is not yet called due, so a
# run that starts just before the window opens and reports just after it does not fail.
GRACE = timedelta(hours=1)


@dataclass
class Owed:
    due: list[str] = field(default_factory=list)  # forecasts to publish before a deadline
    missed: list[str] = field(default_factory=list)  # deadline passed with no forecast
    scores: list[str] = field(default_factory=list)  # decided, forecast published, not scored
    next_window: str = ""


def decisions(cfg: dict[str, Any]) -> list[tuple[str, str, datetime]]:
    """(bank, decision date, tz-aware announcement time), sorted by time."""
    out = []
    for bank, c in cfg["calendar"].items():
        hh, mm = (int(x) for x in str(c["time"]).split(":"))
        for d in c["dates"]:
            day = d if isinstance(d, date) else date.fromisoformat(str(d))
            out.append((bank, day.isoformat(),
                        datetime.combine(day, time(hh, mm), tzinfo=ZoneInfo(c["tz"]))))
    return sorted(out, key=lambda r: (r[2], r[0]))


def _utc(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def owed(cfg: dict[str, Any], now: datetime, forecasts: Path = FORECASTS,
         scores: Path = SCORES) -> Owed:
    ahead = timedelta(days=int(cfg["schedule"]["publish_days_before"]))
    notice = timedelta(hours=float(cfg["schedule"]["min_hours_before"]))
    out = Owed()
    for bank, day, ts in decisions(cfg):
        fid = f"{day}_{bank}"
        left = ts - now
        if not (forecasts / f"{fid}.json").exists():
            if notice <= left <= ahead - GRACE:
                out.due.append(f"{fid}: forecast due, deadline {_utc(ts - notice)}")
            elif -LOOKBACK <= left < notice:
                out.missed.append(f"{fid}: no forecast, deadline was {_utc(ts - notice)}")
            elif left > ahead - GRACE and not out.next_window:
                out.next_window = (f"{fid}: forecast window opens {_utc(ts - ahead)}, "
                                   f"deadline {_utc(ts - notice)}")
        elif left < timedelta(0):
            want = ["score1"] + (["score2"] if bank == "fomc" else [])
            missing = [k for k in want if not (scores / f"{fid}_{k}.json").exists()]
            if missing:
                out.scores.append(f"{fid}: decided {_utc(ts)}, waiting for "
                                  f"{', '.join(missing)}")
    return out


def report(o: Owed, no_key: bool) -> tuple[list[tuple[str, str]], str]:
    """(level, message) annotations and the Markdown job summary."""
    notes: list[tuple[str, str]] = []
    if no_key:
        notes.append(("warning", "FRED_API_KEY secret is not set, so this run published no "
                      "forecasts and scored nothing. Add it under Settings > Secrets and "
                      "variables > Actions, or run 'uv run wsl live run' locally and push."))
    notes += [("error", f"{x}. Without the key it will be missed." if no_key else x)
              for x in o.due]
    notes += [("warning", x) for x in o.missed]
    if no_key:
        notes += [("warning", f"{x}. Scoring needs the key.") for x in o.scores]

    md = ["## Live mode", ""]
    if no_key:
        md += ["**Skipped: the `FRED_API_KEY` secret is not set.** Nothing was downloaded, "
               "forecast or scored. Add the secret under Settings > Secrets and variables > "
               "Actions, or run `uv run wsl live run` locally and push.", ""]
    for title, rows in (("Forecasts due", o.due), ("Forecasts missed", o.missed),
                        ("Waiting for scores", o.scores)):
        md += [f"**{title}:** " + ("none" if not rows else ""), *[f"- {r}" for r in rows], ""]
    if o.next_window:
        md += [f"**Next:** {o.next_window}", ""]
    if o.scores and not no_key:
        md += ["Score 1 waits until every 5-day outcome is published (USD/CAD comes from the "
               "weekly H.10 release); score 2 waits for the JK intraday surprise.", ""]
    return notes, "\n".join(md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="What live mode owes right now.")
    ap.add_argument("--no-key", action="store_true",
                    help="the FRED key is missing, so this run did nothing")
    ap.add_argument("--now", help="ISO time to evaluate at (default: now)")
    args = ap.parse_args(argv)
    now = (datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now
           else datetime.now(UTC))
    o = owed(yaml.safe_load(LIVE_CONFIG.read_text()), now, FORECASTS, SCORES)
    notes, md = report(o, args.no_key)
    gha = os.environ.get("GITHUB_ACTIONS") == "true"
    for level, msg in notes:
        print(f"::{level}::{msg}" if gha else f"{level}: {msg}")
    print(md)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as fh:
            fh.write(md + "\n")
    return 1 if o.due else 0


if __name__ == "__main__":
    sys.exit(main())
