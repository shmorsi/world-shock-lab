"""What live mode owes right now, read from the committed files alone (no data, no network).

The live workflow runs this at the end of every scheduled run and writes the result to the
log and the job summary, with the route the run read FRED by (the API when the FRED_API_KEY
secret is set, FRED's public CSV when it is not). A forecast that is still due fails the
run: one that is not published at least ``min_hours_before`` the announcement can never
count, while scores can always be computed later.

    python -m wsl.live.pending [--route csv|api] [--now 2026-11-30T22:30:00Z]

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
# How a run read the two FRED series (wsl/live/data.py: fred_route).
ROUTE_TEXT = {
    "api": "FRED data: the FRED API (the `FRED_API_KEY` secret is set).",
    "csv": "FRED data: FRED's public CSV download, `fredgraph.csv` (the `FRED_API_KEY` secret is "
           "not set). The series and their values are the same as the API's, but FRED's website "
           "refuses some networks, GitHub's hosted runners among them on 2026-11-11: if this run "
           "stopped on those downloads, add the secret.",
}


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


def report(o: Owed, route: str | None = None) -> tuple[list[tuple[str, str]], str]:
    """(level, message) annotations and the Markdown job summary.

    ``route`` is how the run read FRED ("api" or "csv"); anything else leaves the line out.
    """
    notes: list[tuple[str, str]] = [("error", x) for x in o.due]
    notes += [("warning", x) for x in o.missed]

    md = ["## Live mode", ""]
    if route in ROUTE_TEXT:
        md += [ROUTE_TEXT[route], ""]
    for title, rows in (("Forecasts due", o.due), ("Forecasts missed", o.missed),
                        ("Waiting for scores", o.scores)):
        md += [f"**{title}:** " + ("none" if not rows else ""), *[f"- {r}" for r in rows], ""]
    if o.next_window:
        md += [f"**Next:** {o.next_window}", ""]
    if o.scores:
        md += ["Score 1 waits until every 5-day outcome is published (USD/CAD comes from the "
               "weekly H.10 release); score 2 waits for the JK intraday surprise.", ""]
    return notes, "\n".join(md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="What live mode owes right now.")
    ap.add_argument("--route", default="",
                    help="how the run read FRED: 'api' (key set) or 'csv' (public download)")
    ap.add_argument("--now", help="ISO time to evaluate at (default: now)")
    args = ap.parse_args(argv)
    now = (datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now
           else datetime.now(UTC))
    o = owed(yaml.safe_load(LIVE_CONFIG.read_text()), now, FORECASTS, SCORES)
    notes, md = report(o, args.route)
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
