"""BoC announcement-list construction on synthetic releases (the real list is checked by the
`data`-marked integration test)."""

from __future__ import annotations

import pandas as pd
import pytest

from wsl.shocks.boc_fad import (
    Release,
    assemble,
    is_decision_title,
    schedule_time,
    stated_next_date,
    title_direction,
    validate,
)


@pytest.mark.parametrize(("title", "expected"), [
    ("Bank of Canada maintains the policy rate at 2¼%", "hold"),
    ("Bank of Canada raises overnight rate target by 1/4 percentage point to 4 1/2 per cent",
     "hike"),
    ("Bank of Canada lowers overnight rate target by 1/4 percentage point to 1/4 per cent and, "
     "conditional on the inflation outlook, commits to hold current policy rate", "cut"),
    ("Bank of Canada keeps Bank Rate at 6 per cent", "hold"),
    ("Bank of Canada lowers Bank Rate to 5 3/4 per cent", "cut"),
])
def test_title_direction(title: str, expected: str) -> None:
    assert title_direction(title) == expected
    assert is_decision_title(title)


def test_schedule_releases_are_not_decisions() -> None:
    assert not is_decision_title(
        "Bank of Canada releases 2008 schedule of dates for its policy interest rate announcements")


@pytest.mark.parametrize(("text", "expected"), [
    ("... The next scheduled date for announcing the overnight rate target is 5 September 2007.",
     "2007-09-05"),
    ("The next scheduled date for announcing the overnight rate target is October 28, 2026. The",
     "2026-10-28"),
    ("The next scheduled date for announcing the overnight rate target is Wednesday, "
     "December 9, 2026.", "2026-12-09"),
    ("No date here.", None),
])
def test_stated_next_date(text: str, expected: str | None) -> None:
    got = stated_next_date(text)
    assert (got.strftime("%Y-%m-%d") if got is not None else None) == expected


def test_schedule_time() -> None:
    assert schedule_time("All announcements will continue to be made at 9 a.m. (ET).") == (
        "09:00", True)
    assert schedule_time("All interest rate announcements will take place at 09:45 (ET), and "
                         "the Monetary Policy Report will be published at 10:30 (ET)") == (
        "09:45", False)
    # The time for the rest of the current year and survey times are not the schedule's time.
    assert schedule_time("All these remaining 2012 rate announcements will continue to be made "
                         "at 09:00 (ET) and the October MPR will be published at 10:30 (ET)") \
        is None
    assert schedule_time("All survey releases will take place at 10:30 (ET).") is None
    assert schedule_time("no time") is None


def _target(points: dict[str, float]) -> pd.Series:
    """Business-daily target with step changes on the given dates."""
    idx = pd.bdate_range("2022-01-03", "2022-04-29")
    s = pd.Series(1.0, index=idx)
    for d, v in points.items():
        s[s.index >= pd.Timestamp(d)] = v
    return s


def _rel(date: str, title: str, nxt: str | None) -> Release:
    text = f"The next scheduled date for announcing the overnight rate target is {nxt}." \
        if nxt else ""
    return Release(url=f"https://www.bankofcanada.ca/{date}/", title=title,
                    date=pd.Timestamp(date), text=text)


def test_assemble_handles_next_day_effective_changes_and_flags_unscheduled() -> None:
    # Hike announced Wed 2022-03-02, recorded in the target series on Thu 2022-03-03.
    target = _target({"2022-03-03": 1.5, "2022-04-14": 2.0})
    decisions = [
        _rel("2022-01-26", "Bank of Canada maintains policy rate", "March 2, 2022"),
        _rel("2022-03-02", "Bank of Canada raises policy rate", "April 13, 2022"),
        _rel("2022-04-13", "Bank of Canada raises policy rate", None),
        _rel("2022-03-18", "Bank of Canada maintains policy rate", None),  # not named: unscheduled
    ]
    ev = assemble(decisions, [], target).set_index("date")
    assert ev.loc["2022-03-02", "change_bp"] == 50
    assert ev.loc["2022-03-02", "target_effective_date"] == pd.Timestamp("2022-03-03")
    assert ev.loc["2022-04-13", "decision"] == "hike"
    assert ev.loc["2022-01-26", "decision"] == "hold"
    assert not ev.loc["2022-03-18", "scheduled"]
    problems = validate(ev.reset_index(), target)
    assert [p for p in problems if not p.startswith("(d)")] == []


def test_validate_catches_missing_event_wrong_direction_and_broken_chain() -> None:
    target = _target({"2022-03-03": 1.5, "2022-04-14": 2.0, "2022-04-21": 2.5})
    decisions = [
        _rel("2022-01-26", "Bank of Canada maintains policy rate", "March 9, 2022"),  # wrong
        _rel("2022-03-02", "Bank of Canada lowers policy rate", None),  # title contradicts data
        _rel("2022-04-13", "Bank of Canada raises policy rate", None),
    ]  # and the 2022-04-21 change has no release at all
    ev = assemble(decisions, [], target)
    problems = validate(ev, target)
    assert any(p.startswith("(a)") and "2022-04-21" in p for p in problems)
    assert any(p.startswith("(b)") and "2022-03-02" in p for p in problems)
    assert any(p.startswith("(c)") and "2022-03-09" in p for p in problems)
