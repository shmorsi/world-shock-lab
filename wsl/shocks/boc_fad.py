"""Bank of Canada rate-announcement list, built from official press releases.

There is no official machine-readable list of past announcement dates (data/SOURCES.md §2.2),
so we construct one:

1. crawl the BoC press-release listing (bankofcanada.ca/press/press-releases/?mt_page=N), and
   the monthly archives (bankofcanada.ca/YYYY/MM/) for months the listing does not reach
   (before mid-2001) or where a rate change has no matching release;
2. keep releases whose title announces a rate decision, plus the yearly schedule releases;
3. parse each decision release: date, title, the "next scheduled date" it names;
4. take the decision itself (change in bp, rate after) from the BoC's daily target series
   (Valet CBC20210), not from free text. Rate decisions applied immediately until 2020; from
   2021 the target "will take effect on the business day following each rate announcement"
   (BoC 2021 schedule release, 2020-07-30), and the Friday 2020-03-13 cut shows up on Monday.
   The change is therefore measured from the last value before the announcement to the first
   value after it, which is correct under both conventions;
5. validate:
   (a) every target-rate change since 2001 falls on a listed date,
   (b) the title's verb (raise / lower / maintain) agrees with the sign of that change,
   (c) each release's stated "next scheduled date" is the next scheduled event in the list,
   (d) scheduled events per year.

An event is *scheduled* if the previous release named its date as the next scheduled date.
Announcement times come from the yearly schedule releases ("All announcements will continue
to be made at 9 a.m. (ET)"); ``time_basis`` says whether the time is stated for that year or
inferred (2013: the 2014 release says announcements "continue" at 10:00; 2021: 10:00 stated
for 2020 and 2022 and no change release found). Unscheduled events keep time_et = "unknown"
and are excluded from the BoC proxy.

Crawled pages go through the raw cache with their own manifest
(data/manifest_boc_web.json) so every page used is hashed. The output CSV is committed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from wsl.data.cache import IntegrityError, RawCache
from wsl.data.http import Http
from wsl.paths import EVENTS_DIR

LISTING = "https://www.bankofcanada.ca/press/press-releases/"
BOC_WEB_MANIFEST = Path("data/manifest_boc_web.json")
OUT_CSV = EVENTS_DIR / "boc_fad.csv"
FIRST_DATE = pd.Timestamp("2000-12-01")  # fixed-date system starts Dec 2000 (BoC, Sept 2000)

_DECISION_TITLE = re.compile(
    r"(overnight rate|policy rate|policy interest rate|key interest rate|target for the overnight"
    r"|bank rate|interest rate (?:reduction|increase)s?)",
    re.I,
)
_SCHEDULE_TITLE = re.compile(r"schedule", re.I)
_VERB = {
    "hike": re.compile(r"\b(raises?|increases?|hikes?)\b", re.I),
    "cut": re.compile(r"\b(lowers?|reduces?|reductions?|cuts?|decreases?)\b", re.I),
    "hold": re.compile(r"\b(maintains?|keeps?|holds?|leaves?|unchanged)\b", re.I),
}
_NEXT = re.compile(
    r"next scheduled (?:date|announcement)[^.]{0,120}?\bis\s+(?:on\s+)?"
    r"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday),?\s+)?"
    r"(\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{1,2},\s+\d{4})"
)
_TIME = re.compile(
    r"(?:made|take place)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(?:a\.m\.\s*)?\(ET\)", re.I
)
_YEAR_IN_TITLE = re.compile(r"\b(20\d{2})\b")


@dataclass(frozen=True)
class Release:
    url: str
    title: str
    date: pd.Timestamp
    text: str


def _cache() -> RawCache:
    return RawCache(manifest=BOC_WEB_MANIFEST)


def _get(http: Http, cache: RawCache, key: str, url: str, crawl: bool) -> bytes:
    if not crawl:
        try:
            return cache.read(key)[0]
        except IntegrityError:
            pass
    got = http.get(url)
    cache.store(key, got.content, url=got.url, ext="html", repin=True)
    return got.content


# ------------------------------------------------------------------------------- crawl
def crawl_listing(http: Http, cache: RawCache, crawl: bool) -> list[tuple[str, str]]:
    """All (url, title) pairs in the press-release listing, newest first."""
    seen: dict[str, str] = {}
    page = 1
    while True:
        html = _get(http, cache, f"boc_web/listing/page{page:03d}", f"{LISTING}?mt_page={page}",
                    crawl)
        soup = BeautifulSoup(html, "html.parser")
        # The paginated results are <article class="media"> items; ".mtt-result" is a
        # featured block repeated on every page and must be ignored.
        items = soup.select("article.media h3.media-heading a[href]")
        new = 0
        for a in items:
            href = str(a["href"])
            if href not in seen:
                seen[href] = a.get_text(" ", strip=True)
                new += 1
        if not items or new == 0:
            break
        page += 1
    return list(seen.items())


def crawl_month(http: Http, cache: RawCache, month: pd.Timestamp,
                crawl: bool) -> list[tuple[str, str]]:
    """(url, title) pairs from the monthly archive bankofcanada.ca/YYYY/MM/ (all pages)."""
    seen: dict[str, str] = {}
    page = 1
    while True:
        base = f"https://www.bankofcanada.ca/{month:%Y/%m}/"
        url = base if page == 1 else f"{base}page/{page}/"
        html = _get(http, cache, f"boc_web/archive/{month:%Y-%m}/page{page:02d}", url, crawl)
        soup = BeautifulSoup(html, "html.parser")
        new = 0
        for a in soup.select("article.media h3.media-heading a[href]"):
            href = str(a["href"])
            if href not in seen:
                seen[href] = a.get_text(" ", strip=True)
                new += 1
        if new == 0 or not soup.find("a", href=f"{base}page/{page + 1}/"):
            break
        page += 1
    return list(seen.items())


def _release(http: Http, cache: RawCache, url: str, crawl: bool) -> Release:
    slug = url.rstrip("/").split("bankofcanada.ca/")[-1].replace("/", "__")
    html = _get(http, cache, f"boc_web/release/{slug}", url, crawl)
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", attrs={"name": "publication_date"})
    if meta is None:
        raise ValueError(f"no publication_date on {url}")
    title_tag = soup.find("h1") or soup.title
    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    title = re.sub(r"\s*-\s*Bank of Canada\s*$", "", title)
    main = soup.select_one("main") or soup.body
    text = main.get_text(" ", strip=True) if main else ""
    return Release(url=url, title=title, date=pd.Timestamp(str(meta["content"])), text=text)


# ------------------------------------------------------------------------------- parse
def title_direction(title: str) -> str | None:
    """Direction of the first decision verb in the title ("lowers ... commits to hold" -> cut)."""
    hits = [(m.start(), k) for k, rx in _VERB.items() if (m := rx.search(title))]
    return min(hits)[1] if hits else None


def stated_next_date(text: str) -> pd.Timestamp | None:
    m = _NEXT.search(text)
    if not m:
        return None
    try:
        return pd.Timestamp(pd.to_datetime(m.group(1)))
    except (ValueError, TypeError):
        return None


def schedule_time(text: str) -> tuple[str, bool] | None:
    """(HH:MM, says 'continue') for the rate-announcement time stated in a schedule release.

    Skips times that refer to the remaining announcements of the current year or to survey
    and MPR releases.
    """
    for m in _TIME.finditer(text):
        context = text[max(0, m.start() - 160):m.start()]
        context = context[context.rfind("(ET)") + 1:] if "(ET)" in context else context
        if "announcement" not in context.lower() or re.search(r"remaining|survey", context, re.I):
            continue
        return f"{int(m.group(1)):02d}:{m.group(2) or '00'}", "continue" in context.lower()
    return None


def is_decision_title(title: str) -> bool:
    return bool(_DECISION_TITLE.search(title)) and not _SCHEDULE_TITLE.search(title) \
        and title_direction(title) is not None


# --------------------------------------------------------------------------- assemble
def assemble(
    decisions: list[Release], schedules: list[Release], target: pd.Series
) -> pd.DataFrame:
    """Event table from parsed releases and the daily target rate (pct, indexed by date)."""
    target = target.dropna().sort_index()
    rows = []
    for r in sorted(decisions, key=lambda x: x.date):
        if r.date < FIRST_DATE:
            continue
        before = target[target.index < r.date]
        after = target[target.index > r.date]
        change, rate_after, effective = None, None, None
        if not before.empty and not after.empty:
            rate_after = float(after.iloc[0])
            change = round((rate_after - float(before.iloc[-1])) * 100)
            effective = target[(target.index >= r.date) & (target == rate_after)].index[0]
        rows.append({
            "date": r.date.normalize(),
            "title": r.title,
            "title_direction": title_direction(r.title),
            "change_bp": change,
            "rate_after_pct": rate_after,
            "target_effective_date": effective if change else None,
            "next_scheduled_stated": stated_next_date(r.text),
            "url": r.url,
        })
    ev = pd.DataFrame(rows).drop_duplicates(subset=["date"], keep="first")
    ev = ev.sort_values("date").reset_index(drop=True)

    named = set(ev["next_scheduled_stated"].dropna())
    ev["scheduled"] = ev["date"].isin(named)
    ev.loc[ev.index == 0, "scheduled"] = True  # first FAD (Dec 2000): no previous release

    stated: dict[int, tuple[str, bool, str]] = {}
    for s in schedules:
        t = schedule_time(s.text)
        years = [int(y) for y in _YEAR_IN_TITLE.findall(s.title)]
        if t and years:
            stated[years[0]] = (t[0], t[1], s.url)
    times = _year_times(stated, range(ev["date"].dt.year.min(), ev["date"].dt.year.max() + 1))
    ev["time_et"], ev["time_basis"], ev["time_source"] = "unknown", "", ""
    for i, row in ev.iterrows():
        if row["scheduled"] and row["date"].year in times:
            ev.loc[i, ["time_et", "time_basis", "time_source"]] = times[row["date"].year]

    def _decision(bp: float | None) -> str | None:
        if bp is None or bp != bp:  # None or NaN
            return None
        return "hike" if bp > 0 else "cut" if bp < 0 else "hold"

    ev["decision"] = ev["change_bp"].map(_decision)
    cols = ["date", "time_et", "time_basis", "scheduled", "decision", "change_bp",
            "rate_after_pct", "target_effective_date", "title_direction", "next_scheduled_stated",
            "title", "url", "time_source"]
    return ev[cols]


def _year_times(stated: dict[int, tuple[str, bool, str]],
                years: range) -> dict[int, tuple[str, str, str]]:
    """Announcement time per year: stated, or inferred only from documentary wording."""
    out: dict[int, tuple[str, str, str]] = {}
    for y in years:
        if y in stated:
            out[y] = (stated[y][0], "stated", stated[y][2])
        elif y + 1 in stated and stated[y + 1][1]:
            out[y] = (stated[y + 1][0], "inferred: next year's release says announcements "
                      "'continue' at this time", stated[y + 1][2])
        elif y - 1 in stated and y + 1 in stated and stated[y - 1][0] == stated[y + 1][0]:
            out[y] = (stated[y + 1][0], "inferred: same time stated for previous and next year",
                      stated[y + 1][2])
    return out


# -------------------------------------------------------------------------- validate
def unmatched_changes(ev: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Target changes with no listed event between the previous observation and the change
    date, inclusive (so a change effective the business day after the announcement, as in the
    series from March 2022, is matched to its announcement)."""
    target = target.dropna().sort_index()
    prev_obs = pd.Series(target.index, index=target.index).shift(1)
    changed = target[target.diff().fillna(0) != 0]
    dates = ev["date"].sort_values().to_numpy()
    keep = []
    for d in changed.index:
        lo = prev_obs.loc[d]
        hit = ((dates >= lo) & (dates <= d)).any() if pd.notna(lo) else (dates <= d).any()
        keep.append(not hit)
    return changed[keep]


def validate(ev: pd.DataFrame, target: pd.Series) -> list[str]:
    """Return a list of human-readable problems (empty = all checks pass)."""
    problems: list[str] = []
    target = target.dropna().sort_index()
    unmatched = unmatched_changes(ev, target)
    for d, v in zip(unmatched.index, unmatched.to_numpy(), strict=True):
        problems.append(f"(a) target changed to {v} on {pd.Timestamp(d).date()} "
                        "but no listed event")
    for _, r in ev.iterrows():
        if pd.notna(r["decision"]) and r["title_direction"] != r["decision"]:
            problems.append(f"(b) {r['date'].date()}: title says {r['title_direction']}, "
                            f"target series says {r['decision']}")
    listed, last = set(ev["date"]), ev["date"].max()
    for _, r in ev.iterrows():
        stated = r["next_scheduled_stated"]
        if pd.notna(stated) and stated <= last and stated not in listed:
            problems.append(f"(c) {r['date'].date()} names next date "
                            f"{pd.Timestamp(stated).date()}, which is not in the list")
    sched = ev[ev["scheduled"]].reset_index(drop=True)
    dates = [pd.Timestamp(d) for d in sched["date"]]
    stated_next = list(sched["next_scheduled_stated"])
    for i in range(len(sched) - 1):
        stated = stated_next[i]
        if pd.notna(stated) and pd.Timestamp(stated) != dates[i + 1]:
            problems.append(f"(c) {dates[i].date()} names next date "
                            f"{pd.Timestamp(stated).date()}, next scheduled event is "
                            f"{dates[i + 1].date()}")
    per_year = pd.Series([d.year for d in dates]).value_counts().sort_index()
    for y, n in zip(per_year.index.astype(int), per_year.to_numpy(), strict=True):
        if 2001 <= y <= 2025 and n != 8:
            problems.append(f"(d) {y}: {n} scheduled events (expected 8)")
    return problems


def load_target_rate() -> pd.Series:
    """Daily BoC target rate (Valet CBC20210) from the pinned raw cache."""
    content, _ = RawCache().read("valet/CBC20210")
    rows = json.loads(content)["observations"]
    s = pd.Series({pd.Timestamp(r["d"]): float(r["CBC20210"]["v"]) for r in rows
                   if r.get("CBC20210", {}).get("v")})
    return s.sort_index()


def _split(entries: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    dec = [u for u, t in entries if is_decision_title(t) or "fad-press-release" in u]
    sch = [u for u, t in entries if _SCHEDULE_TITLE.search(t)
           and re.search(r"(announcement|rate)", t, re.I)]
    return dec, sch


def build_fad_list(crawl: bool = False) -> list[str]:
    http, cache = Http(), _cache()
    listing = crawl_listing(http, cache, crawl)
    dec_urls, sch_urls = _split(listing)
    target = load_target_rate()

    def _assemble(dec: list[str], sch: list[str]) -> tuple[pd.DataFrame, list[Release]]:
        releases = [_release(http, cache, u, crawl) for u in dict.fromkeys(dec)]
        releases = [r for r in releases if is_decision_title(r.title)
                    or "fad-press-release" in r.url]
        schedules = [_release(http, cache, u, crawl) for u in dict.fromkeys(sch)]
        return assemble(releases, schedules, target), releases

    ev, releases = _assemble(dec_urls, sch_urls)
    # Months the listing does not reach, plus months with a rate change but no release.
    oldest = ev["date"].min()
    months = set(pd.date_range(FIRST_DATE, oldest, freq="MS"))
    months |= {d.to_period("M").to_timestamp() for d in unmatched_changes(ev, target).index}
    archive_entries = [e for m in sorted(months) for e in crawl_month(http, cache, m, crawl)]
    extra_dec, extra_sch = _split(archive_entries)
    ev, releases = _assemble(dec_urls + extra_dec, sch_urls + extra_sch)
    problems = validate(ev, target)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = ev.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    for col in ("next_scheduled_stated", "target_effective_date"):
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    out.to_csv(OUT_CSV, index=False)
    lines = [
        f"listing entries: {len(listing)}; archive months crawled: {len(months)}; "
        f"decision releases: {len(releases)}",
        f"events: {len(ev)} ({int(ev['scheduled'].sum())} scheduled, "
        f"{int((~ev['scheduled']).sum())} unscheduled), "
        f"{ev['date'].min().date()} -> {ev['date'].max().date()}",
        f"wrote {OUT_CSV}",
        f"validation problems: {len(problems)}",
    ]
    return lines + [f"  {p}" for p in problems]
