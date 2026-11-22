"""The statement corpus: the text of every policy statement behind the lab's event lists.

Bank of Canada: one press release per row of ``data/events/boc_fad.csv`` (built from the
BoC's own press-release listing, see ``wsl.shocks.boc_fad``). Federal Reserve: every FOMC
statement linked from the Fed's yearly "FOMC historical" pages (2000 to 2020), and every
"Federal Reserve issues FOMC statement" release in the Fed's press-release index (2021 on).

Each page goes through the raw cache with its own manifest (``data/manifest_rag.json``), so
every page used is hashed and a revised page fails loudly instead of changing the corpus.
The output, ``data/rag/statements.jsonl``, is committed: one statement per line with its
paragraphs, official URL and the SHA-256 of the page it came from. Fed statements are US
government works (public domain); BoC terms allow copying with attribution
(data/SOURCES.md §8).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from wsl.data.cache import IntegrityError, RawCache
from wsl.data.http import Http
from wsl.paths import DATA_DIR, EVENTS_DIR

RAG_MANIFEST = Path("data/manifest_rag.json")
RAG_DATA = DATA_DIR / "rag"
STATEMENTS = RAG_DATA / "statements.jsonl"
FIRST_YEAR = 2000
FED = "https://www.federalreserve.gov"
FED_INDEX = f"{FED}/json/ne-press.json"
FED_HISTORICAL_LAST_YEAR = 2020  # later years are on the calendar page, not a historical page

_FED_PATH = re.compile(r"(\d{8})")
_NOISE = re.compile(
    r"^(for immediate release|for release at|share$|for media inquiries|implementation note"
    r"|related information|last update|\d{4} monetary policy|home \|)",
    re.I,
)
_DATELINE = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$")


@dataclass(frozen=True)
class Statement:
    doc_id: str
    bank: str  # "boc" | "fed"
    date: str  # YYYY-MM-DD, the announcement date
    title: str
    url: str
    sha256: str  # of the page as downloaded
    text_sha256: str  # of the extracted paragraphs: the check that a statement is unchanged
    paragraphs: list[str]


def text_sha256(paragraphs: list[str]) -> str:
    return hashlib.sha256("\n\n".join(paragraphs).encode()).hexdigest()


def _cache() -> RawCache:
    return RawCache(manifest=RAG_MANIFEST)


def _get(http: Http, cache: RawCache, key: str, url: str, refresh: bool) -> tuple[bytes, str]:
    """Bytes for ``key`` and their SHA-256, downloading only when the local cache lacks them.

    Pages carry a per-request bot-check snippet, so the bytes of a re-download differ even
    when the statement does not. The page hash is kept as the record of what was read; the
    check that a statement has not changed is on its extracted text (``text_sha256``)."""
    if not refresh:
        try:
            content, rec = cache.read(key)
            return content, rec.sha256
        except IntegrityError:
            pass
    got = http.get(url)
    res = cache.store(key, got.content, url=got.url, ext="html", repin=True)
    return got.content, res.record.sha256


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


def _keep(text: str) -> bool:
    return bool(text) and not _NOISE.match(text) and not _DATELINE.match(text)


_P_SPLIT = re.compile(r"<p\b[^>]*>|</p>|<br\s*/?>\s*<br\s*/?>|</?t[dr]\b[^>]*>", re.I)


def _split_paragraphs(html: str) -> list[str]:
    """Paragraph texts from raw HTML, split on the tags themselves: pages from 2000 to 2005
    leave <p> unclosed, so a parsed tree nests every paragraph inside the one before it."""
    out: list[str] = []
    for piece in _P_SPLIT.split(html):
        text = _clean(BeautifulSoup(piece, "html.parser").get_text(" ", strip=True))
        text = re.sub(r"^For immediate release\s*", "", text, flags=re.I)
        if _keep(text) and text not in out:
            out.append(text)
    return out


def parse_boc(html: bytes) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("div.post-content")
    if body is None:
        raise ValueError("BoC page has no div.post-content")
    title = _clean(soup.title.get_text()) if soup.title else ""
    title = re.sub(r"\s*-\s*Bank of Canada$", "", title)
    paras = [t for li_or_p in body.find_all(["p", "li"])
             if (t := _clean(li_or_p.get_text(" ", strip=True))) and _keep(t)]
    return title, paras


def parse_fed(html: bytes) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = _clean(soup.title.get_text()) if soup.title else ""
    title = re.sub(r"^(Federal Reserve Board - |FRB: Press Release\s*--\s*)", "", title)
    article = soup.select_one("#article")
    if article is not None:
        cols = [d for d in article.find_all("div", recursive=False)
                if "col-sm-8" in (d.get("class") or []) and "heading" not in (d.get("class") or [])]
        root = str(cols[0] if cols else article)
    else:
        # 2000-2005: the statement is the table cell holding the most paragraph text
        cells = soup.find_all("td")
        root = str(max(cells, key=lambda c: len(c.get_text())) if cells else soup)
    paras = [p for p in _split_paragraphs(root)
             if not re.match(r"^(an official website|release date:)", p, re.I)]
    if not paras:
        raise ValueError("Fed page has no paragraphs")
    return title, paras


_FAQ = re.compile(r"^(frequently asked questions|current faqs|faqs:)", re.I)


def tidy(paragraphs: list[str]) -> list[str]:
    """Drop what is page furniture rather than statement: link titles and contact lines
    (short, no closing punctuation), and FAQ sections appended to a few Fed releases."""
    out: list[str] = []
    for p in paragraphs:
        if _FAQ.match(p):
            break
        if len(p) < 160 and not p.rstrip().endswith((".", ":", ";", ")", '"', "”")):
            continue
        out.append(p)
    return out


def boc_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_DIR / "boc_fad.csv", dtype=str)
    return ev[ev["url"].notna()][["date", "title", "url"]].reset_index(drop=True)


def fed_statement_links(http: Http, cache: RawCache, refresh: bool,
                        last_year: int) -> list[tuple[str, str]]:
    """(date, absolute url) for every FOMC statement from FIRST_YEAR to ``last_year``."""
    found: dict[str, str] = {}
    for year in range(FIRST_YEAR, FED_HISTORICAL_LAST_YEAR + 1):
        html, _ = _get(http, cache, f"rag_web/fed_index/historical{year}",
                       f"{FED}/monetarypolicy/fomchistorical{year}.htm", refresh)
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            if a.get_text(strip=True) != "Statement":
                continue
            href = str(a["href"])
            m = _FED_PATH.search(href)
            if m and href.startswith(("/boarddocs/press/", "/newsevents/")):
                date = datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d")
                # 2006-2008 links use an old path that redirects to plain http; use the new one
                href = re.sub(r"^/newsevents/press/monetary/(\d{8}[a-z]?)\.htm$",
                              r"/newsevents/pressreleases/monetary\1.htm", href)
                found.setdefault(date, FED + href)
    raw, _ = _get(http, cache, "rag_web/fed_index/ne-press", FED_INDEX, refresh)
    for item in json.loads(raw.decode("utf-8-sig")):
        if item.get("t") != "Federal Reserve issues FOMC statement":
            continue
        date = datetime.strptime(item["d"].split(" ")[0], "%m/%d/%Y").strftime("%Y-%m-%d")
        if FED_HISTORICAL_LAST_YEAR < int(date[:4]) <= last_year:
            found.setdefault(date, FED + item["l"])
    return sorted(found.items())


def build(refresh: bool = False, last_date: str | None = None,
          repin: bool = False) -> list[Statement]:
    """Download (or read from the pinned cache) and parse every statement."""
    http, cache = Http(), _cache()
    out: list[Statement] = []
    boc = boc_events()
    end = last_date or max(boc["date"])
    for row in boc.itertuples():
        html, sha = _get(http, cache, f"rag_web/boc/{row.date}", str(row.url), refresh)
        title, paras = parse_boc(html)
        paras = tidy(paras)
        out.append(Statement(f"boc-{row.date}", "boc", str(row.date), title or str(row.title),
                             str(row.url), sha, text_sha256(paras), paras))
    for date, url in fed_statement_links(http, cache, refresh, int(end[:4])):
        if date > end:
            continue
        html, sha = _get(http, cache, f"rag_web/fed/{date}", url, refresh)
        title, paras = parse_fed(html)
        paras = tidy(paras)
        out.append(Statement(f"fed-{date}", "fed", date, title, url, sha, text_sha256(paras),
                             paras))
    out.sort(key=lambda s: (s.date, s.bank))
    if STATEMENTS.exists() and not repin:
        before = {s.doc_id: s.text_sha256 for s in load()}
        changed = [s.doc_id for s in out
                   if s.doc_id in before and before[s.doc_id] != s.text_sha256]
        if changed:
            raise IntegrityError(
                f"{len(changed)} statement(s) now read differently from data/rag/statements.jsonl "
                f"(first: {changed[0]}); inspect them, then rerun with repin=True")
    return out


def write(statements: list[Statement], path: Path = STATEMENTS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in statements:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return path


def load(path: Path = STATEMENTS) -> list[Statement]:
    with path.open() as f:
        return [Statement(**json.loads(line)) for line in f if line.strip()]
