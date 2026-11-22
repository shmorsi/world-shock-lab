"""Chunks and keyword tokens, exactly as config/rag_protocol.yaml fixes them.

The website runs the same chunker output and a TypeScript twin of ``tokenize``
(web/src/ask/text.ts); tests/fixtures/rag_tokens.json holds shared cases both must pass.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from functools import cache
from typing import Any

import yaml

from wsl.paths import CONFIG_DIR
from wsl.rag.corpus import Statement

PROTOCOL = CONFIG_DIR / "rag_protocol.yaml"
QUESTIONS = CONFIG_DIR / "rag_questions.yaml"


@cache
def protocol() -> dict[str, Any]:
    out: dict[str, Any] = yaml.safe_load(PROTOCOL.read_text())
    return out


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc_id>#<n>"
    doc_id: str
    bank: str
    date: str
    header: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.header}\n{self.body}"


def header(s: Statement) -> str:
    p = protocol()["chunking"]
    d = date.fromisoformat(s.date)
    return str(p["header"]).format(bank_name=p["bank_names"][s.bank], day=d.day,
                              Month=d.strftime("%B"), year=d.year, title=s.title)


def merge_short(paragraphs: list[str], min_chars: int) -> list[str]:
    """Join a paragraph shorter than ``min_chars`` to the next one (the last, to the previous)."""
    out: list[str] = []
    carry = ""
    for p in paragraphs:
        text = f"{carry} {p}".strip() if carry else p
        if len(text) < min_chars:
            carry = text
            continue
        out.append(text)
        carry = ""
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
    return out


def chunks(statements: list[Statement]) -> list[Chunk]:
    min_chars = int(protocol()["chunking"]["merge_below_chars"])
    out: list[Chunk] = []
    for s in statements:
        h = header(s)
        for i, body in enumerate(merge_short(s.paragraphs, min_chars)):
            out.append(Chunk(f"{s.doc_id}#{i}", s.doc_id, s.bank, s.date, h, body))
    return out


@cache
def stopwords() -> frozenset[str]:
    return frozenset(protocol()["retrievers"]["bm25"]["stopwords"])


def s_stem(word: str) -> str:
    """Harman's S-stemmer (1991)."""
    if word.endswith("ies") and not word.endswith(("eies", "aies")):
        return word[:-3] + "y"
    if word.endswith("es") and not word.endswith(("aes", "ees", "oes")):
        return word[:-1]
    if word.endswith("s") and not word.endswith(("us", "ss")):
        return word[:-1]
    return word


_SPLIT = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    folded = "".join(c for c in unicodedata.normalize("NFKD", text)
                     if not unicodedata.combining(c)).lower()
    stop = stopwords()
    return [s_stem(t) for t in _SPLIT.split(folded) if t and t not in stop]
