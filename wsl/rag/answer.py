"""Answers that cite the statements they come from, and cannot invent a number.

``ask`` retrieves the top statements, then writes an answer with Claude when credentials are
available, or quotes the best-matching sentence when they are not. Either way the rule in
config/rag_protocol.yaml applies: every answer cites numbered passages, and a digit run that
is not in a cited passage or in the lab's market data for that announcement withholds the
answer, so the reader sees the passages instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from typing import Any

from wsl.paths import ROOT
from wsl.rag.retrieve import Searcher
from wsl.rag.text import Chunk, protocol, tokenize

MODEL = "claude-opus-5-5"
REPLAY = ROOT / "web" / "public" / "data" / "replay.json"
_DIGITS = re.compile(r"\d+")
_CITE = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"])")

SYSTEM = """You answer questions about Federal Reserve and Bank of Canada policy statements.
Use only the numbered passages you are given. After each claim, cite the passage it comes
from in square brackets, like [2]. Use only numbers that appear in the passages or in the
market data block. If the passages do not answer the question, say so in one sentence.
Answer in at most four sentences of plain prose."""


@dataclass
class Answer:
    question: str
    text: str
    method: str  # "claude" | "extract" | "withheld"
    passages: list[Chunk]
    cited: list[int] = field(default_factory=list)  # 1-based passage numbers
    unsupported: list[str] = field(default_factory=list)
    market: list[dict[str, Any]] = field(default_factory=list)


@cache
def _replay() -> dict[str, dict[str, Any]]:
    if not REPLAY.exists():
        return {}
    events = json.loads(REPLAY.read_text())["replay"]
    return {e["event"].split(" ")[0]: e for e in events}


def market_moves(doc_id: str) -> list[dict[str, Any]]:
    """The lab's recorded first-observation moves after this announcement (2019 on), or []."""
    bank, day = doc_id.split("-", 1)
    ev = _replay().get(f"{'fomc' if bank == 'fed' else 'boc'}:{day}")
    if ev is None:
        return []
    units = {"US2Y": "bp", "USDCAD": "%", "WTI": "%"}
    names = {"US2Y": "US 2-year yield", "USDCAD": "CAD per USD", "WTI": "WTI oil"}
    return [{"asset": a, "name": names[a], "units": units[a],
             "value": round(float(cell["actual"]), 2)}
            for key, cell in ev["cells"].items() if (a := key.split("|")[0]) in units
            and key.endswith("|h1") and cell.get("actual") is not None]


def market_text(moves: list[dict[str, Any]]) -> str:
    return "; ".join(f"{m['name']} {m['value']:+.2f}{' ' if m['units'] == 'bp' else ''}"
                     f"{m['units']}" for m in moves)


def unsupported_numbers(text: str, sources: list[str]) -> list[str]:
    allowed = {n for s in sources for n in _DIGITS.findall(s)}
    return sorted({n for n in _DIGITS.findall(text) if n not in allowed})


def _label(c: Chunk) -> str:
    names = protocol()["chunking"]["bank_names"]
    d = date.fromisoformat(c.date)
    return f"{names[c.bank]}, {d.day} {d:%B} {d.year}"


def check(text: str, passages: list[Chunk],
          moves: dict[int, list[dict[str, Any]]]) -> tuple[list[int], list[str]]:
    """Cited passage numbers, and the digit runs not supported by what is cited."""
    cited = sorted({int(n) for n in _CITE.findall(text) if 1 <= int(n) <= len(passages)})
    sources = [passages[i - 1].text for i in cited]
    sources += [market_text(moves.get(i, [])) for i in cited]
    body = _CITE.sub("", text)
    return cited, unsupported_numbers(body, sources)


def extract(question: str, passages: list[Chunk], idf: dict[str, float] | None = None) -> str:
    """No model: the sentence of the best passage that shares the rarest words with the
    question (summed BM25 idf; the first sentence wins a tie). The browser does the same."""
    if not passages:
        return "No statement matched the question."
    q = set(tokenize(question))
    weight = idf or {}
    sentences = _SENTENCE.split(passages[0].body)
    scores = [sum(weight.get(t, 1.0) for t in q & set(tokenize(s))) for s in sentences]
    best = sentences[scores.index(max(scores))]
    return f"{_label(passages[0])}: “{best.strip()}” [1]"


def _prompt(question: str, passages: list[Chunk], moves: dict[int, list[dict[str, Any]]]) -> str:
    lines = []
    for i, c in enumerate(passages, start=1):
        lines.append(f"[{i}] {_label(c)} ({c.doc_id})\n{c.body}")
        if moves.get(i):
            lines.append(f"Market data for [{i}], first observation after the announcement: "
                         f"{market_text(moves[i])}")
    return "Passages:\n\n" + "\n\n".join(lines) + f"\n\nQuestion: {question}"


def generate(client: Any, question: str, passages: list[Chunk],
             moves: dict[int, list[dict[str, Any]]]) -> str | None:
    """Claude's answer, or None when it declined. Server-side fallbacks are on, so a policy
    decline is retried on another model inside the same call."""
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": _prompt(question, passages, moves)}],
    )
    if response.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or None


def ask(question: str, searcher: Searcher | None = None, kind: str = "hybrid", k: int = 5,
        client: Any = None, bank: str | None = None) -> Answer:
    s = searcher or Searcher()
    passages = s.passages(kind, question, k, bank) if bank else s.passages(kind, question, k)
    moves = {i: m for i, c in enumerate(passages, start=1) if (m := market_moves(c.doc_id))}
    text, method = None, "extract"
    if client is not None and passages:
        import anthropic

        try:
            text = generate(client, question, passages, moves)
            method = "claude"
        except (anthropic.APIError, anthropic.AnthropicError):
            text = None  # no credentials, network or API error: quote instead
    if text is None:
        idf = s.bm25.idf if isinstance(s, Searcher) else None
        text, method = extract(question, passages, idf), "extract"
    cited, bad = check(text, passages, moves)
    if method == "claude" and (bad or not cited):
        method = "withheld"
        text = ("The generated answer was withheld: "
                + (f"it printed numbers not in its sources ({', '.join(bad)})." if bad
                   else "it cited no passage.")
                + " The passages below are what the search found.")
    return Answer(question, text, method, passages, cited, bad,
                  [m for i in cited for m in moves.get(i, [])])


def default_client() -> Any:
    """An Anthropic client when credentials resolve, else None (answers fall back to quotes)."""
    try:
        import anthropic

        return anthropic.Anthropic()
    except Exception:  # no credentials configured
        return None
