"""Ask the decisions: parsers, tokenizers, BM25, fusion, metrics and the test-split gate.

No network and no model download: the static-model checks use a small vocabulary cut from the
real one, stored in tests/fixtures/rag_cases.json, which the TypeScript twin also reads
(web/src/ask/ask.test.ts).
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from wsl.rag import evaluate
from wsl.rag.bm25 import BM25
from wsl.rag.corpus import load, parse_boc, parse_fed, text_sha256, tidy
from wsl.rag.retrieve import rrf, to_statements, top_by_score
from wsl.rag.static import StaticModel, normalize, pre_tokenize, wordpiece
from wsl.rag.text import Chunk, chunks, header, merge_short, protocol, s_stem, tokenize

FIXTURES = Path(__file__).parent / "fixtures"
CASES = FIXTURES / "rag_cases.json"

# -- corpus ---------------------------------------------------------------------------------

BOC_HTML = b"""<html><head>
<title>Bank of Canada holds policy rate at 2\xc2\xbe% - Bank of Canada</title></head>
<body><div class="post-content">
<p>The Bank of Canada today maintained its target for the overnight rate at 2.75%.</p>
<p>Information note:</p>
<p>The next scheduled date for announcing the overnight rate target is June 4, 2025.</p>
</div></body></html>"""

OLD_FED_HTML = b"""<HTML><HEAD>
<TITLE>FRB: Press Release -- FOMC statement -- February 2, 2000</TITLE></HEAD>
<BODY><TABLE><TR><TD><FONT>For immediate release</FONT></TD></TR></TABLE>
<TABLE><TR><TD><p>
The Federal Open Market Committee voted today to raise its target by 25 basis points.
<p>
The Committee remains concerned that over time increases in demand will exceed supply.
<p></TD></TR></TABLE><a href="/x">2000 Monetary policy</a> Home | News and events
<script>window.__CF$cv$params={r:'abc'};</script></BODY></HTML>"""

NEW_FED_HTML = b"""<html><head><title>Federal Reserve Board - Federal Reserve issues FOMC statement
</title></head><body><div id="article">
<div class="heading col-xs-12 col-sm-8 col-md-8">
<p class="article__time">October 28, 2026</p></div>
<div class="col-xs-12 col-sm-8 col-md-8">
<p>The Committee decided to maintain the target range for the federal funds rate.</p>
<p>Voting for the monetary policy action were Jerome H. Powell, Chair.</p>
<p>Frequently Asked Questions: Something</p><p>What is it?</p>
</div></div></body></html>"""


def test_parse_boc_keeps_statement_and_drops_furniture() -> None:
    title, paras = parse_boc(BOC_HTML)
    assert title == "Bank of Canada holds policy rate at 2¾%"
    assert tidy(paras) == [
        "The Bank of Canada today maintained its target for the overnight rate at 2.75%.",
        "Information note:",
        "The next scheduled date for announcing the overnight rate target is June 4, 2025.",
    ]


def test_parse_fed_splits_unclosed_paragraphs_of_old_pages() -> None:
    title, paras = parse_fed(OLD_FED_HTML)
    assert title.startswith("FOMC statement")
    assert paras == [
        "The Federal Open Market Committee voted today to raise its target by 25 basis points.",
        "The Committee remains concerned that over time increases in demand will exceed supply.",
    ]


def test_parse_fed_new_layout_stops_at_faq() -> None:
    title, paras = parse_fed(NEW_FED_HTML)
    assert title == "Federal Reserve issues FOMC statement"
    assert tidy(paras) == [
        "The Committee decided to maintain the target range for the federal funds rate.",
        "Voting for the monetary policy action were Jerome H. Powell, Chair.",
    ]


def test_committed_corpus_is_consistent() -> None:
    docs = load()
    assert len(docs) == protocol()["corpus"]["statements"]
    assert len({d.doc_id for d in docs}) == len(docs)
    assert all(d.text_sha256 == text_sha256(d.paragraphs) for d in docs)
    assert all(d.url.startswith(("https://www.bankofcanada.ca/", "https://www.federalreserve.gov/"))
               for d in docs)


# -- chunks and keyword tokens ---------------------------------------------------------------

def test_merge_short() -> None:
    assert merge_short(["a" * 250, "short", "b" * 250], 200) == ["a" * 250, "short " + "b" * 250]
    assert merge_short(["a" * 250, "tail"], 200) == ["a" * 250 + " tail"]
    assert merge_short(["x", "y"], 200) == ["x y"]


def test_chunk_header_and_ids() -> None:
    d = next(s for s in load() if s.doc_id == "boc-2022-07-13")
    assert header(d).startswith("Bank of Canada, 13 July 2022: ")
    cs = chunks([d])
    assert [c.chunk_id for c in cs] == [f"boc-2022-07-13#{i}" for i in range(len(cs))]
    assert all(len(c.body) >= 200 for c in cs)


@pytest.mark.parametrize(("word", "stem"), [
    ("policies", "policy"), ("series", "sery"), ("s", ""), ("cases", "case"), ("goes", "goe"),
    ("rates", "rate"), ("bus", "bus"), ("pass", "pass"), ("aies", "aie"), ("shoes", "shoe"),
])
def test_s_stemmer(word: str, stem: str) -> None:
    assert s_stem(word) == stem


def test_tokenize() -> None:
    # The possessive "s" stems to "" as the frozen rule says; it is a term like any other.
    assert tokenize("The Bank's rates: 2¼% — Café policies!") == [
        "bank", "", "rate", "21", "4", "cafe", "policy"]  # NFKD: 2¼ -> 21⁄4


# -- BM25, ranking and fusion ----------------------------------------------------------------

def test_bm25_matches_the_formula() -> None:
    docs = [["rate", "cut", "rate"], ["oil", "price"], ["rate", "hike"]]
    bm = BM25(docs, k1=1.2, b=0.75)
    avg = 7 / 3
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    tf_part = 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / avg))
    assert bm.scores(["rate"])[0] == pytest.approx(idf * tf_part)
    assert bm.scores(["rate", "rate"])[0] == pytest.approx(idf * tf_part)  # set of query terms
    assert bm.scores(["missing"]).sum() == 0


def test_top_by_score_breaks_ties_by_index_and_drops_zero() -> None:
    assert top_by_score(np.array([0.5, 0.9, 0.5, 0.0, 0.9]), 10) == [1, 4, 0, 2]
    assert top_by_score(np.array([0.5, 0.9, 0.5]), 2) == [1, 0]


def test_rrf() -> None:
    assert rrf([[1, 2, 3], [3, 1]], 60) == [1, 3, 2]
    assert rrf([[5], [6]], 60) == [5, 6]  # equal score: earlier best rank, then first list


def test_to_statements_keeps_best_chunk_order() -> None:
    cs = [Chunk(f"{d}#{i}", d, "fed", "2020-01-01", "", "") for d, i in
          [("a", 0), ("a", 1), ("b", 0)]]
    assert to_statements([1, 2, 0], cs) == ["a", "b"]


# -- the static model's tokenizer, against cases cut from the real model --------------------

def _cases() -> dict:  # type: ignore[type-arg]
    return json.loads(CASES.read_text())


def test_wordpiece_and_static_encoding_match_the_fixture() -> None:
    c = _cases()["static"]
    vocab = c["vocab"]
    vectors = np.array(c["vectors"], dtype=np.float32)
    model = StaticModel(vocab=vocab, vectors=vectors)
    for case in c["cases"]:
        assert model.token_ids(case["text"]) == case["ids"], case["text"]
        assert model.encode([case["text"]])[0] == pytest.approx(case["vector"], abs=1e-6)


def test_keyword_fixture_cases() -> None:
    c = _cases()
    for case in c["tokenize"]:
        assert tokenize(case["text"]) == case["tokens"]
    bm = BM25(c["bm25"]["docs"])
    assert bm.scores(c["bm25"]["query"]) == pytest.approx(c["bm25"]["scores"])
    for case in c["rrf"]:
        assert rrf(case["lists"], 60) == case["fused"]


def test_normalize_and_pre_tokenize() -> None:
    assert normalize("Café\tCRÈME 東京") == "cafe creme  東  京 "
    assert pre_tokenize("u.s.-mexico 2¼%") == ["u", ".", "s", ".", "-", "mexico", "2¼", "%"]
    assert wordpiece("x" * 101, {"x": 1}) is None


# -- scoring and the gate ---------------------------------------------------------------------

def test_metrics() -> None:
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert evaluate.per_question(ranked, ["c", "z"]) == {
        "hit@1": 0.0, "hit@5": 1.0, "hit@10": 1.0, "mrr@10": pytest.approx(1 / 3)}
    assert evaluate.per_question(ranked, ["z"])["mrr@10"] == 0.0


def test_bootstrap_and_verdict() -> None:
    a = np.array([1.0] * 30 + [0.0] * 10)
    b = np.array([0.0] * 30 + [0.0] * 10)
    ci = evaluate.paired_bootstrap(a, b, 2000, 1)
    assert ci["difference"] == pytest.approx(0.75)
    assert evaluate.verdict(ci) == "helps"
    assert evaluate.verdict({"lo": -0.1, "hi": 0.2}) == "no detectable difference"
    assert evaluate.verdict({"lo": -0.3, "hi": -0.1}) == "hurts"
    assert evaluate.paired_bootstrap(a, b, 500, 7) == evaluate.paired_bootstrap(a, b, 500, 7)


def test_question_set_is_as_registered() -> None:
    ids = {d.doc_id for d in load()}
    for split, n in (("dev", 16), ("test", 48)):
        qs = evaluate.questions(split)
        assert len(qs) == n
        assert all(g in ids and g.startswith(q["bank"]) for q in qs for g in q["gold"])


def test_gate_refuses_without_the_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_git(*args: str) -> subprocess.CompletedProcess[str]:
        code = 1 if args[0] == "rev-parse" else 0
        return subprocess.CompletedProcess(args, code, "", "")

    monkeypatch.setattr(evaluate, "_git", fake_git)
    with pytest.raises(evaluate.GateError, match="not found"):
        evaluate.check_gate()


def test_gate_refuses_when_frozen_files_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1 if args[0] == "diff" else 0, "", "")

    monkeypatch.setattr(evaluate, "_git", fake_git)
    with pytest.raises(evaluate.GateError, match="differ"):
        evaluate.check_gate()


def test_a_second_test_scoring_needs_a_reason(monkeypatch: pytest.MonkeyPatch,
                                              tmp_path: Path) -> None:
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    monkeypatch.setattr(evaluate, "check_gate", lambda: None)
    (tmp_path / "test.json").write_text("{}")
    with pytest.raises(evaluate.GateError, match="reason"):
        evaluate.run("test")


# -- answers ----------------------------------------------------------------------------------

class _Block:
    def __init__(self, text: str) -> None:
        self.type, self.text = "text", text


class _FakeClient:
    """Stands in for anthropic.Anthropic(): returns a fixed reply, records the request."""

    def __init__(self, reply: str, stop_reason: str = "end_turn") -> None:
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []
        self.beta = self
        self.messages = self

    def create(self, **kw):  # type: ignore[no-untyped-def]
        self.calls.append(kw)
        return type("R", (), {"stop_reason": self.stop_reason,
                              "content": [_Block(self.reply)]})()


class _OneDocSearcher:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def passages(self, kind: str, query: str, k: int = 5) -> list[Chunk]:
        return self.chunks[:k]


def _july_2022() -> list[Chunk]:
    d = next(s for s in load() if s.doc_id == "boc-2022-07-13")
    return chunks([d])[:2]


def test_answer_with_supported_numbers_and_a_citation_passes() -> None:
    from wsl.rag.answer import ask

    client = _FakeClient("The Bank raised its policy rate by 100 basis points to 2½% [1].")
    a = ask("full point hike?", _OneDocSearcher(_july_2022()), client=client)  # type: ignore[arg-type]
    assert a.method == "claude" and a.cited == [1] and a.unsupported == []
    req = client.calls[0]
    assert req["model"] == "claude-opus-5-5" and req["fallbacks"] == "default"
    assert "[1] Bank of Canada, 13 July 2022" in req["messages"][0]["content"]


def test_answer_with_an_invented_number_is_withheld() -> None:
    from wsl.rag.answer import ask

    client = _FakeClient("The Bank raised by 150 basis points [1].")
    a = ask("full point hike?", _OneDocSearcher(_july_2022()), client=client)  # type: ignore[arg-type]
    assert a.method == "withheld" and a.unsupported == ["150"]


def test_answer_without_citation_is_withheld_and_refusal_falls_back_to_a_quote() -> None:
    from wsl.rag.answer import ask

    s = _OneDocSearcher(_july_2022())
    assert ask("q", s, client=_FakeClient("No citation here.")).method == "withheld"  # type: ignore[arg-type]
    a = ask("what deposit rate did it set", s,  # type: ignore[arg-type]
            client=_FakeClient("", stop_reason="refusal"))
    assert a.method == "extract" and a.text.startswith("Bank of Canada, 13 July 2022: “")
    assert "deposit rate" in a.text and a.unsupported == []


def test_market_moves_come_from_the_lab_replay() -> None:
    from wsl.rag.answer import market_moves, market_text

    moves = market_moves("boc-2019-01-09")
    assert {m["asset"] for m in moves} == {"US2Y", "USDCAD", "WTI"}
    assert "US 2-year yield -2.00 bp" in market_text(moves)
    assert market_moves("boc-2001-01-23") == []


def test_langchain_retriever_returns_documents_with_sources() -> None:
    from wsl.rag.lc import StatementRetriever

    r = StatementRetriever(searcher=_OneDocSearcher(_july_2022()), k=1)
    r.searcher.statements = load()  # what the real Searcher carries
    docs = r.invoke("anything")
    assert len(docs) == 1
    assert docs[0].metadata["url"].startswith("https://www.bankofcanada.ca/2022/07/")
    assert docs[0].metadata["date"] == "2022-07-13"


# -- the generated write-up -------------------------------------------------------------------

def test_rag_documents_match_the_results_and_type_no_number_by_hand() -> None:
    from wsl.rag.report import END, START, render, unregistered

    out, nums = render()
    for path, text in out.items():
        assert path.read_text() == text, f"{path.name} is stale; run `uv run wsl rag report`"
        part = text.split(START, 1)[1].split(END, 1)[0] if START in text else text
        assert unregistered(part, nums) == [], path.name
