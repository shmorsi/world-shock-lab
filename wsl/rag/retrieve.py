"""The five retrievers of config/rag_protocol.yaml, ranking statements for a question.

Each retriever ranks chunks; a statement's rank is the rank of its best chunk. Ties keep corpus
order, so the ranking is deterministic and the browser can reproduce it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from wsl.rag import corpus
from wsl.rag.bm25 import BM25
from wsl.rag.dense import INDEX_DIR, QdrantStore, TransformerEncoder, model_dir
from wsl.rag.static import StaticModel
from wsl.rag.text import Chunk, chunks, protocol, tokenize

KINDS = ("bm25", "static", "transformer", "hybrid_static", "hybrid")
HYBRIDS = {"hybrid_static": ("bm25", "static"), "hybrid": ("bm25", "transformer")}


def top_by_score(scores: np.ndarray, depth: int) -> list[int]:
    """Indices of the ``depth`` highest positive scores; ties keep index order."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    return [int(i) for i in order[:depth] if scores[i] > 0]


def rrf(rankings: Sequence[Sequence[int]], k: int) -> list[int]:
    """Reciprocal rank fusion: score = sum over lists of 1 / (k + rank), rank from 1."""
    score: dict[int, float] = {}
    first: dict[int, tuple[int, int]] = {}
    for li, ranking in enumerate(rankings):
        for r, idx in enumerate(ranking, start=1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + r)
            first.setdefault(idx, (r, li))
    return sorted(score, key=lambda i: (-score[i], first[i], i))


def to_statements(ranked_chunks: Sequence[int], chunk_list: Sequence[Chunk]) -> list[str]:
    seen: dict[str, None] = {}
    for i in ranked_chunks:
        seen.setdefault(chunk_list[i].doc_id, None)
    return list(seen)


@dataclass
class Searcher:
    statements: list[corpus.Statement] = field(default_factory=corpus.load)

    def __post_init__(self) -> None:
        self.chunks = chunks(self.statements)
        r = protocol()["retrievers"]
        self.depth = int(r["fusion"]["depth"])
        self.k = int(r["fusion"]["k"])
        self.key = hashlib.sha256("\n".join(c.text for c in self.chunks).encode()).hexdigest()[:16]

    # -- retrievers, built on first use ---------------------------------------------------
    @cached_property
    def bm25(self) -> BM25:
        p = protocol()["retrievers"]["bm25"]
        return BM25([tokenize(c.text) for c in self.chunks], k1=p["k1"], b=p["b"])

    @cached_property
    def static_model(self) -> StaticModel:
        return StaticModel.load(model_dir("static"))

    @cached_property
    def static_vectors(self) -> np.ndarray:
        return self._cached("static", lambda: self.static_model.encode(
            [c.text for c in self.chunks]))

    @cached_property
    def encoder(self) -> TransformerEncoder:
        return TransformerEncoder()

    @cached_property
    def store(self) -> QdrantStore:
        vectors = self._cached("transformer", lambda: self.encoder.encode_passages(
            [c.text for c in self.chunks]))
        store = QdrantStore(f"statements_{self.key}", path=INDEX_DIR / "qdrant")
        if not store.client.collection_exists(store.collection) or store.count() != len(vectors):
            store.build(vectors, [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "bank": c.bank,
                                   "date": c.date} for c in self.chunks])
        return store

    def _cached(self, name: str, make: object) -> np.ndarray:
        path = INDEX_DIR / f"{name}_{self.key}.npy"
        if path.exists():
            loaded: np.ndarray = np.load(path)
            return loaded
        assert callable(make)
        vectors: np.ndarray = make()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    # -- ranking --------------------------------------------------------------------------
    def rank_chunks(self, kind: str, query: str) -> list[int]:
        if kind == "bm25":
            return top_by_score(self.bm25.scores(tokenize(query)), self.depth)
        if kind == "static":
            q = self.static_model.encode([query])[0]
            return top_by_score(self.static_vectors @ q, self.depth)
        if kind == "transformer":
            q = self.encoder.encode_queries([query])[0]
            return [i for i, _ in self.store.search(q, self.depth)]
        if kind in HYBRIDS:
            return rrf([self.rank_chunks(part, query) for part in HYBRIDS[kind]], self.k)
        raise ValueError(f"unknown retriever {kind!r}; one of {KINDS}")

    def rank(self, kind: str, query: str) -> list[str]:
        """Statement ids, best first."""
        return to_statements(self.rank_chunks(kind, query), self.chunks)

    def passages(self, kind: str, query: str, k: int = 5,
                 bank: str | None = None) -> list[Chunk]:
        """The best chunk of each of the top ``k`` statements, optionally of one bank only
        (a filter after ranking; the scored results use no filter)."""
        out: list[Chunk] = []
        seen: set[str] = set()
        for i in self.rank_chunks(kind, query):
            c = self.chunks[i]
            if bank is not None and c.bank != bank:
                continue
            if c.doc_id not in seen:
                seen.add(c.doc_id)
                out.append(c)
            if len(out) == k:
                break
        return out
