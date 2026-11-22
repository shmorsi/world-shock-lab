"""Writes tests/fixtures/rag_cases.json: cases the Python code and its TypeScript twin
(web/src/ask/) must both reproduce exactly. Run ``uv run wsl rag fixtures`` after changing
either implementation; ``tests/test_rag.py`` and ``web/src/ask/ask.test.ts`` read the file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wsl.paths import ROOT
from wsl.rag.bm25 import BM25
from wsl.rag.corpus import load
from wsl.rag.dense import model_dir
from wsl.rag.retrieve import rrf
from wsl.rag.static import StaticModel, normalize, pre_tokenize
from wsl.rag.text import chunks, tokenize

OUT = ROOT / "tests" / "fixtures" / "rag_cases.json"

TRICKY = [
    "The Bank's rates: 2¼% — Café policies!",
    "U.S.-Mexico-Canada Agreement (USMCA) and the Fed's 3‑1/2 to 3‑3/4 percent range",
    "Naïve résumé façade; ÉPICENTRE coöperation",
    "COVID-19 pandemic, Omicron variant, 東京 and Zürich",
    "When did the Bank of Canada raise its rate by a full percentage point at once?",
    "series cases goes buses passes policies aies",
    "tab\tnew\nline nbsp thin​zero-width",
    "",
]


def _sub_vocab(model: StaticModel, texts: list[str]) -> dict[str, int]:
    """Every vocabulary piece greedy matching could try on these texts, so a reduced vocabulary
    gives the same pieces as the full one."""
    keep: dict[str, int] = {}
    for t in texts:
        for w in pre_tokenize(normalize(t)):
            for i in range(len(w)):
                for j in range(i + 1, len(w) + 1):
                    piece = w[i:j] if i == 0 else "##" + w[i:j]
                    if piece in model.vocab:
                        keep[piece] = model.vocab[piece]
    return keep


def build() -> dict[str, object]:
    docs = load()
    sample = [c.text for c in chunks(docs)[::97]]
    texts = TRICKY + sample[:12]
    full = StaticModel.load(model_dir("static"))
    vocab = _sub_vocab(full, texts)
    ids = sorted(set(vocab.values()))
    remap = {old: new for new, old in enumerate(ids)}
    small = StaticModel(vocab={p: remap[i] for p, i in vocab.items()},
                        vectors=np.round(full.vectors[ids][:, :8], 4))  # 8 dims keep it small
    cases = [{"text": t, "ids": small.token_ids(t),
              "vector": [round(float(x), 6) for x in small.encode([t])[0]]} for t in texts]
    bm_docs = [tokenize(t) for t in sample[:6]]
    query = tokenize("oil prices and inflation in Canada")
    return {
        "tokenize": [{"text": t, "tokens": tokenize(t)} for t in texts],
        "bm25": {"docs": bm_docs, "query": query,
                 "scores": [round(float(x), 10) for x in BM25(bm_docs).scores(query)]},
        "rrf": [{"lists": lists, "fused": rrf(lists, 60)} for lists in
                ([[1, 2, 3], [3, 1]], [[5], [6]], [[4, 9, 2, 7], [2, 4, 8], [9]])],
        "static": {"vocab": small.vocab, "vectors": small.vectors.tolist(), "cases": cases},
    }


def write(path: Path = OUT) -> Path:
    path.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n")
    return path
