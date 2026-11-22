"""The files the web app's Ask view searches, in the browser, with no server:

- ``web/public/data/ask.json``: statements (with official URLs and the lab's market moves),
  the protocol's BM25 and fusion parameters, and the scored results (small: the view draws
  from it at once).
- ``web/public/data/ask_text.json``: the chunk texts, as [statement index, text].
- ``web/public/data/ask_vocab.json``: the static model's word pieces (index = id).
- ``web/public/data/ask_static.bin``: its vectors as int8 with one float32 scale per row.
- ``web/public/data/ask_chunks.bin``: the chunks' static vectors, quantized the same way.

Binary layout: uint32 rows, uint32 dims, float32 scales[rows], int8 values[rows * dims].
Also writes ``tests/fixtures/rag_site_rankings.json``, Python's float32 rankings that the
browser's int8 search is checked against (web/src/ask/ask.test.ts).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from wsl.paths import ROOT
from wsl.rag.answer import market_moves
from wsl.rag.evaluate import RESULTS, questions
from wsl.rag.retrieve import Searcher
from wsl.rag.text import header, protocol

OUT = ROOT / "web" / "public" / "data"
RANKINGS = ROOT / "tests" / "fixtures" / "rag_site_rankings.json"


def quantize(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 per row: value ~= scale * q."""
    scale = np.abs(m).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(m / scale[:, None]), -127, 127).astype(np.int8)
    return scale.astype(np.float32), q


def write_bin(path: Path, m: np.ndarray) -> None:
    scale, q = quantize(m.astype(np.float32))
    with path.open("wb") as f:
        f.write(struct.pack("<II", *q.shape))
        f.write(scale.astype("<f4").tobytes())
        f.write(q.tobytes())


def _scores() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("dev", "test"):
        d = json.loads((RESULTS / f"{split}.json").read_text())
        out[split] = {"n": d["n_questions"], "summary": d["summary"], "primary": d["primary"],
                      "site_vs_bm25": d["site_vs_bm25"],
                      "git_sha": d["provenance"]["git_sha"]}
    return out


def export(searcher: Searcher | None = None) -> list[Path]:
    s = searcher or Searcher()
    p = protocol()
    index = {st.doc_id: i for i, st in enumerate(s.statements)}
    statements = [{"id": st.doc_id, "bank": st.bank, "date": st.date, "title": st.title,
                   "url": st.url, "header": header(st), "moves": market_moves(st.doc_id)}
                  for st in s.statements]
    doc = {
        "protocol_tag": "rag-v1",
        "params": {"k1": p["retrievers"]["bm25"]["k1"], "b": p["retrievers"]["bm25"]["b"],
                   "stopwords": p["retrievers"]["bm25"]["stopwords"],
                   "rrf_k": p["retrievers"]["fusion"]["k"],
                   "depth": p["retrievers"]["fusion"]["depth"],
                   "static_model": p["retrievers"]["static"]["model"],
                   "static_revision": p["retrievers"]["static"]["revision"]},
        "scores": _scores(),
        "statements": statements,
    }
    text = [[index[c.doc_id], c.body] for c in s.chunks]
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [OUT / "ask.json", OUT / "ask_vocab.json", OUT / "ask_static.bin",
             OUT / "ask_chunks.bin", OUT / "ask_text.json"]
    paths[0].write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n")
    paths[4].write_text(json.dumps(text, ensure_ascii=False, separators=(",", ":")) + "\n")
    model = s.static_model
    vocab = [""] * len(model.vocab)
    for piece, i in model.vocab.items():
        vocab[i] = piece
    paths[1].write_text(json.dumps(vocab, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_bin(paths[2], model.vectors)
    write_bin(paths[3], s.static_vectors)
    rankings = {q["id"]: {"question": q["question"], "gold": q["gold"],
                          "hybrid_static": s.rank("hybrid_static", q["question"])[:10],
                          "bm25": s.rank("bm25", q["question"])[:10]}
                for split in ("dev", "test") for q in questions(split)}
    RANKINGS.write_text(json.dumps(rankings, indent=1) + "\n")
    return [*paths, RANKINGS]
