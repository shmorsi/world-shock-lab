"""Scoring the retrievers on the fixed questions (config/rag_protocol.yaml).

``dev`` can be scored any number of times. ``test`` is scored once: only on a commit that has
the ``rag-v1`` tag in its history, with the protocol, questions and corpus unchanged since
the tag, and never again unless a reason is logged in the result file.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wsl.paths import ROOT
from wsl.rag.corpus import STATEMENTS
from wsl.rag.retrieve import KINDS, Searcher
from wsl.rag.text import PROTOCOL, QUESTIONS, protocol

RESULTS = ROOT / "results" / "rag"
TAG = "rag-v1"
FROZEN = [PROTOCOL, QUESTIONS, STATEMENTS]


class GateError(RuntimeError):
    pass


def questions(split: str) -> list[dict[str, Any]]:
    qs: list[dict[str, Any]] = yaml.safe_load(QUESTIONS.read_text())["questions"]
    return [q for q in qs if q["split"] == split]


def hit_at(ranked: list[str], gold: list[str], k: int) -> float:
    return float(any(d in gold for d in ranked[:k]))


def rr_at(ranked: list[str], gold: list[str], k: int) -> float:
    for r, d in enumerate(ranked[:k], start=1):
        if d in gold:
            return 1.0 / r
    return 0.0


def per_question(ranked: list[str], gold: list[str]) -> dict[str, float]:
    return {"hit@1": hit_at(ranked, gold, 1), "hit@5": hit_at(ranked, gold, 5),
            "hit@10": hit_at(ranked, gold, 10), "mrr@10": rr_at(ranked, gold, 10)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int, seed: int,
                     level: float = 0.90) -> dict[str, float]:
    """Mean of a - b with a percentile interval from resampling questions."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"difference": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def verdict(ci: dict[str, float]) -> str:
    if ci["lo"] > 0:
        return "helps"
    if ci["hi"] < 0:
        return "hurts"
    return "no detectable difference"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def check_gate() -> None:
    if _git("rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}").returncode != 0:
        raise GateError(f"tag {TAG} not found; fetch tags (git fetch --tags) first")
    if _git("merge-base", "--is-ancestor", TAG, "HEAD").returncode != 0:
        raise GateError(f"{TAG} is not in this commit's history; the test split stays closed")
    rel = [str(p.relative_to(ROOT)) for p in FROZEN]
    if _git("diff", "--quiet", TAG, "--", *rel).returncode != 0:
        raise GateError(f"protocol, questions or corpus differ from {TAG}; the test split "
                        "stays closed (a change of method needs its own tag)")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict[str, Any]:
    r = protocol()["retrievers"]
    pkgs = ["sentence-transformers", "transformers", "torch", "qdrant-client", "numpy"]
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git("rev-parse", "HEAD").stdout.strip(),
        "protocol_tag": TAG,
        "protocol_sha256": _sha(PROTOCOL),
        "questions_sha256": _sha(QUESTIONS),
        "corpus_sha256": _sha(STATEMENTS),
        "models": {k: {"model": r[k]["model"], "revision": r[k]["revision"]}
                   for k in ("static", "transformer")},
        "packages": {p: md.version(p) for p in pkgs},
    }


def score(split: str, searcher: Searcher | None = None) -> dict[str, Any]:
    s = searcher or Searcher()
    qs = questions(split)
    rows: dict[str, list[dict[str, Any]]] = {k: [] for k in KINDS}
    for q in qs:
        for kind in KINDS:
            ranked = s.rank(kind, q["question"])
            rows[kind].append({"id": q["id"], "top5": ranked[:5],
                               **per_question(ranked, q["gold"])})
    summary = {k: {m: float(np.mean([r[m] for r in rows[k]]))
                   for m in ("hit@1", "hit@5", "hit@10", "mrr@10")} for k in KINDS}
    pc = protocol()["primary_comparison"]
    boot = pc["interval"]
    n, seed = 10000, 20261122
    assert str(n) in boot and str(seed) in boot  # the protocol's numbers, not new ones

    def compare(a: str, b: str) -> dict[str, Any]:
        ci = paired_bootstrap(np.array([r["hit@5"] for r in rows[a]]),
                              np.array([r["hit@5"] for r in rows[b]]), n, seed)
        return {"a": a, "b": b, "metric": "hit@5", **ci, "verdict": verdict(ci)}

    return {
        "split": split,
        "n_questions": len(qs),
        "summary": summary,
        "primary": compare(pc["a"], pc["b"]),
        "site_vs_bm25": compare("hybrid_static", "bm25"),
        "per_question": rows,
    }


def run(split: str, reason: str | None = None) -> Path:
    path = RESULTS / f"{split}.json"
    log: list[dict[str, str]] = []
    if split == "test":
        check_gate()
        if path.exists():
            if not reason:
                raise GateError(f"{path.name} is already scored; a second scoring needs --reason")
            log = json.loads(path.read_text()).get("rescoring_log", [])
            log.append({"at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "reason": reason})
    out = {"name": f"rag/{split}", "provenance": provenance(), **score(split)}
    if log:
        out["rescoring_log"] = log
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    return path
