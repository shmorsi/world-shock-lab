"""Embedding retrieval: model downloads at pinned revisions, the transformer encoder, and the
Qdrant vector store the chunk vectors live in."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from wsl.paths import DATA_DIR
from wsl.rag.text import protocol

INDEX_DIR = DATA_DIR / "rag" / "index"  # git-ignored: vectors and the Qdrant store


def model_dir(kind: str) -> Path:
    """Local snapshot of the protocol's model at its pinned revision (downloaded once)."""
    # Plain HTTPS downloads: the Xet transfer client needs extra hosts and adds nothing here.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    spec = protocol()["retrievers"][kind]
    patterns = (["tokenizer.json", "model.safetensors", "config.json"] if kind == "static"
                else None)
    return Path(snapshot_download(spec["model"], revision=spec["revision"],
                                  allow_patterns=patterns,
                                  ignore_patterns=["onnx/*", "pytorch_model.bin"]))


class TransformerEncoder:
    """bge-small-en-v1.5 through sentence-transformers (CLS pooling, normalised)."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(model_dir("transformer")), device="cpu")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        out: np.ndarray = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                            convert_to_numpy=True, show_progress_bar=False)
        return out.astype(np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_passages([self.QUERY_PREFIX + t for t in texts])


class QdrantStore:
    """Chunk vectors in Qdrant. Local mode by default (a folder, or memory); pass ``url`` to
    use a Qdrant server instead, with the same calls."""

    def __init__(self, collection: str, path: Path | None = None, url: str | None = None) -> None:
        from qdrant_client import QdrantClient

        if url is not None:
            self.client = QdrantClient(url=url)
        elif path is not None:
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(location=":memory:")
        self.collection = collection

    def build(self, vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=int(vectors.shape[1]),
                                               distance=models.Distance.COSINE))
        self.client.upload_points(self.collection, [
            models.PointStruct(id=i, vector=v.tolist(), payload=p)
            for i, (v, p) in enumerate(zip(vectors, payloads, strict=True))])

    def search(self, query: np.ndarray, limit: int) -> list[tuple[int, float]]:
        from qdrant_client import models

        # Local mode is always exact; ask a server for exact search too.
        params = None if self.client.init_options.get("url") is None else models.SearchParams(
            exact=True)
        hits = self.client.query_points(
            self.collection, query=query.tolist(), limit=limit, with_payload=False,
            search_params=params).points
        return [(int(h.id), float(h.score)) for h in hits]

    def count(self) -> int:
        return int(self.client.count(self.collection).count)
