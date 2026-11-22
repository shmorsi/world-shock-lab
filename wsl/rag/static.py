"""Static embeddings: a vector per word piece, averaged. Small and fast enough to run in a
browser, which is why the website can search by meaning without a server.

The WordPiece tokenizer is written out here (and in web/src/ask/wordpiece.ts) rather than
taken from a library, so Python and the browser split text identically; a test checks it
against the Hugging Face tokenizer on the corpus.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UNK = "[UNK]"
MAX_WORD_CHARS = 100


def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


def normalize(text: str) -> str:
    """BertNormalizer(clean_text, handle_chinese_chars, lowercase, strip accents)."""
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp == 0 or cp == 0xFFFD or (unicodedata.category(ch).startswith("C")
                                        and ch not in "\t\n\r"):
            continue
        if ch in "\t\n\r" or unicodedata.category(ch) == "Zs":
            out.append(" ")
        elif _is_cjk(cp):
            out.append(f" {ch} ")
        else:
            out.append(ch)
    lowered = "".join(out).lower()
    return "".join(c for c in unicodedata.normalize("NFD", lowered)
                   if unicodedata.category(c) != "Mn")


def pre_tokenize(text: str) -> list[str]:
    """BertPreTokenizer: split on whitespace, and every punctuation mark is its own word."""
    words: list[str] = []
    for chunk in text.split():
        cur = ""
        for ch in chunk:
            if _is_punct(ch):
                if cur:
                    words.append(cur)
                    cur = ""
                words.append(ch)
            else:
                cur += ch
        if cur:
            words.append(cur)
    return words


def wordpiece(word: str, vocab: dict[str, int]) -> list[int] | None:
    """Greedy longest-match-first. None when the word cannot be covered (it maps to [UNK])."""
    if len(word) > MAX_WORD_CHARS:
        return None
    ids: list[int] = []
    start = 0
    while start < len(word):
        end = len(word)
        found = None
        while start < end:
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                found = vocab[piece]
                break
            end -= 1
        if found is None:
            return None
        ids.append(found)
        start = end
    return ids


@dataclass
class StaticModel:
    vocab: dict[str, int]
    vectors: np.ndarray  # (vocab, dim) float32

    @classmethod
    def load(cls, directory: Path) -> StaticModel:
        from safetensors.numpy import load_file

        tok = json.loads((directory / "tokenizer.json").read_text())
        vectors = load_file(str(directory / "model.safetensors"))["embeddings"]
        return cls(vocab=dict(tok["model"]["vocab"]), vectors=vectors.astype(np.float32))

    def token_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for w in pre_tokenize(normalize(text)):
            pieces = wordpiece(w, self.vocab)
            if pieces is not None:  # unknown words are dropped, as model2vec does
                ids.extend(pieces)
        return ids

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.vectors.shape[1]), dtype=np.float32)
        for i, t in enumerate(texts):
            ids = self.token_ids(t)
            if ids:
                v = self.vectors[ids].mean(axis=0)
                n = float(np.linalg.norm(v))
                out[i] = v / n if n > 0 else v
        return out
