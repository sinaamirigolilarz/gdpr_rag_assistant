"""FAISS-backed vector store with its chunk metadata and build info.

A flat inner-product index is used deliberately: the corpus is tiny (a few
hundred chunks), so exact search is instant and avoids the tuning and recall
loss of approximate indexes. Because embeddings are L2-normalised, inner
product equals cosine similarity.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .models import Chunk

_INDEX_FILE = "index.faiss"
_CHUNKS_FILE = "chunks.json"
_META_FILE = "meta.json"


class VectorStore:
    """An in-memory FAISS index plus the chunks it was built from."""

    def __init__(self, index: faiss.Index, chunks: list[Chunk], meta: dict | None = None):
        self.index = index
        self.chunks = chunks
        self.meta = meta or {}

    @classmethod
    def build(cls, chunks: list[Chunk], embeddings: np.ndarray, meta: dict) -> "VectorStore":
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(index, list(chunks), meta)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Return up to ``k`` (chunk, similarity) pairs, best first."""
        k = min(k, len(self.chunks))
        scores, indices = self.index.search(query_vector.reshape(1, -1), k)
        return [
            (self.chunks[index], float(score))
            for score, index in zip(scores[0], indices[0])
            if index != -1
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / _INDEX_FILE))
        chunks_payload = [asdict(chunk) for chunk in self.chunks]
        (directory / _CHUNKS_FILE).write_text(
            json.dumps(chunks_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (directory / _META_FILE).write_text(
            json.dumps(self.meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        index = faiss.read_index(str(directory / _INDEX_FILE))
        chunks_data = json.loads((directory / _CHUNKS_FILE).read_text(encoding="utf-8"))
        chunks = [Chunk(**data) for data in chunks_data]
        meta = json.loads((directory / _META_FILE).read_text(encoding="utf-8"))
        return cls(index, chunks, meta)
