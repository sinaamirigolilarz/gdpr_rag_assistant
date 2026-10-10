"""Retrieval over the vector store: dense, or hybrid (dense + BM25).

Dense retrieval uses cosine similarity from the embedding model. Hybrid
retrieval additionally scores every chunk with BM25 (lexical) and fuses the two
rankings with Reciprocal Rank Fusion (RRF), which needs no score normalisation
and is robust when the two scorers live on different scales.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import EmbeddingModel
from .models import Chunk
from .vector_store import VectorStore

# Standard RRF dampening constant; larger values flatten the rank contribution.
_RRF_K = 60

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved chunk with its score and 1-based rank."""

    chunk: Chunk
    score: float
    rank: int


class Retriever:
    """Retrieves the top-k chunks for a query in ``dense`` or ``hybrid`` mode."""

    def __init__(self, store: VectorStore, embedder: EmbeddingModel, mode: str = "dense"):
        self.store = store
        self.embedder = embedder
        self.mode = mode
        self._bm25: BM25Okapi | None = None
        if mode == "hybrid":
            corpus = [_tokenize(chunk.text) for chunk in store.chunks]
            self._bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        if self.mode == "hybrid":
            return self._retrieve_hybrid(query, k)
        return self._retrieve_dense(query, k)

    def _retrieve_dense(self, query: str, k: int) -> list[RetrievalResult]:
        query_vector = self.embedder.encode_queries([query])[0]
        hits = self.store.search(query_vector, k)
        return [
            RetrievalResult(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(hits, start=1)
        ]

    def _retrieve_hybrid(self, query: str, k: int) -> list[RetrievalResult]:
        dense_rank = self._dense_ranks(query)
        bm25_rank = self._bm25_ranks(query)

        fused_scores = {
            chunk.chunk_id: _rrf_score(dense_rank.get(chunk.chunk_id))
            + _rrf_score(bm25_rank.get(chunk.chunk_id))
            for chunk in self.store.chunks
        }
        ordered = sorted(
            self.store.chunks,
            key=lambda chunk: fused_scores[chunk.chunk_id],
            reverse=True,
        )
        return [
            RetrievalResult(chunk=chunk, score=fused_scores[chunk.chunk_id], rank=rank)
            for rank, chunk in enumerate(ordered[:k], start=1)
        ]

    def _dense_ranks(self, query: str) -> dict[str, int]:
        query_vector = self.embedder.encode_queries([query])[0]
        hits = self.store.search(query_vector, len(self.store.chunks))
        return {chunk.chunk_id: rank for rank, (chunk, _) in enumerate(hits)}

    def _bm25_ranks(self, query: str) -> dict[str, int]:
        assert self._bm25 is not None  # guaranteed in hybrid mode
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1]
        return {self.store.chunks[index].chunk_id: rank for rank, index in enumerate(order)}


def _rrf_score(rank: int | None) -> float:
    """Reciprocal rank fusion contribution for a 0-based rank (0.0 if absent)."""
    if rank is None:
        return 0.0
    return 1.0 / (_RRF_K + rank)
