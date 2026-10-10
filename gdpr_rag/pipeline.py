"""High-level wiring that ties ingestion, chunking, embedding and retrieval.

These helpers are the single place where the moving parts are assembled, so the
CLI and the evaluation share exactly the same construction logic.
"""

from __future__ import annotations

from .chunking import build_chunks
from .config import Settings
from .embeddings import EmbeddingModel
from .ingest import load_or_ingest
from .retriever import Retriever
from .vector_store import VectorStore


def build_index(
    settings: Settings,
    *,
    strategy: str | None = None,
    embedding_model: str | None = None,
) -> VectorStore:
    """Build and persist the vector index, returning the in-memory store."""
    strategy = strategy or settings.chunk_strategy
    embedding_model = embedding_model or settings.embedding_model

    articles = load_or_ingest(settings)
    chunks = build_chunks(articles, strategy)

    embedder = EmbeddingModel(embedding_model)
    embeddings = embedder.encode_passages([chunk.text for chunk in chunks])

    meta = {
        "embedding_model": embedding_model,
        "chunk_strategy": strategy,
        "dimension": embedder.dimension,
        "num_chunks": len(chunks),
    }
    store = VectorStore.build(chunks, embeddings, meta)
    store.save(settings.index_dir)
    return store


def load_retriever(settings: Settings) -> Retriever:
    """Load the persisted index and build a retriever matching its embed model."""
    store = VectorStore.load(settings.index_dir)
    embedding_model = store.meta.get("embedding_model", settings.embedding_model)
    embedder = EmbeddingModel(embedding_model)
    return Retriever(store, embedder, mode=settings.retrieval_mode)
