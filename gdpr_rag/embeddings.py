"""Local sentence-embedding model wrapper.

Default model: ``BAAI/bge-small-en-v1.5`` - a 33M-parameter English model that
runs comfortably on CPU, yet ranks strongly on the MTEB retrieval benchmark.
Its small size keeps indexing and querying fast on a 16 GB CPU-only machine,
which is why it is preferred here over larger bge/e5 variants.

Both BGE and E5 models expect asymmetric query/passage prefixes; applying the
right one materially improves retrieval, so the prefixes are selected from the
model name automatically.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# (query_prefix, passage_prefix) per model family.
_BGE_PREFIXES = ("Represent this sentence for searching relevant passages: ", "")
_E5_PREFIXES = ("query: ", "passage: ")
_NO_PREFIXES = ("", "")


def _prefixes_for(model_name: str) -> tuple[str, str]:
    name = model_name.lower()
    if "bge" in name:
        return _BGE_PREFIXES
    if "e5" in name:
        return _E5_PREFIXES
    return _NO_PREFIXES


class EmbeddingModel:
    """Encodes passages and queries into L2-normalised vectors (cosine-ready)."""

    def __init__(self, model_name: str, local_files_only: bool = False) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, local_files_only=local_files_only)
        self._query_prefix, self._passage_prefix = _prefixes_for(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self._passage_prefix + text for text in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self._query_prefix + text for text in texts])

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")
