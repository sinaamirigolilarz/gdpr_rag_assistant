"""Central configuration, loaded from environment variables with safe defaults.

A single ``Settings`` object is passed around instead of reading ``os.environ``
in many places, which keeps the rest of the code testable and side-effect free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# GDPR Chapters 1-3 cover Articles 1-23 (data-processing principles and the
# rights of the data subject). Later chapters are out of scope for this task.
MAX_ARTICLE_NUMBER = 23

# Official English consolidated text of the GDPR (CELEX 32016R0679).
GDPR_HTML_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNK_STRATEGIES = ("paragraph", "article")
RETRIEVAL_MODES = ("dense", "hybrid")
LLM_BACKENDS = ("openai", "ollama")


@dataclass(frozen=True)
class Settings:
    """All tunable knobs for ingestion, retrieval and answering."""

    # Data and index locations.
    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = PROJECT_ROOT / "index"

    # Retrieval.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_strategy: str = "paragraph"
    retrieval_mode: str = "dense"
    top_k: int = 5

    # Answer generation.
    llm_backend: str = "openai"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.2:3b"
    ollama_host: str | None = None

    @property
    def raw_dir(self) -> Path:
        """Directory holding the downloaded source HTML (not committed)."""
        return self.data_dir / "raw"

    @property
    def articles_path(self) -> Path:
        """Parsed, structured articles cached as JSON (committed, reproducible)."""
        return self.data_dir / "articles.json"


def load_settings() -> Settings:
    """Build :class:`Settings` from a local ``.env`` file and the environment."""
    load_dotenv(PROJECT_ROOT / ".env")

    return Settings(
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),
        chunk_strategy=os.getenv("CHUNK_STRATEGY", Settings.chunk_strategy),
        retrieval_mode=os.getenv("RETRIEVAL_MODE", Settings.retrieval_mode),
        top_k=int(os.getenv("TOP_K", Settings.top_k)),
        llm_backend=os.getenv("LLM_BACKEND", Settings.llm_backend),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        openai_model=os.getenv("OPENAI_MODEL", Settings.openai_model),
        ollama_model=os.getenv("OLLAMA_MODEL", Settings.ollama_model),
        ollama_host=os.getenv("OLLAMA_HOST"),
    )
