"""Domain models for GDPR articles and retrievable chunks.

These are plain, immutable data holders. Keeping them free of parsing or
embedding logic lets every other module depend on a small, stable contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Paragraph:
    """A single numbered paragraph inside an article.

    ``number`` is ``None`` for articles that are not split into numbered
    paragraphs in the official text (e.g. Article 4 "Definitions", Article 10).
    """

    number: int | None
    text: str


@dataclass(frozen=True)
class Article:
    """One GDPR article together with the chapter it belongs to."""

    number: int
    title: str
    chapter_number: str
    chapter_title: str
    paragraphs: tuple[Paragraph, ...]

    @property
    def full_text(self) -> str:
        """The article body as a single string (all paragraphs joined)."""
        return "\n".join(paragraph.text for paragraph in self.paragraphs)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with the metadata needed for citation.

    ``text`` is what gets embedded and shown to the LLM; it includes a short
    context header (chapter + article + title) so a paragraph stays meaningful
    on its own. ``reference`` is the human-facing citation, e.g. ``Article 6(1)``.
    """

    chunk_id: str
    reference: str
    article_number: int
    article_title: str
    chapter_number: str
    chapter_title: str
    paragraph_number: int | None
    text: str
