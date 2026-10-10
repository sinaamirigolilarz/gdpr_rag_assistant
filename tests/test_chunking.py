"""Tests for the article-level and paragraph-level chunking strategies."""

from __future__ import annotations

import pytest

from gdpr_rag.chunking import build_chunks, chunk_by_article, chunk_by_paragraph
from gdpr_rag.models import Article, Paragraph

_ARTICLE_6 = Article(
    number=6,
    title="Lawfulness of processing",
    chapter_number="II",
    chapter_title="Principles",
    paragraphs=(
        Paragraph(number=1, text="1. Processing shall be lawful only if:\n(f) legitimate interests."),
        Paragraph(number=2, text="2. Member States may maintain provisions."),
    ),
)

_ARTICLE_10 = Article(
    number=10,
    title="Criminal convictions",
    chapter_number="II",
    chapter_title="Principles",
    paragraphs=(Paragraph(number=None, text="Processing relating to criminal convictions."),),
)

_ARTICLES = [_ARTICLE_6, _ARTICLE_10]


def test_article_chunking_yields_one_chunk_per_article():
    chunks = chunk_by_article(_ARTICLES)
    assert [chunk.reference for chunk in chunks] == ["Article 6", "Article 10"]


def test_article_chunk_carries_metadata_and_context_header():
    chunk = chunk_by_article(_ARTICLES)[0]
    assert chunk.article_number == 6
    assert chunk.article_title == "Lawfulness of processing"
    assert chunk.chapter_number == "II"
    assert chunk.paragraph_number is None
    # The embedded text starts with a self-describing header.
    assert chunk.text.startswith("GDPR Chapter II (Principles) - Article 6:")
    # An article chunk contains all of its paragraphs.
    assert "Member States" in chunk.text


def test_paragraph_chunking_references_each_numbered_paragraph():
    chunks = chunk_by_paragraph(_ARTICLES)
    references = [chunk.reference for chunk in chunks]
    # Article 6 has two numbered paragraphs; Article 10 has none.
    assert references == ["Article 6(1)", "Article 6(2)", "Article 10"]


def test_paragraph_chunk_keeps_subpoints_for_precise_reference():
    first_chunk = chunk_by_paragraph(_ARTICLES)[0]
    assert first_chunk.paragraph_number == 1
    assert "(f) legitimate interests." in first_chunk.text


def test_unnumbered_article_paragraph_has_article_level_reference():
    article_10_chunk = chunk_by_paragraph(_ARTICLES)[-1]
    assert article_10_chunk.reference == "Article 10"
    assert article_10_chunk.paragraph_number is None


def test_build_chunks_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        build_chunks(_ARTICLES, "sentence")
