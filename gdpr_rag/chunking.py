"""Turn structured articles into retrievable chunks.

Two strategies are offered so the evaluation can compare them (task section 4.4):

* ``article``   - one chunk per article. Fewer, broader chunks.
* ``paragraph`` - one chunk per numbered paragraph, keeping its lettered
  sub-points together. This keeps references precise (e.g. ``Article 6(1)``)
  while still giving the embedding enough surrounding context to match
  conversational questions.

Every chunk carries a short context header (chapter, article number and title)
so a lone paragraph is still self-describing once retrieved.
"""

from __future__ import annotations

from .models import Article, Chunk


def _context_header(article: Article) -> str:
    return (
        f"GDPR Chapter {article.chapter_number} ({article.chapter_title}) "
        f"- Article {article.number}: {article.title}"
    )


def chunk_by_article(articles: list[Article]) -> list[Chunk]:
    """One chunk per article, referenced as ``Article N``."""
    chunks = []
    for article in articles:
        text = f"{_context_header(article)}\n{article.full_text}"
        chunks.append(
            Chunk(
                chunk_id=f"art{article.number}",
                reference=f"Article {article.number}",
                article_number=article.number,
                article_title=article.title,
                chapter_number=article.chapter_number,
                chapter_title=article.chapter_title,
                paragraph_number=None,
                text=text,
            )
        )
    return chunks


def chunk_by_paragraph(articles: list[Article]) -> list[Chunk]:
    """One chunk per numbered paragraph, referenced as ``Article N(M)``.

    Articles without numbered paragraphs (e.g. Article 4) yield a single chunk
    referenced simply as ``Article N``.
    """
    chunks = []
    for article in articles:
        for paragraph in article.paragraphs:
            if paragraph.number is None:
                reference = f"Article {article.number}"
                chunk_id = f"art{article.number}"
            else:
                reference = f"Article {article.number}({paragraph.number})"
                chunk_id = f"art{article.number}_p{paragraph.number}"
            text = f"{_context_header(article)}\n{paragraph.text}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    reference=reference,
                    article_number=article.number,
                    article_title=article.title,
                    chapter_number=article.chapter_number,
                    chapter_title=article.chapter_title,
                    paragraph_number=paragraph.number,
                    text=text,
                )
            )
    return chunks


_CHUNKERS = {
    "article": chunk_by_article,
    "paragraph": chunk_by_paragraph,
}


def build_chunks(articles: list[Article], strategy: str) -> list[Chunk]:
    """Dispatch to the requested chunking strategy."""
    if strategy not in _CHUNKERS:
        raise ValueError(
            f"Unknown chunk strategy {strategy!r}; expected one of {sorted(_CHUNKERS)}"
        )
    return _CHUNKERS[strategy](articles)
