"""Download the official GDPR HTML and parse it into structured articles.

The eur-lex consolidated text uses a predictable layout:

    <div id="cpt_II">                         chapter
        <p class="oj-ti-section-1">CHAPTER II</p>
        <div class="eli-title"><p class="oj-ti-section-2">Principles</p></div>
        <div class="eli-subdivision" id="art_6">   article
            <p class="oj-ti-art">Article 6</p>
            <div class="eli-title"><p class="oj-sti-art">Lawfulness ...</p></div>
            <div id="006.001"> ... </div>          numbered paragraph
            <div id="006.002"> ... </div>
        </div>
    </div>

A few short articles (e.g. Article 4 "Definitions", Article 10) have no numbered
paragraph ``<div>``s; their body sits directly under the article node. Both cases
are handled below so every in-scope article is captured.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .config import GDPR_HTML_URL, MAX_ARTICLE_NUMBER, Settings
from .models import Article, Paragraph

_CHAPTER_ID = re.compile(r"^cpt_[IVXLCDM]+$")
_ARTICLE_ID = re.compile(r"^art_\d+$")
_PARAGRAPH_ID = re.compile(r"^\d{3}\.\d{3}$")

_REQUEST_TIMEOUT_SECONDS = 60


def _clean_text(raw: str) -> str:
    """Normalise whitespace and smart quotes coming from the source HTML."""
    text = raw.replace("\xa0", " ").replace("‘", "'").replace("’", "'")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _block_text(node: BeautifulSoup) -> str:
    """Text of a paragraph node: lead sentence plus each lettered sub-point.

    Direct ``<p>`` and ``<table>`` children are joined with newlines so that
    sub-points such as ``(a)``/``(f)`` stay on their own lines and remain
    traceable to references like ``Article 6(1)(f)``.
    """
    parts = []
    for child in node.find_all(["p", "table"], recursive=False):
        part = _clean_text(child.get_text(" ", strip=True))
        if part:
            parts.append(part)
    return "\n".join(parts)


def _loose_content_text(article_div: BeautifulSoup) -> str:
    """Body text for articles without numbered paragraph ``<div>``s.

    Skips the ``Article N`` heading and keeps the intro sentence and any
    point tables (e.g. the 26 definitions of Article 4).
    """
    parts = []
    for child in article_div.find_all(["p", "table"], recursive=False):
        if "oj-ti-art" in (child.get("class") or []):
            continue
        part = _clean_text(child.get_text(" ", strip=True))
        if part:
            parts.append(part)
    return "\n".join(parts)


def _extract_paragraphs(article_div: BeautifulSoup) -> tuple[Paragraph, ...]:
    """Return the article's paragraphs, numbered where the source provides them."""
    paragraph_divs = [
        div
        for div in article_div.find_all("div", id=_PARAGRAPH_ID)
    ]
    if paragraph_divs:
        return tuple(
            Paragraph(number=int(div["id"].split(".")[1]), text=_block_text(div))
            for div in paragraph_divs
        )
    return (Paragraph(number=None, text=_loose_content_text(article_div)),)


def _article_title(article_div: BeautifulSoup) -> str:
    title_node = article_div.find("p", class_="oj-sti-art")
    return _clean_text(title_node.get_text(" ")) if title_node else ""


def parse_articles(html: str) -> list[Article]:
    """Parse GDPR Articles 1-23 (Chapters I-III) from the consolidated HTML."""
    soup = BeautifulSoup(html, "html.parser")
    articles: list[Article] = []

    for chapter_div in soup.find_all("div", id=_CHAPTER_ID):
        chapter_number = chapter_div["id"].removeprefix("cpt_")
        title_node = chapter_div.find("p", class_="oj-ti-section-2")
        chapter_title = _clean_text(title_node.get_text(" ")) if title_node else ""

        for article_div in chapter_div.find_all(
            "div", class_="eli-subdivision", id=_ARTICLE_ID
        ):
            number = int(article_div["id"].removeprefix("art_"))
            if number > MAX_ARTICLE_NUMBER:
                continue
            articles.append(
                Article(
                    number=number,
                    title=_article_title(article_div),
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    paragraphs=_extract_paragraphs(article_div),
                )
            )

    articles.sort(key=lambda article: article.number)
    return articles


def fetch_html(url: str, cache_path: Path, *, force_download: bool = False) -> str:
    """Download the source HTML, caching it on disk for reproducible runs."""
    if cache_path.exists() and not force_download:
        return cache_path.read_text(encoding="utf-8")

    response = requests.get(
        url,
        timeout=_REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": "Mozilla/5.0 (GDPR-RAG-assistant)"},
    )
    response.raise_for_status()
    response.encoding = "utf-8"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text, encoding="utf-8")
    return response.text


def _articles_to_dicts(articles: list[Article]) -> list[dict]:
    return [
        {
            "number": article.number,
            "title": article.title,
            "chapter_number": article.chapter_number,
            "chapter_title": article.chapter_title,
            "paragraphs": [
                {"number": paragraph.number, "text": paragraph.text}
                for paragraph in article.paragraphs
            ],
        }
        for article in articles
    ]


def _dicts_to_articles(data: list[dict]) -> list[Article]:
    return [
        Article(
            number=item["number"],
            title=item["title"],
            chapter_number=item["chapter_number"],
            chapter_title=item["chapter_title"],
            paragraphs=tuple(
                Paragraph(number=p["number"], text=p["text"])
                for p in item["paragraphs"]
            ),
        )
        for item in data
    ]


def save_articles(articles: list[Article], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _articles_to_dicts(articles)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_articles(path: Path) -> list[Article]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _dicts_to_articles(data)


def ingest(settings: Settings, *, force_download: bool = False) -> list[Article]:
    """Fetch, parse and cache the structured articles. Returns them in order."""
    html = fetch_html(
        GDPR_HTML_URL,
        settings.raw_dir / "gdpr.html",
        force_download=force_download,
    )
    articles = parse_articles(html)
    save_articles(articles, settings.articles_path)
    return articles


def load_or_ingest(settings: Settings) -> list[Article]:
    """Load cached articles if present, otherwise download and parse them."""
    if settings.articles_path.exists():
        return load_articles(settings.articles_path)
    return ingest(settings)
