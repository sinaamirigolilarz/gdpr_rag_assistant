"""Parser tests on a small HTML fixture mirroring the eur-lex structure."""

from __future__ import annotations

from gdpr_rag.ingest import parse_articles

# Mirrors the real eur-lex layout: a chapter containing one article with
# numbered paragraphs and lettered sub-points, and one article without
# numbered paragraphs (like Article 10).
_FIXTURE_HTML = """
<div id="cpt_II">
  <p class="oj-ti-section-1"><span class="oj-italic">CHAPTER II</span></p>
  <div class="eli-title" id="cpt_II.tit_1">
    <p class="oj-ti-section-2"><span class="oj-bold"><span class="oj-italic">Principles</span></span></p>
  </div>
  <div class="eli-subdivision" id="art_6">
    <p class="oj-ti-art">Article 6</p>
    <div class="eli-title" id="art_6.tit_1"><p class="oj-sti-art">Lawfulness of processing</p></div>
    <div id="006.001">
      <p class="oj-normal">1.   Processing shall be lawful only if:</p>
      <table><tbody><tr>
        <td><p class="oj-normal">(a)</p></td>
        <td><p class="oj-normal">the data subject has given consent;</p></td>
      </tr></tbody></table>
      <table><tbody><tr>
        <td><p class="oj-normal">(f)</p></td>
        <td><p class="oj-normal">processing is necessary for legitimate interests.</p></td>
      </tr></tbody></table>
    </div>
    <div id="006.002"><p class="oj-normal">2.   Member States may maintain provisions.</p></div>
  </div>
  <div class="eli-subdivision" id="art_10">
    <p class="oj-ti-art">Article 10</p>
    <div class="eli-title" id="art_10.tit_1"><p class="oj-sti-art">Criminal convictions</p></div>
    <p class="oj-normal">Processing relating to criminal convictions shall be carried out only under control.</p>
  </div>
</div>
"""


def test_parses_all_articles_in_order():
    articles = parse_articles(_FIXTURE_HTML)
    assert [article.number for article in articles] == [6, 10]


def test_keeps_chapter_and_article_metadata():
    article = parse_articles(_FIXTURE_HTML)[0]
    assert article.chapter_number == "II"
    assert article.chapter_title == "Principles"
    assert article.title == "Lawfulness of processing"


def test_numbered_paragraphs_are_extracted_with_subpoints():
    article = parse_articles(_FIXTURE_HTML)[0]
    assert [p.number for p in article.paragraphs] == [1, 2]
    first_paragraph = article.paragraphs[0].text
    assert first_paragraph.startswith("1. Processing shall be lawful")
    # Sub-points stay attached so references like Article 6(1)(f) remain traceable.
    assert "(a)" in first_paragraph
    assert "(f)" in first_paragraph


def test_article_without_numbered_paragraphs_becomes_single_paragraph():
    article = parse_articles(_FIXTURE_HTML)[1]
    assert len(article.paragraphs) == 1
    assert article.paragraphs[0].number is None
    assert "criminal convictions" in article.paragraphs[0].text


def test_whitespace_is_normalised():
    first_paragraph = parse_articles(_FIXTURE_HTML)[0].paragraphs[0].text
    assert " " not in first_paragraph
    assert "   " not in first_paragraph
