"""Grounded answer generation with article citations.

The LLM is instructed to use only the retrieved context and, when the answer is
not supported there, to reply with the exact sentinel ``Not found in the GDPR``.
This keeps the system honest about its single data source (Articles 1-23) rather
than answering from the model's own prior knowledge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LLMClient
from .retriever import RetrievalResult, Retriever

NOT_FOUND_MESSAGE = "Not found in the GDPR"

_SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about the EU GDPR using "
    "only the provided context excerpts from GDPR Articles 1-23.\n"
    "Rules:\n"
    "1. Use ONLY the context. Do not rely on outside knowledge.\n"
    f"2. If the context does not contain the answer, reply with exactly "
    f"'{NOT_FOUND_MESSAGE}' and nothing else.\n"
    "3. Otherwise answer in 1-3 sentences and cite the relevant article(s) "
    "inline in the form 'Article N'."
)

_ARTICLE_PATTERN = re.compile(r"Article\s+(\d+)")


def _format_context(results: list[RetrievalResult]) -> str:
    return "\n\n".join(
        f"[{result.chunk.reference}]\n{result.chunk.text}" for result in results
    )


def build_user_prompt(question: str, results: list[RetrievalResult]) -> str:
    return (
        f"Context:\n{_format_context(results)}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def extract_cited_articles(answer_text: str) -> list[int]:
    """Article numbers cited in the answer, de-duplicated and sorted."""
    return sorted({int(match) for match in _ARTICLE_PATTERN.findall(answer_text)})


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    is_not_found: bool
    cited_articles: list[int]
    retrieved: list[RetrievalResult]


def answer_question(
    question: str, retriever: Retriever, llm: LLMClient, k: int
) -> AnswerResult:
    """Retrieve context, ask the LLM, and parse the grounded answer."""
    results = retriever.retrieve(question, k)
    raw_answer = llm.complete(_SYSTEM_PROMPT, build_user_prompt(question, results))

    is_not_found = NOT_FOUND_MESSAGE.lower() in raw_answer.lower()
    cited_articles = [] if is_not_found else extract_cited_articles(raw_answer)

    return AnswerResult(
        question=question,
        answer=raw_answer,
        is_not_found=is_not_found,
        cited_articles=cited_articles,
        retrieved=results,
    )
