"""Evaluation suite: retrieval recall, citation accuracy and comparisons.

Running ``python -m gdpr_rag.cli evaluate`` (or this module) reports, over the
15-question gold set:

* retrieval recall@k, comparing paragraph-level vs article-level chunking;
* retrieval recall@k, comparing dense vs hybrid (BM25 + dense) retrieval (bonus);
* answer citation accuracy on in-scope questions and "Not found" accuracy on
  out-of-scope questions;
* a short error analysis of the questions that failed.

The retrieval tables are built from in-memory indexes so the suite is
self-contained and deterministic; only the answer tables require a configured
LLM backend, and they are skipped with a clear message if none is available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .answer import answer_question
from .chunking import build_chunks
from .config import PROJECT_ROOT, Settings
from .embeddings import EmbeddingModel
from .ingest import load_or_ingest
from .llm import build_llm_client
from .models import Article
from .retriever import Retriever
from .vector_store import VectorStore

_DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
_RESULTS_PATH = PROJECT_ROOT / "eval" / "results.md"


@dataclass(frozen=True)
class Question:
    id: int
    question: str
    gold_articles: list[int]
    in_scope: bool
    style: str


def load_questions(path: Path | None = None) -> list[Question]:
    path = path or _DEFAULT_QUESTIONS_PATH
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Question(**item) for item in payload["questions"]]


def _in_memory_retriever(
    articles: list[Article], strategy: str, embedder: EmbeddingModel, mode: str
) -> Retriever:
    chunks = build_chunks(articles, strategy)
    embeddings = embedder.encode_passages([chunk.text for chunk in chunks])
    meta = {"chunk_strategy": strategy, "num_chunks": len(chunks)}
    store = VectorStore.build(chunks, embeddings, meta)
    return Retriever(store, embedder, mode=mode)


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RecallResult:
    """Per-question ranking outcomes for the in-scope questions.

    ``best_ranks`` holds, for each in-scope question, the 1-based rank of the
    first retrieved chunk whose article matches a gold article, or ``None`` if
    no gold article appeared in the top-k. Recall@1/@k and MRR are derived from
    it so the chunking comparison stays informative even when recall@k ties.
    """

    best_ranks: list[int | None]
    failures: list[tuple[Question, list[int]]]

    @property
    def total(self) -> int:
        return len(self.best_ranks)

    @property
    def hits(self) -> int:
        return sum(1 for rank in self.best_ranks if rank is not None)

    @property
    def recall(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def recall_at_1(self) -> float:
        if not self.total:
            return 0.0
        return sum(1 for rank in self.best_ranks if rank == 1) / self.total

    @property
    def mrr(self) -> float:
        if not self.total:
            return 0.0
        return sum(1.0 / rank for rank in self.best_ranks if rank) / self.total


def retrieval_recall(
    retriever: Retriever, questions: list[Question], k: int
) -> RecallResult:
    """Rank the gold article for each in-scope question within the top-k."""
    in_scope = [question for question in questions if question.in_scope]
    best_ranks: list[int | None] = []
    failures: list[tuple[Question, list[int]]] = []
    for question in in_scope:
        retrieved = [
            result.chunk.article_number
            for result in retriever.retrieve(question.question, k)
        ]
        best_rank = next(
            (
                rank
                for rank, article in enumerate(retrieved, start=1)
                if article in question.gold_articles
            ),
            None,
        )
        best_ranks.append(best_rank)
        if best_rank is None:
            failures.append((question, retrieved))
    return RecallResult(best_ranks=best_ranks, failures=failures)


# --------------------------------------------------------------------------- #
# Answer metrics
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AnswerMetrics:
    citation_hits: int
    in_scope_total: int
    not_found_hits: int
    out_of_scope_total: int
    failures: list[str]

    @property
    def citation_accuracy(self) -> float:
        return self.citation_hits / self.in_scope_total if self.in_scope_total else 0.0

    @property
    def not_found_accuracy(self) -> float:
        return self.not_found_hits / self.out_of_scope_total if self.out_of_scope_total else 0.0


def evaluate_answers(
    retriever: Retriever, llm, questions: list[Question], k: int
) -> AnswerMetrics:
    citation_hits = in_scope_total = not_found_hits = out_of_scope_total = 0
    failures: list[str] = []

    for question in questions:
        result = answer_question(question.question, retriever, llm, k)
        if question.in_scope:
            in_scope_total += 1
            is_correct = not result.is_not_found and bool(
                set(result.cited_articles) & set(question.gold_articles)
            )
            if is_correct:
                citation_hits += 1
            else:
                failures.append(
                    f"Q{question.id} (gold Article {question.gold_articles}): "
                    f"cited {result.cited_articles or 'none'} -> {result.answer[:90]!r}"
                )
        else:
            out_of_scope_total += 1
            if result.is_not_found:
                not_found_hits += 1
            else:
                failures.append(
                    f"Q{question.id} (out of scope): expected 'Not found', "
                    f"got {result.answer[:90]!r}"
                )

    return AnswerMetrics(
        citation_hits=citation_hits,
        in_scope_total=in_scope_total,
        not_found_hits=not_found_hits,
        out_of_scope_total=out_of_scope_total,
        failures=failures,
    )


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #

def _format_report(
    settings: Settings,
    k: int,
    chunking_recall: dict[str, RecallResult],
    retrieval_mode_recall: dict[str, RecallResult],
    answer_metrics: AnswerMetrics | None,
) -> str:
    lines = ["# Evaluation results", ""]
    lines.append(f"- Embedding model: `{settings.embedding_model}`")
    lines.append(f"- Top-k: {k}")
    lines.append("")

    lines.append(f"## 1. Retrieval - chunking comparison (dense)")
    lines.append("")
    lines.append(f"| Chunking | Recall@{k} | Recall@1 | MRR | Hits / Total |")
    lines.append("| --- | --- | --- | --- | --- |")
    for strategy, result in chunking_recall.items():
        lines.append(
            f"| {strategy} | {result.recall:.2%} | {result.recall_at_1:.2%} | "
            f"{result.mrr:.3f} | {result.hits}/{result.total} |"
        )
    lines.append("")

    lines.append(f"## 2. Retrieval - dense vs hybrid (bonus, paragraph chunking)")
    lines.append("")
    lines.append(f"| Retrieval | Recall@{k} | Recall@1 | MRR | Hits / Total |")
    lines.append("| --- | --- | --- | --- | --- |")
    for mode, result in retrieval_mode_recall.items():
        lines.append(
            f"| {mode} | {result.recall:.2%} | {result.recall_at_1:.2%} | "
            f"{result.mrr:.3f} | {result.hits}/{result.total} |"
        )
    lines.append("")

    lines.append("## 3. Answer quality")
    lines.append("")
    if answer_metrics is None:
        lines.append("_Skipped: no LLM backend configured (see README)._")
    else:
        lines.append("| Metric | Score | Count |")
        lines.append("| --- | --- | --- |")
        lines.append(
            f"| Correct citation (in-scope) | {answer_metrics.citation_accuracy:.2%} | "
            f"{answer_metrics.citation_hits}/{answer_metrics.in_scope_total} |"
        )
        lines.append(
            f"| Correct 'Not found' (out-of-scope) | {answer_metrics.not_found_accuracy:.2%} | "
            f"{answer_metrics.not_found_hits}/{answer_metrics.out_of_scope_total} |"
        )
    lines.append("")

    lines.append("## 4. Error analysis")
    lines.append("")
    primary_failures = chunking_recall[settings.chunk_strategy].failures
    if primary_failures:
        lines.append(f"Retrieval misses ({settings.chunk_strategy} chunking):")
        for question, retrieved in primary_failures:
            lines.append(
                f"- Q{question.id} (gold {question.gold_articles}): "
                f"retrieved articles {retrieved}"
            )
    else:
        lines.append(f"Retrieval: no misses with {settings.chunk_strategy} chunking.")
    lines.append("")
    if answer_metrics is not None:
        if answer_metrics.failures:
            lines.append("Answer failures:")
            for failure in answer_metrics.failures:
                lines.append(f"- {failure}")
        else:
            lines.append("Answers: no failures.")
    lines.append("")
    return "\n".join(lines)


def run_evaluation(settings: Settings, questions_path: str | None = None) -> None:
    """Execute the full evaluation and print + persist a markdown report."""
    questions = load_questions(Path(questions_path) if questions_path else None)
    articles = load_or_ingest(settings)
    embedder = EmbeddingModel(
        settings.embedding_model, local_files_only=settings.hf_local_only
    )
    k = settings.top_k

    chunking_recall = {
        strategy: retrieval_recall(
            _in_memory_retriever(articles, strategy, embedder, mode="dense"),
            questions,
            k,
        )
        for strategy in ("paragraph", "article")
    }

    primary_strategy = settings.chunk_strategy
    retrieval_mode_recall = {
        mode: retrieval_recall(
            _in_memory_retriever(articles, primary_strategy, embedder, mode=mode),
            questions,
            k,
        )
        for mode in ("dense", "hybrid")
    }

    answer_metrics = _maybe_evaluate_answers(settings, articles, embedder, questions, k)

    report = _format_report(
        settings, k, chunking_recall, retrieval_mode_recall, answer_metrics
    )
    print(report)
    _RESULTS_PATH.write_text(report, encoding="utf-8")
    print(f"\nSaved report to {_RESULTS_PATH.relative_to(PROJECT_ROOT)}")


def _maybe_evaluate_answers(
    settings: Settings,
    articles: list[Article],
    embedder: EmbeddingModel,
    questions: list[Question],
    k: int,
) -> AnswerMetrics | None:
    """Run the answer evaluation, or return ``None`` if no LLM is available."""
    try:
        llm = build_llm_client(settings)
    except ValueError as error:
        print(f"[answer eval skipped] {error}")
        return None

    retriever = _in_memory_retriever(
        articles, settings.chunk_strategy, embedder, mode=settings.retrieval_mode
    )
    try:
        return evaluate_answers(retriever, llm, questions, k)
    except Exception as error:  # noqa: BLE001 - surface any backend failure clearly
        print(f"[answer eval skipped] LLM call failed: {error}")
        return None


if __name__ == "__main__":
    from .config import load_settings

    run_evaluation(load_settings())
