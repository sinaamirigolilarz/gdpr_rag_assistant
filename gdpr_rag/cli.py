"""Command-line entry point: ``build-index``, ``ask`` and ``evaluate``.

Examples
--------
    python -m gdpr_rag.cli build-index
    python -m gdpr_rag.cli ask "Can my old employer keep my CV forever?"
    python -m gdpr_rag.cli evaluate
"""

from __future__ import annotations

import argparse
import dataclasses

from .answer import answer_question
from .config import CHUNK_STRATEGIES, RETRIEVAL_MODES, Settings, load_settings
from .ingest import ingest
from .llm import build_llm_client
from .pipeline import build_index, load_retriever


def _cmd_build_index(args: argparse.Namespace, settings: Settings) -> None:
    if args.force_download:
        ingest(settings, force_download=True)
    store = build_index(
        settings, strategy=args.strategy, embedding_model=args.embedding_model
    )
    print(
        f"Indexed {store.meta['num_chunks']} chunks "
        f"(strategy={store.meta['chunk_strategy']}, "
        f"model={store.meta['embedding_model']})."
    )


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> None:
    settings = dataclasses.replace(settings, retrieval_mode=args.mode, top_k=args.k)
    retriever = load_retriever(settings)
    llm = build_llm_client(settings)
    result = answer_question(args.question, retriever, llm, settings.top_k)

    print(result.answer)
    print("\nRetrieved context:")
    for item in result.retrieved:
        print(f"  {item.rank}. {item.chunk.reference}  (score={item.score:.3f})")


def _cmd_evaluate(args: argparse.Namespace, settings: Settings) -> None:
    # Imported lazily so that `build-index` / `ask` do not require eval deps.
    from .evaluation import run_evaluation

    run_evaluation(settings, questions_path=args.questions)


def _build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gdpr_rag", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index", help="Build the vector index.")
    build.add_argument("--strategy", choices=CHUNK_STRATEGIES, default=settings.chunk_strategy)
    build.add_argument("--embedding-model", default=settings.embedding_model)
    build.add_argument("--force-download", action="store_true", help="Re-fetch source HTML.")
    build.set_defaults(func=_cmd_build_index)

    ask = subparsers.add_parser("ask", help="Answer a single question with citations.")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=settings.top_k)
    ask.add_argument("--mode", choices=RETRIEVAL_MODES, default=settings.retrieval_mode)
    ask.set_defaults(func=_cmd_ask)

    evaluate = subparsers.add_parser("evaluate", help="Run the evaluation suite.")
    evaluate.add_argument("--questions", default=None, help="Path to questions JSON.")
    evaluate.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> None:
    settings = load_settings()
    parser = _build_parser(settings)
    args = parser.parse_args(argv)
    args.func(args, settings)


if __name__ == "__main__":
    main()
