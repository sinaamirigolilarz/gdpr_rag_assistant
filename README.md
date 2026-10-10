# GDPR RAG Assistant

A small, CPU-friendly Retrieval-Augmented Generation (RAG) system that answers
employees' questions about their personal data using **GDPR Articles 1-23**
(Chapters 1-3: principles of processing and rights of the data subject) and
cites the reference **Article**.

Example:

> **Q:** How do I get a company to delete my data?
> **A:** Ask the company to erase your personal data. It must do so without undue
> delay if, for example, the data is no longer needed, you withdraw consent ...
> (**Article 17**). You can also ask which recipients received your data ...
> (**Article 19**).

If the answer is not contained in Articles 1-23, the system replies exactly
`Not found in the GDPR` instead of inventing an answer.

---

## Contents

- [Design decisions](#design-decisions)
- [Setup](#setup)
- [How to run](#how-to-run)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Assumptions & limitations](#assumptions--limitations)

---

## Design decisions

### 1. Source and chunking

The official consolidated HTML (CELEX `32016R0679`) is downloaded from eur-lex
and parsed into structured articles. The eur-lex markup is predictable
(`cpt_*` chapter divs, `art_*` article divs, `NNN.MMM` numbered-paragraph divs),
so parsing is done with BeautifulSoup and keeps, for every chunk:

- article number, article title, and chapter number + title (used for citation
  and as a context header on each chunk).

**Two chunking strategies** are provided so they can be compared:

| Strategy | Unit | Reference | Count |
| --- | --- | --- | --- |
| `paragraph` (default) | one numbered paragraph | `Article 6(1)` | 76 chunks |
| `article` | whole article | `Article 6` | 23 chunks |

**How are long, multi-paragraph articles chunked (e.g. Article 6(1)(f))?**
Each **numbered paragraph** becomes one chunk, and its lettered sub-points
`(a)`, `(b)`, ... `(f)` are kept **inside** that chunk rather than split further.
Reasoning:

- The citation granularity the task cares about is the **article/paragraph**
  (e.g. `Article 6(1)`), so a paragraph is the natural retrieval unit. The
  sub-point text stays available in the chunk, so `Article 6(1)(f)` remains
  fully traceable.
- Splitting every `(a)/(b)/(f)` into its own chunk would fragment the meaning
  (sub-points only make sense under their lead sentence "Processing shall be
  lawful only if ...") and hurt retrieval.
- A short **context header** (`GDPR Chapter II (Principles) - Article 6:
  Lawfulness of processing`) is prepended to each chunk so a lone paragraph is
  still self-describing once retrieved.

A few short articles have no numbered paragraphs in the official text
(Article 4 "Definitions", Article 10). These become a single chunk referenced as
`Article N`.

### 2. Embedding model

Default: **`BAAI/bge-small-en-v1.5`**. It offers strong MTEB retrieval quality at
only 33M parameters / 384 dimensions, so indexing and querying are fast on a
CPU-only 16 GB machine; BGE's asymmetric query prefix is applied automatically
to improve matching of conversational questions. `intfloat/e5-small-v2` is a
drop-in alternative (`EMBEDDING_MODEL=...`).

### 3. Vector store and retrieval

- **FAISS `IndexFlatIP`** (exact search). The corpus is tiny (76 chunks), so an
  exact index is instant and avoids the recall loss and tuning of approximate
  indexes. Embeddings are L2-normalised, so inner product = cosine similarity.
- `Retriever.retrieve(query, k)` returns the top-k chunks with their similarity
  score and article number, in two modes:
  - `dense` (default): embedding cosine similarity.
  - `hybrid` (bonus): BM25 (lexical) + dense, fused with Reciprocal Rank Fusion.

### 4. Answer generation

Retrieved chunks are passed to an LLM with a strict prompt: answer **only** from
the context, be concise, cite `Article N`, and otherwise reply exactly
`Not found in the GDPR`. The backend is pluggable (`LLM_BACKEND`):

- `openai` - any OpenAI-compatible Chat Completions API (OpenAI, Groq,
  OpenRouter, a local gateway, ...).
- `ollama` - a fully local model, no API key.

---

## Setup

Requires Python 3.10+ (developed on 3.12).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Create your `.env` from the template and fill in your LLM credentials:

```bash
cp .env.example .env
# then edit .env
```

No secrets are stored in the repository; only `.env.example` (placeholders) is
committed. The embedding model (~130 MB) is downloaded automatically from
Hugging Face the first time you build the index, and is **not** committed.

---

## How to run

All three commands are run from the repository root.

**1. Build the vector index** (downloads the embedding model on first run):

```bash
python -m gdpr_rag.cli build-index
```

**2. Ask a single question:**

```bash
python -m gdpr_rag.cli ask "Can my old employer keep my CV forever?"
```

Optional flags: `--k 5`, `--mode dense|hybrid`.

**3. Run the evaluation:**

```bash
python -m gdpr_rag.cli evaluate
```

The evaluation is self-contained (it builds its own in-memory indexes), so it
can be run without `build-index` as long as the source data is available.

---

## Evaluation

The gold set is [`eval/questions.json`](eval/questions.json): **15 questions**,
each with its correct article(s):

- **12 in-scope** questions answerable within Articles 1-23, of which **6 are
  conversational** and avoid the regulation's own vocabulary
  (e.g. "Can my old employer keep my CV forever?" instead of "storage
  limitation").
- **3 out-of-scope** questions whose answers are **not** in Articles 1-23
  (minimum wage; GDPR fines, which live in Article 83; the Data Protection
  Officer, Articles 37-39). The correct behaviour is `Not found in the GDPR`.

Metrics reported:

- **Retrieval** - Recall@5 (is a gold article among the top 5?), plus Recall@1
  and MRR so the chunking comparison stays informative when Recall@5 ties.
- **Answers** - correct-citation rate on in-scope questions, and
  correct-`Not found` rate on out-of-scope questions.
- **Comparison** - paragraph vs article chunking, and (bonus) dense vs hybrid.

---

## Results

Embedding model `BAAI/bge-small-en-v1.5`, k = 5, LLM `claude-opus-4-8` via an
OpenAI-compatible endpoint. Reproduce with `python -m gdpr_rag.cli evaluate`
(the report is also written to [`eval/results.md`](eval/results.md)).

### Retrieval - chunking comparison (dense)

| Chunking | Recall@5 | Recall@1 | MRR | Hits / Total |
| --- | --- | --- | --- | --- |
| paragraph | 91.67% | 83.33% | 0.854 | 11/12 |
| article | 91.67% | 83.33% | 0.875 | 11/12 |

Both reach the same Recall@5. Article-level chunking has a slightly higher MRR
(0.875 vs 0.854) because a whole-article chunk matches a query on more terms and
tends to rank first. Paragraph-level chunking gives up a little ranking quality
in exchange for **more precise citations** (`Article 6(1)` instead of
`Article 6`), which is the point of the task, so it is the default.

### Retrieval - dense vs hybrid (bonus)

| Retrieval | Recall@5 | Recall@1 | MRR | Hits / Total |
| --- | --- | --- | --- | --- |
| dense | 91.67% | 83.33% | 0.854 | 11/12 |
| hybrid (BM25 + dense) | 75.00% | 66.67% | 0.694 | 9/12 |

**Hybrid did not help here - it hurt.** On conversational questions the user's
words rarely match the legal vocabulary, so BM25 injects lexical noise
(e.g. "right" pulls up *Article 21 Right to object*, "birthday" pulls up
*Article 8*), which RRF then blends into the ranking and pushes the correct
article out of the top-5 on two questions. On a small, single-domain corpus with
paraphrased queries, pure dense retrieval is the stronger choice. (Reported as a
finding rather than hidden.)

### Answer quality

| Metric | Score | Count |
| --- | --- | --- |
| Correct citation (in-scope) | 91.67% | 11/12 |
| Correct `Not found` (out-of-scope) | 100.00% | 3/3 |

### Error analysis

- **Q1 "Can my old employer keep my CV forever?" (gold Article 5).** This is the
  only retrieval miss and the only citation miss. The answer lives in the
  *storage limitation* principle, Article 5(1)(e) ("kept ... for no longer than
  is necessary"), but the conversational phrasing "keep ... forever" embeds
  closer to *Article 17 (right to erasure)*, which dominates the top-5. Because
  retrieval misses Article 5, the answer is grounded on Article 17 and cites it
  instead - a cascading error that starts at retrieval. (Article 17 is a
  reasonable neighbour, but it is not the gold principle.)
- All **out-of-scope** questions were answered `Not found in the GDPR`, including
  the two GDPR-adjacent traps (fines, DPO) that live outside Articles 1-23 - the
  system does not pull them in from the model's own knowledge.

---

## Project structure

```
gdpr_rag/
  config.py       # settings from env, paths, constants
  models.py       # Article / Paragraph / Chunk dataclasses
  ingest.py       # download + parse GDPR HTML -> articles.json
  chunking.py     # article-level & paragraph-level chunkers
  embeddings.py   # local embedding model wrapper (bge / e5)
  vector_store.py # FAISS index build / save / load / search
  retriever.py    # dense and hybrid (BM25 + RRF) retrieval
  llm.py          # pluggable LLM backend (openai / ollama)
  answer.py       # grounded answer + citation + not-found
  pipeline.py     # build_index / load_retriever wiring
  cli.py          # build-index | ask | evaluate
  evaluation.py   # metrics, comparisons, report
eval/
  questions.json  # 15 gold questions
  results.md      # generated evaluation report
tests/
  test_ingest.py
  test_chunking.py
data/
  articles.json   # parsed Articles 1-23 (committed, reproducible)
```

Build artifacts (`index/`, downloaded models, `data/raw/`) are git-ignored.

---

## Testing

```bash
pytest
```

Meaningful tests cover the scored core: article/chapter parsing and metadata
retention from an HTML fixture ([`tests/test_ingest.py`](tests/test_ingest.py)),
and both chunking strategies including precise `Article 6(1)` references with
their sub-points ([`tests/test_chunking.py`](tests/test_chunking.py)).

---

## Assumptions & limitations

- **Scope** is Articles 1-23 only, as specified; Recitals are excluded.
- **Citation granularity** is article/paragraph. Sub-points `(a)/(f)` are kept
  in the chunk text but are not independently retrievable units.
- **Retrieval similarity score**: `dense` returns cosine similarity; `hybrid`
  returns the fused RRF score (not directly comparable to cosine).
- `articles.json` is committed for reproducibility (EU legislation is public and
  not subject to copyright restrictions on reuse). Re-generate it any time with
  `python -m gdpr_rag.cli build-index --force-download`.
- The source HTML is cached under `data/raw/` (git-ignored); the first
  `build-index` needs internet access to eur-lex and Hugging Face.
