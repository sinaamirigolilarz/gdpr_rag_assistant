# Commands & Tests

A step-by-step runbook mapping each task section to the exact commands that
exercise it. Run everything from the repository root.

> On Windows, either activate the virtual environment first
> (`.venv\Scripts\Activate.ps1`) and use `python`, or call the interpreter
> directly with `.venv\Scripts\python.exe`. The examples below assume the venv
> is active.

---

## 0. One-time setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# Create your local secrets file and fill in the LLM credentials:
cp .env.example .env
```

---

## Part 1 & 2 — Text preparation, chunking, embeddings & retrieval

Build the vector index. This parses GDPR Articles 1-23 from `data/articles.json`
(or downloads and parses the source HTML if it is missing), chunks them, embeds
the chunks with the local model, and writes the FAISS index to `index/`.

```bash
python -m gdpr_rag.cli build-index
```

Expected: `Indexed 76 chunks (strategy=paragraph, model=BAAI/bge-small-en-v1.5).`

Optional variants:

```bash
# Article-level chunking instead of paragraph-level
python -m gdpr_rag.cli build-index --strategy article

# Force a fresh download of the source HTML from eur-lex
python -m gdpr_rag.cli build-index --force-download
```

---

## Part 3 — Answer with citation

Ask a question whose answer is in Articles 1-23 (returns a short answer with the
cited article(s) plus the retrieved context):

```bash
python -m gdpr_rag.cli ask "How do I get a company to delete my data?"
```

Ask an out-of-scope question (must reply exactly `Not found in the GDPR`):

```bash
python -m gdpr_rag.cli ask "What is the minimum wage in France?"
```

Optional flags: `--k 5` (number of chunks), `--mode dense|hybrid`.

---

## Part 4 & 5 — Evaluation (metrics, comparison, hybrid bonus)

Runs the 15-question gold set and prints the report (also saved to
`eval/results.md`): retrieval Recall@5 / Recall@1 / MRR, the paragraph-vs-article
chunking comparison, the dense-vs-hybrid comparison (bonus), answer citation
accuracy and "Not found" accuracy, and an error analysis.

```bash
python -m gdpr_rag.cli evaluate
```

---

## Tests

```bash
pytest -v
```

Expected: `11 passed`. Tests cover HTML parsing / metadata retention
(`tests/test_ingest.py`) and both chunking strategies with precise
`Article 6(1)` references (`tests/test_chunking.py`).

---

## Notes for restricted or offline networks

- **Embedding model**: the first `build-index` downloads `BAAI/bge-small-en-v1.5`
  (~130 MB) from Hugging Face. Afterwards, set `HF_LOCAL_ONLY=1` in `.env` to load
  it from the local cache and skip the online update check.
- **LLM API**: if your provider must be reached directly (not through a system
  proxy), clear the proxy variables in the shell before `ask` / `evaluate`:
  - PowerShell: `$env:HTTP_PROXY=""; $env:HTTPS_PROXY=""`
  - bash: prefix with `HTTP_PROXY= HTTPS_PROXY=`
- **Transient gateway errors** (502/503/504) from the API are retried
  automatically; if they persist, switch `OPENAI_MODEL` in `.env` to a faster
  model offered by your provider.
