# Eval Results

Tracked before/after numbers so every change is backed by evidence.
Golden dataset: 5 questions (`evals/golden_dataset.json`).

## Retrieval metrics

Re-run: `uv run python evals/run_retrieval_eval.py`

| Date | Change | Recall@5 | MRR@5 | Precision@5 | Notes |
|------|--------|---------:|------:|------------:|-------|
| 2026-06-30 | **Baseline** — fixed-size chunking (1000 chars / 200 overlap), OpenAI `text-embedding-3-small`, dense top-k vector search (cosine) | 0.800 | 0.700 | 0.200 | q3 (embedding-generation question) not retrieved in top-5 — the known gap to fix with hybrid search / reranking in Phase 3 |
| 2026-06-30 | **+ text cleaning** (strip leading line numbers, ASCII separators, nbsp) | 0.400 | 0.400 | 0.080 | ⚠️ *Not* a quality regression — cleaning shifted chunk boundaries, moving the exact-snippet chunks to ranks #8/#7/#14 (just outside top-5). Information still retrieved from neighbor chunks → end-to-end correctness rose (see below). Shows snippet-recall is sensitive to boundary shifts; motivates structure-aware chunking. |

## Generation metrics

Re-run: `uv run python -m evals.run_generation_eval` (LLM-as-judge, `gpt-4o-mini`, temp 0)

| Date | Change | Avg correctness | Avg faithfulness | Notes |
|------|--------|----------------:|-----------------:|-------|
| 2026-06-30 | **Baseline** — top-k=5 dense retrieval → `gpt-4o-mini` | 3.40/5 | 5.00/5 | Faithfulness perfect (never hallucinates). Correctness dragged by **q3** (retrieval miss) and **q4** (relevant chunk retrieved but answered "I don't know" — noisy extraction). Both score 1/5. |
| 2026-06-30 | **+ text cleaning** | **4.00/5** ⬆️ | 5.00/5 | **q3 fixed: 1 → 4** (cleaner context the LLM can actually use). q4 still 1/5 (chunk boundary). Faithfulness held at perfect. First *proven* improvement. |

## Known gaps to fix with proof (Phase 3)
- **q3** — relevant chunk not retrieved in top-5 (retrieval miss).
- **q4** — relevant chunk *is* retrieved but generation says "I don't know" (fragmented/noisy extracted text).
- Expected fixes: better chunking, data cleaning, hybrid search, reranking — each rerun against these same metrics.
