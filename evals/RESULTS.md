# Eval Results

Tracked before/after numbers so every change is backed by evidence.
Golden dataset: 5 questions (`evals/golden_dataset.json`).

## Retrieval metrics

Re-run: `uv run python evals/run_retrieval_eval.py`

| Date | Change | Recall@5 | MRR@5 | Precision@5 | Notes |
|------|--------|---------:|------:|------------:|-------|
| 2026-06-30 | **Baseline** — fixed-size chunking (1000 chars / 200 overlap), OpenAI `text-embedding-3-small`, dense top-k vector search (cosine) | 0.800 | 0.700 | 0.200 | q3 (embedding-generation question) not retrieved in top-5 — the known gap to fix with hybrid search / reranking in Phase 3 |

## Generation metrics

_(coming next — faithfulness, answer relevance, correctness)_
