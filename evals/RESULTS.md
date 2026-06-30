# Eval Results

Tracked before/after numbers so every change is backed by evidence.
Golden dataset: 5 questions (`evals/golden_dataset.json`).

**Journey so far:** Recall@5 0.80 → **1.00** · MRR 0.70 → **0.75** · correctness 3.40 → **5.00** · faithfulness 5.00 → 5.00 — via text cleaning → structure-aware chunking → hybrid search → cross-encoder reranking, each change measured.

## Retrieval metrics

Re-run: `uv run python evals/run_retrieval_eval.py`

| Date | Change | Recall@5 | MRR@5 | Precision@5 | Notes |
|------|--------|---------:|------:|------------:|-------|
| 2026-06-30 | **Baseline** — fixed-size chunking (1000 chars / 200 overlap), OpenAI `text-embedding-3-small`, dense top-k vector search (cosine) | 0.800 | 0.700 | 0.200 | q3 (embedding-generation question) not retrieved in top-5 — the known gap to fix with hybrid search / reranking in Phase 3 |
| 2026-06-30 | **+ text cleaning** (strip leading line numbers, ASCII separators, nbsp) | 0.400 | 0.400 | 0.080 | ⚠️ *Not* a quality regression — cleaning shifted chunk boundaries, moving the exact-snippet chunks to ranks #8/#7/#14 (just outside top-5). Information still retrieved from neighbor chunks → end-to-end correctness rose (see below). Shows snippet-recall is sensitive to boundary shifts; motivates structure-aware chunking. |
| 2026-06-30 | **+ structure-aware chunking** (whole-paragraph packing) | **1.000** ⬆️ | 0.557 | 0.200 | All 5 questions hit. Coherent chunks pulled q1/q3/q5 back into the top-5 (q3 — the long-standing miss — now hits at #5). MRR is below the noisy-baseline 0.700 because relevant chunks rank #3–#5 rather than #1, but all are retrieved. 85 coherent chunks (was 90). |
| 2026-06-30 | **+ hybrid search** (vector + Postgres FTS, fused via RRF) | 1.000 | 0.600 | 0.200 | Held recall; MRR slightly up. Neutral on this semantic-heavy dataset — its real role is a higher-recall **candidate pool** for reranking. Also pinned generation `temperature=0` so eval comparisons are deterministic (an earlier correctness 4.00 was generation noise, not hybrid). |
| 2026-06-30 | **+ cross-encoder reranking** (rerank the hybrid pool with MiniLM cross-encoder) | 1.000 | **0.750** ⬆️ | 0.240 | Best ranking yet — reranker puts the right chunk first (q1→#1, q5→#2) and pulled q4's answer chunk (~#6 in the pool, vocab mismatch) into the top-5. |

## Generation metrics

Re-run: `uv run python -m evals.run_generation_eval` (LLM-as-judge, `gpt-4o-mini`, temp 0)

| Date | Change | Avg correctness | Avg faithfulness | Notes |
|------|--------|----------------:|-----------------:|-------|
| 2026-06-30 | **Baseline** — top-k=5 dense retrieval → `gpt-4o-mini` | 3.40/5 | 5.00/5 | Faithfulness perfect (never hallucinates). Correctness dragged by **q3** (retrieval miss) and **q4** (relevant chunk retrieved but answered "I don't know" — noisy extraction). Both score 1/5. |
| 2026-06-30 | **+ text cleaning** | **4.00/5** ⬆️ | 5.00/5 | **q3 fixed: 1 → 4** (cleaner context the LLM can actually use). q4 still 1/5 (chunk boundary). Faithfulness held at perfect. First *proven* improvement. |
| 2026-06-30 | **+ structure-aware chunking** | **4.20/5** ⬆️ | 5.00/5 | **q3 now 5/5** (fully fixed from the original 1/5). Only **q4** remains broken (1/5) — retrieved at rank 1 but still answered poorly. Next target. |
| 2026-06-30 | **+ hybrid search** (generation `temperature=0`) | 4.20/5 | 5.00/5 | Neutral vs dense on this semantic-heavy set. q4 still 1/5 — vocabulary mismatch (question says "augmentation step"; the answer chunk says "combine query + context"); both retrievers rank it ~#7–#9, so hybrid alone can't fix it. Needs reranking / query rewrite. |
| 2026-06-30 | **+ cross-encoder reranking** | **5.00/5** ⬆️ | 5.00/5 | **q4 fixed: 1 → 5.** Cross-encoder reads query+chunk together → recognized the answer despite vocabulary mismatch. Perfect correctness + faithfulness across all 5 questions. |

## Known gaps to fix with proof (Phase 3)
- **q3** — relevant chunk not retrieved in top-5 (retrieval miss).
- **q4** — relevant chunk *is* retrieved but generation says "I don't know" (fragmented/noisy extracted text).
- Expected fixes: better chunking, data cleaning, hybrid search, reranking — each rerun against these same metrics.
