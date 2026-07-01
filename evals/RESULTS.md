# Eval Results

Tracked before/after numbers so every change is backed by evidence.
Golden dataset: **13 questions** (`evals/golden_dataset.json`) — expanded from 5 once the metric saturated.

**Journey:** On the original 5-question set, Recall 0.80 → **1.00** and correctness 3.40 → **5.00** via text cleaning → structure-aware chunking → hybrid → cross-encoder reranking (each measured). Then expanded to **13** harder questions to de-saturate; current pipeline (hybrid + rerank): Recall@5 **0.923**, MRR 0.673, correctness **4.85**, faithfulness 5.00. **Multi-query expansion** was built + A/B tested → **zero lift**, so it's left out of the default pipeline (measure, don't cargo-cult).

## Retrieval metrics

Re-run: `uv run python evals/run_retrieval_eval.py`

| Date | Change | Recall@5 | MRR@5 | Precision@5 | Notes |
|------|--------|---------:|------:|------------:|-------|
| 2026-06-30 | **Baseline** — fixed-size chunking (1000 chars / 200 overlap), OpenAI `text-embedding-3-small`, dense top-k vector search (cosine) | 0.800 | 0.700 | 0.200 | q3 (embedding-generation question) not retrieved in top-5 — the known gap to fix with hybrid search / reranking in Phase 3 |
| 2026-06-30 | **+ text cleaning** (strip leading line numbers, ASCII separators, nbsp) | 0.400 | 0.400 | 0.080 | ⚠️ *Not* a quality regression — cleaning shifted chunk boundaries, moving the exact-snippet chunks to ranks #8/#7/#14 (just outside top-5). Information still retrieved from neighbor chunks → end-to-end correctness rose (see below). Shows snippet-recall is sensitive to boundary shifts; motivates structure-aware chunking. |
| 2026-06-30 | **+ structure-aware chunking** (whole-paragraph packing) | **1.000** ⬆️ | 0.557 | 0.200 | All 5 questions hit. Coherent chunks pulled q1/q3/q5 back into the top-5 (q3 — the long-standing miss — now hits at #5). MRR is below the noisy-baseline 0.700 because relevant chunks rank #3–#5 rather than #1, but all are retrieved. 85 coherent chunks (was 90). |
| 2026-06-30 | **+ hybrid search** (vector + Postgres FTS, fused via RRF) | 1.000 | 0.600 | 0.200 | Held recall; MRR slightly up. Neutral on this semantic-heavy dataset — its real role is a higher-recall **candidate pool** for reranking. Also pinned generation `temperature=0` so eval comparisons are deterministic (an earlier correctness 4.00 was generation noise, not hybrid). |
| 2026-06-30 | **+ cross-encoder reranking** (rerank the hybrid pool with MiniLM cross-encoder) | 1.000 | **0.750** ⬆️ | 0.240 | Best ranking yet — reranker puts the right chunk first (q1→#1, q5→#2) and pulled q4's answer chunk (~#6 in the pool, vocab mismatch) into the top-5. *(measured on the 5-Q set)* |
| 2026-06-30 | **expanded dataset → 13 questions** (hybrid + rerank) | 0.923 | 0.673 | 0.215 | New baseline on harder questions. Only q13 misses retrieval (its exact chunk is below the top-20 pool) — yet still answers 5/5 from neighbor chunks. |
| 2026-06-30 | **multi-query expansion** *(A/B → removed)* | 0.923 | 0.673 | 0.215 | **Identical to hybrid+rerank** (same recall, MRR, same q13 miss) → **zero lift** on this corpus; the reranker already bridges vocab mismatch. Built + measured for learning; kept the module but removed from the default pipeline (cost without benefit). |

## Generation metrics

Re-run: `uv run python -m evals.run_generation_eval` (LLM-as-judge, `gpt-4o-mini`, temp 0)

| Date | Change | Avg correctness | Avg faithfulness | Notes |
|------|--------|----------------:|-----------------:|-------|
| 2026-06-30 | **Baseline** — top-k=5 dense retrieval → `gpt-4o-mini` | 3.40/5 | 5.00/5 | Faithfulness perfect (never hallucinates). Correctness dragged by **q3** (retrieval miss) and **q4** (relevant chunk retrieved but answered "I don't know" — noisy extraction). Both score 1/5. |
| 2026-06-30 | **+ text cleaning** | **4.00/5** ⬆️ | 5.00/5 | **q3 fixed: 1 → 4** (cleaner context the LLM can actually use). q4 still 1/5 (chunk boundary). Faithfulness held at perfect. First *proven* improvement. |
| 2026-06-30 | **+ structure-aware chunking** | **4.20/5** ⬆️ | 5.00/5 | **q3 now 5/5** (fully fixed from the original 1/5). Only **q4** remains broken (1/5) — retrieved at rank 1 but still answered poorly. Next target. |
| 2026-06-30 | **+ hybrid search** (generation `temperature=0`) | 4.20/5 | 5.00/5 | Neutral vs dense on this semantic-heavy set. q4 still 1/5 — vocabulary mismatch (question says "augmentation step"; the answer chunk says "combine query + context"); both retrievers rank it ~#7–#9, so hybrid alone can't fix it. Needs reranking / query rewrite. |
| 2026-06-30 | **+ cross-encoder reranking** | **5.00/5** ⬆️ | 5.00/5 | **q4 fixed: 1 → 5.** Cross-encoder reads query+chunk together → recognized the answer despite vocabulary mismatch. Perfect correctness + faithfulness across all 5 questions. |
| 2026-06-30 | **expanded → 13 questions** (hybrid + rerank) | 4.85/5 | 5.00/5 | New harder questions (q9–q13). q7 & q12 at 4/5 (minor completeness); q13 answers correctly despite a retrieval miss. Multi-query added no lift (see retrieval table). |

## Known gaps to fix with proof (Phase 3)
- **q3** — relevant chunk not retrieved in top-5 (retrieval miss).
- **q4** — relevant chunk *is* retrieved but generation says "I don't know" (fragmented/noisy extracted text).
- Expected fixes: better chunking, data cleaning, hybrid search, reranking — each rerun against these same metrics.

## Agent faithfulness (Summary & Essay)

Reference-free faithfulness (LLM-judge, temp 0) — is the output grounded in its source, with nothing invented?
Re-run: `uv run python -m evals.run_summary_eval` · `uv run python -m evals.run_essay_eval`

| Date | Agent | Faithfulness | Notes |
|------|-------|-------------:|-------|
| 2026-07-01 | **Summary** | **5.00/5** | Judged against the full document. Same grounding discipline as Q&A. |
| 2026-07-01 | **Essay** | **5.00/5** (3 topics) | Judged against retrieved sources. Grounded even at `temperature=0.5`. |
