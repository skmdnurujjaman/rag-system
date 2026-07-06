from evals.judge import judge_faithfulness
from rag.agents.summary import summarize_document
from rag.documents import get_document_chunks

DOCUMENT_IDS = [1]  # add more as you ingest more docs
EVAL_TENANT = 1

async def evaluate() -> dict:
    scores = []
    for doc_id in DOCUMENT_IDS:
        summary = await summarize_document(doc_id, max_words=150, tenant_id=EVAL_TENANT)
        f = await judge_faithfulness(await get_document_chunks(doc_id, tenant_id=EVAL_TENANT), summary)
        scores.append(f.score)
        print(f"doc {doc_id}: faithfulness={f.score}/5 — {f.reasoning}")
    avg = sum(scores) / len(scores)
    print("-" * 50)
    print(f"Avg summary faithfulness: {avg:.2f}/5")
    return {"summary_faithfulness": avg}


if __name__ == "__main__":
    import asyncio

    from rag.db.pool import pool
    async def _main():
        async with pool:
            await evaluate()
    asyncio.run(_main())