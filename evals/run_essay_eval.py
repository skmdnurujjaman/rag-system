from evals.judge import judge_faithfulness
from rag.agents.essay import write_essay

TOPICS = [
    "Why retrieval-augmented generation matters for LLMs",
    "How chunking and text splitting affect RAG quality",
    "The difference between a vector store and a vector database",
]
EVAL_TENANT = 1

async def evaluate() -> dict:
    scores = []
    for topic in TOPICS:
        result = await write_essay(topic, max_words=300, tenant_id=EVAL_TENANT)
        chunks = [c["content"] for c in result["sources"]]
        f = await judge_faithfulness(chunks, result["essay"])
        scores.append(f.score)
        print(f"'{topic[:45]}...': faithfulness={f.score}/5 — {f.reasoning}")
    avg = sum(scores) / len(scores)
    print("-" * 50)
    print(f"Avg essay faithfulness: {avg:.2f}/5")
    return {"essay_faithfulness": avg}


if __name__ == "__main__":
    import asyncio

    from rag.db.pool import pool
    async def _main():
        async with pool:
            await evaluate()
    asyncio.run(_main())
