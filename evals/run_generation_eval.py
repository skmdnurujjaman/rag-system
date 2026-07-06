import json
from pathlib import Path

from pydantic import BaseModel

from evals.judge import judge_correctness, judge_faithfulness
from rag.agents.qa import answer_question

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
TOP_K = 5


class GoldenItem(BaseModel):
    id: str
    question: str
    expected_answer: str
    relevant_snippets: list[str]


def load_dataset() -> list[GoldenItem]:
    return [GoldenItem(**x) for x in json.loads(DATASET_PATH.read_text())]


async def evaluate(top_k: int = TOP_K) -> dict:
    dataset = load_dataset()
    corr_scores, faith_scores = [], []

    for item in dataset:
        result = await answer_question(item.question, top_k=top_k)
        answer = result["answer"]
        chunks = [c["content"] for c in result["chunks"]]

        c = await judge_correctness(item.question, item.expected_answer, answer)
        f = await judge_faithfulness(chunks, answer)
        corr_scores.append(c.score)
        faith_scores.append(f.score)

        print(f"{item.id}: correctness={c.score}/5  faithfulness={f.score}/5")

    n = len(dataset)
    metrics = {
        "correctness": sum(corr_scores) / n,
        "faithfulness": sum(faith_scores) / n,
    }
    print("-" * 50)
    print(f"Questions:         {n}")
    print(f"Avg correctness:   {metrics['correctness']:.2f}/5")
    print(f"Avg faithfulness:  {metrics['faithfulness']:.2f}/5")
    return metrics


if __name__ == "__main__":
    import asyncio

    from rag.db.pool import pool
    async def _main():
        async with pool:
            await evaluate()
    asyncio.run(_main())
