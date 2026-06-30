import json
from pathlib import Path

from pydantic import BaseModel

from rag.retrieval.search import search

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
TOP_K = 5


class GoldenItem(BaseModel):
    id: str
    question: str
    expected_answer: str
    relevant_snippets: list[str]


def load_dataset() -> list[GoldenItem]:
    data = json.loads(DATASET_PATH.read_text())
    return [GoldenItem(**item) for item in data]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def is_relevant(chunk_content: str, snippets: list[str]) -> bool:
    """A chunk is relevant if it contains ANY golden snippet (OR semantics)."""
    content = normalize(chunk_content)
    return any(normalize(s) in content for s in snippets)


def evaluate(top_k: int = TOP_K) -> dict:
    dataset = load_dataset()
    recalls, mrrs, precisions = [], [], []

    for item in dataset:
        results = search(item.question, top_k=top_k)
        relevance = [is_relevant(r["content"], item.relevant_snippets) for r in results]

        recall = 1.0 if any(relevance) else 0.0

        rr = 0.0
        for rank, rel in enumerate(relevance, start=1):
            if rel:
                rr = 1.0 / rank
                break

        precision = sum(relevance) / len(relevance) if relevance else 0.0

        recalls.append(recall)
        mrrs.append(rr)
        precisions.append(precision)

        first_rank = next((i + 1 for i, rel in enumerate(relevance) if rel), None)
        print(f"{item.id}: hit={'Y' if recall else 'N'}  first_rank={first_rank}  precision={precision:.2f}")

    n = len(dataset)
    metrics = {
        "recall_at_k": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "precision_at_k": sum(precisions) / n,
    }
    print("-" * 50)
    print(f"Questions:     {n}")
    print(f"Recall@{top_k}:      {metrics['recall_at_k']:.3f}")
    print(f"MRR@{top_k}:         {metrics['mrr']:.3f}")
    print(f"Precision@{top_k}:   {metrics['precision_at_k']:.3f}")
    return metrics


if __name__ == "__main__":
    evaluate()
