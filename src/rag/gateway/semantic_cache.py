import logging

from rag.gateway.llm import embed

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class SemanticCache:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self._embeddings: list[list[float]] = []
        self._values: list[object] = []

    async def get(self, query: str) -> object | None:
        if not self._embeddings:
            return None
        q = (await embed([query]))[0]
        best_i, best_sim = -1, -1.0
        for i, e in enumerate(self._embeddings):
            sim = _cosine(q, e)
            if sim > best_sim:
                best_i, best_sim = i, sim
        logger.info("semantic_cache best_sim=%.3f threshold=%.2f", best_sim, self.threshold)
        return self._values[best_i] if best_sim >= self.threshold else None

    async def set(self, query: str, value: object) -> None:
        self._embeddings.append((await embed([query]))[0])
        self._values.append(value)
