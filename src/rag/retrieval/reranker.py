import os

#use the local cache and skip the Hub entirely
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)  # loaded once, reused
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Re-score candidate chunks with a cross-encoder; return the best top_k."""
    if not chunks:
        return []
    model = _get_model()
    pairs = [(query, c["content"]) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [{**chunk, "rerank_score": float(score)} for chunk, score in ranked[:top_k]]
