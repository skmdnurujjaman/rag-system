from rag.gateway.semantic_cache import SemanticCache
from rag.generation.answer import generate_answer
from rag.retrieval.search import retrieve
from rag.observability import tracer

_qa_cache = SemanticCache(threshold=0.82)


def answer_question(question: str, top_k: int = 5) -> dict:
    with tracer.start_as_current_span("qa.answer"):
        cached = _qa_cache.get(question)
        if cached is not None:
            return cached
        chunks = retrieve(question, top_k=top_k)
        answer = generate_answer(question, chunks)
        result = {"answer": answer, "chunks": chunks}
        _qa_cache.set(question, result)
        return result
