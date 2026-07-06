from rag.gateway.semantic_cache import SemanticCache
from rag.generation.answer import generate_answer
from rag.observability import tracer
from rag.retrieval.search import retrieve

_qa_caches: dict[int, SemanticCache] = {}

def _cache_for(tenant_id: int) -> SemanticCache:
    if tenant_id not in _qa_caches:
        _qa_caches[tenant_id] = SemanticCache(threshold=0.82)
    return _qa_caches[tenant_id]

async def answer_question(question: str, top_k: int = 5, *, tenant_id: int) -> dict:
    with tracer.start_as_current_span("qa.answer"):
        cache = _cache_for(tenant_id)
        cached = await cache.get(question)
        if cached is not None:
            return cached
        chunks = await retrieve(question, top_k=top_k, tenant_id=tenant_id)
        answer = await generate_answer(question, chunks)
        result = {"answer": answer, "chunks": chunks}
        await cache.set(question, result)
        return result
