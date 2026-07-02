import psycopg
from pgvector.psycopg import register_vector

from rag.config import settings
from rag.ingestion.embedder import embed_texts
from rag.retrieval.reranker import rerank
from rag.observability import tracer

def _rrf_fuse(result_lists: list[list[dict]], top_k: int, rrf_k: int = 60) -> list[dict]:
    """Fuse any number of ranked result lists via Reciprocal Rank Fusion."""
    scores: dict[int, float] = {}
    chunks: dict[int, dict] = {}
    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            cid = r["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunks[cid] = r
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        {"id": cid, "document_id": chunks[cid]["document_id"],
         "content": chunks[cid]["content"], "rrf_score": scores[cid]}
        for cid in ranked
    ]
    
def retrieve(query: str, top_k: int = 5, candidates: int = 20) -> list[dict]:
    """Pipeline: hybrid retrieve a candidate pool → cross-encoder rerank."""
    with tracer.start_as_current_span("retrieve") as span:
        span.set_attribute("retrieve.top_k", top_k)
        pool = hybrid_search(query, top_k=candidates)
        return rerank(query, pool, top_k=top_k)

def search(query: str, top_k: int = 5) -> list[dict]:
    """Find the top_k chunks most similar to the query."""
    query_embedding = embed_texts([query])[0]

    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, document_id, content, embedding <=> %s::vector AS distance
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()

    return [
        {"id": r[0], "document_id": r[1], "content": r[2], "distance": r[3]}
        for r in rows
    ]

def keyword_search(query: str, top_k: int = 5) -> list[dict]:
    """Sparse keyword search via Postgres full-text search (OR over query terms)."""
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH q AS (
                    SELECT to_tsquery(
                        'english',
                        array_to_string(
                            tsvector_to_array(to_tsvector('english', %s)),
                            ' | '
                        )
                    ) AS query
                )
                SELECT c.id, c.document_id, c.content,
                       ts_rank_cd(c.content_tsv, q.query) AS score
                FROM chunks c, q
                WHERE c.content_tsv @@ q.query
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, top_k),
            )
            rows = cur.fetchall()
    return [
        {"id": r[0], "document_id": r[1], "content": r[2], "score": r[3]}
        for r in rows
    ]

def hybrid_search(
    query: str, top_k: int = 5, candidates: int = 20, rrf_k: int = 60
) -> list[dict]:
    """Fuse dense (vector) + sparse (keyword) results with Reciprocal Rank Fusion."""
    dense = search(query, top_k=candidates)            # vector results
    sparse = keyword_search(query, top_k=candidates)   # keyword results

    scores: dict[int, float] = {}
    chunks: dict[int, dict] = {}
    for results in (dense, sparse):
        for rank, r in enumerate(results, start=1):
            cid = r["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunks[cid] = r

    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        {
            "id": cid,
            "document_id": chunks[cid]["document_id"],
            "content": chunks[cid]["content"],
            "rrf_score": scores[cid],
        }
        for cid in ranked
    ]
