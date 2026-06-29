import psycopg
from pgvector.psycopg import register_vector

from rag.config import settings
from rag.ingestion.embedder import embed_texts


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

