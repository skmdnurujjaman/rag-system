from pgvector.psycopg import register_vector
from rag.db.pool import pool

def store_document(filename: str, chunks: list[str], embeddings: list[list[float]]) -> int:
    """Store a document plus its chunks and embeddings. Returns the new document id."""
    with pool.connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (filename) VALUES (%s) RETURNING id",
                (filename,),
            )
            document_id = cur.fetchone()[0]

            for index, (content, embedding) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (document_id, index, content, embedding),
                )
    return document_id
