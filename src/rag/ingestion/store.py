from pgvector.psycopg import register_vector_async

from rag.db.pool import pool


async def store_document(filename: str, chunks: list[str], embeddings: list[list[float]], *, tenant_id: int) -> int:
    """Store a document plus its chunks and embeddings. Returns the new document id."""
    async with pool.connection() as conn:
        await register_vector_async(conn)
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO documents (filename, tenant_id) VALUES (%s, %s) RETURNING id",
                (filename,tenant_id),
            )
            row = await cur.fetchone()
            document_id = row[0]

            for index, (content, embedding) in enumerate(zip(chunks, embeddings)):
                await cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding, tenant_id)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (document_id, index, content, embedding, tenant_id),
                )
    return document_id
