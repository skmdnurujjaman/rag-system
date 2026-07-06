from pgvector.psycopg import register_vector_async
from rag.db.pool import pool

async def store_document(filename: str, chunks: list[str], embeddings: list[list[float]]) -> int:
    """Store a document plus its chunks and embeddings. Returns the new document id."""
    async with pool.connection() as conn:
        await register_vector_async(conn)
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO documents (filename) VALUES (%s) RETURNING id",
                (filename,),
            )
            row = await cur.fetchone()
            document_id = row[0]

            for index, (content, embedding) in enumerate(zip(chunks, embeddings)):
                await cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (document_id, index, content, embedding),
                )
    return document_id
