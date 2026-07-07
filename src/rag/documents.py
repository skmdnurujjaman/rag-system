from rag.db.pool import pool


async def get_document_chunks(document_id: int, *, tenant_id: int) -> list[str]:
    """Return a document's chunks, in order."""
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT content FROM chunks WHERE document_id = %s AND tenant_id = %s ORDER BY chunk_index",
            (document_id, tenant_id),
        )
        rows = await cur.fetchall()
        return [content for (content,) in rows]


async def get_document_text(document_id: int, *, tenant_id: int) -> str:
    """Reassemble a document's full text from its chunks, in order."""
    return "\n\n".join(await get_document_chunks(document_id, tenant_id=tenant_id))
