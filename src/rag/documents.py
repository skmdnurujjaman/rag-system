from rag.db.pool import pool

def get_document_chunks(document_id: int) -> list[str]:
    """Return a document's chunks, in order."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT content FROM chunks WHERE document_id = %s ORDER BY chunk_index",
            (document_id,),
        )
        return [content for (content,) in cur.fetchall()]


def get_document_text(document_id: int) -> str:
    """Reassemble a document's full text from its chunks, in order."""
    return "\n\n".join(get_document_chunks(document_id))
