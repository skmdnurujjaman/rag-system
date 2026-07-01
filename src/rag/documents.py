import psycopg

from rag.config import settings


def get_document_text(document_id: int) -> str:
    """Reassemble a document's full text from its stored chunks, in order."""
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT content FROM chunks WHERE document_id = %s ORDER BY chunk_index",
            (document_id,),
        )
        return "\n\n".join(content for (content,) in cur.fetchall())
