import secrets

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse
from rag.config import settings
from rag.retrieval.search import retrieve
from rag.db.pool import pool

mcp = FastMCP("rag", host="127.0.0.1", port=8001)   # 8001 so it won't clash with the RAG API on 8000


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> str:
    """Search the ingested documents and return the most relevant passages for a query."""
    chunks = retrieve(query, top_k=top_k)
    return "\n\n".join(f"[{i + 1}] {c['content']}" for i, c in enumerate(chunks))


@mcp.tool()
def delete_document(document_id: int, confirm: bool = False) -> str:
    """Permanently delete a document and all its chunks. HIGH-STAKES and irreversible.

    Safety: do NOT delete without user approval. Call once with confirm=false to preview the
    action and show the warning to the user; only if they approve, call again with confirm=true.
    """
    if not confirm:
        return (f"⚠️ This will PERMANENTLY delete document {document_id} and all its chunks — "
                f"this cannot be undone. If the user approves, call again with confirm=true.")

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        deleted = cur.rowcount
    if deleted:
        return f"Deleted document {document_id} ({deleted} row; chunks removed via cascade)."
    return f"No document {document_id} found — nothing deleted."


class BearerAuth:
    """Pure-ASGI middleware: require 'Authorization: Bearer <token>' on every HTTP request."""
    def __init__(self, app, token):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode()
            if not secrets.compare_digest(auth, f"Bearer {self.token}"):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


# HTTP (streamable) ASGI app, protected by the bearer token
app = BearerAuth(mcp.streamable_http_app(), settings.mcp_auth_token.get_secret_value())


if __name__ == "__main__":
    mcp.run()   # STDIO transport (local testing) still works via `python src/rag/mcp_server.py`
