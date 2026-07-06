import hashlib
from fastapi import Header, HTTPException
from rag.db.pool import pool

def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

async def resolve_tenant(api_key: str) -> int | None:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SELECT id FROM tenants WHERE api_key_hash = %s", (_hash(api_key),))
        row = await cur.fetchone()
    return row[0] if row else None

async def require_tenant(x_api_key: str | None = Header(default=None)) -> int:
    """FastAPI dependency: resolve the caller's tenant from X-API-Key, or 401."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")
    tenant_id = await resolve_tenant(x_api_key)
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return tenant_id
