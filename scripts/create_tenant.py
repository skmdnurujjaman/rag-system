import asyncio, hashlib, secrets, sys
from rag.db.pool import pool

def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

async def create_tenant(name: str) -> tuple[int, str]:
    api_key = "rag_" + secrets.token_urlsafe(24)          # plaintext — shown ONCE, never stored
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO tenants (name, api_key_hash) VALUES (%s, %s) RETURNING id",
            (name, _hash(api_key)),
        )
        tenant_id = (await cur.fetchone())[0]
    return tenant_id, api_key

if __name__ == "__main__":
    async def _main():
        async with pool:
            tid, key = await create_tenant(sys.argv[1] if len(sys.argv) > 1 else "tenant")
            print(f"tenant_id={tid}\nAPI key (save it — shown once): {key}")
    asyncio.run(_main())
