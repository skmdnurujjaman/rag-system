from psycopg_pool import AsyncConnectionPool

from rag.config import settings

# One pool for the whole process. Opens min_size connections up front and reuses them.
pool = AsyncConnectionPool(
    conninfo=settings.database_url,
    min_size=2,          # keep a couple warm
    max_size=10,         # cap concurrent DB connections (protects Postgres)
    open=False,
)
