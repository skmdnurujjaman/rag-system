from arq import cron  # noqa (optional; shown for awareness)
from arq.connections import RedisSettings
from rag.config import settings
from rag.db.pool import pool
from rag.ingestion.pipeline import ingest_pdf
from rag.observability import log

async def ingest_document(ctx, path: str) -> int:
    """Background task: run the full async ingestion pipeline for one file."""
    log.info("worker.ingest_start", path=path)
    doc_id = await ingest_pdf(path)          # our async pipeline, unchanged
    log.info("worker.ingest_done", path=path, document_id=doc_id)
    return doc_id                             # arq stores the return value as the job result

async def on_startup(ctx):
    await pool.open()                         # the worker needs the DB pool too

async def on_shutdown(ctx):
    await pool.close()

class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    on_startup = on_startup
    on_shutdown = on_shutdown
