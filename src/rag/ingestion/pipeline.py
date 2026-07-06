import logging
from pathlib import Path

from rag.ingestion.chunker import chunk_text
from rag.ingestion.cleaner import clean_text
from rag.ingestion.embedder import embed_texts
from rag.ingestion.loader import load_pdf
from rag.ingestion.store import store_document

logger = logging.getLogger(__name__)


async def ingest_pdf(path: str) -> int:
    """Run the full ingestion pipeline: load -> chunk -> embed -> store. Returns the document id."""
    filename = Path(path).name
    logger.info("ingestion.start filename=%s", filename)

    text = load_pdf(path)
    logger.info("ingestion.loaded filename=%s chars=%d", filename, len(text))
    
    text = clean_text(text)
    logger.info("ingestion.cleaned filename=%s chars=%d", filename, len(text))

    chunks = chunk_text(text)
    logger.info("ingestion.chunked filename=%s chunks=%d", filename, len(chunks))

    embeddings = await embed_texts(chunks)
    logger.info("ingestion.embedded filename=%s vectors=%d", filename, len(embeddings))

    document_id = await store_document(filename, chunks, embeddings)
    logger.info("ingestion.stored filename=%s document_id=%d", filename, document_id)

    return document_id


if __name__ == "__main__":
    import asyncio
    import sys

    from rag.db.pool import pool
    async def _main():
        async with pool:                        # opens + closes the async pool
            for path in sys.argv[1:]:
                print("doc id:", await ingest_pdf(path))
    asyncio.run(_main())