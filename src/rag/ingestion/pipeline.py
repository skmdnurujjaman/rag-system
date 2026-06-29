import logging
from pathlib import Path

from rag.ingestion.chunker import chunk_text
from rag.ingestion.embedder import embed_texts
from rag.ingestion.loader import load_pdf
from rag.ingestion.store import store_document

logger = logging.getLogger(__name__)


def ingest_pdf(path: str) -> int:
    """Run the full ingestion pipeline: load -> chunk -> embed -> store. Returns the document id."""
    filename = Path(path).name
    logger.info("ingestion.start filename=%s", filename)

    text = load_pdf(path)
    logger.info("ingestion.loaded filename=%s chars=%d", filename, len(text))

    chunks = chunk_text(text)
    logger.info("ingestion.chunked filename=%s chunks=%d", filename, len(chunks))

    embeddings = embed_texts(chunks)
    logger.info("ingestion.embedded filename=%s vectors=%d", filename, len(embeddings))

    document_id = store_document(filename, chunks, embeddings)
    logger.info("ingestion.stored filename=%s document_id=%d", filename, document_id)

    return document_id
