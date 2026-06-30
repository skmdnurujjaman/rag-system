-- Create vector extention if not exist 
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per uploaded file
CREATE TABLE IF NOT EXISTS documents (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename    TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Many rows per document
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id  BIGINT       NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT          NOT NULL,
    content      TEXT         NOT NULL,
    embedding    VECTOR(1536) NOT NULL,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- Fast approximate nearest-neighbor search using cosine distance
-- Hierarchical Navigable Small World index
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks
    USING hnsw (embedding vector_cosine_ops);
