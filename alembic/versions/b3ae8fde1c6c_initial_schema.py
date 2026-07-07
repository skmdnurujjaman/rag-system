"""initial schema

Revision ID: b3ae8fde1c6c
Revises: 
Create Date: 2026-07-06 22:54:28.948949

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b3ae8fde1c6c'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("""
        CREATE TABLE documents (
            id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            filename    TEXT        NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE chunks (
            id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            document_id  BIGINT       NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index  INT          NOT NULL,
            content      TEXT         NOT NULL,
            embedding    VECTOR(1536) NOT NULL,
            content_tsv  tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
            UNIQUE (document_id, chunk_index)
        )
    """)
    op.execute("CREATE INDEX chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX chunks_content_tsv_gin ON chunks USING gin (content_tsv)")

def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunks")       # child first (FK)
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP EXTENSION IF EXISTS vector")

