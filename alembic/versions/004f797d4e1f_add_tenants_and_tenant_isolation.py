"""add tenants and tenant isolation

Revision ID: 004f797d4e1f
Revises: b3ae8fde1c6c
Create Date: 2026-07-06 23:16:28.028141

"""
import hashlib
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '004f797d4e1f'
down_revision: Union[str, Sequence[str], None] = 'b3ae8fde1c6c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def upgrade() -> None:
    op.execute("""
        CREATE TABLE tenants (
            id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            name         TEXT        NOT NULL,
            api_key_hash TEXT        NOT NULL UNIQUE,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    # seed a default tenant (local dev API key = 'dev-tenant-key' — rotate/remove in prod)
    op.execute(f"INSERT INTO tenants (name, api_key_hash) VALUES ('default', '{_hash('dev-tenant-key')}')")
    # add tenant_id, backfill existing rows to 'default', then enforce NOT NULL
    op.execute("ALTER TABLE documents ADD COLUMN tenant_id BIGINT REFERENCES tenants(id)")
    op.execute("ALTER TABLE chunks    ADD COLUMN tenant_id BIGINT REFERENCES tenants(id)")
    op.execute("UPDATE documents SET tenant_id = (SELECT id FROM tenants WHERE name='default')")
    op.execute("UPDATE chunks    SET tenant_id = (SELECT id FROM tenants WHERE name='default')")
    op.execute("ALTER TABLE documents ALTER COLUMN tenant_id SET NOT NULL")
    op.execute("ALTER TABLE chunks    ALTER COLUMN tenant_id SET NOT NULL")
    op.execute("CREATE INDEX chunks_tenant_id_idx ON chunks (tenant_id)")   # every search filters on this

def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_tenant_id_idx")
    op.execute("ALTER TABLE chunks    DROP COLUMN IF EXISTS tenant_id")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS tenant_id")
    op.execute("DROP TABLE IF EXISTS tenants")
