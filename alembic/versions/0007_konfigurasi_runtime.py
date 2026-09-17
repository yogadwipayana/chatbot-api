"""Parameter retrieval dan chunking yang dapat diubah dari dashboard.

Sebelum ini setiap penyetelan ambang atau ukuran potongan berarti menyunting
`.env` di server lalu me-restart layanan -- tidak mungkin dilakukan admin
konten. Tabel ini menyimpan nilai yang ditimpa saja; parameter yang tidak ada
barisnya tetap mengikuti `.env`.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-17
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runtime_config",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("value", sa.String(64), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("updated_by", sa.String(255)),
    )


def downgrade() -> None:
    op.drop_table("runtime_config")
