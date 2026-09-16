"""Selaraskan NOT NULL dengan models.py.

`alembic check` setelah migrasi 0001 menemukan tujuh kolom yang NOT NULL di
model tetapi nullable di database. Model yang benar:

- `created_at` / `updated_at` punya server default `now()`; NULL di sana hanya
  mungkin lewat INSERT yang salah dan merusak urutan log serta statistik AD-5.
- `chunks.embedding` tanpa nilai membuat chunk tidak pernah terambil vector
  search -- dokumen tampak terindeks padahal separuh retrieval tidak melihatnya.

Aman dijalankan selama tabel belum berisi NULL pada kolom tersebut.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-15
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

KOLOM_WAKTU = (
    ("admins", "created_at"),
    ("conversations", "created_at"),
    ("documents", "updated_at"),
    ("feedback", "created_at"),
    ("messages", "created_at"),
    ("unanswered", "created_at"),
)


def upgrade() -> None:
    for tabel, kolom in KOLOM_WAKTU:
        op.alter_column(
            tabel,
            kolom,
            existing_type=sa.DateTime(timezone=True),
            existing_server_default=sa.func.now(),
            nullable=False,
        )
    # Tipe `vector` tidak dikenal SQLAlchemy inti; pakai SQL langsung.
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding SET NOT NULL")


def downgrade() -> None:
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding DROP NOT NULL")
    for tabel, kolom in KOLOM_WAKTU:
        op.alter_column(
            tabel,
            kolom,
            existing_type=sa.DateTime(timezone=True),
            existing_server_default=sa.func.now(),
            nullable=True,
        )
