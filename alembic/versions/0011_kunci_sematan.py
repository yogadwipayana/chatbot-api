"""Tabel `embed_keys`: satu kunci untuk setiap situs yang memasang asisten.

Sebelumnya situs yang boleh memasang asisten diatur `EMBED_ALLOWED_ORIGINS` di
portal (`client/`), yang dibaca saat build: menambah satu situs berarti build
ulang portal, dan tidak ada cara mencabut satu situs tanpa menyentuh yang lain.
Kini setiap situs membawa kuncinya sendiri di tag `<script>`, dan portal
menanyakan kunci itu ke API setiap kali panel dibuka.

Kunci BUKAN rahasia -- ia tertulis di kode sumber situs penyemat. Karena itu
disimpan apa adanya, tanpa hash. Yang benar-benar membatasi pemakaiannya adalah
`asal_diizinkan` (ditegakkan peramban lewat CSP `frame-ancestors`) dan
`is_active`.

`conversations.embed_key` mencatat dari situs mana sebuah percakapan datang,
untuk jumlah pertanyaan per situs di dashboard. Menghapus kunci tidak menghapus
percakapannya, hanya melepas penandanya.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-26
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "embed_keys",
        sa.Column("kunci", sa.String(64), primary_key=True),
        sa.Column("nama", sa.String(200), nullable=False),
        sa.Column(
            "asal_diizinkan",
            postgresql.ARRAY(sa.String(255)),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("dibuat_oleh", sa.String(255)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.add_column(
        "conversations",
        sa.Column(
            "embed_key",
            sa.String(64),
            sa.ForeignKey(
                "embed_keys.kunci", name="fk_conversations_embed_key", ondelete="SET NULL"
            ),
        ),
    )
    # Jumlah pertanyaan per kunci di halaman Sematan menyaring percakapan
    # menurut kolom ini; tanpa index ia menjadi sequential scan seluruh log.
    op.create_index("ix_conversations_embed_key", "conversations", ["embed_key"])


def downgrade() -> None:
    op.drop_index("ix_conversations_embed_key", table_name="conversations")
    op.drop_column("conversations", "embed_key")
    op.drop_table("embed_keys")
