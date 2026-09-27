"""Hapus `usage_log`: embedding kini model self-hosted tanpa tarif per token.

0008 membuat buku biaya untuk embedding saat ingestion, dengan anggapan setiap
panggilan berbayar dan tidak punya baris pesan untuk ditumpangi. Anggapan itu
gugur sejak `EMBED_MODEL` menjadi `intfloat/multilingual-e5-small` di gateway
sendiri: tarifnya nol (`costs.PRICES_PER_MTOK`), dan ongkos servernya biaya
tetap yang tidak dapat dibagi per panggilan. Tabelnya pun tidak pernah punya
penulis -- `app/ingestion/` tidak pernah mengisinya.

Bila kelak kembali ke embedding berbayar, mengganti `EMBED_MODEL` sudah berarti
re-index seluruh dokumen; membuat ulang tabel ini menjadi bagian dari pekerjaan
itu.

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-26
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_usage_log_created_at", table_name="usage_log")
    op.drop_table("usage_log")


def downgrade() -> None:
    # Hanya strukturnya yang kembali, sama persis dengan 0008. Barisnya tidak:
    # tabel ini tidak pernah punya penulis, jadi memang tidak ada isi yang hilang.
    op.create_table(
        "usage_log",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("operasi", sa.String(20), nullable=False),
        sa.Column("model", sa.String(200), nullable=False),
        sa.Column("model_dilaporkan", sa.String(200)),
        sa.Column("tokens", sa.Integer()),
        sa.Column("biaya_usd", sa.Float()),
        sa.Column("biaya_sumber", sa.String(20)),
        sa.Column("is_byok", sa.Boolean()),
        sa.Column(
            "document_id",
            sa.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
        ),
        sa.Column("keterangan", sa.String(500)),
        sa.CheckConstraint("operasi IN ('ingest', 'reindex')", name="ck_usage_log_operasi"),
        sa.CheckConstraint(
            "biaya_sumber IS NULL OR biaya_sumber IN ('provider', 'estimasi')",
            name="ck_usage_log_biaya_sumber_nilai",
        ),
        sa.CheckConstraint(
            "(biaya_usd IS NULL) = (biaya_sumber IS NULL)",
            name="ck_usage_log_biaya_lengkap",
        ),
    )
    op.create_index("ix_usage_log_created_at", "usage_log", ["created_at"])
