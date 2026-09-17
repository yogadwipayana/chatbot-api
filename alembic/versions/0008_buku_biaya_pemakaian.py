"""Buku biaya panggilan model di luar percakapan.

Biaya chat menumpang `messages.meta`, tetapi embedding saat ingestion terjadi
ketika tidak ada mahasiswa yang bertanya -- tidak ada baris pesan yang bisa
dititipi. Tanpa tabel ini halaman Biaya AD-5 diam-diam hanya melaporkan
sebagian dari yang benar-benar dibelanjakan.

Kolom di `documents` sempat dipertimbangkan dan ditolak: menghapus dokumen akan
mengecilkan total bulan yang sudah lewat, dan menyunting entri tanya jawab akan
menimpa biaya ingestion pertamanya. Buku biaya yang berubah surut tidak dapat
menjawab "bulan lalu habis berapa". `document_id` karena itu memakai SET NULL,
alasan yang sama dengan `unanswered.message_id` di 0003.

`biaya_usd` sengaja tidak punya batas presisi: satu panggilan embedding bisa
berharga $0,00000018, dan pembulatan enam desimal menjadikannya nol bulat.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-17
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
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
        # Angka biaya tanpa asal-usul tidak dapat ditafsirkan, dan asal-usul
        # tanpa angka tidak ada artinya. Keduanya ada, atau keduanya tidak ada.
        sa.CheckConstraint(
            "(biaya_usd IS NULL) = (biaya_sumber IS NULL)",
            name="ck_usage_log_biaya_lengkap",
        ),
    )
    op.create_index("ix_usage_log_created_at", "usage_log", ["created_at"])


def downgrade() -> None:
    # Seluruh riwayat biaya ingestion ikut hilang: angkanya tidak tersimpan di
    # tempat lain dan tidak dapat dihitung ulang dari `documents`.
    op.drop_index("ix_usage_log_created_at", table_name="usage_log")
    op.drop_table("usage_log")
