"""Entri tanya jawab: sumber jawaban tanpa berkas PDF.

Admin sering hanya perlu menuliskan satu pertanyaan beserta jawabannya --
memaksanya membuat PDF lebih dulu membuat perbaikan AD-4 tertunda berhari-hari.

Entri tanya jawab tinggal di tabel `documents` yang sama, bukan tabel sendiri,
supaya seluruh mesin yang sudah ada berlaku tanpa perubahan: filter dokumen
aktif (FR-2), masa berlaku, unit, chunk, embedding, dan sitasi. Yang membedakan
hanya `jenis`:

- `pdf`: `file_path` menunjuk berkas di penyimpanan objek, `jawaban` NULL.
- `tanya_jawab`: tanpa berkas, `judul` berisi pertanyaannya dan `jawaban`
  berisi jawabannya apa adanya seperti diketik admin.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-16
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("jenis", sa.String(20), nullable=False, server_default="pdf"),
    )
    op.add_column("documents", sa.Column("jawaban", sa.Text()))
    # Seluruh baris yang sudah ada adalah PDF, jadi kolomnya tetap terisi;
    # NOT NULL dilepas hanya untuk baris tanya jawab yang memang tak berberkas.
    op.alter_column("documents", "file_path", nullable=True)

    op.create_check_constraint(
        "ck_documents_jenis", "documents", "jenis IN ('pdf', 'tanya_jawab')"
    )
    op.create_check_constraint(
        "ck_documents_isi_sesuai_jenis",
        "documents",
        "(jenis = 'pdf' AND file_path IS NOT NULL AND jawaban IS NULL)"
        " OR (jenis = 'tanya_jawab' AND file_path IS NULL AND jawaban IS NOT NULL)",
    )


def downgrade() -> None:
    # Baris tanya jawab tidak punya berkas, jadi tidak ada cara mengubahnya
    # menjadi dokumen PDF yang sah: hapus beserta chunk-nya (ON DELETE CASCADE).
    op.execute("DELETE FROM documents WHERE jenis = 'tanya_jawab'")
    op.drop_constraint("ck_documents_isi_sesuai_jenis", "documents", type_="check")
    op.drop_constraint("ck_documents_jenis", "documents", type_="check")
    op.alter_column("documents", "file_path", nullable=False)
    op.drop_column("documents", "jawaban")
    op.drop_column("documents", "jenis")
