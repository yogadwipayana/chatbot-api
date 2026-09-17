"""Simpan nama asli PDF untuk tampilan dan unduhan.

`file_path` tetap menjadi key objek berbasis UUID. Nama unggahan disimpan
terpisah agar URL internal tidak berubah, tetapi browser tidak lagi menampilkan
UUID saat PDF dibuka.

Revision ID: 0006
Revises: 0005
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("nama_file", sa.String(255), nullable=True))
    # Nama asli tidak dapat dipulihkan dari objek lama. Judul menjadi fallback
    # yang lebih berguna daripada UUID sampai dokumen tersebut diunggah ulang.
    op.execute(
        "UPDATE documents SET nama_file = CASE "
        "WHEN lower(right(judul, 4)) = '.pdf' THEN judul "
        "ELSE judul || '.pdf' END "
        "WHERE jenis = 'pdf' AND nama_file IS NULL"
    )


def downgrade() -> None:
    op.drop_column("documents", "nama_file")
