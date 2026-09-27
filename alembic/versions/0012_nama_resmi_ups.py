"""Deskripsi unit UPS memakai nama resminya: Unit Pelaksana Sertifikasi.

Migrasi 0009 mengisinya "Unit Pelayanan Sertifikasi", sedangkan FAQ resmi
kampus (instiki.ac.id/faq) menyebut unit itu Unit Pelaksana Sertifikasi (UPS),
pengelola ujian sertifikasi kompetensi mahasiswa. Deskripsi ini tampil di menu
unit chatbot, jadi mahasiswa membacanya langsung.

Hanya baris yang masih berisi teks lama yang diubah: deskripsi yang sudah
disunting admin lewat dashboard dibiarkan.

Revision ID: 0012
Revises: 0011
Create Date: 2026-09-26
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None

LAMA = "Unit Pelayanan Sertifikasi"
BARU = "Unit Pelaksana Sertifikasi"


def _ganti(dari: str, ke: str) -> None:
    op.get_bind().execute(
        sa.text("UPDATE units SET deskripsi = :ke WHERE nama = 'UPS' AND deskripsi = :dari"),
        {"dari": dari, "ke": ke},
    )


def upgrade() -> None:
    _ganti(LAMA, BARU)


def downgrade() -> None:
    _ganti(BARU, LAMA)
