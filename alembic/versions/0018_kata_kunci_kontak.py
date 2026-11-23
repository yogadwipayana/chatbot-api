"""Kolom `unit_contacts.keywords`: kontak yang hanya tampil untuk urusannya.

Unit Akademik punya dua kontak sejak 0017, PIC BAAK dan Admin PLK, dan
keduanya tampil di setiap penolakan -- termasuk pertanyaan surat-menyurat yang
tidak ada hubungannya dengan PLK, sehingga mahasiswa bisa salah kontak
(2026-11-23). BAAK adalah pintu pertama urusan akademik; nomor PLK cukup
muncul bila pertanyaannya menyangkut PLK.

Kontak tanpa kata kunci tetap selalu tampil. Yang berkata kunci hanya tampil
bila pertanyaan menyebut salah satunya (`app.rag.risk.pilih_kontak`).

Isi: Admin PLK di unit SELAIN unit PLK sendiri diberi kata kunci PLK. Dicocokkan
lewat label (tanpa membedakan huruf besar), bukan nama unit: unit Akademik
pernah bernama BAAK, dan di unit PLK nomor itu memang kontak utamanya.

Tambah kolom saja dengan default kosong: image lama yang belum mengenal kolom
ini tetap jalan dan menampilkan semua kontak seperti sebelumnya.

Revision ID: 0018
Revises: 0017
Create Date: 2026-11-23
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None

LABEL_PLK = "Admin PLK INSTIKI"
UNIT_PLK = "PLK"
KATA_KUNCI_PLK = ("PLK", "pembelajaran di luar kampus", "pembelajaran luar kampus")
"""Singkatan dan kepanjangannya (`app/rag/glossary.py`), dengan dan tanpa "di"."""


def upgrade() -> None:
    op.add_column(
        "unit_contacts",
        sa.Column(
            "keywords",
            postgresql.ARRAY(sa.String(100)),
            server_default="{}",
            nullable=False,
        ),
    )
    op.get_bind().execute(
        sa.text(
            "UPDATE unit_contacts SET keywords = :kata_kunci,"
            " updated_at = now(), updated_by = 'migrasi 0018'"
            " WHERE lower(label) = lower(:label) AND lower(unit) <> lower(:unit_plk)"
        ).bindparams(
            sa.bindparam("kata_kunci", type_=postgresql.ARRAY(sa.String(100))),
        ),
        {"kata_kunci": list(KATA_KUNCI_PLK), "label": LABEL_PLK, "unit_plk": UNIT_PLK},
    )


def downgrade() -> None:
    op.drop_column("unit_contacts", "keywords")
