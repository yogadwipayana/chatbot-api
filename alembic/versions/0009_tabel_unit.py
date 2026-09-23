"""Tabel `units`: daftar tetap unit layanan, dirujuk dokumen dan akun staf.

Mahasiswa akan memilih unit di menu chatbot, lalu retrieval hanya mencari di
dokumen unit itu. Selama `documents.unit` diketik bebas, filter itu tidak dapat
dipercaya: dokumen berlabel "Bagian Keuangan" tidak akan pernah terambil untuk
pilihan "Keuangan", dan mahasiswa menerima penolakan padahal jawabannya ada.

Memecah `chunks` menjadi satu tabel per unit sempat dipertimbangkan dan
ditolak: pertanyaan lintas unit (cuti akademik menyangkut BAAK dan Keuangan)
menjadi UNION, index HNSW dan GIN berlipat sembilan, dan memindah unit sebuah
dokumen berarti memindah seluruh chunk-nya antar-tabel.

Nilai lama dipetakan ke nama baru lewat `ALIAS`. Nilai yang tidak dikenali
menggagalkan migrasi alih-alih ditebak: dokumen yang diam-diam masuk unit yang
salah justru hilang dari menu unit yang benar.

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-22
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

UNIT_AWAL: tuple[tuple[str, str | None], ...] = (
    ("BAAK", None),
    ("FO", "Front Office"),
    ("Keuangan", None),
    ("Kemahasiswaan", None),
    ("Prodi", None),
    ("Fakultas", None),
    ("PLK", None),
    ("UPS", "Unit Pelayanan Sertifikasi"),
    ("Akademik", None),
)
"""(nama, deskripsi), dalam urutan tampil di menu."""

ALIAS = {
    "biro administrasi akademik": "BAAK",
    "biro administrasi akademik dan kemahasiswaan": "BAAK",
    "bagian keuangan": "Keuangan",
    "biro keuangan": "Keuangan",
    "bagian kemahasiswaan": "Kemahasiswaan",
    "front office": "FO",
    "unit pelayanan sertifikasi": "UPS",
}
"""Nama yang pernah diketik bebas -> nama resmi. Kuncinya sudah dinormalisasi
seperti `permissions.normalize_unit`."""


def _normal(nama: str) -> str:
    return " ".join(nama.split()).casefold()


def upgrade() -> None:
    units = op.create_table(
        "units",
        sa.Column("nama", sa.String(200), primary_key=True),
        sa.Column("deskripsi", sa.String(500)),
        sa.Column("urutan", sa.Integer(), server_default="0", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
    )
    op.bulk_insert(
        units,
        [
            {"nama": nama, "deskripsi": deskripsi, "urutan": i}
            for i, (nama, deskripsi) in enumerate(UNIT_AWAL, start=1)
        ],
    )

    peta = {_normal(nama): nama for nama, _ in UNIT_AWAL} | ALIAS
    conn = op.get_bind()
    lama = conn.execute(
        sa.text(
            "SELECT unit FROM documents UNION SELECT unit FROM admins WHERE unit IS NOT NULL"
        )
    ).scalars()
    tak_dikenal = []
    for nilai in lama:
        baru = peta.get(_normal(nilai))
        if baru is None:
            tak_dikenal.append(nilai)
            continue
        if baru != nilai:
            for tabel in ("documents", "admins"):
                conn.execute(
                    sa.text(f"UPDATE {tabel} SET unit = :baru WHERE unit = :lama"),
                    {"baru": baru, "lama": nilai},
                )
    if tak_dikenal:
        raise RuntimeError(
            "Nilai unit berikut tidak dapat dipetakan ke unit resmi: "
            + ", ".join(repr(n) for n in sorted(tak_dikenal))
            + ". Ubah dulu ke salah satu dari "
            + ", ".join(nama for nama, _ in UNIT_AWAL)
            + " (UPDATE documents/admins SET unit = ...), atau tambahkan ke ALIAS"
            " di migrasi ini, lalu jalankan ulang."
        )

    op.create_foreign_key(
        "fk_documents_unit_units",
        "documents",
        "units",
        ["unit"],
        ["nama"],
        onupdate="CASCADE",
    )
    op.create_foreign_key(
        "fk_admins_unit_units",
        "admins",
        "units",
        ["unit"],
        ["nama"],
        onupdate="CASCADE",
    )
    # Foreign key di Postgres tidak otomatis ber-index. Tanpa ini, filter unit
    # di retrieval dan daftar dokumen staf menjadi sequential scan.
    op.create_index("ix_documents_unit", "documents", ["unit"])


def downgrade() -> None:
    # Nama unit yang sudah dipetakan (mis. "Bagian Keuangan" -> "Keuangan")
    # TIDAK dikembalikan: nilai lamanya tidak disimpan di mana pun.
    op.drop_index("ix_documents_unit", table_name="documents")
    op.drop_constraint("fk_admins_unit_units", "admins", type_="foreignkey")
    op.drop_constraint("fk_documents_unit_units", "documents", type_="foreignkey")
    op.drop_table("units")
