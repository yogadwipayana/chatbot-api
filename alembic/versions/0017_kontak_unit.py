"""Tabel `unit_contacts`: kontak PIC per unit, dikelola dari dashboard.

Sebelumnya PIC ditulis di kode (`UNIT_CONTACTS` di app/rag/risk.py) dengan
kunci nama unit. Mengganti PIC berarti deploy ulang, dan mengganti nama unit
dari dashboard diam-diam memutus kontaknya (T69, 2026-11-18). Di sini kontak
merujuk `units.name` dengan ON UPDATE CASCADE, jadi ikut berganti nama.

Isi awal adalah salinan `UNIT_CONTACTS` terakhir (daftar "Kontak PIC per
bidang", 2026-10-12), dipasang ke setiap unit yang namanya -- atau nama lamanya
di `UNIT_ALIASES` -- cocok tanpa membedakan huruf besar, persis seperti
`contacts_for_unit` dulu. Salinan, bukan impor: konstanta itu dihapus dari kode
bersama migrasi ini.

Tambah tabel saja: image lama yang belum mengenal revisi ini tetap jalan
(ia masih membaca PIC dari kodenya sendiri), dan `app.db.migrasi` melewatinya.

Downgrade membuang tabel beserta perubahan yang dibuat dari dashboard.

Revision ID: 0017
Revises: 0016
Create Date: 2026-11-23
"""

from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None

# (label, telepon, WhatsApp, jam layanan)
_Kontak = tuple[str, str | None, str | None, str | None]

_PIC_BAAK: _Kontak = ("PIC BAAK", None, "+62 813-5346-2032", None)
_PIC_PLK: _Kontak = ("Admin PLK INSTIKI", None, "+62 851-1931-5346", None)

ISI_AWAL: dict[str, tuple[_Kontak, ...]] = {
    "AKADEMIK": (_PIC_BAAK, _PIC_PLK),
    "KEMAHASISWAAN": (
        ("PIC Kemahasiswaan", None, "+62 813-3936-5103", None),
        ("Bidang Pengembangan Karakter Mahasiswa (Ibu Asti)", None, "+62 812-3673-3230", None),
        ("Bidang Prestasi Mahasiswa (Pak Redy)", None, "+62 819-9843-9707", None),
        ("Beasiswa (Ibu Suci Ariantini)", None, "+62 812-3608-8993", None),
    ),
    "UNIT PELAYANAN SERTIFIKASI": (
        ("Layanan Pendidikan (UPS)", None, "+62 813-3734-8654", None),
    ),
    "PLK": (_PIC_PLK,),
    "KEUANGAN": (("Keuangan INSTIKI", None, "+62 851-7426-1765", None),),
    "FRONT OFFICE": (
        (
            "Front Office INSTIKI",
            "(0361) 256995",
            "+62 813-3982-2383",
            "Senin-Sabtu, 08.00-22.00 WITA",
        ),
    ),
}
"""Salinan `UNIT_CONTACTS` per 2026-11-18."""

ALIAS: dict[str, str] = {
    "BAAK": "AKADEMIK",
    "UPS": "UNIT PELAYANAN SERTIFIKASI",
    "FO": "FRONT OFFICE",
}
"""Salinan `UNIT_ALIASES`: database yang belum mengganti nama unitnya tetap terisi."""


def _isi_untuk(nama: str) -> tuple[_Kontak, ...]:
    kunci = nama.strip().casefold()
    kunci = next((b.casefold() for a, b in ALIAS.items() if a.casefold() == kunci), kunci)
    return next((isi for k, isi in ISI_AWAL.items() if k.casefold() == kunci), ())


def upgrade() -> None:
    tabel = op.create_table(
        "unit_contacts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "unit",
            sa.String(200),
            sa.ForeignKey("units.name", onupdate="CASCADE", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("label", sa.String(200), nullable=False),
        sa.Column("phone", sa.String(40)),
        sa.Column("whatsapp", sa.String(40)),
        sa.Column("email", sa.String(255)),
        sa.Column("service_hours", sa.String(200)),
        sa.Column("sort_order", sa.Integer, server_default="0", nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("updated_by", sa.String(255)),
        sa.CheckConstraint(
            "phone IS NOT NULL OR whatsapp IS NOT NULL OR email IS NOT NULL",
            name="ck_unit_contacts_channel",
        ),
    )
    op.create_index(
        "ix_unit_contacts_unit_label",
        "unit_contacts",
        ["unit", sa.text("lower(label)")],
        unique=True,
    )

    units = op.get_bind().execute(sa.text("SELECT name FROM units")).scalars().all()
    baris = [
        {
            "id": uuid.uuid4(),
            "unit": unit,
            "label": label,
            "phone": phone,
            "whatsapp": whatsapp,
            "email": None,
            "service_hours": jam,
            "sort_order": urutan,
            "updated_by": "migrasi 0017",
        }
        for unit in units
        for urutan, (label, phone, whatsapp, jam) in enumerate(_isi_untuk(unit), start=1)
    ]
    if baris:
        op.bulk_insert(tabel, baris)


def downgrade() -> None:
    op.drop_index("ix_unit_contacts_unit_label", table_name="unit_contacts")
    op.drop_table("unit_contacts")
