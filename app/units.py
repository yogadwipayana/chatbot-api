"""Daftar unit layanan kampus (tabel `units`), isi menu topik chatbot, dan
kontak PIC per unit (tabel `unit_contacts`).

Dibaca dua pihak: menu topik di chatbot mahasiswa, dan setiap isian unit di
dashboard admin. Keduanya harus berpegang pada daftar yang sama -- itulah yang
membuat filter unit di retrieval dapat dipercaya.

Kontak PIC tampil di banner eskalasi dan penolakan (FE-3) untuk unit yang
dipilih mahasiswa. Unit tanpa kontak memakai cadangan di `app.rag.risk`.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.permissions import normalize_unit
from app.db.models import DocumentType
from app.rag.filters import active_document_clause
from app.rag.risk import UnitContact

_PERTANYAAN_SQL = text(
    f"""
    SELECT d.title FROM documents d
    WHERE d.type = :type
      AND {active_document_clause("d")}
      AND (CAST(:unit AS text) IS NULL OR d.unit = CAST(:unit AS text))
    ORDER BY d.updated_at DESC, d.id
    LIMIT :limit
    """
)


@dataclass(frozen=True)
class UnitInfo:
    name: str
    description: str | None = None


@dataclass(frozen=True)
class UnitRecord:
    """Satu baris `units` lengkap, untuk halaman kelola unit di dashboard."""

    name: str
    description: str | None
    sort_order: int
    is_active: bool
    document_count: int = 0
    account_count: int = 0
    """Akun dashboard berunit ini -- menonaktifkan unit tidak mengeluarkan
    mereka, tetapi staf tidak lagi dapat memilih unitnya untuk isian baru."""
    contact_count: int = 0
    """Kontak PIC unit ini. Nol = penolakan dan banner memakai Front Office."""


class DuplicateUnitError(ValueError):
    """Nama unit bentrok dengan unit lain setelah dinormalisasi."""


class DuplicateContactError(ValueError):
    """Label kontak bentrok dengan kontak lain di unit yang sama."""


EDITABLE_FIELDS = ("name", "description", "sort_order", "is_active")

CONTACT_EDITABLE_FIELDS = (
    "label",
    "phone",
    "whatsapp",
    "email",
    "service_hours",
    "sort_order",
    "keywords",
)

CONTACT_CHANNELS = ("phone", "whatsapp", "email")
"""Minimal satu terisi (`ck_unit_contacts_channel`)."""


def susun_kontak(phone: str | None, whatsapp: str | None, email: str | None) -> str:
    """Teks kontak yang tampil di widget dan di teks penolakan.

    Labelnya bukan hiasan: widget membuka nomor yang didahului "WhatsApp" di
    WhatsApp dan nomor lain lewat `tel:` (client `kontak.ts`), dan hanya alamat
    email yang berdiri sendiri yang menjadi tautan `mailto:`.
    """
    bagian = []
    if phone:
        bagian.append(f"Telepon {phone}")
    if whatsapp:
        bagian.append(f"WhatsApp {whatsapp}")
    if email:
        bagian.append(email)
    return " / ".join(bagian)


@dataclass(frozen=True)
class ContactRecord:
    """Satu baris `unit_contacts`."""

    id: uuid.UUID
    unit: str
    label: str
    phone: str | None
    whatsapp: str | None
    email: str | None
    service_hours: str | None
    sort_order: int
    updated_at: datetime | None = None
    updated_by: str | None = None
    keywords: tuple[str, ...] = ()
    """Kosong = selalu tampil untuk unit ini; terisi = hanya bila pertanyaan
    menyebut salah satunya (`app.rag.risk.pilih_kontak`)."""

    def __post_init__(self) -> None:
        # asyncpg mengembalikan list, API mengirim list; tuple supaya tetap
        # dapat di-hash seperti field lain dataclass beku ini.
        object.__setattr__(self, "keywords", tuple(self.keywords or ()))

    @property
    def display(self) -> str:
        return susun_kontak(self.phone, self.whatsapp, self.email)

    def untuk_tampil(self) -> UnitContact:
        """Bentuk yang dipakai pipeline (`PipelineOutcome.contacts`)."""
        return UnitContact(
            unit=self.label,
            jam_layanan=self.service_hours or "",
            kontak=self.display,
            kata_kunci=self.keywords,
        )


def label_bentrok(
    contacts: Iterable[ContactRecord], label: str, kecuali: uuid.UUID | None = None
) -> bool:
    """Sama dengan index unik `ix_unit_contacts_unit_label`: tanpa membedakan huruf besar."""
    kunci = label.casefold()
    return any(k.id != kecuali and k.label.casefold() == kunci for k in contacts)


_SEMUA_SQL = """
    SELECT u.name, u.description, u.sort_order, u.is_active,
           (SELECT count(*) FROM documents d WHERE d.unit = u.name) AS document_count,
           (SELECT count(*) FROM admins a WHERE a.unit = u.name) AS account_count,
           (SELECT count(*) FROM unit_contacts k WHERE k.unit = u.name) AS contact_count
    FROM units u
"""

_KONTAK_SQL = """
    SELECT id, unit, label, phone, whatsapp, email, service_hours, sort_order,
           updated_at, updated_by, keywords
    FROM unit_contacts
"""


def bentrok(
    units: Iterable[UnitInfo | UnitRecord], nama: str, kecuali: str | None = None
) -> str | None:
    """Nama unit lain yang sama dengan `nama` setelah dinormalisasi, atau None.

    "keuangan" di samping "Keuangan" adalah dua baris yang lolos kunci utama
    Postgres, tetapi `cocokkan` hanya akan pernah menemukan salah satunya --
    dokumen di unit satunya lagi tidak dapat dipilih siapa pun.
    """
    kunci = normalize_unit(nama)
    return next(
        (u.name for u in units if u.name != kecuali and normalize_unit(u.name) == kunci),
        None,
    )


def cocokkan(units: Iterable[UnitInfo], nama: str | None) -> str | None:
    """Nama resmi unit yang cocok dengan `nama`, atau None.

    Dicocokkan lewat `normalize_unit` -- "keuangan" dan " Keuangan " sama-sama
    menjadi "Keuangan" -- supaya yang tersimpan dan yang dikirim ke retrieval
    selalu ejaan resmi. Filter retrieval membandingkan persis, tanpa normalisasi.
    """
    kunci = normalize_unit(nama)
    if not kunci:
        return None
    return next((u.name for u in units if normalize_unit(u.name) == kunci), None)


class SqlUnitDirectory:
    """Unit aktif, langsung dari database. Di-override di test."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self) -> list[UnitInfo]:
        rows = await self.session.execute(
            text(
                "SELECT name, description FROM units WHERE is_active ORDER BY sort_order, name"
            )
        )
        return [UnitInfo(r.name, r.description) for r in rows]

    async def pertanyaan(self, unit: str | None, limit: int) -> list[str]:
        """Pertanyaan entri tanya jawab yang sedang berlaku, terbaru lebih dulu.

        Hanya entri yang juga terambil retrieval (filter dokumen aktif yang
        sama), jadi setiap pertanyaan yang ditawarkan di menu pasti punya
        jawaban -- mengkliknya tidak pernah berakhir dengan penolakan.
        """
        rows = await self.session.execute(
            _PERTANYAAN_SQL,
            {"type": DocumentType.TANYA_JAWAB.value, "unit": unit, "limit": limit},
        )
        return list(rows.scalars())

    async def resolve(self, nama: str | None) -> str | None:
        """Tabelnya hanya segelintir baris, jadi dicocokkan di Python, bukan SQL:
        aturannya tetap satu dengan `normalize_unit` yang dipakai pembatasan
        akses staf."""
        return cocokkan(await self.list(), nama)

    # --- Kelola unit (dashboard, khusus superadmin) ---------------------------

    async def semua(self) -> list[UnitRecord]:
        """Termasuk unit nonaktif, beserta pemakaiannya."""
        rows = await self.session.execute(text(_SEMUA_SQL + " ORDER BY u.sort_order, u.name"))
        return [UnitRecord(**r) for r in rows.mappings()]

    async def ambil(self, name: str) -> UnitRecord | None:
        """Persis menurut kunci utama: path API membawa ejaan resmi dari `semua`."""
        sql = text(_SEMUA_SQL + " WHERE u.name = :name")
        row = (await self.session.execute(sql, {"name": name})).mappings().first()
        return UnitRecord(**row) if row else None

    async def buat(
        self, *, name: str, description: str | None, sort_order: int | None
    ) -> UnitRecord:
        if bentrok(await self.semua(), name):
            raise DuplicateUnitError(name)
        try:
            await self.session.execute(
                text(
                    "INSERT INTO units (name, description, sort_order, is_active)"
                    " VALUES (:name, :description,"
                    " COALESCE(:sort_order,"
                    " (SELECT COALESCE(max(sort_order), 0) + 1 FROM units)),"
                    " true)"
                ),
                {"name": name, "description": description, "sort_order": sort_order},
            )
        except IntegrityError:
            # Balapan dua pembuatan dengan nama sama.
            await self.session.rollback()
            raise DuplicateUnitError(name) from None
        await self.session.commit()
        unit = await self.ambil(name)
        assert unit is not None
        return unit

    async def ubah(self, name: str, changes: dict[str, Any]) -> UnitRecord | None:
        """Ganti nama ikut mengalir ke `documents.unit` dan `admins.unit` lewat
        `ON UPDATE CASCADE` -- satu UPDATE di sini cukup.

        `changes` memakai nama kolom (`EDITABLE_FIELDS`), sama dengan field API."""
        kolom = [k for k in EDITABLE_FIELDS if k in changes]
        if not kolom:
            return await self.ambil(name)
        baru = changes.get("name", name)
        if baru != name and bentrok(await self.semua(), baru, kecuali=name):
            raise DuplicateUnitError(baru)
        try:
            row = (
                await self.session.execute(
                    text(
                        f"UPDATE units SET {', '.join(f'{k} = :{k}' for k in kolom)}"
                        " WHERE name = :lama RETURNING name"
                    ),
                    {**{k: changes[k] for k in kolom}, "lama": name},
                )
            ).first()
        except IntegrityError:
            await self.session.rollback()
            raise DuplicateUnitError(baru) from None
        if row is None:
            await self.session.rollback()
            return None
        await self.session.commit()
        return await self.ambil(baru)

    # --- Kontak PIC per unit (dashboard, khusus superadmin) -------------------

    async def kontak(self, unit: str) -> list[ContactRecord]:
        """Kontak `unit` (ejaan resmi) dalam urutan tampil."""
        rows = await self.session.execute(
            text(_KONTAK_SQL + " WHERE unit = :unit ORDER BY sort_order, label"),
            {"unit": unit},
        )
        return [ContactRecord(**r) for r in rows.mappings()]

    async def kontak_tampil(self, unit: str | None) -> tuple[UnitContact, ...]:
        """Kontak PIC unit pilihan mahasiswa untuk pipeline; kosong bila tidak ada.

        Kosong bukan galat: pipeline lalu memakai kontak per topik risiko atau
        Front Office (`app.rag.graph._kontak`)."""
        if not unit:
            return ()
        return tuple(k.untuk_tampil() for k in await self.kontak(unit))

    async def ambil_kontak(self, unit: str, contact_id: uuid.UUID) -> ContactRecord | None:
        sql = text(_KONTAK_SQL + " WHERE unit = :unit AND id = :id")
        hasil = await self.session.execute(sql, {"unit": unit, "id": contact_id})
        row = hasil.mappings().first()
        return ContactRecord(**row) if row else None

    async def tambah_kontak(
        self, unit: str, data: dict[str, Any], by: str | None
    ) -> ContactRecord:
        """`data` memakai nama kolom (`CONTACT_EDITABLE_FIELDS`); `sort_order`
        kosong = paling akhir di unit itu."""
        if label_bentrok(await self.kontak(unit), data["label"]):
            raise DuplicateContactError(data["label"])
        contact_id = uuid.uuid4()
        nilai = {k: data.get(k) for k in CONTACT_EDITABLE_FIELDS}
        nilai["keywords"] = list(nilai["keywords"] or [])
        try:
            # CAST: `:unit` dipakai dua kali, dan asyncpg menolak parameter yang
            # tipenya tersimpul berbeda (varchar untuk kolom, text di subquery).
            await self.session.execute(
                text(
                    "INSERT INTO unit_contacts (id, unit, label, phone, whatsapp, email,"
                    " service_hours, sort_order, keywords, updated_by)"
                    " VALUES (:id, CAST(:unit AS varchar), :label, :phone, :whatsapp, :email,"
                    " :service_hours,"
                    " COALESCE(CAST(:sort_order AS integer),"
                    " (SELECT COALESCE(max(sort_order), 0) + 1 FROM unit_contacts"
                    " WHERE unit = CAST(:unit AS varchar))),"
                    " :keywords, :by)"
                ),
                {**nilai, "id": contact_id, "unit": unit, "by": by},
            )
        except IntegrityError:
            # Balapan dua penambahan dengan label sama.
            await self.session.rollback()
            raise DuplicateContactError(data["label"]) from None
        await self.session.commit()
        kontak = await self.ambil_kontak(unit, contact_id)
        assert kontak is not None
        return kontak

    async def ubah_kontak(
        self, unit: str, contact_id: uuid.UUID, changes: dict[str, Any], by: str | None
    ) -> ContactRecord | None:
        """Hanya kolom di `changes` yang diubah. Pemanggil sudah memastikan minimal
        satu kanal tetap terisi sesudah perubahan."""
        kolom = [k for k in CONTACT_EDITABLE_FIELDS if k in changes]
        if not kolom:
            return await self.ambil_kontak(unit, contact_id)
        if "label" in changes and label_bentrok(
            await self.kontak(unit), changes["label"], kecuali=contact_id
        ):
            raise DuplicateContactError(changes["label"])
        try:
            row = (
                await self.session.execute(
                    text(
                        "UPDATE unit_contacts SET "
                        + ", ".join(f"{k} = :{k}" for k in kolom)
                        + ", updated_at = now(), updated_by = :by"
                        " WHERE unit = :unit AND id = :id RETURNING id"
                    ),
                    {
                        **{k: changes[k] for k in kolom},
                        "by": by,
                        "unit": unit,
                        "id": contact_id,
                    },
                )
            ).first()
        except IntegrityError:
            await self.session.rollback()
            raise DuplicateContactError(changes.get("label", "")) from None
        if row is None:
            await self.session.rollback()
            return None
        await self.session.commit()
        return await self.ambil_kontak(unit, contact_id)

    async def hapus_kontak(self, unit: str, contact_id: uuid.UUID) -> bool:
        row = (
            await self.session.execute(
                text("DELETE FROM unit_contacts WHERE unit = :unit AND id = :id RETURNING id"),
                {"unit": unit, "id": contact_id},
            )
        ).first()
        if row is None:
            await self.session.rollback()
            return False
        await self.session.commit()
        return True
