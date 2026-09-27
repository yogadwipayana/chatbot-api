"""Daftar unit layanan kampus (tabel `units`) dan isi menu topik chatbot.

Dibaca dua pihak: menu topik di chatbot mahasiswa, dan setiap isian unit di
dashboard admin. Keduanya harus berpegang pada daftar yang sama -- itulah yang
membuat filter unit di retrieval dapat dipercaya.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.permissions import normalize_unit
from app.db.models import DocumentType
from app.rag.filters import active_document_clause

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


class DuplicateUnitError(ValueError):
    """Nama unit bentrok dengan unit lain setelah dinormalisasi."""


EDITABLE_FIELDS = ("name", "description", "sort_order", "is_active")

_SEMUA_SQL = """
    SELECT u.name, u.description, u.sort_order, u.is_active,
           (SELECT count(*) FROM documents d WHERE d.unit = u.name) AS document_count,
           (SELECT count(*) FROM admins a WHERE a.unit = u.name) AS account_count
    FROM units u
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
