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
from app.db.models import JenisDokumen
from app.rag.filters import active_document_clause

_PERTANYAAN_SQL = text(
    f"""
    SELECT d.judul FROM documents d
    WHERE d.jenis = :jenis
      AND {active_document_clause("d")}
      AND (CAST(:unit AS text) IS NULL OR d.unit = CAST(:unit AS text))
    ORDER BY d.updated_at DESC, d.id
    LIMIT :limit
    """
)


@dataclass(frozen=True)
class UnitInfo:
    nama: str
    deskripsi: str | None = None


@dataclass(frozen=True)
class UnitRecord:
    """Satu baris `units` lengkap, untuk halaman kelola unit di dashboard."""

    nama: str
    deskripsi: str | None
    urutan: int
    is_active: bool
    jumlah_dokumen: int = 0
    jumlah_akun: int = 0
    """Akun dashboard berunit ini -- menonaktifkan unit tidak mengeluarkan
    mereka, tetapi staf tidak lagi dapat memilih unitnya untuk isian baru."""


class DuplicateUnitError(ValueError):
    """Nama unit bentrok dengan unit lain setelah dinormalisasi."""


EDITABLE_FIELDS = ("nama", "deskripsi", "urutan", "is_active")

_SEMUA_SQL = """
    SELECT u.nama, u.deskripsi, u.urutan, u.is_active,
           (SELECT count(*) FROM documents d WHERE d.unit = u.nama) AS jumlah_dokumen,
           (SELECT count(*) FROM admins a WHERE a.unit = u.nama) AS jumlah_akun
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
        (u.nama for u in units if u.nama != kecuali and normalize_unit(u.nama) == kunci),
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
    return next((u.nama for u in units if normalize_unit(u.nama) == kunci), None)


class SqlUnitDirectory:
    """Unit aktif, langsung dari database. Di-override di test."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self) -> list[UnitInfo]:
        rows = await self.session.execute(
            text("SELECT nama, deskripsi FROM units WHERE is_active ORDER BY urutan, nama")
        )
        return [UnitInfo(r.nama, r.deskripsi) for r in rows]

    async def pertanyaan(self, unit: str | None, limit: int) -> list[str]:
        """Pertanyaan entri tanya jawab yang sedang berlaku, terbaru lebih dulu.

        Hanya entri yang juga terambil retrieval (filter dokumen aktif yang
        sama), jadi setiap pertanyaan yang ditawarkan di menu pasti punya
        jawaban -- mengkliknya tidak pernah berakhir dengan penolakan.
        """
        rows = await self.session.execute(
            _PERTANYAAN_SQL,
            {"jenis": JenisDokumen.TANYA_JAWAB.value, "unit": unit, "limit": limit},
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
        rows = await self.session.execute(text(_SEMUA_SQL + " ORDER BY u.urutan, u.nama"))
        return [UnitRecord(**r) for r in rows.mappings()]

    async def ambil(self, nama: str) -> UnitRecord | None:
        """Persis menurut kunci utama: path API membawa ejaan resmi dari `semua`."""
        sql = text(_SEMUA_SQL + " WHERE u.nama = :nama")
        row = (await self.session.execute(sql, {"nama": nama})).mappings().first()
        return UnitRecord(**row) if row else None

    async def buat(
        self, *, nama: str, deskripsi: str | None, urutan: int | None
    ) -> UnitRecord:
        if bentrok(await self.semua(), nama):
            raise DuplicateUnitError(nama)
        try:
            await self.session.execute(
                text(
                    "INSERT INTO units (nama, deskripsi, urutan, is_active)"
                    " VALUES (:nama, :deskripsi,"
                    " COALESCE(:urutan, (SELECT COALESCE(max(urutan), 0) + 1 FROM units)),"
                    " true)"
                ),
                {"nama": nama, "deskripsi": deskripsi, "urutan": urutan},
            )
        except IntegrityError:
            # Balapan dua pembuatan dengan nama sama.
            await self.session.rollback()
            raise DuplicateUnitError(nama) from None
        await self.session.commit()
        unit = await self.ambil(nama)
        assert unit is not None
        return unit

    async def ubah(self, nama: str, changes: dict[str, Any]) -> UnitRecord | None:
        """Ganti nama ikut mengalir ke `documents.unit` dan `admins.unit` lewat
        `ON UPDATE CASCADE` -- satu UPDATE di sini cukup."""
        kolom = [k for k in EDITABLE_FIELDS if k in changes]
        if not kolom:
            return await self.ambil(nama)
        baru = changes.get("nama", nama)
        if baru != nama and bentrok(await self.semua(), baru, kecuali=nama):
            raise DuplicateUnitError(baru)
        try:
            row = (
                await self.session.execute(
                    text(
                        f"UPDATE units SET {', '.join(f'{k} = :{k}' for k in kolom)}"
                        " WHERE nama = :lama RETURNING nama"
                    ),
                    {**{k: changes[k] for k in kolom}, "lama": nama},
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
