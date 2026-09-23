"""Daftar unit layanan kampus (tabel `units`) dan isi menu topik chatbot.

Dibaca dua pihak: menu topik di chatbot mahasiswa, dan setiap isian unit di
dashboard admin. Keduanya harus berpegang pada daftar yang sama -- itulah yang
membuat filter unit di retrieval dapat dipercaya.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import text
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
