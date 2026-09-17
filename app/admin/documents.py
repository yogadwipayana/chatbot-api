"""Query dokumen untuk dashboard admin (AD-2, AD-3)."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.permissions import normalize_unit
from app.db.models import JenisDokumen

STALE_AFTER_MONTHS = 6

UNIT_MATCH = r"lower(regexp_replace(btrim(d.unit), '\s+', ' ', 'g')) = :unit"
"""Sepadan dengan `permissions.normalize_unit`: huruf kecil, spasi dirapikan."""

PDF_SAJA = "d.jenis = :jenis"
"""Tabel `documents` juga menampung entri tanya jawab (lihat `admin.faq`)."""


def stale_clause(alias: str = "d") -> str:
    """Predikat 'perlu ditinjau' (AD-2).

    Lebih dari 6 bulan tidak diperbarui, atau masa berlakunya sudah lewat.
    Batas masa berlaku memakai perbandingan yang persis berlawanan dengan
    `filters.active_document_clause`, sehingga dokumen yang diberi badge
    kedaluwarsa di sini tepat dokumen yang sudah berhenti terambil retrieval.
    """
    if not alias.isidentifier():
        raise ValueError(f"alias tabel tidak valid: {alias!r}")
    return (
        f"({alias}.updated_at < now() - interval '{STALE_AFTER_MONTHS} months' "
        f"OR ({alias}.valid_until IS NOT NULL AND {alias}.valid_until <= now()))"
    )


_STALE = stale_clause("d")

_KOLOM = f"""
    d.id::text AS id, d.judul, d.unit, d.jenis, d.tahun_berlaku, d.valid_until, d.updated_at,
    d.is_active, d.uploaded_by, d.file_path,
    (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS jumlah_chunk,
    {_STALE} AS stale
"""

EDITABLE_FIELDS = ("judul", "unit", "tahun_berlaku", "valid_until", "is_active")

CONTENT_FIELDS = frozenset({"judul", "unit", "tahun_berlaku", "valid_until"})
"""Mengubah salah satunya memperbarui `updated_at`. `is_active` tidak: menyalakan
ulang dokumen lama tidak boleh menghapus badge 'perlu ditinjau'-nya."""


async def list_documents(
    session: AsyncSession,
    *,
    include_inactive: bool,
    only_stale: bool,
    limit: int,
    offset: int,
    unit: str | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return: (baris, total sesuai filter, jumlah dokumen aktif yang perlu ditinjau).

    `unit` membatasi seluruh angka -- termasuk `jumlah_stale` -- pada satu unit;
    dipakai untuk staf/dosen. None berarti semua unit.

    Entri tanya jawab tidak ikut: ia berbagi tabel ini, tetapi dikelola di menu
    sendiri dan tidak punya berkas yang bisa dibuka di halaman dokumen.
    """
    lingkup: list[str] = [PDF_SAJA]
    params: dict[str, Any] = {"jenis": JenisDokumen.PDF.value}
    if unit is not None:
        lingkup.append(UNIT_MATCH)
        params["unit"] = normalize_unit(unit)

    kondisi = list(lingkup)
    if not include_inactive:
        kondisi.append("d.is_active")
    if only_stale:
        kondisi.append(_STALE)
    where = ("WHERE " + " AND ".join(kondisi)) if kondisi else ""

    rows = await session.execute(
        text(
            f"SELECT {_KOLOM} FROM documents d {where}"
            " ORDER BY d.is_active DESC, stale DESC, d.updated_at DESC, d.id"
            " LIMIT :limit OFFSET :offset"
        ),
        {**params, "limit": limit, "offset": offset},
    )
    total = (
        await session.execute(text(f"SELECT count(*) FROM documents d {where}"), params)
    ).scalar_one()
    where_stale = " AND ".join(["d.is_active", _STALE, *lingkup])
    jumlah_stale = (
        await session.execute(
            text(f"SELECT count(*) FROM documents d WHERE {where_stale}"), params
        )
    ).scalar_one()
    return [dict(r) for r in rows.mappings()], total, jumlah_stale


async def get_document(session: AsyncSession, document_id: uuid.UUID) -> dict[str, Any] | None:
    """Hanya dokumen PDF. Id entri tanya jawab dijawab None, sehingga router
    menganggapnya tidak ada -- bukan menampilkan baris tanpa berkas."""
    row = (
        (
            await session.execute(
                text(f"SELECT {_KOLOM} FROM documents d WHERE d.id = :id AND {PDF_SAJA}"),
                {"id": document_id, "jenis": JenisDokumen.PDF.value},
            )
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


async def update_document(
    session: AsyncSession, document_id: uuid.UUID, changes: dict[str, Any]
) -> dict[str, Any] | None:
    """Terapkan perubahan metadata. Return: dokumen terbaru, atau None bila tidak ada.

    Perubahan `is_active` dan `valid_until` langsung berlaku pada retrieval
    berikutnya: filter dokumen aktif dievaluasi di WHERE setiap query (FR-2).
    """
    kolom = [k for k in EDITABLE_FIELDS if k in changes]
    if not kolom:
        return await get_document(session, document_id)

    set_clause = [f"{k} = :{k}" for k in kolom]
    if CONTENT_FIELDS.intersection(kolom):
        set_clause.append("updated_at = now()")

    hasil = await session.execute(
        text(
            f"UPDATE documents d SET {', '.join(set_clause)}"
            f" WHERE d.id = :id AND {PDF_SAJA} RETURNING d.id"
        ),
        {**{k: changes[k] for k in kolom}, "id": document_id, "jenis": JenisDokumen.PDF.value},
    )
    if hasil.first() is None:
        await session.rollback()
        return None
    await session.commit()
    return await get_document(session, document_id)


async def delete_document(session: AsyncSession, document_id: uuid.UUID) -> str | None:
    """Hapus dokumen beserta chunk-nya (ON DELETE CASCADE). Return: kunci objek berkasnya."""
    row = (
        await session.execute(
            text(
                f"DELETE FROM documents d WHERE d.id = :id AND {PDF_SAJA}"
                " RETURNING file_path"
            ),
            {"id": document_id, "jenis": JenisDokumen.PDF.value},
        )
    ).first()
    if row is None:
        await session.rollback()
        return None
    await session.commit()
    return row.file_path


async def document_exists(session: AsyncSession, document_id: uuid.UUID) -> bool:
    return bool(
        (
            await session.execute(
                text("SELECT 1 FROM documents WHERE id = :id"), {"id": document_id}
            )
        ).scalar()
    )


async def list_chunks(
    session: AsyncSession, document_id: uuid.UUID, *, limit: int, offset: int
) -> list[dict[str, Any]]:
    rows = await session.execute(
        text(
            "SELECT id::text AS id, konten, halaman, urutan FROM chunks"
            " WHERE document_id = :id ORDER BY urutan LIMIT :limit OFFSET :offset"
        ),
        {"id": document_id, "limit": limit, "offset": offset},
    )
    return [dict(r) for r in rows.mappings()]
