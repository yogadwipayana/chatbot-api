"""Entri tanya jawab: sumber jawaban yang diketik langsung, tanpa PDF.

Satu entri adalah satu baris `documents` berjenis `tanya_jawab` (lihat
`JenisDokumen`): `judul` menyimpan pertanyaannya, `jawaban` menyimpan
jawabannya, dan chunk-nya dibangkitkan dari keduanya. Karena tabelnya sama,
entri tanya jawab ikut terambil retrieval hibrida, ikut tunduk pada filter
dokumen aktif dan masa berlaku (FR-2), dan ikut menjadi kartu sitasi -- tanpa
satu baris pun perubahan di sisi chatbot.

Berbeda dari PDF, isinya dapat disunting. Setiap suntingan pada pertanyaan atau
jawaban membuang chunk lama dan menghitung ulang embedding-nya: indeks yang
masih memuat kalimat versi lama akan menjawab mahasiswa dengan aturan yang
sudah dicabut, dan itu justru risiko yang diangkat PRD §12.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.documents import UNIT_MATCH, stale_clause
from app.admin.permissions import normalize_unit
from app.db.models import JenisDokumen
from app.ingestion.chunker import split_qa
from app.ingestion.embedder import embed_and_store

TANYA_JAWAB_SAJA = "d.jenis = :jenis"

_STALE = stale_clause("d")

_KOLOM = f"""
    d.id::text AS id, d.judul AS pertanyaan, d.jawaban, d.unit, d.valid_until,
    d.updated_at, d.is_active, d.uploaded_by,
    (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS jumlah_chunk,
    {_STALE} AS stale
"""

EDITABLE_FIELDS = ("pertanyaan", "jawaban", "unit", "valid_until", "is_active")

KOLOM_DB = {"pertanyaan": "judul"}
"""Nama field API yang berbeda dari nama kolomnya di database."""

REINDEX_FIELDS = frozenset({"pertanyaan", "jawaban"})
"""Mengubah salah satunya membuat chunk lama tidak lagi mewakili isinya."""

CONTENT_FIELDS = frozenset({"pertanyaan", "jawaban", "unit", "valid_until"})
"""Mengubah salah satunya dianggap peninjauan dan memperbarui `updated_at`.
`is_active` tidak, sama seperti dokumen PDF."""


def _params(**extra: Any) -> dict[str, Any]:
    return {"jenis": JenisDokumen.TANYA_JAWAB.value, **extra}


async def list_entries(
    session: AsyncSession,
    *,
    include_inactive: bool,
    limit: int,
    offset: int,
    unit: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return: (baris, total sesuai filter). `unit` membatasi keduanya (staf/dosen)."""
    kondisi = [TANYA_JAWAB_SAJA]
    params = _params()
    if unit is not None:
        kondisi.append(UNIT_MATCH)
        params["unit"] = normalize_unit(unit)
    if not include_inactive:
        kondisi.append("d.is_active")
    where = "WHERE " + " AND ".join(kondisi)

    rows = await session.execute(
        text(
            f"SELECT {_KOLOM} FROM documents d {where}"
            " ORDER BY d.is_active DESC, d.updated_at DESC, d.id"
            " LIMIT :limit OFFSET :offset"
        ),
        {**params, "limit": limit, "offset": offset},
    )
    total = (
        await session.execute(text(f"SELECT count(*) FROM documents d {where}"), params)
    ).scalar_one()
    return [dict(r) for r in rows.mappings()], total


async def get_entry(session: AsyncSession, entry_id: uuid.UUID) -> dict[str, Any] | None:
    row = (
        (
            await session.execute(
                text(
                    f"SELECT {_KOLOM} FROM documents d"
                    f" WHERE d.id = :id AND {TANYA_JAWAB_SAJA}"
                ),
                _params(id=entry_id),
            )
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


async def create_entry(
    session: AsyncSession,
    *,
    pertanyaan: str,
    jawaban: str,
    unit: str,
    embeddings: Any,
    valid_until: date | None = None,
    uploaded_by: str | None = None,
    chunk_size: int = 700,
    chunk_overlap: int = 105,
) -> dict[str, Any]:
    """Simpan entri baru beserta chunk dan embedding-nya, dalam satu transaksi."""
    chunks = split_qa(
        pertanyaan,
        jawaban,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata={"unit": unit},
    )
    entry_id = uuid.uuid4()
    try:
        await session.execute(
            text(
                "INSERT INTO documents"
                " (id, judul, jawaban, unit, jenis, file_path, valid_until, uploaded_by,"
                "  updated_at, is_active)"
                " VALUES (:id, :pertanyaan, :jawaban, :unit, :jenis, NULL, :valid_until,"
                " :uploaded_by, now(), true)"
            ),
            _params(
                id=entry_id,
                pertanyaan=pertanyaan,
                jawaban=jawaban,
                unit=unit,
                valid_until=valid_until,
                uploaded_by=uploaded_by,
            ),
        )
        await embed_and_store(session, entry_id, chunks, embeddings)
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    hasil = await get_entry(session, entry_id)
    if hasil is None:  # pragma: no cover - baris baru saja di-commit
        raise RuntimeError(f"entri tanya jawab {entry_id} hilang setelah disimpan")
    return hasil


async def update_entry(
    session: AsyncSession,
    entry_id: uuid.UUID,
    changes: dict[str, Any],
    *,
    embeddings: Any,
    chunk_size: int = 700,
    chunk_overlap: int = 105,
) -> dict[str, Any] | None:
    """Terapkan perubahan. Return: entri terbaru, atau None bila tidak ada.

    Bila pertanyaan atau jawabannya berubah, chunk lama dihapus dan embedding
    dihitung ulang dalam transaksi yang sama: gagal meng-embed berarti entri
    tetap seperti semula, bukan tersimpan dengan indeks yang sudah kosong.
    """
    lama = await get_entry(session, entry_id)
    if lama is None:
        return None

    kolom = [k for k in EDITABLE_FIELDS if k in changes]
    if not kolom:
        return lama

    set_clause = [f"{KOLOM_DB.get(k, k)} = :{k}" for k in kolom]
    if CONTENT_FIELDS.intersection(kolom):
        set_clause.append("updated_at = now()")

    baru = {**lama, **{k: changes[k] for k in kolom}}
    try:
        await session.execute(
            text(
                f"UPDATE documents d SET {', '.join(set_clause)}"
                f" WHERE d.id = :id AND {TANYA_JAWAB_SAJA}"
            ),
            _params(id=entry_id, **{k: changes[k] for k in kolom}),
        )
        if REINDEX_FIELDS.intersection(kolom):
            await session.execute(
                text("DELETE FROM chunks WHERE document_id = :id"), {"id": entry_id}
            )
            await embed_and_store(
                session,
                entry_id,
                split_qa(
                    baru["pertanyaan"],
                    baru["jawaban"],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    metadata={"unit": baru["unit"]},
                ),
                embeddings,
            )
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    return await get_entry(session, entry_id)


async def delete_entry(session: AsyncSession, entry_id: uuid.UUID) -> bool:
    """Hapus entri beserta chunk-nya (ON DELETE CASCADE). Return: False bila tidak ada."""
    hasil = await session.execute(
        text(
            f"DELETE FROM documents d WHERE d.id = :id AND {TANYA_JAWAB_SAJA} RETURNING d.id"
        ),
        _params(id=entry_id),
    )
    if hasil.first() is None:
        await session.rollback()
        return False
    await session.commit()
    return True
