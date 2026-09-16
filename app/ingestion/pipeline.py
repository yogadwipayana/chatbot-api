"""Pipeline ingestion utuh (FR-1, AD-3).

Satu dokumen masuk -> halaman -> chunk -> embedding -> baris Postgres.
Seluruhnya dalam satu transaksi: dokumen yang gagal di tengah jalan tidak
boleh meninggalkan indeks setengah terisi, karena chunk yatim akan tetap
terambil retrieval tanpa induk dokumen yang sah.
"""

from __future__ import annotations

import functools
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import anyio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunker import split_pages
from app.ingestion.embedder import embed_and_store
from app.ingestion.loader import load_pdf
from app.storage import ObjectStorage, document_key


class EmptyDocumentError(ValueError):
    """PDF terbaca, tetapi tidak menghasilkan satu chunk pun."""


@dataclass(frozen=True)
class IngestionResult:
    document_id: uuid.UUID
    jumlah_halaman: int
    jumlah_chunk: int


async def ingest_document(
    session: AsyncSession,
    *,
    path: str | Path,
    judul: str,
    unit: str,
    embeddings: Any,
    storage: ObjectStorage,
    tahun_berlaku: int | None = None,
    valid_until: date | None = None,
    uploaded_by: str | None = None,
    chunk_size: int = 700,
    chunk_overlap: int = 105,
) -> IngestionResult:
    """Muat, pecah, embed, simpan berkas ke penyimpanan objek, lalu catat ke DB.

    `path` adalah berkas sementara hasil unggahan admin. Isinya disalin ke
    `storage` dengan kunci deterministik dari `document_id`, sehingga kolom
    `documents.file_path` menyimpan kunci objek -- bukan lintasan disk. Berpindah
    dari disk lokal ke R2 karena itu tidak memaksa migrasi data.

    Ekstraksi dan chunking dijalankan di threadpool: PyMuPDF sinkron dan
    panduan akademik bisa ratusan halaman, sehingga memanggilnya langsung akan
    membekukan event loop untuk seluruh mahasiswa yang sedang bertanya.

    Raises:
        ScannedPdfError: PDF hasil scan tanpa lapisan teks (FR-1).
        UnreadablePdfError: berkas rusak atau bukan PDF.
        EmptyDocumentError: tidak ada chunk yang dihasilkan.
        FileNotFoundError: berkas tidak ada.
    """
    halaman = await anyio.to_thread.run_sync(load_pdf, path)
    chunks = await anyio.to_thread.run_sync(
        functools.partial(
            split_pages,
            halaman,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata={"unit": unit, "tahun_berlaku": tahun_berlaku},
        )
    )
    if not chunks:
        raise EmptyDocumentError(f"'{Path(path).name}' tidak menghasilkan satu chunk pun")

    document_id = uuid.uuid4()
    key = document_key(str(document_id))

    # Berkas diunggah lebih dulu, baris DB menyusul. Urutan ini disengaja:
    # penyimpanan objek berada di luar transaksi database, jadi salah satunya
    # pasti bisa gagal sendirian. Objek yatim (ada di bucket, tidak ada di DB)
    # hanya memakan tempat; baris yatim (ada di DB, berkasnya hilang) membuat
    # kartu sitasi FE-2 menunjuk ke ketiadaan -- dan itu terlihat oleh mahasiswa.
    isi = await anyio.to_thread.run_sync(Path(path).read_bytes)
    await storage.save(key, isi, content_type="application/pdf")

    try:
        return await _catat_dokumen(
            session,
            document_id=document_id,
            key=key,
            judul=judul,
            unit=unit,
            tahun_berlaku=tahun_berlaku,
            valid_until=valid_until,
            uploaded_by=uploaded_by,
            halaman=halaman,
            chunks=chunks,
            embeddings=embeddings,
        )
    except Exception:
        # Bersihkan objek yang sudah terlanjur terunggah, lalu teruskan galatnya.
        await session.rollback()
        await storage.delete(key)
        raise


async def _catat_dokumen(
    session: AsyncSession,
    *,
    document_id: uuid.UUID,
    key: str,
    judul: str,
    unit: str,
    tahun_berlaku: int | None,
    valid_until: date | None,
    uploaded_by: str | None,
    halaman: list,
    chunks: list,
    embeddings: Any,
) -> IngestionResult:
    await session.execute(
        text(
            "INSERT INTO documents (id, judul, unit, file_path, tahun_berlaku,"
            " valid_until, uploaded_by, updated_at, is_active)"
            " VALUES (:id, :judul, :unit, :file_path, :tahun_berlaku,"
            " :valid_until, :uploaded_by, now(), true)"
        ),
        {
            "id": document_id,
            "judul": judul,
            "unit": unit,
            "file_path": key,
            "tahun_berlaku": tahun_berlaku,
            "valid_until": valid_until,
            "uploaded_by": uploaded_by,
        },
    )

    jumlah = await embed_and_store(session, document_id, chunks, embeddings)
    await session.commit()

    return IngestionResult(
        document_id=document_id,
        jumlah_halaman=len(halaman),
        jumlah_chunk=jumlah,
    )
