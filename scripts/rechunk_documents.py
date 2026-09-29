"""Pecah ulang dokumen PDF yang sudah diunggah dengan chunker saat ini.

Wajib dijalankan setelah `app.ingestion.chunker` atau `loader` berubah: chunk
lama tetap tersimpan dengan pemecahan lama sampai dokumennya diunggah ulang,
dan retrieval tidak memberi tahu bahwa indeksnya usang. Contoh: jejak judul BAB
(T9, 2026-09-29) baru berlaku bagi dokumen yang dipecah sesudahnya.

Berkas asli diambil dari penyimpanan objek (`documents.file_path`), jadi PDF
tidak perlu diunggah ulang. Per dokumen, dalam SATU transaksi: chunk lama
dihapus, chunk baru di-embed dan disimpan. Gagal di tengah dokumen berarti
dokumen itu tetap dengan chunk lamanya. ID chunk berganti, sehingga
`messages.retrieved_chunk_ids` percakapan lama tidak lagi menunjuk ke chunk
yang ada -- sama seperti saat entri tanya jawab diubah.

    python -m scripts.rechunk_documents --dry-run   # jumlah chunk lama -> baru
    python -m scripts.rechunk_documents
    python -m scripts.rechunk_documents --dokumen <uuid> [<uuid> ...]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
import uuid
from pathlib import Path

import anyio
from sqlalchemy import text

from app.config import get_settings
from app.db.session import SessionLocal, engine
from app.ingestion.chunker import split_pages
from app.ingestion.embedder import embed_and_store
from app.ingestion.loader import load_pdf
from app.rag.providers import build_embeddings
from app.storage import build_storage

DOKUMEN_SQL = text(
    """
    SELECT d.id, d.title, d.unit, d.file_path, d.effective_year,
           (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS jumlah
    FROM documents d
    WHERE d.type = 'pdf'
      AND (CAST(:ids AS uuid[]) IS NULL OR d.id = ANY(CAST(:ids AS uuid[])))
    ORDER BY d.unit, d.title
    """
)


async def _pecah(storage, row) -> list:
    isi = await storage.load(row.file_path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dokumen.pdf"
        path.write_bytes(isi)
        halaman = await anyio.to_thread.run_sync(load_pdf, path)
    return split_pages(
        halaman, metadata={"unit": row.unit, "tahun_berlaku": row.effective_year}
    )


async def jalankan(dry_run: bool, ids: list[uuid.UUID] | None) -> int:
    settings = get_settings()
    storage = build_storage(settings)
    embeddings = build_embeddings(settings)
    try:
        async with SessionLocal() as session:
            rows = (
                await session.execute(DOKUMEN_SQL, {"ids": [str(i) for i in ids] if ids else None})
            ).all()
        if not rows:
            print("Tidak ada dokumen PDF yang cocok.")
            return 1
        print(f"Dokumen : {len(rows)}{' (--dry-run: tidak ada yang diubah)' if dry_run else ''}")

        mulai = time.perf_counter()
        for row in rows:
            chunks = await _pecah(storage, row)
            print(f"  [{row.unit}] {row.title}: {row.jumlah} -> {len(chunks)} chunk")
            if dry_run or not chunks:
                continue
            async with SessionLocal() as session:
                await session.execute(
                    text("DELETE FROM chunks WHERE document_id = :id"), {"id": row.id}
                )
                await embed_and_store(session, row.id, chunks, embeddings)
                await session.commit()
        if not dry_run:
            print(f"Selesai dalam {time.perf_counter() - mulai:.1f} detik.")
    finally:
        await engine.dispose()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="hitung chunk tanpa mengubah")
    parser.add_argument(
        "--dokumen", nargs="+", type=uuid.UUID, help="hanya dokumen ini (default: semua PDF)"
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(jalankan(args.dry_run, args.dokumen)))


if __name__ == "__main__":
    main()
