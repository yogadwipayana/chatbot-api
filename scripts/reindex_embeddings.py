"""Hitung ulang embedding seluruh chunk dengan EMBED_PROVIDER/EMBED_MODEL saat ini.

Wajib dijalankan setelah model embedding diganti (mis. ke multilingual-e5).
Vektor dari dua model berbeda tidak dapat dibandingkan: pertanyaan yang di-embed
model baru terhadap chunk yang di-embed model lama menghasilkan skor acak, dan
tidak ada galat yang memberi tahu.

Hanya kolom `embedding` yang diperbarui. Teks chunk, `tsv` (fulltext), dan
pemecahan dokumen tidak disentuh -- tidak perlu mengunggah ulang PDF.

Seluruh pembaruan berjalan dalam SATU transaksi: gagal di tengah jalan berarti
tidak ada yang berubah, bukan indeks yang separuh model lama separuh model baru.

    python -m scripts.reindex_embeddings --dry-run   # hitung chunk, uji model
    python -m scripts.reindex_embeddings
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from sqlalchemy import text

from app.config import get_settings
from app.db.models import EMBEDDING_DIM
from app.db.session import SessionLocal, engine
from app.ingestion.embedder import BATCH_SIZE
from app.rag.providers import build_embeddings
from app.rag.retriever import vector_literal

UPDATE_SQL = text("UPDATE chunks SET embedding = (:embedding)::vector WHERE id = :id")


async def jalankan(dry_run: bool, batch_size: int) -> int:
    settings = get_settings()
    embeddings = build_embeddings(settings)
    print(f"Model   : {settings.embed_model} ({settings.embed_provider})")

    uji = await embeddings.aembed_documents(["uji dimensi"])
    if len(uji[0]) != EMBEDDING_DIM:
        print(f"GAGAL: model menghasilkan {len(uji[0])} dimensi, kolom butuh {EMBEDDING_DIM}")
        return 1

    try:
        async with SessionLocal() as session:
            rows = (
                await session.execute(text("SELECT id, konten FROM chunks ORDER BY id"))
            ).all()
            print(f"Chunk   : {len(rows)}")
            if dry_run:
                print("--dry-run: tidak ada yang diubah.")
                return 0

            mulai = time.perf_counter()
            for awal in range(0, len(rows), batch_size):
                batch = rows[awal : awal + batch_size]
                vektor = await embeddings.aembed_documents([r.konten for r in batch])
                for row, vec in zip(batch, vektor, strict=True):
                    await session.execute(
                        UPDATE_SQL, {"id": row.id, "embedding": vector_literal(vec)}
                    )
                print(f"  {min(awal + batch_size, len(rows))}/{len(rows)}")
            await session.commit()
            print(f"Selesai dalam {time.perf_counter() - mulai:.1f} detik.")
    finally:
        await engine.dispose()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="hitung chunk tanpa mengubah")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    sys.exit(asyncio.run(jalankan(args.dry_run, args.batch_size)))


if __name__ == "__main__":
    main()
