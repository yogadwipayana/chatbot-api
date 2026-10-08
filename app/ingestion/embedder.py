"""Penyimpanan chunk + embedding ke Postgres (FR-1).

Ditulis dengan SQL langsung, bukan `vectorstore.add_documents()`, karena PRD §6
menuntut kendali penuh atas kolom metadata kustom (`page`, `position`,
`document_id`) dan atas pembuatan `tsvector` -- keduanya tidak terjangkau lewat
abstraksi VectorStore.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import EMBEDDING_DIM
from app.ingestion.chunker import PreparedChunk
from app.rag.providers import GalatGateway
from app.rag.retriever import vector_literal


class EmbeddingDimensionError(RuntimeError):
    """Model embedding menghasilkan dimensi yang berbeda dari kolom `chunks.embedding`."""


def galat_layanan_ai(exc: BaseException) -> bool:
    """True bila galat berasal dari API model AI, bukan dari kode kita.

    Dipakai router untuk memisahkan "layanan luar sedang bermasalah, coba lagi"
    (502) dari bug yang harus tetap menjadi 500 dan terlihat di log.

    `openai.APIError` mencakup status galat, batas waktu, dan koneksi putus;
    `GalatGateway` adalah galat yang dikirim gateway sebagai isi jawaban.
    """
    if isinstance(exc, GalatGateway):
        return True
    try:
        import openai
    except ModuleNotFoundError:  # pragma: no cover - langchain-openai selalu membawanya
        return False
    return isinstance(exc, openai.APIError)


BATCH_SIZE = 32
"""Jumlah chunk per panggilan embedding. Terlalu besar berisiko kena batas
ukuran request penyedia; terlalu kecil membuat ingestion lambat dan mahal.

32 adalah batas bawaan Text Embeddings Inference (`--max-client-batch-size`)
yang melayani e5: batch 33 ke atas ditolak 422, dan setiap dokumen yang
potongannya lebih dari itu gagal dipasang dengan 502."""

INSERT_CHUNK_SQL = text(
    """
    INSERT INTO chunks (id, document_id, content, page, position, embedding)
    VALUES (
        :id, :document_id, :content, :page, :position,
        (:embedding)::vector
    )
    """
)
"""`tsv` sengaja tidak diisi di sini: trigger `trg_chunks_tsv` yang membentuknya
dari judul dokumen dan isi potongan (migrasi 0014, 0015). Nilai yang diisi
di sini selalu ditimpa trigger, dan dua rumus di dua tempat cepat menyimpang."""


async def embed_and_store(
    session: AsyncSession,
    document_id: uuid.UUID,
    chunks: Sequence[PreparedChunk],
    embeddings: Any,
    *,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Hitung embedding lalu simpan seluruh chunk. Return: jumlah chunk tersimpan.

    Pemanggil bertanggung jawab atas commit, supaya satu dokumen gagal di
    tengah jalan tidak meninggalkan indeks yang setengah terisi.
    """
    tersimpan = 0
    for awal in range(0, len(chunks), batch_size):
        batch = chunks[awal : awal + batch_size]
        vektor = await embeddings.aembed_documents([c.konten for c in batch])

        for chunk, vec in zip(batch, vektor, strict=True):
            # Diperiksa sebelum INSERT: galat dimensi dari pgvector baru muncul
            # sebagai DBAPIError generik yang tidak menyebut EMBED_MODEL sama sekali.
            if len(vec) != EMBEDDING_DIM:
                raise EmbeddingDimensionError(
                    f"model embedding menghasilkan {len(vec)} dimensi, kolom "
                    f"chunks.embedding butuh {EMBEDDING_DIM}. Periksa EMBED_MODEL."
                )
            await session.execute(
                INSERT_CHUNK_SQL,
                {
                    "id": uuid.uuid4(),
                    "document_id": document_id,
                    "content": chunk.konten,
                    "page": chunk.halaman,
                    "position": chunk.urutan,
                    # Format yang sama persis dengan yang dipakai retriever saat
                    # mencari; dua format berbeda adalah sumber bug yang sulit dilacak.
                    "embedding": vector_literal(vec),
                },
            )
            tersimpan += 1
    return tersimpan
