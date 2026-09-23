"""PostgresHybridRetriever -- retrieval hibrida (FR-2).

Subclass `BaseRetriever` supaya tetap bisa dirangkai dengan LCEL, tetapi
seluruh isinya SQL langsung. PRD §6 menolak `EnsembleRetriever` karena BM25
bawaannya in-memory, sementara sistem ini harus memakai Postgres FTS, dan
menolak abstraksi `VectorStore` karena menyembunyikan skor yang justru
dibutuhkan FR-3.

Alur: vector search (top 20) dan fulltext search (top 20) dijalankan paralel,
masing-masing di sesi database sendiri, digabung dengan RRF, dipotong menjadi
top 5. Skor mentah tiap sumber ikut dibawa di `Document.metadata` agar tahap
threshold dapat membacanya.

Mahasiswa yang memilih unit di menu chatbot mempersempit KEDUA pencarian ke
dokumen unit itu, lewat WHERE yang sama -- bukan disaring setelah hasilnya
kembali, yang bisa menyisakan nol kandidat padahal unit itu punya jawabannya.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sqlalchemy import text

from app.rag.filters import active_document_clause
from app.rag.fusion import RankedHit, reciprocal_rank_fusion
from app.rag.threshold import LEXICAL_SOURCE, VECTOR_SOURCE

FTS_CONFIG = "indonesian"
"""Konfigurasi text search Postgres. Verifikasi tersedia di instans target:
`SELECT cfgname FROM pg_ts_config;` -- bila tidak ada, turunkan ke 'simple'
dan catat dampaknya pada evaluasi."""

_ACTIVE = active_document_clause("d")

_UNIT = "(CAST(:unit AS text) IS NULL OR d.unit = CAST(:unit AS text))"
"""NULL = semua unit. Satu SQL untuk kedua kasus, bukan dua varian query yang
bisa menyimpang. CAST wajib: asyncpg tidak dapat menebak tipe parameter yang
hanya muncul di `IS NULL`."""

ITERATIVE_SCAN_SQL = text("SET LOCAL hnsw.iterative_scan = strict_order")
"""pgvector >= 0.8. HNSW menyaring SETELAH index dipindai, dan `hnsw.ef_search`
bawaannya 40: bila satu unit hanya ~1/9 isi indeks, rata-rata cuma 4-5 dari 40
kandidat yang lolos filter -- jauh di bawah `candidates`. Iterative scan terus
memindai index sampai LIMIT terpenuhi.

`strict_order`, bukan `relaxed_order`: RRF memakai peringkat, jadi urutan
jarak harus tepat. LOCAL: berlaku sampai transaksi sesi ini selesai, tidak ikut
terbawa ke koneksi pool berikutnya."""

VECTOR_SQL = text(
    f"""
    SELECT c.id::text AS chunk_id,
           c.konten,
           c.halaman,
           d.id::text AS document_id,
           d.judul,
           d.jenis,
           d.file_path,
           1 - (c.embedding <=> (:query_embedding)::vector) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE {_ACTIVE}
      AND {_UNIT}
    ORDER BY c.embedding <=> (:query_embedding)::vector
    LIMIT :limit
    """
)

FULLTEXT_SQL = text(
    f"""
    SELECT c.id::text AS chunk_id,
           c.konten,
           c.halaman,
           d.id::text AS document_id,
           d.judul,
           d.jenis,
           d.file_path,
           ts_rank(c.tsv, websearch_to_tsquery('{FTS_CONFIG}', :query)) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.tsv @@ websearch_to_tsquery('{FTS_CONFIG}', :query)
      AND {_ACTIVE}
      AND {_UNIT}
    ORDER BY score DESC
    LIMIT :limit
    """
)


def vector_literal(embedding: Sequence[float]) -> str:
    """Bentuk literal teks pgvector, mis. `[0.1,0.2]`.

    asyncpg tidak punya codec untuk tipe `vector`, jadi parameternya harus
    berupa teks yang di-cast `::vector` di SQL. Mengirim list Python langsung
    gagal dengan "expected str, got list" -- tertangkap saat retriever pertama
    kali dijalankan terhadap PostgreSQL sungguhan.
    """
    if not embedding:
        raise ValueError("embedding kosong")
    nilai = [float(x) for x in embedding]
    if not all(math.isfinite(x) for x in nilai):
        raise ValueError("embedding mengandung NaN atau tak hingga")
    return "[" + ",".join(repr(x) for x in nilai) + "]"


class PostgresHybridRetriever(BaseRetriever):
    """Retriever hibrida di atas satu instans PostgreSQL + pgvector."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_factory: Callable[[], Any]
    """Pembuat sesi async, mis. `async_sessionmaker`.

    Setiap pencarian membuka sesinya sendiri. Satu koneksi asyncpg menolak dua
    query yang berjalan bersamaan ("another operation is in progress"), jadi
    paralelisme FR-2 hanya mungkin dengan dua koneksi terpisah."""

    embed_query: Any
    """Callable async: `str -> list[float]`. Disuntikkan agar penyedia
    embedding dapat diganti tanpa menyentuh kelas ini, dan agar test dapat
    memakai embedding palsu tanpa memanggil API."""

    candidates: int = 20
    top_n: int = 5
    rrf_k: int = 60
    weight_vector: float = 1.0
    weight_fulltext: float = 1.0

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,
        unit: str | None = None,
    ) -> list[Document]:
        """`unit`: nama resmi dari tabel `units` (lihat `app.units`), atau None
        untuk semua unit. Diteruskan lewat `ainvoke(query, unit=...)`."""
        embedding = await self.embed_query(query)

        vector_rows, fulltext_rows = await asyncio.gather(
            self._vector_search(embedding, unit),
            self._fulltext_search(query, unit),
        )

        fused = reciprocal_rank_fusion(
            {
                VECTOR_SOURCE: [
                    RankedHit(r["chunk_id"], float(r["score"])) for r in vector_rows
                ],
                LEXICAL_SOURCE: [
                    RankedHit(r["chunk_id"], float(r["score"])) for r in fulltext_rows
                ],
            },
            weights={
                VECTOR_SOURCE: self.weight_vector,
                LEXICAL_SOURCE: self.weight_fulltext,
            },
            k=self.rrf_k,
            top_n=self.top_n,
        )

        by_id = {r["chunk_id"]: r for r in (*vector_rows, *fulltext_rows)}
        return [
            Document(
                id=hit.chunk_id,
                page_content=by_id[hit.chunk_id]["konten"],
                metadata={
                    "chunk_id": hit.chunk_id,
                    "document_id": by_id[hit.chunk_id]["document_id"],
                    "judul": by_id[hit.chunk_id]["judul"],
                    "jenis": by_id[hit.chunk_id].get("jenis"),
                    "halaman": by_id[hit.chunk_id]["halaman"],
                    "file_path": by_id[hit.chunk_id]["file_path"],
                    "rrf_score": hit.rrf_score,
                    "raw_scores": hit.raw_scores,
                    "ranks": hit.ranks,
                },
            )
            for hit in fused
        ]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
        unit: str | None = None,
    ) -> list[Document]:
        # `unit` harus ada juga di sini: LangChain hanya meneruskan argumen
        # tambahan `ainvoke` bila tanda tangan metode SINKRON ini memintanya.
        raise NotImplementedError(
            "Retriever ini hanya mendukung mode async; pakai `ainvoke`. "
            "Jalur sinkron sengaja tidak disediakan agar dua pencarian tetap paralel."
        )

    async def _vector_search(
        self, embedding: Sequence[float], unit: str | None = None
    ) -> list[dict[str, Any]]:
        return await self._jalankan(
            VECTOR_SQL,
            {
                "query_embedding": vector_literal(embedding),
                "limit": self.candidates,
                "unit": unit,
            },
            # Hanya saat difilter: tanpa filter unit, jalur ini tetap persis
            # seperti sebelum menu unit ada.
            persiapan=(ITERATIVE_SCAN_SQL,) if unit is not None else (),
        )

    async def _fulltext_search(
        self, query: str, unit: str | None = None
    ) -> list[dict[str, Any]]:
        # Tanpa iterative scan: index GIN mencocokkan secara pasti, dan filter
        # unit di atasnya tidak pernah membuang kandidat yang sah.
        return await self._jalankan(
            FULLTEXT_SQL,
            {"query": query, "limit": self.candidates, "unit": unit},
        )

    async def _jalankan(
        self, sql: Any, params: dict[str, Any], *, persiapan: Sequence[Any] = ()
    ) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            for perintah in persiapan:
                await session.execute(perintah)
            result = await session.execute(sql, params)
            return [dict(row) for row in result.mappings()]
