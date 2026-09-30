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

Bila reranker dipasang, RRF dipotong menjadi `rerank_candidates` dulu, lalu
reranker memilih top 5 dari situ (`app.rag.reranker`).

Terakhir, beberapa hasil teratas diberi potongan lanjutannya dari dokumen yang
sama (`neighbors`, lihat `NEIGHBOR_SQL`), supaya prosedur yang terbelah antar
halaman sampai ke LLM utuh.

Jalur fulltext memperluas query dengan kamus sinonim kampus
(`app.rag.glossary`): "STIKI" ikut mencari "INSTIKI", "UPS" ikut mencari
"Unit Pelaksana Sertifikasi", dan sebaliknya. Setiap varian lalu dibersihkan
dari kata tanya dan kata sambung dan digabung dengan `or` (`app.rag.fts_query`):
tanpa itu pertanyaan utuh hampir tidak pernah cocok dengan potongan mana pun.

Mahasiswa yang memilih unit di menu chatbot mempersempit KEDUA pencarian ke
dokumen unit itu, lewat WHERE yang sama -- bukan disaring setelah hasilnya
kembali, yang bisa menyisakan nol kandidat padahal unit itu punya jawabannya.
"""

from __future__ import annotations

import asyncio
import functools
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
from app.rag.fts_query import fulltext_queries
from app.rag.fusion import RankedHit, reciprocal_rank_fusion
from app.rag.reranker import rerank_documents
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
           c.content,
           c.page,
           d.id::text AS document_id,
           d.title,
           d.type,
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


def _param_varian(i: int) -> str:
    return "query" if i == 0 else f"query_{i}"


@functools.cache
def fulltext_sql(varian: int = 1) -> Any:
    """Query fulltext untuk `varian` bentuk query (lihat `app.rag.glossary`).

    Baris cocok bila cocok dengan SALAH SATU varian (tsquery digabung `||`).
    Skornya varian yang paling cocok (`GREATEST`), BUKAN `ts_rank` atas tsquery
    gabungan: pada gabungan, `ts_rank` ikut menghitung leksem varian lain yang
    tidak ada di chunk, sehingga skor chunk yang cocok dengan query asli anjlok
    (terukur 0,099 -> 0,024 untuk "pembayaran ukt") -- jatuh di bawah
    `ThresholdPolicy.lexical_threshold` dan mengubah jawaban menjadi penolakan.
    Dengan `GREATEST`, skor terhadap query asli tidak pernah turun.

    Dengan satu varian, SQL-nya sama persis seperti sebelum kamus ada.

    Varian berupa query `or` (`app.rag.fts_query`). Untuk query `or`, `ts_rank`
    dibagi rata dengan jumlah katanya: potongan yang cocok dengan "toeic"
    bernilai 0,076, dan hanya 0,019 bila tiga kata lain yang tidak ada di
    potongan itu ikut ditanyakan.
    Urutan di dalam satu pertanyaan tidak berubah, tetapi varian panjang (frasa
    kamus yang diurai) selalu kalah skor dari varian pendek -- satu alasan lagi
    memakai `GREATEST`.
    """
    if varian < 1:
        raise ValueError(f"jumlah varian minimal 1, bukan {varian}")
    tsquery = [
        f"websearch_to_tsquery('{FTS_CONFIG}', :{_param_varian(i)})" for i in range(varian)
    ]
    if varian == 1:
        skor, cocok = f"ts_rank(c.tsv, {tsquery[0]})", tsquery[0]
    else:
        skor = "GREATEST(" + ", ".join(f"ts_rank(c.tsv, {q})" for q in tsquery) + ")"
        cocok = "(" + " || ".join(tsquery) + ")"
    return text(
        f"""
    SELECT c.id::text AS chunk_id,
           c.content,
           c.page,
           d.id::text AS document_id,
           d.title,
           d.type,
           d.file_path,
           {skor} AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.tsv @@ {cocok}
      AND {_ACTIVE}
      AND {_UNIT}
    ORDER BY score DESC
    LIMIT :limit
    """
    )


FULLTEXT_SQL = fulltext_sql(1)

NEIGHBOR_SQL = text(
    """
    SELECT c.id::text AS source_id,
           n.id::text AS chunk_id,
           n.content,
           n.page,
           d.id::text AS document_id,
           d.title,
           d.type,
           d.file_path
    FROM chunks c
    JOIN chunks n ON n.document_id = c.document_id AND n.position = c.position + 1
    JOIN documents d ON d.id = n.document_id
    WHERE c.id = ANY(CAST(:ids AS text[])::uuid[])
    """
)
"""Potongan sesudah (`position + 1`) setiap chunk sumber, dari dokumen yang sama.

Dokumen dipecah per halaman, sehingga satu prosedur sering terbelah: langkah
7-8 panduan KRS MBKM ada di halaman 5 sendirian, format SMS pembayaran VA
menyambung potongan "SMS Banking" tanpa mengulang judulnya. Potongan lanjutan
seperti itu kalah peringkat karena tidak memuat kata kunci pertanyaannya,
dan LLM lalu menjawab prosedur yang bolong di tengah. Status aktif tidak perlu
diperiksa lagi: dokumennya sama dengan chunk sumber yang sudah lolos filter."""


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


async def _tanpa_hasil() -> list[dict[str, Any]]:
    """Pengganti pencarian untuk sumber yang dimatikan (bobot 0)."""
    return []


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
    reranker: Any = None
    """`app.rag.reranker.Reranker`, atau None untuk memakai urutan RRF langsung."""
    rerank_candidates: int = 20
    neighbors: int = 0
    """Berapa hasil teratas yang diberi potongan lanjutannya (`NEIGHBOR_SQL`).
    0 = mati. Potongan lanjutan disisipkan tepat sesudah sumbernya dan tidak
    membawa skor, jadi tidak ikut menentukan keputusan threshold FR-3."""

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,
        unit: str | None = None,
    ) -> list[Document]:
        """`unit`: nama resmi dari tabel `units` (lihat `app.units`), atau None
        untuk semua unit. Diteruskan lewat `ainvoke(query, unit=...)`."""
        # Bobot 0 mematikan sumbernya (lihat `reciprocal_rank_fusion`), jadi
        # pencariannya -- dan untuk vektor, panggilan embedding-nya -- dilewati.
        cari_vektor = self.weight_vector != 0
        cari_kata = self.weight_fulltext != 0
        embedding = await self.embed_query(query) if cari_vektor else None

        vector_rows, fulltext_rows = await asyncio.gather(
            self._vector_search(embedding, unit) if embedding is not None else _tanpa_hasil(),
            self._fulltext_search(query, unit) if cari_kata else _tanpa_hasil(),
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
            top_n=(
                max(self.rerank_candidates, self.top_n)
                if self.reranker is not None
                else self.top_n
            ),
        )

        by_id = {r["chunk_id"]: r for r in (*vector_rows, *fulltext_rows)}
        documents = [
            Document(
                id=hit.chunk_id,
                page_content=by_id[hit.chunk_id]["content"],
                metadata={
                    "chunk_id": hit.chunk_id,
                    "document_id": by_id[hit.chunk_id]["document_id"],
                    # Kunci metadata sengaja tetap (internal, dibaca sitasi);
                    # kolom sumbernya `title`, `type`, dan `page`.
                    "judul": by_id[hit.chunk_id]["title"],
                    "jenis": by_id[hit.chunk_id].get("type"),
                    "halaman": by_id[hit.chunk_id]["page"],
                    "file_path": by_id[hit.chunk_id]["file_path"],
                    "rrf_score": hit.rrf_score,
                    "raw_scores": hit.raw_scores,
                    "ranks": hit.ranks,
                },
            )
            for hit in fused
        ]
        documents = await rerank_documents(
            query, documents, self.reranker, top_n=self.top_n
        )
        return await self._with_neighbors(documents)

    async def _with_neighbors(self, documents: list[Document]) -> list[Document]:
        if self.neighbors <= 0 or not documents:
            return documents
        sumber = [doc.metadata["chunk_id"] for doc in documents[: self.neighbors]]
        rows = await self._jalankan(NEIGHBOR_SQL, {"ids": sumber})

        sudah = {doc.metadata["chunk_id"] for doc in documents}
        lanjutan = {r["source_id"]: r for r in rows if r["chunk_id"] not in sudah}
        hasil: list[Document] = []
        for doc in documents:
            hasil.append(doc)
            row = lanjutan.get(doc.metadata["chunk_id"])
            if row is None:
                continue
            hasil.append(
                Document(
                    id=row["chunk_id"],
                    page_content=row["content"],
                    metadata={
                        "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"],
                        "judul": row["title"],
                        "jenis": row.get("type"),
                        "halaman": row["page"],
                        "file_path": row["file_path"],
                        # Tanpa skor: potongan ini ikut karena sumbernya, bukan
                        # karena mirip pertanyaan, dan tidak boleh meloloskan
                        # threshold atas namanya sendiri.
                        "rrf_score": 0.0,
                        "raw_scores": {},
                        "ranks": {},
                        "neighbor_of": doc.metadata["chunk_id"],
                    },
                )
            )
        return hasil

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
        #
        # Hanya jalur ini yang memakai kamus sinonim. Jalur vektor sudah
        # menangkap kemiripan makna, dan menambah teks pada query embedding
        # akan menggeser distribusi skor yang dipakai kalibrasi threshold.
        #
        # Setiap varian sudah berupa "a or b or c" tanpa kata tanya dan kata
        # sambung (`app.rag.fts_query`). Kosong = semua kata umum ("apa itu?"):
        # tidak ada yang layak dicocokkan, vektor saja yang memutuskan.
        varian = fulltext_queries(query)
        if not varian:
            return []
        return await self._jalankan(
            fulltext_sql(len(varian)),
            {
                **{_param_varian(i): teks for i, teks in enumerate(varian)},
                "limit": self.candidates,
                "unit": unit,
            },
        )

    async def _jalankan(
        self, sql: Any, params: dict[str, Any], *, persiapan: Sequence[Any] = ()
    ) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            for perintah in persiapan:
                await session.execute(perintah)
            result = await session.execute(sql, params)
            return [dict(row) for row in result.mappings()]
