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

Top 5 dibatasi paling banyak `max_per_document` potongan dari satu dokumen
(T59), supaya ringkasan TRANSKRIP yang cocok dengan banyak kata tidak menutup
dokumen lain yang memuat jawabannya.

Terakhir, beberapa hasil teratas diberi potongan lanjutannya dari dokumen yang
sama (`neighbors`, lihat `NEIGHBOR_SQL`), supaya prosedur yang terbelah antar
halaman sampai ke LLM utuh. Dokumen yang hasilnya tersebar di beberapa bab
diberi daftar babnya (`outline`, lihat `app.rag.outline`), supaya pertanyaan
daftar ("jenis beasiswa apa saja?") terjawab lengkap.

Bila pertanyaan ditulis ulang (FR-4), pertanyaan asli ikut dicari: kedua jalur
dijalankan untuk masing-masing bentuk, dan keempat daftar digabung dalam satu
RRF (`fuse_ranked_lists`).

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

from app.db.models import DocumentType
from app.ingestion.chunker import LANJUTAN
from app.rag.filters import active_document_clause
from app.rag.fts_query import fulltext_queries
from app.rag.fusion import RankedHit, fuse_ranked_lists
from app.rag.outline import susun_daftar_bab
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
           n.position - c.position AS arah,
           n.id::text AS chunk_id,
           n.content,
           n.page,
           d.id::text AS document_id,
           d.title,
           d.type,
           d.file_path
    FROM chunks c
    JOIN chunks n ON n.document_id = c.document_id
                 AND n.position IN (c.position - 1, c.position + 1)
    JOIN documents d ON d.id = n.document_id
    WHERE c.id = ANY(CAST(:ids AS text[])::uuid[])
    """
)
"""Potongan sebelum (`arah` -1) dan sesudah (`arah` 1) setiap chunk sumber,
dari dokumen yang sama.

Dokumen dipecah per halaman, sehingga satu prosedur sering terbelah: langkah
7-8 panduan KRS MBKM ada di halaman 5 sendirian, format SMS pembayaran VA
menyambung potongan "SMS Banking" tanpa mengulang judulnya. Potongan lanjutan
seperti itu kalah peringkat karena tidak memuat kata kunci pertanyaannya,
dan LLM lalu menjawab prosedur yang bolong di tengah. Arah sebaliknya sama:
"SKP wajib apa saja?" menemukan butir d-f daftar kegiatan wajib di halaman 21,
sedangkan butir a-c (PKKMB dst.) ada di potongan sebelumnya (T40). Potongan
sebelumnya hanya dipakai bila sebagian yang sama (`_judul_potongan`), lihat
`_with_neighbors`. Status aktif tidak perlu diperiksa lagi: dokumennya sama
dengan chunk sumber yang sudah lolos filter."""


OUTLINE_SQL = text(
    """
    SELECT c.document_id::text AS document_id,
           c.id::text AS chunk_id,
           c.page,
           split_part(c.content, chr(10), 1) AS kepala
    FROM chunks c
    WHERE c.document_id = ANY(CAST(:ids AS text[])::uuid[])
    ORDER BY c.document_id, c.position
    """
)
"""Baris pertama -- jejak judul -- setiap potongan dokumen, bahan daftar bab
(`app.rag.outline.susun_daftar_bab`). Hanya baris pertamanya: pedoman beasiswa
148 potongan, isinya tidak perlu ikut terkirim."""


def _batasi_per_dokumen(
    documents: Sequence[Document], top_n: int, maks: int
) -> list[Document]:
    """`top_n` hasil teratas dengan paling banyak `maks` potongan per dokumen.

    T59: "sertifikasi dasar DKV apa dan berapa?" mengisi kelima kursi dengan
    TRANSKRIP UPS, yang menyebut nama sertifikasinya tetapi tidak harganya.
    Dokumen HARGA SERTIFIKASI ada di peringkat 6, sehingga dijawab "dokumen
    resmi tidak mencantumkan biayanya" (4/4).

    Kursi yang tersisa karena dokumen lain kehabisan hasil diisi kembali oleh
    potongan yang tadi dilewati: unit berdokumen tunggal (Keuangan) tetap
    mendapat `top_n` potongan. Urutan peringkat dipertahankan. `maks` 0 = tanpa
    batas.
    """
    if maks <= 0 or len(documents) <= top_n:
        return list(documents[:top_n])
    terpilih: list[int] = []
    dilewati: list[int] = []
    jumlah: dict[str, int] = {}
    for i, doc in enumerate(documents):
        dokumen = doc.metadata["document_id"]
        if jumlah.get(dokumen, 0) < maks:
            jumlah[dokumen] = jumlah.get(dokumen, 0) + 1
            terpilih.append(i)
            if len(terpilih) == top_n:
                break
        else:
            dilewati.append(i)
    terpilih += dilewati[: top_n - len(terpilih)]
    return [documents[i] for i in sorted(terpilih)]


def _judul_potongan(isi: str) -> str:
    """Baris pertama potongan -- jejak judul bagiannya -- tanpa penanda lanjutan."""
    baris = isi.split("\n", 1)[0].strip()
    return baris.removesuffix(LANJUTAN).rstrip()


def _tetangga(row: dict[str, Any], sumber: str) -> Document:
    return Document(
        id=row["chunk_id"],
        page_content=row["content"],
        metadata={
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "judul": row["title"],
            "jenis": row.get("type"),
            "halaman": row["page"],
            "file_path": row["file_path"],
            # Tanpa skor: potongan ini ikut karena sumbernya, bukan karena mirip
            # pertanyaan, dan tidak boleh meloloskan threshold atas namanya sendiri.
            "rrf_score": 0.0,
            "raw_scores": {},
            "ranks": {},
            "neighbor_of": sumber,
        },
    )


def _kunci(teks: str) -> str:
    return " ".join(teks.split()).casefold()


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
    max_per_document: int = 0
    """Paling banyak berapa potongan satu dokumen di antara `top_n` hasil (T59,
    `_batasi_per_dokumen`). 0 = tanpa batas."""
    neighbors: int = 0
    """Berapa hasil teratas yang diberi potongan lanjutannya (`NEIGHBOR_SQL`).
    0 = mati. Potongan lanjutan disisipkan tepat sesudah sumbernya dan tidak
    membawa skor, jadi tidak ikut menentukan keputusan threshold FR-3."""
    outline: int = 0
    """Paling banyak berapa dokumen yang daftar babnya ikut ke konteks (T9,
    `app.rag.outline`). 0 = mati. Hanya dokumen yang hasilnya menyentuh dua bab
    atau lebih: pertanyaan satu bab ("syarat KIP") tidak butuh gambaran
    seluruh dokumen. Daftar bab ditaruh paling akhir dan, seperti potongan
    lanjutan, tidak membawa skor."""

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,
        unit: str | None = None,
        original_query: str | None = None,
    ) -> list[Document]:
        """`unit`: nama resmi dari tabel `units` (lihat `app.units`), atau None
        untuk semua unit. Diteruskan lewat `ainvoke(query, unit=...)`.

        `original_query`: pertanyaan mahasiswa sebelum ditulis ulang (FR-4),
        bila `query` hasil rewrite. Keduanya dicari dan digabung dalam satu RRF
        (T40). Rewrite melengkapi rujukan dari riwayat, tetapi kerap ikut
        mengganti kata mahasiswa ("harga" menjadi "biaya") atau menambah kata
        umum ("mahasiswa"), dan potongan yang cocok dengan kata aslinya
        terlempar dari top 5. Pertanyaan asli menjaga kata-kata itu tetap dicari.
        """
        queries = [query]
        if original_query and _kunci(original_query) != _kunci(query):
            queries.append(original_query)

        # Bobot 0 mematikan sumbernya (lihat `reciprocal_rank_fusion`), jadi
        # pencariannya -- dan untuk vektor, panggilan embedding-nya -- dilewati.
        cari_vektor = self.weight_vector != 0
        cari_kata = self.weight_fulltext != 0
        embeddings = (
            await asyncio.gather(*(self.embed_query(q) for q in queries))
            if cari_vektor
            else []
        )
        vektor = [self._vector_search(e, unit) for e in embeddings] or [_tanpa_hasil()]
        kata = (
            [self._fulltext_search(q, unit) for q in queries]
            if cari_kata
            else [_tanpa_hasil()]
        )
        hasil = await asyncio.gather(*vektor, *kata)
        daftar = [
            *((VECTOR_SOURCE, rows) for rows in hasil[: len(vektor)]),
            *((LEXICAL_SOURCE, rows) for rows in hasil[len(vektor) :]),
        ]

        fused = fuse_ranked_lists(
            [
                (sumber, [RankedHit(r["chunk_id"], float(r["score"])) for r in rows])
                for sumber, rows in daftar
            ],
            weights={
                VECTOR_SOURCE: self.weight_vector,
                LEXICAL_SOURCE: self.weight_fulltext,
            },
            k=self.rrf_k,
            # Tanpa reranker seluruh hasil fusi dibawa: batas per dokumen bisa
            # mengambil pengganti dari peringkat berapa pun.
            top_n=(
                max(self.rerank_candidates, self.top_n) if self.reranker is not None else None
            ),
        )

        by_id = {r["chunk_id"]: r for _, rows in daftar for r in rows}
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
            query, documents, self.reranker, top_n=len(documents)
        )
        documents = _batasi_per_dokumen(documents, self.top_n, self.max_per_document)
        lengkap, daftar_bab = await asyncio.gather(
            self._with_neighbors(documents), self._daftar_bab(documents)
        )
        return [*lengkap, *daftar_bab]

    async def _with_neighbors(self, documents: list[Document]) -> list[Document]:
        """Sisipkan potongan sesudah tiap hasil teratas, dan potongan sebelumnya
        bila hasil itu lanjutan dari bagian yang sama (judul bagiannya sama)."""
        if self.neighbors <= 0 or not documents:
            return documents
        sumber = [doc.metadata["chunk_id"] for doc in documents[: self.neighbors]]
        rows = await self._jalankan(NEIGHBOR_SQL, {"ids": sumber})

        by_id = {doc.metadata["chunk_id"]: doc for doc in documents}
        sudah = set(by_id)
        sebelum: dict[str, dict[str, Any]] = {}
        sesudah: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row["chunk_id"] in sudah:
                continue
            if row["arah"] > 0:
                sesudah[row["source_id"]] = row
            elif _judul_potongan(row["content"]) == _judul_potongan(
                by_id[row["source_id"]].page_content
            ):
                sebelum[row["source_id"]] = row

        hasil: list[Document] = []
        for doc in documents:
            chunk_id = doc.metadata["chunk_id"]
            # Satu potongan bisa menjadi tetangga dua hasil (sesudah yang satu,
            # sebelum yang lain); cukup disisipkan sekali.
            depan = sebelum.get(chunk_id)
            if depan is not None and depan["chunk_id"] not in sudah:
                hasil.append(_tetangga(depan, chunk_id))
                sudah.add(depan["chunk_id"])
            hasil.append(doc)
            belakang = sesudah.get(chunk_id)
            if belakang is not None and belakang["chunk_id"] not in sudah:
                hasil.append(_tetangga(belakang, chunk_id))
                sudah.add(belakang["chunk_id"])
        return hasil

    async def _daftar_bab(self, documents: list[Document]) -> list[Document]:
        """Daftar bab dokumen yang hasil teratasnya tersebar di dua bab atau lebih.

        Dokumen diurutkan menurut hasil terbaiknya. Dokumen dengan satu hasil
        saja tidak diperiksa: satu potongan hanya menyentuh satu bab."""
        if self.outline <= 0:
            return []
        per_dokumen: dict[str, list[Document]] = {}
        for doc in documents:
            # Entri tanya jawab satu potongan per dokumen, tanpa bab.
            if doc.metadata.get("jenis") == DocumentType.TANYA_JAWAB:
                continue
            per_dokumen.setdefault(doc.metadata["document_id"], []).append(doc)
        calon = [d for d, docs in per_dokumen.items() if len(docs) >= 2]
        if not calon:
            return []
        rows = await self._jalankan(OUTLINE_SQL, {"ids": calon})

        kepala: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            kepala.setdefault(row["document_id"], []).append(row)
        hasil: list[Document] = []
        for document_id in calon:
            docs = per_dokumen[document_id]
            daftar = susun_daftar_bab(kepala.get(document_id, []))
            ids = [d.metadata["chunk_id"] for d in docs]
            if daftar is None or len(daftar.tersentuh(ids)) < 2:
                continue
            meta = docs[0].metadata
            hasil.append(
                Document(
                    page_content=daftar.teks(meta["judul"]),
                    metadata={
                        # Tanpa `chunk_id`: bukan potongan, jadi tidak dicatat
                        # sebagai potongan terambil (log, tabel Uji coba).
                        "document_id": document_id,
                        "judul": meta["judul"],
                        "jenis": meta.get("jenis"),
                        "halaman": daftar.bab[0].halaman,
                        "file_path": meta["file_path"],
                        "rrf_score": 0.0,
                        "raw_scores": {},
                        "ranks": {},
                        "daftar_bab": [
                            {"judul": b.judul, "halaman": b.halaman} for b in daftar.bab
                        ],
                    },
                )
            )
            if len(hasil) >= self.outline:
                break
        return hasil

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
        unit: str | None = None,
        original_query: str | None = None,
    ) -> list[Document]:
        # `unit` dan `original_query` harus ada juga di sini: LangChain hanya
        # meneruskan argumen tambahan `ainvoke` bila tanda tangan metode
        # SINKRON ini memintanya.
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
