"""FR-2 -- PostgresHybridRetriever terhadap PostgreSQL sungguhan.

Yang diuji di sini tidak bisa diuji dengan mock: bahwa SQL-nya sah, bahwa
operator pgvector bekerja, dan -- yang paling penting -- bahwa dokumen
kedaluwarsa dan dokumen nonaktif benar-benar tidak pernah terambil.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.rag.retriever import FTS_CONFIG, PostgresHybridRetriever

pytestmark = pytest.mark.integration


@pytest.fixture
def session_factory(engine):
    """Retriever membuka satu sesi per pencarian, jadi ia butuh pabrik sesi."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
def retriever(session_factory, embed_query) -> PostgresHybridRetriever:
    return PostgresHybridRetriever(
        session_factory=session_factory, embed_query=embed_query, candidates=20, top_n=5
    )


class TestPrasyaratPostgres:
    async def test_ekstensi_vector_terpasang(self, session):
        hasil = await session.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        )
        assert hasil.scalar() == 1

    async def test_konfigurasi_text_search_indonesian_tersedia(self, session):
        """Kalau konfigurasi ini tidak ada, seluruh jalur FTS diam-diam gagal.

        Bila instans target tidak menyediakannya, turunkan `FTS_CONFIG` ke
        'simple' dan catat dampaknya pada set evaluasi -- jangan biarkan
        setengah sistem retrieval mati tanpa diketahui.
        """
        hasil = await session.execute(
            text("SELECT 1 FROM pg_ts_config WHERE cfgname = :nama"),
            {"nama": FTS_CONFIG},
        )
        assert hasil.scalar() == 1, f"konfigurasi text search '{FTS_CONFIG}' tidak ada"


class TestFilterDokumenAktif:
    async def test_dokumen_kedaluwarsa_tidak_pernah_terambil(self, retriever, seed):
        """Metrik PRD §3: 0 dokumen kedaluwarsa aktif di indeks."""
        docs = await retriever.ainvoke("pengisian KRS")
        judul = {d.metadata["judul"] for d in docs}
        assert "Panduan Akademik 2019" not in judul

    async def test_dokumen_nonaktif_tidak_pernah_terambil(self, retriever, seed):
        docs = await retriever.ainvoke("pengisian KRS")
        judul = {d.metadata["judul"] for d in docs}
        assert "Draf Panduan Internal" not in judul

    async def test_dokumen_aktif_tetap_terambil(self, retriever, seed):
        docs = await retriever.ainvoke("pengisian KRS")
        assert "Panduan Akademik 2025" in {d.metadata["judul"] for d in docs}

    async def test_filter_berlaku_pada_jalur_vektor(self, retriever, seed):
        rows = await retriever._vector_search([1.0, 0.0] + [0.0] * 6)
        assert len(rows) == 1

    async def test_filter_berlaku_pada_jalur_fulltext(self, retriever, seed):
        rows = await retriever._fulltext_search("pengisian KRS")
        assert len(rows) == 1


class TestHasilRetrieval:
    async def test_mengembalikan_document_dengan_metadata_lengkap(self, retriever, seed):
        docs = await retriever.ainvoke("pengisian KRS")
        meta = docs[0].metadata
        assert {"chunk_id", "document_id", "judul", "halaman", "file_path"} <= set(meta)

    async def test_skor_mentah_kedua_sumber_ikut_terbawa(self, retriever, seed):
        """FR-3 menilai skor mentah; kalau hilang di sini, ambang jadi buta.

        Kedua sumber harus terisi -- ini sekaligus membuktikan vector search
        dan fulltext search sama-sama berhasil saat dijalankan paralel.
        """
        docs = await retriever.ainvoke("pengisian KRS")
        assert set(docs[0].metadata["raw_scores"]) == {"vector", "fulltext"}

    async def test_skor_kemiripan_dalam_rentang_wajar(self, retriever, seed):
        rows = await retriever._vector_search([1.0, 0.0] + [0.0] * 6)
        assert 0.0 <= rows[0]["score"] <= 1.0

    async def test_isi_chunk_ikut_dikembalikan(self, retriever, seed):
        docs = await retriever.ainvoke("pengisian KRS")
        assert "KRS" in docs[0].page_content

    async def test_top_n_dihormati(self, session_factory, embed_query, seed):
        kecil = PostgresHybridRetriever(
            session_factory=session_factory, embed_query=embed_query, candidates=20, top_n=1
        )
        assert len(await kecil.ainvoke("pengisian KRS")) <= 1

    async def test_query_tanpa_kecocokan_tidak_menggagalkan(self, retriever, seed):
        docs = await retriever.ainvoke("resep rendang padang")
        skor = [d.metadata["raw_scores"].get("vector", 0.0) for d in docs]
        assert all(s <= 1.0 for s in skor)


class TestModeSinkron:
    def test_jalur_sinkron_ditolak_dengan_pesan_jelas(self, retriever):
        """Dua pencarian harus tetap paralel; jalur sinkron akan membuatnya
        berurutan dan melipatgandakan latency."""
        with pytest.raises(NotImplementedError, match="async"):
            retriever.invoke("pengisian KRS")
