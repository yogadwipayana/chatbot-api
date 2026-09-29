"""PostgresHybridRetriever tanpa database -- bentuk parameter dan paralelisme.

Dua bug lolos dari seluruh unit test lain dan baru tertangkap saat retriever
dijalankan terhadap PostgreSQL sungguhan:

1. embedding dikirim sebagai list, padahal asyncpg butuh teks untuk `vector`;
2. dua query dijalankan bersamaan di atas SATU sesi, yang ditolak asyncpg.

Sesi palsu di sini meniru perilaku koneksi asyncpg -- menolak query kedua
selagi yang pertama masih berjalan -- sehingga kedua bug itu terjaga tanpa
perlu database.
"""

from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager

import pytest

from app.rag.fts_query import fulltext_queries
from app.rag.retriever import (
    FULLTEXT_SQL,
    ITERATIVE_SCAN_SQL,
    NEIGHBOR_SQL,
    VECTOR_SQL,
    PostgresHybridRetriever,
    fulltext_sql,
    vector_literal,
)


def baris(chunk_id: str, skor: float, judul: str = "Panduan Akademik 2025") -> dict:
    return {
        "chunk_id": chunk_id,
        "content": f"isi {chunk_id}",
        "page": 12,
        "document_id": "d1",
        "title": judul,
        "file_path": "documents/d1.pdf",
        "score": skor,
    }


class HasilPalsu:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def mappings(self) -> list[dict]:
        return self._rows


class SesiPalsu:
    """Meniru satu koneksi asyncpg: menolak operasi kedua selagi yang pertama berjalan."""

    def __init__(self, pabrik: PabrikSesiPalsu) -> None:
        self.pabrik = pabrik
        self.sibuk = False
        self.panggilan: list[tuple] = []

    async def execute(self, sql, params=None):
        if self.sibuk:
            raise RuntimeError("cannot perform operation: another operation is in progress")
        self.sibuk = True
        self.panggilan.append((sql, params))
        self.pabrik.berjalan += 1
        self.pabrik.puncak = max(self.pabrik.puncak, self.pabrik.berjalan)
        try:
            await asyncio.sleep(0.05)
            return HasilPalsu(self.pabrik.baris_untuk(sql))
        finally:
            self.pabrik.berjalan -= 1
            self.sibuk = False


class PabrikSesiPalsu:
    def __init__(self, vector_rows: list[dict], fulltext_rows: list[dict]) -> None:
        self.vector_rows = vector_rows
        self.fulltext_rows = fulltext_rows
        self.sesi: list[SesiPalsu] = []
        self.berjalan = 0
        self.puncak = 0

    def baris_untuk(self, sql) -> list[dict]:
        return self.vector_rows if sql is VECTOR_SQL else self.fulltext_rows

    def buat_sesi(self) -> SesiPalsu:
        sesi = SesiPalsu(self)
        self.sesi.append(sesi)
        return sesi

    @asynccontextmanager
    async def __call__(self):
        yield self.buat_sesi()


class PabrikSatuSesi(PabrikSesiPalsu):
    """Desain lama: semua pencarian berbagi satu sesi."""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._bersama: SesiPalsu | None = None

    def buat_sesi(self) -> SesiPalsu:
        if self._bersama is None:
            self._bersama = super().buat_sesi()
        return self._bersama


async def embed(_: str) -> list[float]:
    return [0.1] * 1024


def retriever_dengan(pabrik, **kw) -> PostgresHybridRetriever:
    return PostgresHybridRetriever(session_factory=pabrik, embed_query=embed, **kw)


@pytest.fixture
def pabrik() -> PabrikSesiPalsu:
    return PabrikSesiPalsu(
        vector_rows=[baris("a", 0.91), baris("b", 0.80)],
        fulltext_rows=[baris("b", 0.40), baris("c", 0.30)],
    )


class TestVectorLiteral:
    def test_bentuk_literal_pgvector(self):
        assert vector_literal([1.0, 0.5, -0.25]) == "[1.0,0.5,-0.25]"

    def test_selalu_teks_bukan_list(self):
        assert isinstance(vector_literal([0.1, 0.2]), str)

    def test_bilangan_bulat_dijadikan_float(self):
        assert vector_literal([1, 2]) == "[1.0,2.0]"

    def test_nilai_kecil_tetap_terbaca_kembali(self):
        teks = vector_literal([1e-07, 0.123456789])
        kembali = [float(x) for x in teks.strip("[]").split(",")]
        assert kembali == [1e-07, 0.123456789]

    def test_embedding_kosong_ditolak(self):
        with pytest.raises(ValueError, match="kosong"):
            vector_literal([])

    @pytest.mark.parametrize("buruk", [math.nan, math.inf, -math.inf])
    def test_nan_dan_tak_hingga_ditolak(self, buruk):
        """pgvector menolak NaN; lebih jelas gagal di sini daripada di SQL."""
        with pytest.raises(ValueError, match="NaN"):
            vector_literal([0.1, buruk])


class TestParameterQuery:
    async def test_embedding_dikirim_sebagai_teks(self, pabrik):
        await retriever_dengan(pabrik).ainvoke("pengisian KRS")
        params = next(p for s in pabrik.sesi for sql, p in s.panggilan if sql is VECTOR_SQL)
        assert isinstance(params["query_embedding"], str)
        assert params["query_embedding"].startswith("[")

    async def test_pertanyaan_dikirim_ke_fulltext(self, pabrik):
        await retriever_dengan(pabrik).ainvoke("pengisian KRS")
        params = next(p for s in pabrik.sesi for sql, p in s.panggilan if sql is FULLTEXT_SQL)
        assert params["query"] == "pengisian or krs"

    async def test_kata_tanya_dibuang_dan_digabung_or(self, pabrik):
        """T17: dengan DAN atas setiap kata, "berapa harga sertifikasi TOEIC?"
        tidak cocok dengan satu potongan pun."""
        await retriever_dengan(pabrik).ainvoke("berapa harga sertifikasi TOEIC?")
        params = next(p for s in pabrik.sesi for sql, p in s.panggilan if sql is FULLTEXT_SQL)
        assert params["query"] == "harga or sertifikasi or toeic"

    async def test_semua_kata_umum_melewati_fulltext(self, pabrik):
        """Tidak ada yang layak dicocokkan; vektor saja yang memutuskan."""
        docs = await retriever_dengan(pabrik).ainvoke("apa itu?")
        dijalankan = [sql for s in pabrik.sesi for sql, _ in s.panggilan]
        assert VECTOR_SQL in dijalankan
        assert all(sql is VECTOR_SQL or sql is ITERATIVE_SCAN_SQL for sql in dijalankan)
        assert [d.metadata["chunk_id"] for d in docs] == ["a", "b"]

    async def test_jumlah_kandidat_diteruskan(self, pabrik):
        await retriever_dengan(pabrik, candidates=7).ainvoke("pengisian KRS")
        assert all(p["limit"] == 7 for s in pabrik.sesi for _, p in s.panggilan)


class TestKamusSinonim:
    def panggilan_fulltext(self, pabrik) -> list[tuple]:
        return [
            (q, p)
            for s in pabrik.sesi
            for q, p in s.panggilan
            if q is not VECTOR_SQL and q is not ITERATIVE_SCAN_SQL
        ]

    async def test_istilah_kampus_mengirim_semua_varian(self, pabrik):
        await retriever_dengan(pabrik).ainvoke("akreditasi STIKI", unit="BAAK")
        [(sql, params)] = self.panggilan_fulltext(pabrik)
        varian = fulltext_queries("akreditasi STIKI")
        assert sql is fulltext_sql(len(varian))
        assert params["query"] == "akreditasi or stiki"
        assert '"institut bisnis dan teknologi indonesia"' in " ".join(varian)
        assert [params[f"query_{i}"] for i in range(1, len(varian))] == varian[1:]
        assert params["unit"] == "BAAK"

    async def test_jalur_vektor_tetap_memakai_query_asli(self, pabrik):
        """Teks tambahan di query embedding akan menggeser skor vektor yang
        dipakai kalibrasi threshold."""
        teks: list[str] = []

        async def rekam(q: str) -> list[float]:
            teks.append(q)
            return [0.1] * 1024

        await PostgresHybridRetriever(session_factory=pabrik, embed_query=rekam).ainvoke(
            "akreditasi STIKI"
        )
        assert teks == ["akreditasi STIKI"]


class TestSqlFulltext:
    def test_satu_varian_sama_dengan_sql_lama(self):
        assert fulltext_sql(1) is FULLTEXT_SQL
        assert "GREATEST" not in str(FULLTEXT_SQL)
        assert "||" not in str(FULLTEXT_SQL)

    def test_banyak_varian_digabung_dan_skor_terbaik(self):
        """Skor `ts_rank` atas tsquery gabungan anjlok untuk chunk yang cocok
        dengan query asli; skor harus diambil per varian lalu yang terbaik."""
        sql = str(fulltext_sql(3))
        assert sql.count("websearch_to_tsquery") == 6  # 3 di skor, 3 di WHERE
        assert "GREATEST(ts_rank" in sql
        assert "|| websearch_to_tsquery" in sql
        for nama in (":query)", ":query_1)", ":query_2)"):
            assert sql.count(nama) == 2

    def test_banyak_varian_tetap_menyaring_dokumen(self):
        sql = str(fulltext_sql(4))
        assert "is_active" in sql
        assert "valid_until" in sql
        assert ":unit" in sql

    def test_nol_varian_ditolak(self):
        with pytest.raises(ValueError, match="minimal 1"):
            fulltext_sql(0)


class TestFilterUnit:
    def panggilan(self, pabrik, sql) -> list[dict]:
        return [p for s in pabrik.sesi for q, p in s.panggilan if q is sql]

    async def test_tanpa_unit_kedua_query_menerima_null(self, pabrik):
        """NULL = semua unit; parameter tetap dikirim karena SQL-nya menyebutnya."""
        await retriever_dengan(pabrik).ainvoke("pengisian KRS")
        assert self.panggilan(pabrik, VECTOR_SQL)[0]["unit"] is None
        assert self.panggilan(pabrik, FULLTEXT_SQL)[0]["unit"] is None

    async def test_unit_diteruskan_ke_kedua_query(self, pabrik):
        """Keduanya, bukan hanya vektor: chunk unit lain yang lolos lewat
        fulltext tetap akan menjawab pertanyaan dengan dokumen yang salah."""
        await retriever_dengan(pabrik).ainvoke("pengisian KRS", unit="BAAK")
        assert self.panggilan(pabrik, VECTOR_SQL)[0]["unit"] == "BAAK"
        assert self.panggilan(pabrik, FULLTEXT_SQL)[0]["unit"] == "BAAK"

    async def test_iterative_scan_mendahului_query_vektor_di_sesi_yang_sama(self, pabrik):
        """SET LOCAL hanya berlaku di transaksinya sendiri: dikirim di sesi lain,
        query vektor tetap memakai scan biasa yang membuang kandidat."""
        await retriever_dengan(pabrik).ainvoke("pengisian KRS", unit="BAAK")
        sesi_vektor = next(
            s for s in pabrik.sesi if any(q is VECTOR_SQL for q, _ in s.panggilan)
        )
        assert [q for q, _ in sesi_vektor.panggilan] == [ITERATIVE_SCAN_SQL, VECTOR_SQL]

    async def test_iterative_scan_hanya_saat_difilter(self, pabrik):
        await retriever_dengan(pabrik).ainvoke("pengisian KRS")
        assert not any(q is ITERATIVE_SCAN_SQL for s in pabrik.sesi for q, _ in s.panggilan)

    async def test_fulltext_tanpa_iterative_scan(self, pabrik):
        await retriever_dengan(pabrik).ainvoke("pengisian KRS", unit="BAAK")
        sesi_fulltext = next(
            s for s in pabrik.sesi if any(q is FULLTEXT_SQL for q, _ in s.panggilan)
        )
        assert [q for q, _ in sesi_fulltext.panggilan] == [FULLTEXT_SQL]

    async def test_iterative_scan_urutan_ketat(self):
        """RRF memakai peringkat; `relaxed_order` boleh mengacak urutan jarak."""
        assert "strict_order" in str(ITERATIVE_SCAN_SQL)
        assert "SET LOCAL" in str(ITERATIVE_SCAN_SQL)


class TestParalel:
    async def test_setiap_pencarian_memakai_sesi_sendiri(self, pabrik):
        await retriever_dengan(pabrik).ainvoke("pengisian KRS")
        assert len(pabrik.sesi) == 2
        assert all(len(s.panggilan) == 1 for s in pabrik.sesi)

    async def test_kedua_pencarian_berjalan_bersamaan(self, pabrik):
        """FR-2: vector dan fulltext paralel. Puncak 2 berarti keduanya pernah
        berjalan pada saat yang sama, bukan berurutan."""
        await retriever_dengan(pabrik).ainvoke("pengisian KRS")
        assert pabrik.puncak == 2

    async def test_satu_sesi_bersama_ditolak_seperti_asyncpg(self):
        """Mendokumentasikan kenapa desain lama gagal: dua query paralel di atas
        satu sesi ditolak. Kalau test ini lolos, sesi palsunya tidak lagi meniru
        asyncpg dan test paralel di atas kehilangan artinya."""
        pabrik = PabrikSatuSesi([baris("a", 0.9)], [baris("a", 0.4)])
        with pytest.raises(RuntimeError, match="another operation is in progress"):
            await retriever_dengan(pabrik).ainvoke("pengisian KRS")


class TestHasil:
    async def test_digabung_dengan_rrf(self, pabrik):
        """'b' muncul di kedua sumber, jadi menang atas 'a' yang hanya peringkat 1 vektor."""
        docs = await retriever_dengan(pabrik).ainvoke("pengisian KRS")
        assert docs[0].metadata["chunk_id"] == "b"

    async def test_skor_mentah_per_sumber_dipertahankan(self, pabrik):
        docs = await retriever_dengan(pabrik).ainvoke("pengisian KRS")
        b = next(d for d in docs if d.metadata["chunk_id"] == "b")
        assert b.metadata["raw_scores"] == {"vector": 0.80, "fulltext": 0.40}

    async def test_top_n_dihormati(self, pabrik):
        assert len(await retriever_dengan(pabrik, top_n=2).ainvoke("KRS")) == 2

    async def test_metadata_untuk_sitasi_lengkap(self, pabrik):
        meta = (await retriever_dengan(pabrik).ainvoke("KRS"))[0].metadata
        assert {"chunk_id", "document_id", "judul", "halaman", "file_path"} <= set(meta)

    async def test_tanpa_hasil_tidak_menggagalkan(self):
        kosong = PabrikSesiPalsu([], [])
        assert await retriever_dengan(kosong).ainvoke("resep rendang") == []


def baris_lanjutan(sumber: str, chunk_id: str, halaman: int = 13) -> dict:
    return {
        "source_id": sumber,
        "chunk_id": chunk_id,
        "content": f"lanjutan {chunk_id}",
        "page": halaman,
        "document_id": "d1",
        "title": "Panduan Akademik 2025",
        "type": "pdf",
        "file_path": "documents/d1.pdf",
    }


class PabrikDenganLanjutan(PabrikSesiPalsu):
    def __init__(self, *args, lanjutan: list[dict]) -> None:
        super().__init__(*args)
        self.lanjutan = lanjutan

    def baris_untuk(self, sql) -> list[dict]:
        return self.lanjutan if sql is NEIGHBOR_SQL else super().baris_untuk(sql)

    def panggilan_lanjutan(self) -> list[dict]:
        return [p for s in self.sesi for q, p in s.panggilan if q is NEIGHBOR_SQL]


class TestPotonganLanjutan:
    """Prosedur yang terbelah antar halaman harus sampai ke LLM utuh.

    Peringkat awal (lihat fixture `pabrik`): b, a, c.
    """

    def pabrik(self, *lanjutan: dict) -> PabrikDenganLanjutan:
        return PabrikDenganLanjutan(
            [baris("a", 0.91), baris("b", 0.80)],
            [baris("b", 0.40), baris("c", 0.30)],
            lanjutan=list(lanjutan),
        )

    async def test_mati_secara_bawaan_tanpa_query_tambahan(self):
        pabrik = self.pabrik(baris_lanjutan("b", "b2"))
        docs = await retriever_dengan(pabrik).ainvoke("KRS")
        assert [d.metadata["chunk_id"] for d in docs] == ["b", "a", "c"]
        assert pabrik.panggilan_lanjutan() == []

    async def test_hanya_hasil_teratas_yang_dicarikan_lanjutan(self):
        pabrik = self.pabrik()
        await retriever_dengan(pabrik, neighbors=2).ainvoke("KRS")
        assert pabrik.panggilan_lanjutan() == [{"ids": ["b", "a"]}]

    async def test_lanjutan_disisipkan_tepat_sesudah_sumbernya(self):
        pabrik = self.pabrik(baris_lanjutan("b", "b2"), baris_lanjutan("a", "a2"))
        docs = await retriever_dengan(pabrik, neighbors=2).ainvoke("KRS")
        assert [d.metadata["chunk_id"] for d in docs] == ["b", "b2", "a", "a2", "c"]

    async def test_lanjutan_yang_sudah_terambil_tidak_digandakan(self):
        pabrik = self.pabrik(baris_lanjutan("b", "c"))
        docs = await retriever_dengan(pabrik, neighbors=2).ainvoke("KRS")
        assert [d.metadata["chunk_id"] for d in docs] == ["b", "a", "c"]

    async def test_lanjutan_tanpa_skor_supaya_tidak_meloloskan_threshold(self):
        pabrik = self.pabrik(baris_lanjutan("b", "b2"))
        docs = await retriever_dengan(pabrik, neighbors=1).ainvoke("KRS")
        b2 = next(d for d in docs if d.metadata["chunk_id"] == "b2")
        assert b2.metadata["raw_scores"] == {}
        assert b2.metadata["neighbor_of"] == "b"

    async def test_metadata_sitasi_lanjutan_lengkap(self):
        pabrik = self.pabrik(baris_lanjutan("b", "b2", halaman=14))
        docs = await retriever_dengan(pabrik, neighbors=1).ainvoke("KRS")
        meta = next(d for d in docs if d.metadata["chunk_id"] == "b2").metadata
        assert meta["halaman"] == 14
        assert {"chunk_id", "document_id", "judul", "halaman", "file_path"} <= set(meta)

    async def test_tanpa_hasil_tidak_mencari_lanjutan(self):
        pabrik = PabrikDenganLanjutan([], [], lanjutan=[])
        assert await retriever_dengan(pabrik, neighbors=2).ainvoke("x") == []
        assert pabrik.panggilan_lanjutan() == []

    def test_sql_mengambil_posisi_berikutnya_di_dokumen_yang_sama(self):
        sql = str(NEIGHBOR_SQL)
        assert "n.position = c.position + 1" in sql
        assert "n.document_id = c.document_id" in sql
