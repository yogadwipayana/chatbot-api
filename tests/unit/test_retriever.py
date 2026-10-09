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
    OUTLINE_SQL,
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


class TestBobotNol:
    """T29: bobot 0 mematikan sumbernya -- pencariannya tidak dijalankan, jadi
    potongannya tidak masuk konteks dan skornya tidak ikut dinilai threshold."""

    async def test_bobot_kata_nol_tidak_menjalankan_fulltext(self, pabrik):
        docs = await retriever_dengan(pabrik, weight_fulltext=0.0).ainvoke("pengisian KRS")
        dijalankan = [sql for s in pabrik.sesi for sql, _ in s.panggilan]
        assert VECTOR_SQL in dijalankan
        assert all(sql is VECTOR_SQL or sql is ITERATIVE_SCAN_SQL for sql in dijalankan)
        assert [d.metadata["chunk_id"] for d in docs] == ["a", "b"]
        assert all("fulltext" not in d.metadata["raw_scores"] for d in docs)

    async def test_bobot_makna_nol_tidak_memanggil_embedding(self, pabrik):
        dipanggil: list[str] = []

        async def rekam(q: str) -> list[float]:
            dipanggil.append(q)
            return [0.1] * 1024

        docs = await PostgresHybridRetriever(
            session_factory=pabrik, embed_query=rekam, weight_vector=0.0
        ).ainvoke("pengisian KRS")
        assert dipanggil == []
        dijalankan = [sql for s in pabrik.sesi for sql, _ in s.panggilan]
        assert VECTOR_SQL not in dijalankan
        assert [d.metadata["chunk_id"] for d in docs] == ["b", "c"]
        assert all("vector" not in d.metadata["raw_scores"] for d in docs)


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


class SesiPerQuery(SesiPalsu):
    async def execute(self, sql, params=None):
        self.panggilan.append((sql, params))
        return HasilPalsu(self.pabrik.baris_untuk_params(sql, params))


class PabrikPerQuery(PabrikSesiPalsu):
    """Hasil berbeda untuk tiap bentuk pertanyaan: kunci vektor = literal
    embedding-nya, kunci fulltext = query `or`-nya."""

    def __init__(self, vektor: dict[str, list[dict]], kata: dict[str, list[dict]]) -> None:
        super().__init__([], [])
        self.vektor = vektor
        self.kata = kata

    def baris_untuk_params(self, sql, params) -> list[dict]:
        if sql is VECTOR_SQL:
            return self.vektor.get(params["query_embedding"], [])
        return self.kata.get(params["query"], [])

    def buat_sesi(self) -> SesiPalsu:
        sesi = SesiPerQuery(self)
        self.sesi.append(sesi)
        return sesi


class TestPertanyaanAsli:
    """T40: rewrite mengganti "harga" menjadi "biaya", dan potongan HARGA
    SERTIFIKASI terlempar dari top 5. Pertanyaan asli ikut dicari."""

    REWRITE = "berapa biaya sertifikasi DKV?"
    ASLI = "harga sertifikasi DKV"
    EMBEDDING = {REWRITE: [0.1] * 4, ASLI: [0.2] * 4}

    def pabrik(self) -> PabrikPerQuery:
        v_rewrite, v_asli = (
            vector_literal(self.EMBEDDING[q]) for q in (self.REWRITE, self.ASLI)
        )
        return PabrikPerQuery(
            vektor={
                v_rewrite: [baris("pedoman", 0.86), baris("harga", 0.80)],
                v_asli: [baris("harga", 0.88), baris("transkrip", 0.84)],
            },
            kata={
                fulltext_queries(self.REWRITE)[0]: [baris("pedoman", 0.07)],
                fulltext_queries(self.ASLI)[0]: [baris("harga", 0.09), baris("pedoman", 0.03)],
            },
        )

    def retriever(self, pabrik, **kw) -> PostgresHybridRetriever:
        async def embed(q: str) -> list[float]:
            return self.EMBEDDING[q]

        return PostgresHybridRetriever(session_factory=pabrik, embed_query=embed, **kw)

    async def test_kedua_bentuk_dicari_di_kedua_jalur(self):
        pabrik = self.pabrik()
        await self.retriever(pabrik).ainvoke(self.REWRITE, original_query=self.ASLI)
        panggilan = [(q, p) for s in pabrik.sesi for q, p in s.panggilan]
        assert sum(q is VECTOR_SQL for q, _ in panggilan) == 2
        kata = sorted(p["query"] for q, p in panggilan if q is not VECTOR_SQL)
        assert kata == sorted(
            [fulltext_queries(self.REWRITE)[0], fulltext_queries(self.ASLI)[0]]
        )

    async def test_potongan_yang_ditemukan_kedua_bentuk_menang(self):
        docs = await self.retriever(self.pabrik(), top_n=1).ainvoke(
            self.REWRITE, original_query=self.ASLI
        )
        assert [d.metadata["chunk_id"] for d in docs] == ["harga"]

    async def test_skor_mentah_tertinggi_dan_peringkat_terbaik_per_sumber(self):
        docs = await self.retriever(self.pabrik()).ainvoke(
            self.REWRITE, original_query=self.ASLI
        )
        harga = next(d for d in docs if d.metadata["chunk_id"] == "harga")
        assert harga.metadata["raw_scores"] == {"vector": 0.88, "fulltext": 0.09}
        assert harga.metadata["ranks"] == {"vector": 1, "fulltext": 1}

    async def test_tanpa_pertanyaan_asli_hanya_satu_bentuk(self):
        pabrik = self.pabrik()
        docs = await self.retriever(pabrik).ainvoke(self.REWRITE)
        assert len(pabrik.sesi) == 2
        assert docs[0].metadata["chunk_id"] == "pedoman"

    @pytest.mark.parametrize("asli", [REWRITE, "  Berapa BIAYA sertifikasi  DKV? "])
    async def test_pertanyaan_asli_yang_sama_tidak_dicari_dua_kali(self, asli):
        pabrik = self.pabrik()
        await self.retriever(pabrik).ainvoke(self.REWRITE, original_query=asli)
        assert len(pabrik.sesi) == 2


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


def baris_lanjutan(
    sumber: str, chunk_id: str, halaman: int = 13, *, arah: int = 1, isi: str | None = None
) -> dict:
    return {
        "source_id": sumber,
        "arah": arah,
        "chunk_id": chunk_id,
        "content": isi if isi is not None else f"lanjutan {chunk_id}",
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

    def test_sql_mengambil_posisi_sebelum_dan_sesudah_di_dokumen_yang_sama(self):
        sql = str(NEIGHBOR_SQL)
        assert "n.position IN (c.position - 1, c.position + 1)" in sql
        assert "n.position - c.position AS arah" in sql
        assert "n.document_id = c.document_id" in sql


class TestPotonganSebelumnya:
    """T40: "SKP wajib apa saja?" menemukan butir d-f daftar kegiatan wajib
    (halaman 21), sedangkan butir a-c ada di potongan sebelumnya (halaman 20).

    Peringkat awal: b, a, c. Judul bagian `b` adalah "E. PENERAPAN".
    """

    def pabrik(self, *lanjutan: dict) -> PabrikDenganLanjutan:
        b = {**baris("b", 0.80), "content": "E. PENERAPAN\nd. Alumni Pulang Kampus"}
        return PabrikDenganLanjutan(
            [baris("a", 0.91), b],
            [{**b, "score": 0.40}, baris("c", 0.30)],
            lanjutan=list(lanjutan),
        )

    async def test_bagian_yang_sama_disisipkan_sebelum_sumbernya(self):
        pabrik = self.pabrik(
            baris_lanjutan("b", "b0", 12, arah=-1, isi="E. PENERAPAN\na. PKKMB"),
            baris_lanjutan("b", "b2", arah=1),
        )
        docs = await retriever_dengan(pabrik, neighbors=1).ainvoke("KRS")
        assert [d.metadata["chunk_id"] for d in docs] == ["b0", "b", "b2", "a", "c"]
        b0 = docs[0].metadata
        assert b0["neighbor_of"] == "b"
        assert b0["raw_scores"] == {}
        assert b0["halaman"] == 12

    async def test_penanda_lanjutan_diabaikan_saat_membandingkan_judul(self):
        pabrik = self.pabrik(
            baris_lanjutan("b", "b0", arah=-1, isi="E. PENERAPAN (lanjutan)\nc. Seminar")
        )
        docs = await retriever_dengan(pabrik, neighbors=1).ainvoke("KRS")
        assert [d.metadata["chunk_id"] for d in docs][:2] == ["b0", "b"]

    async def test_bagian_lain_tidak_disisipkan(self):
        """Potongan sebelumnya yang judulnya lain adalah bagian lain: biasanya
        tidak relevan, dan hanya memperpanjang konteks."""
        pabrik = self.pabrik(
            baris_lanjutan("b", "b0", arah=-1, isi="D. SISTEM PENILAIAN\nb. Partisipasi")
        )
        docs = await retriever_dengan(pabrik, neighbors=1).ainvoke("KRS")
        assert [d.metadata["chunk_id"] for d in docs] == ["b", "a", "c"]

    async def test_tetangga_dua_hasil_disisipkan_sekali(self):
        """Potongan sesudah `b` sekaligus potongan sebelum `a`."""
        a = {**baris("a", 0.91), "content": "F. PREDIKAT\n2. BAIK"}
        b = {**baris("b", 0.80), "content": "E. PENERAPAN\nd. Alumni"}
        pabrik = PabrikDenganLanjutan(
            [a, b],
            [{**b, "score": 0.40}],
            lanjutan=[
                baris_lanjutan("b", "f1", arah=1, isi="F. PREDIKAT\n1. SANGAT BAIK"),
                baris_lanjutan("a", "f1", arah=-1, isi="F. PREDIKAT\n1. SANGAT BAIK"),
            ],
        )
        docs = await retriever_dengan(pabrik, neighbors=2).ainvoke("KRS")
        assert [d.metadata["chunk_id"] for d in docs] == ["b", "f1", "a"]


def kepala(chunk_id: str, halaman: int, judul: str, dokumen: str = "d1") -> dict:
    return {"document_id": dokumen, "chunk_id": chunk_id, "page": halaman, "kepala": judul}


class PabrikDenganBab(PabrikSesiPalsu):
    def __init__(self, *args, bab: list[dict]) -> None:
        super().__init__(*args)
        self.bab = bab

    def baris_untuk(self, sql) -> list[dict]:
        return self.bab if sql is OUTLINE_SQL else super().baris_untuk(sql)

    def panggilan_bab(self) -> list[dict]:
        return [p for s in self.sesi for q, p in s.panggilan if q is OUTLINE_SQL]


PEDOMAN = [
    kepala("a", 11, "BAB II KIP KULIAH › 2.8 Persyaratan"),
    kepala("x", 12, "BAB II KIP KULIAH › 2.9 Dokumen (lanjutan)"),
    kepala("b", 27, "BAB VI BERPRESTASI › 6.1 Gambaran Umum"),
    kepala("y", 28, "BAB VI BERPRESTASI › 6.5 Kuota"),
    kepala("c", 32, "BAB VII TALENTA › 7.1 Ketentuan Umum"),
]


class TestDaftarBab:
    """T9: "jenis beasiswa apa saja?" -- enam jenis, satu per bab pedoman.

    Peringkat awal (lihat fixture `pabrik`): b, a, c, semuanya dokumen d1,
    masing-masing di bab yang berbeda.
    """

    def pabrik(self, bab: list[dict] = PEDOMAN, vector=None, fulltext=None) -> PabrikDenganBab:
        return PabrikDenganBab(
            vector if vector is not None else [baris("a", 0.91), baris("b", 0.80)],
            fulltext if fulltext is not None else [baris("b", 0.40), baris("c", 0.30)],
            bab=bab,
        )

    @staticmethod
    def daftar(docs) -> list:
        return [d for d in docs if d.metadata.get("daftar_bab")]

    async def test_mati_secara_bawaan_tanpa_query_tambahan(self):
        pabrik = self.pabrik()
        docs = await retriever_dengan(pabrik).ainvoke("beasiswa")
        assert self.daftar(docs) == []
        assert pabrik.panggilan_bab() == []

    async def test_daftar_bab_ditaruh_paling_akhir(self):
        docs = await retriever_dengan(self.pabrik(), outline=2).ainvoke("beasiswa")
        assert [d.metadata.get("chunk_id") for d in docs] == ["b", "a", "c", None]
        assert docs[-1].metadata["daftar_bab"] == [
            {"judul": "BAB II KIP KULIAH", "halaman": 11},
            {"judul": "BAB VI BERPRESTASI", "halaman": 27},
            {"judul": "BAB VII TALENTA", "halaman": 32},
        ]

    async def test_isi_dan_metadata_sitasi(self):
        docs = await retriever_dengan(self.pabrik(), outline=2).ainvoke("beasiswa")
        meta = docs[-1].metadata
        assert meta["judul"] == "Panduan Akademik 2025"
        assert meta["document_id"] == "d1"
        assert meta["file_path"] == "documents/d1.pdf"
        assert meta["halaman"] == 11
        assert "- BAB VI BERPRESTASI (hal. 27)" in docs[-1].page_content

    async def test_tanpa_skor_dan_bukan_potongan(self):
        """Tidak boleh meloloskan threshold atas namanya sendiri, dan tidak
        dicatat sebagai potongan terambil (`chunk_id` dipakai log dan Uji coba)."""
        docs = await retriever_dengan(self.pabrik(), outline=2).ainvoke("beasiswa")
        meta = docs[-1].metadata
        assert meta["raw_scores"] == {}
        assert meta["rrf_score"] == 0.0
        assert "chunk_id" not in meta

    async def test_sql_hanya_untuk_dokumen_dengan_dua_hasil_atau_lebih(self):
        pabrik = self.pabrik()
        await retriever_dengan(pabrik, outline=2).ainvoke("beasiswa")
        assert pabrik.panggilan_bab() == [{"ids": ["d1"]}]

    async def test_hasil_satu_bab_tidak_diberi_daftar(self):
        """ "syarat KIP": semua hasil di BAB II, gambaran dokumen tidak perlu."""
        bab = [
            kepala("a", 11, "BAB II KIP KULIAH › 2.8 Persyaratan"),
            kepala("b", 12, "BAB II KIP KULIAH › 2.9 Dokumen"),
            kepala("c", 12, "BAB II KIP KULIAH › 2.10 Mekanisme"),
            kepala("y", 27, "BAB VI BERPRESTASI › 6.1 Gambaran Umum"),
            kepala("z", 32, "BAB VII TALENTA › 7.1 Ketentuan Umum"),
        ]
        docs = await retriever_dengan(self.pabrik(bab), outline=2).ainvoke("syarat KIP")
        assert self.daftar(docs) == []

    async def test_dokumen_dengan_satu_hasil_tidak_dicari(self):
        pabrik = self.pabrik(
            vector=[{**baris("a", 0.91), "document_id": "d2"}, baris("b", 0.80)],
            fulltext=[baris("b", 0.40), {**baris("c", 0.30), "document_id": "d3"}],
        )
        await retriever_dengan(pabrik, outline=2).ainvoke("beasiswa")
        assert pabrik.panggilan_bab() == []

    async def test_tanya_jawab_dilewati(self):
        tj = {"type": "tanya_jawab"}
        pabrik = self.pabrik(
            vector=[{**baris("a", 0.91), **tj}, {**baris("b", 0.80), **tj}],
            fulltext=[{**baris("b", 0.40), **tj}],
        )
        await retriever_dengan(pabrik, outline=2).ainvoke("beasiswa")
        assert pabrik.panggilan_bab() == []

    async def test_jumlah_dokumen_dibatasi_menurut_hasil_terbaik(self):
        """Peringkat: b (d2), a (d1), c (d1), e (d2)."""
        d2 = {"document_id": "d2", "title": "Buku SKP"}
        bab = [
            *PEDOMAN,
            kepala("b", 19, "SATUAN KREDIT PARTISIPASI › A. PENGERTIAN", "d2"),
            kepala("e", 23, "KEGIATAN WAJIB INSTITUSI", "d2"),
            kepala("e2", 23, "KEGIATAN WAJIB INSTITUSI (lanjutan)", "d2"),
            kepala("f", 38, "TEKNIS PELAKSANAAN SKP", "d2"),
        ]
        vector = [baris("a", 0.91), {**baris("b", 0.80), **d2}]
        fulltext = [{**baris("b", 0.40), **d2}, baris("c", 0.30), {**baris("e", 0.20), **d2}]

        satu = await retriever_dengan(self.pabrik(bab, vector, fulltext), outline=1).ainvoke(
            "skp"
        )
        assert [d.metadata["judul"] for d in self.daftar(satu)] == ["Buku SKP"]

        dua = await retriever_dengan(self.pabrik(bab, vector, fulltext), outline=2).ainvoke(
            "skp"
        )
        assert [d.metadata["judul"] for d in self.daftar(dua)] == [
            "Buku SKP",
            "Panduan Akademik 2025",
        ]

    async def test_bersama_potongan_lanjutan(self):
        """Dua query tambahan, masing-masing di sesinya sendiri."""
        pabrik = PabrikDenganBab(
            [baris("a", 0.91), baris("b", 0.80)],
            [baris("b", 0.40), baris("c", 0.30)],
            bab=PEDOMAN,
        )
        lanjutan = [baris_lanjutan("b", "b2")]
        pabrik.baris_untuk = lambda sql, asli=pabrik.baris_untuk: (
            lanjutan if sql is NEIGHBOR_SQL else asli(sql)
        )
        docs = await retriever_dengan(pabrik, neighbors=1, outline=2).ainvoke("beasiswa")
        assert [d.metadata.get("chunk_id") for d in docs] == ["b", "b2", "a", "c", None]

    def test_sql_hanya_mengambil_baris_pertama_menurut_posisi(self):
        sql = str(OUTLINE_SQL)
        assert "split_part(c.content, chr(10), 1) AS kepala" in sql
        assert "ORDER BY c.document_id, c.position" in sql


def dari(dokumen: str, chunk_id: str, skor: float) -> dict:
    return {**baris(chunk_id, skor), "document_id": dokumen}


class TestBatasPerDokumen:
    """T59: "sertifikasi dasar DKV apa dan berapa?" -- kelima kursi diisi
    TRANSKRIP UPS, HARGA SERTIFIKASI (yang memuat harganya) di peringkat 6."""

    @staticmethod
    def pabrik(*vektor: dict) -> PabrikSesiPalsu:
        return PabrikSesiPalsu(vector_rows=list(vektor), fulltext_rows=[])

    @staticmethod
    def ids(docs) -> list[str]:
        return [d.metadata["chunk_id"] for d in docs]

    async def test_mati_secara_bawaan(self):
        pabrik = self.pabrik(dari("d1", "a", 0.9), dari("d1", "b", 0.8), dari("d2", "e", 0.5))
        assert self.ids(await retriever_dengan(pabrik, top_n=2).ainvoke("q")) == ["a", "b"]

    async def test_pengganti_diambil_dari_luar_top_n(self):
        transkrip = [dari("transkrip", x, 0.9 - i / 100) for i, x in enumerate("abcde")]
        pabrik = self.pabrik(*transkrip, dari("harga", "h", 0.5))
        docs = await retriever_dengan(pabrik, top_n=5, max_per_document=4).ainvoke("dkv")
        assert self.ids(docs) == ["a", "b", "c", "d", "h"]

    async def test_kursi_sisa_diisi_dokumen_yang_sama_menurut_peringkat(self):
        """Peringkat a, b, c (d1), e (d2), f (d1). Batas 2: a, b, e, lalu satu
        kursi tersisa diisi c, potongan d1 terbaik yang dilewati."""
        pabrik = self.pabrik(
            dari("d1", "a", 0.9),
            dari("d1", "b", 0.8),
            dari("d1", "c", 0.7),
            dari("d2", "e", 0.6),
            dari("d1", "f", 0.5),
        )
        docs = await retriever_dengan(pabrik, top_n=4, max_per_document=2).ainvoke("q")
        assert self.ids(docs) == ["a", "b", "c", "e"]

    async def test_unit_berdokumen_tunggal_tetap_mendapat_top_n(self):
        pabrik = self.pabrik(*(dari("va", x, 0.9 - i / 10) for i, x in enumerate("abcd")))
        docs = await retriever_dengan(pabrik, top_n=3, max_per_document=1).ainvoke("atm")
        assert self.ids(docs) == ["a", "b", "c"]

    async def test_tanpa_batas_seluruh_hasil_fusi_tidak_ikut(self):
        """Fusi kini membawa semua hasil; yang keluar tetap `top_n`."""
        pabrik = self.pabrik(*(dari("d1", x, 0.9 - i / 10) for i, x in enumerate("abcd")))
        assert len(await retriever_dengan(pabrik, top_n=2).ainvoke("q")) == 2
