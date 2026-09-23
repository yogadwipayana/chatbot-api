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

from app.rag.retriever import (
    FULLTEXT_SQL,
    ITERATIVE_SCAN_SQL,
    VECTOR_SQL,
    PostgresHybridRetriever,
    vector_literal,
)


def baris(chunk_id: str, skor: float, judul: str = "Panduan Akademik 2025") -> dict:
    return {
        "chunk_id": chunk_id,
        "konten": f"isi {chunk_id}",
        "halaman": 12,
        "document_id": "d1",
        "judul": judul,
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
        assert params["query"] == "pengisian KRS"

    async def test_jumlah_kandidat_diteruskan(self, pabrik):
        await retriever_dengan(pabrik, candidates=7).ainvoke("pengisian KRS")
        assert all(p["limit"] == 7 for s in pabrik.sesi for _, p in s.panggilan)


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
