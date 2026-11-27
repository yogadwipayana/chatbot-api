"""Pemakaian panggilan embedding (FR-8).

`aembed_query` hanya mengembalikan vektor -- antarmuka `Embeddings` LangChain
membuang `response["usage"]`, termasuk jumlah token yang dilaporkan penyedia.
Tanpa test ini, hilangnya angka itu tidak menghasilkan galat apa pun: log
hanya diam-diam kehilangan jumlah tokennya.
"""

from __future__ import annotations

from app.deps import EmbedQuery
from app.rag.providers import embed_with_usage

MODEL = "intfloat/multilingual-e5-small"

# Bentuk yang dikirim gateway proyek ini, termasuk nama model yang berbeda dari
# yang diminta dan kunci di luar spesifikasi OpenAI (`cost`, `is_byok`) yang
# sekarang diabaikan.
USAGE_GATEWAY = {
    "prompt_tokens": 9,
    "total_tokens": 9,
    "cost": 1.8e-07,
    "is_byok": False,
}


class FakeClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.panggilan: list[dict] = []

    async def create(self, **kwargs):
        self.panggilan.append(kwargs)
        return self.response


class FakeEmbeddings:
    """Menyerupai `OpenAIEmbeddings` sejauh yang disentuh `embed_with_usage`."""

    def __init__(self, *, usage: dict | None = USAGE_GATEWAY, model_dilaporkan=MODEL):
        self.model = MODEL
        self.dimensions = 1024
        self.check_embedding_ctx_length = False
        response: dict = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": model_dilaporkan,
        }
        if usage is not None:
            response["usage"] = usage
        self.async_client = FakeClient(response)
        self.dipanggil_lewat_langchain = False

    async def aembed_documents(self, texts):
        self.dipanggil_lewat_langchain = True
        return [[0.1, 0.2, 0.3] for _ in texts]


class TestEmbedWithUsage:
    async def test_membaca_token_dan_model(self):
        hasil = await embed_with_usage(FakeEmbeddings(), ["halo"])
        assert hasil.tokens == 9
        assert hasil.model == MODEL
        assert hasil.vectors == [[0.1, 0.2, 0.3]]

    async def test_dimensions_ikut_dikirim(self):
        """Gateway mengembalikan 1536 dimensi bila `dimensions` tidak disertakan;
        kolom `chunks.embedding` hanya menerima 1024."""
        emb = FakeEmbeddings()
        await embed_with_usage(emb, ["halo"])
        assert emb.async_client.panggilan[0]["dimensions"] == 1024
        assert emb.async_client.panggilan[0]["model"] == MODEL

    async def test_satu_panggilan_untuk_semua_teks(self):
        """Memecah batch di sini membuat usage yang dikembalikan hanya mewakili
        batch terakhir."""
        emb = FakeEmbeddings()
        await embed_with_usage(emb, ["a", "b", "c"])
        assert len(emb.async_client.panggilan) == 1
        assert emb.async_client.panggilan[0]["input"] == ["a", "b", "c"]

    async def test_usage_tidak_dilaporkan_menjadi_none(self):
        """None berarti 'tidak tahu', bukan nol -- gateway lain boleh tidak mengirimnya."""
        hasil = await embed_with_usage(FakeEmbeddings(usage=None), ["halo"])
        assert hasil.tokens is None
        assert hasil.vectors == [[0.1, 0.2, 0.3]]

    async def test_usage_berbentuk_aneh_tidak_menggagalkan(self):
        """Laporan pemakaian di luar spesifikasi OpenAI; bentuknya tidak dijamin."""
        usage = {"prompt_tokens": "sembilan", "cost": None, "is_byok": "mungkin"}
        hasil = await embed_with_usage(FakeEmbeddings(usage=usage), ["halo"])
        assert hasil.tokens is None

    async def test_jalur_len_safe_dilewatkan_ke_langchain(self):
        """Tanpa BASE_URL, langchain-openai mengirim token ID hasil tiktoken dan
        memecah sendiri teks kepanjangan -- jangan ditiru, cukup kehilangan angkanya."""
        emb = FakeEmbeddings()
        emb.check_embedding_ctx_length = True
        hasil = await embed_with_usage(emb, ["halo"])
        assert emb.dipanggil_lewat_langchain is True
        assert hasil.tokens is None


class TestEmbedQuery:
    async def test_mencatat_token_dari_penyedia(self):
        eq = EmbedQuery(FakeEmbeddings(), MODEL)
        vektor = await eq("halo")
        assert vektor == [0.1, 0.2, 0.3]
        assert eq.tokens == 9
        assert eq.panggilan == 1

    async def test_tidak_ada_atribut_biaya(self):
        """Dihapus 2026-11-27: semua model self-hosted."""
        eq = EmbedQuery(FakeEmbeddings(), MODEL)
        await eq("halo")
        assert not any(hasattr(eq, a) for a in ("biaya_usd", "biaya_sumber", "is_byok"))

    async def test_model_diminta_dipertahankan(self):
        """Gateway menjawab `multilingual-e5-small` untuk permintaan
        `intfloat/multilingual-e5-small`; yang dicatat tetap `EMBED_MODEL`."""
        eq = EmbedQuery(FakeEmbeddings(model_dilaporkan="multilingual-e5-small"), MODEL)
        await eq("halo")
        assert eq.model == MODEL
        assert eq.model_dilaporkan == "multilingual-e5-small"

    async def test_model_sama_tidak_dicatat_dua_kali(self):
        eq = EmbedQuery(FakeEmbeddings(model_dilaporkan=MODEL), MODEL)
        await eq("halo")
        assert eq.model_dilaporkan is None

    async def test_akumulasi_antar_panggilan(self):
        """Menimpa berarti diam-diam melaporkan panggilan terakhir saja."""
        eq = EmbedQuery(FakeEmbeddings(), MODEL)
        await eq("satu")
        await eq("dua")
        assert eq.tokens == 18
        assert eq.panggilan == 2

    async def test_token_tak_dilaporkan_tetap_none(self):
        eq = EmbedQuery(FakeEmbeddings(usage=None), MODEL)
        await eq("halo")
        assert eq.panggilan == 1
        assert eq.tokens is None
