"""Pemakaian dan biaya panggilan embedding (FR-8, AD-5).

`aembed_query` hanya mengembalikan vektor -- antarmuka `Embeddings` LangChain
membuang `response["usage"]`, termasuk jumlah token dan biaya yang dilaporkan
penyedia. Tanpa test ini, hilangnya angka itu tidak menghasilkan galat apa pun:
halaman Biaya hanya melaporkan lebih sedikit dari yang sebenarnya dibelanjakan.
"""

from __future__ import annotations

import pytest

from app.deps import EmbedQuery
from app.observability.costs import (
    SUMBER_ESTIMASI,
    SUMBER_PROVIDER,
    biaya_embedding,
    estimate_input_cost,
)
from app.rag.providers import embed_with_usage

MODEL = "openrouter/openai/text-embedding-3-small"
TARIF = {MODEL: (0.02, 0.0)}

# Persis bentuk yang dikirim gateway proyek ini, termasuk nama model yang
# berbeda dari yang diminta dan `cost` yang bukan bagian spesifikasi OpenAI.
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
    async def test_membaca_token_biaya_dan_model(self):
        hasil = await embed_with_usage(FakeEmbeddings(), ["halo"])
        assert hasil.tokens == 9
        assert hasil.biaya_usd == pytest.approx(1.8e-07)
        assert hasil.is_byok is False
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
        assert (hasil.tokens, hasil.biaya_usd, hasil.is_byok) == (None, None, None)
        assert hasil.vectors == [[0.1, 0.2, 0.3]]

    async def test_usage_berbentuk_aneh_tidak_menggagalkan(self):
        """Laporan pemakaian di luar spesifikasi OpenAI; bentuknya tidak dijamin."""
        usage = {"prompt_tokens": "sembilan", "cost": None, "is_byok": "mungkin"}
        hasil = await embed_with_usage(FakeEmbeddings(usage=usage), ["halo"])
        assert (hasil.tokens, hasil.biaya_usd, hasil.is_byok) == (None, None, None)

    async def test_jalur_len_safe_dilewatkan_ke_langchain(self):
        """Tanpa BASE_URL, langchain-openai mengirim token ID hasil tiktoken dan
        memecah sendiri teks kepanjangan -- jangan ditiru, cukup kehilangan angkanya."""
        emb = FakeEmbeddings()
        emb.check_embedding_ctx_length = True
        hasil = await embed_with_usage(emb, ["halo"])
        assert emb.dipanggil_lewat_langchain is True
        assert hasil.tokens is None and hasil.biaya_usd is None


class TestEmbedQuery:
    async def test_mencatat_biaya_dari_penyedia(self):
        eq = EmbedQuery(FakeEmbeddings(), MODEL)
        vektor = await eq("halo")
        assert vektor == [0.1, 0.2, 0.3]
        assert eq.tokens == 9
        assert eq.biaya_usd == pytest.approx(1.8e-07)
        assert eq.biaya_sumber == SUMBER_PROVIDER

    async def test_biaya_tidak_dibulatkan_menjadi_nol(self):
        """`round(usd, 6)` milik `estimate_cost` mengubah biaya satu pertanyaan
        menjadi nol bulat. Ribuan pertanyaan gratis bukan laporan biaya."""
        eq = EmbedQuery(FakeEmbeddings(), MODEL)
        await eq("halo")
        assert eq.biaya_usd > 0
        assert round(eq.biaya_usd, 6) == 0.0

    async def test_model_diminta_dipertahankan_sebagai_kunci_tarif(self):
        """Gateway menjawab `text-embedding-3-small` untuk permintaan
        `openrouter/openai/text-embedding-3-small`; nama pendek itu tidak
        terdaftar di PRICES_PER_MTOK, jadi memakainya menghapus tarifnya."""
        eq = EmbedQuery(FakeEmbeddings(model_dilaporkan="text-embedding-3-small"), MODEL)
        await eq("halo")
        assert eq.model == MODEL
        assert eq.model_dilaporkan == "text-embedding-3-small"

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
        assert eq.biaya_usd == pytest.approx(3.6e-07)

    async def test_jatuh_ke_taksiran_bila_biaya_tak_dilaporkan(self, monkeypatch):
        monkeypatch.setattr("app.observability.costs.PRICES_PER_MTOK", TARIF)
        emb = FakeEmbeddings(usage={"prompt_tokens": 9, "total_tokens": 9})
        eq = EmbedQuery(emb, MODEL)
        await eq("halo")
        assert eq.biaya_usd == pytest.approx(9 / 1_000_000 * 0.02)
        assert eq.biaya_sumber == SUMBER_ESTIMASI

    async def test_campuran_sumber_dilaporkan_sebagai_taksiran(self, monkeypatch):
        """Total yang separuhnya taksiran, secara keseluruhan, tetap taksiran."""
        monkeypatch.setattr("app.observability.costs.PRICES_PER_MTOK", TARIF)
        eq = EmbedQuery(FakeEmbeddings(), MODEL)
        await eq("halo")
        assert eq.biaya_sumber == SUMBER_PROVIDER
        eq._embeddings = FakeEmbeddings(usage={"prompt_tokens": 9})
        await eq("halo")
        assert eq.biaya_sumber == SUMBER_ESTIMASI

    async def test_tarif_belum_terdaftar_tidak_menjadi_nol(self, monkeypatch):
        """Biaya yang tidak diketahui harus None, bukan gratis -- AD-5 melaporkannya
        terpisah sebagai peringatan."""
        monkeypatch.setattr("app.observability.costs.PRICES_PER_MTOK", {})
        eq = EmbedQuery(FakeEmbeddings(usage={"prompt_tokens": 9}), "model-antah-berantah")
        await eq("halo")
        assert eq.tokens == 9
        assert eq.biaya_usd is None and eq.biaya_sumber is None


class TestBiayaEmbedding:
    def test_penyedia_mengalahkan_taksiran(self, monkeypatch):
        """Tarif penyedia berlaku saat itu juga; PRICES_PER_MTOK salinan yang bisa usang."""
        monkeypatch.setattr("app.observability.costs.PRICES_PER_MTOK", TARIF)
        assert biaya_embedding(MODEL, 9, 5.0) == (5.0, SUMBER_PROVIDER)

    def test_biaya_nol_dari_penyedia_bukan_berarti_tak_dilaporkan(self, monkeypatch):
        """Model gratis benar-benar ada; 0.0 tidak boleh memicu taksiran."""
        monkeypatch.setattr("app.observability.costs.PRICES_PER_MTOK", TARIF)
        assert biaya_embedding(MODEL, 9, 0.0) == (0.0, SUMBER_PROVIDER)

    def test_tanpa_token_dan_tanpa_biaya(self):
        assert biaya_embedding(MODEL, None, None) == (None, None)

    def test_estimate_input_cost_tidak_dibulatkan(self, monkeypatch):
        monkeypatch.setattr("app.observability.costs.PRICES_PER_MTOK", TARIF)
        assert estimate_input_cost(MODEL, 9) == pytest.approx(1.8e-07)

    def test_estimate_input_cost_abaikan_sisi_output(self, monkeypatch):
        monkeypatch.setattr("app.observability.costs.PRICES_PER_MTOK", {MODEL: (0.02, 99.0)})
        assert estimate_input_cost(MODEL, 1_000_000) == pytest.approx(0.02)

    def test_token_negatif_ditolak(self, monkeypatch):
        monkeypatch.setattr("app.observability.costs.PRICES_PER_MTOK", TARIF)
        with pytest.raises(ValueError):
            estimate_input_cost(MODEL, -1)
