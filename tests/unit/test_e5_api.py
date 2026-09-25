"""multilingual-e5 lewat endpoint OpenAI-compatible (`EMBED_PROVIDER=api`).

Endpoint hanya menerima teks dan mengembalikan vektor apa adanya. Awalan e5 dan
padding ke kolom 1024 dimensi harus dipasang di sisi kita -- di jalur LangChain
(`aembed_*`, dipakai ingestion dan re-index) maupun di jalur yang memanggil klien
langsung untuk membaca biaya (`embed_with_usage`, dipakai setiap pertanyaan).
Kegagalan di salah satunya tidak menghasilkan galat: retrieval hanya memburuk.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.db.models import EMBEDDING_DIM
from app.rag.providers import build_embeddings, embed_with_usage

E5 = "intfloat/multilingual-e5-small"


class FakeClient:
    """Endpoint tiruan: vektor 384 dimensi seperti e5-small, satu per teks."""

    def __init__(self) -> None:
        self.panggilan: list[dict] = []

    async def create(self, **kwargs):
        self.panggilan.append(kwargs)
        return {
            "data": [
                {"embedding": [0.5] * 384, "index": i} for i, _ in enumerate(kwargs["input"])
            ],
            "model": E5,
            "usage": {"prompt_tokens": 7, "total_tokens": 7, "cost": 1e-08},
        }


def settings(model: str = E5) -> Settings:
    return Settings(
        embed_provider="api", embed_model=model, base_url="https://gateway.test/v1", api_key="k"
    )


@pytest.fixture
def e5():
    emb = build_embeddings(settings())
    emb.async_client = FakeClient()
    return emb


class TestPembangun:
    def test_e5_tanpa_dimensions(self, e5):
        assert e5.dimensions is None
        assert e5.awalan_e5 is True

    def test_model_lain_tidak_berubah(self):
        emb = build_embeddings(settings("openrouter/openai/text-embedding-3-small"))
        assert emb.dimensions == EMBEDDING_DIM
        assert not getattr(emb, "awalan_e5", False)


class TestJalurLangChain:
    async def test_dokumen_berawalan_passage_dan_dipad(self, e5):
        vektor = await e5.aembed_documents(["Pengisian KRS dibuka 1 Agustus."])
        kirim = e5.async_client.panggilan[0]
        assert kirim["input"] == ["passage: Pengisian KRS dibuka 1 Agustus."]
        assert "dimensions" not in kirim
        assert len(vektor[0]) == EMBEDDING_DIM
        assert vektor[0][:384] == [0.5] * 384 and set(vektor[0][384:]) == {0.0}

    async def test_query_berawalan_query_bukan_passage(self, e5):
        vektor = await e5.aembed_query("kapan KRS dibuka?")
        assert e5.async_client.panggilan[0]["input"] == ["query: kapan KRS dibuka?"]
        assert len(vektor) == EMBEDDING_DIM


class TestJalurBiaya:
    async def test_pertanyaan_berawalan_query_dan_dipad(self, e5):
        hasil = await embed_with_usage(e5, ["kapan KRS dibuka?"], sebagai_query=True)
        kirim = e5.async_client.panggilan[0]
        assert kirim["input"] == ["query: kapan KRS dibuka?"]
        assert "dimensions" not in kirim
        assert len(hasil.vectors[0]) == EMBEDDING_DIM
        assert hasil.tokens == 7
        assert hasil.biaya_usd == pytest.approx(1e-08)

    async def test_dokumen_berawalan_passage(self, e5):
        await embed_with_usage(e5, ["isi dokumen"])
        assert e5.async_client.panggilan[0]["input"] == ["passage: isi dokumen"]

    async def test_vektor_melebihi_kolom_ditolak(self, e5):
        class ClientBesar(FakeClient):
            async def create(self, **kwargs):
                return {"data": [{"embedding": [0.1] * 1536, "index": 0}]}

        e5.async_client = ClientBesar()
        with pytest.raises(ValueError, match="1536"):
            await embed_with_usage(e5, ["x"], sebagai_query=True)
