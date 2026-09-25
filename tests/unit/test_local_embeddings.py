"""Embedding lokal (multilingual-e5): awalan, padding, dan jalur biaya."""

from __future__ import annotations

import math

import pytest

from app.config import Settings
from app.deps import EmbedQuery
from app.rag.local_embeddings import LocalEmbeddings, pad, pakai_awalan_e5
from app.rag.providers import build_embeddings, embed_with_usage


class EncoderPalsu:
    """Meniru `SentenceTransformer.encode`: vektor 4 dimensi ternormalisasi."""

    def __init__(self) -> None:
        self.masukan: list[list[str]] = []

    def encode(self, texts, normalize_embeddings=False):
        assert normalize_embeddings, "vektor wajib dinormalisasi sebelum dipadding"
        self.masukan.append(list(texts))
        hasil = []
        for t in texts:
            v = [float(len(t)), 1.0, float(t.count("a")), 0.5]
            n = math.sqrt(sum(x * x for x in v))
            hasil.append([x / n for x in v])
        return hasil


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.fixture
def encoder() -> EncoderPalsu:
    return EncoderPalsu()


@pytest.fixture
def e5(encoder) -> LocalEmbeddings:
    return LocalEmbeddings("intfloat/multilingual-e5-small", 8, encoder=encoder)


class TestAwalan:
    def test_query_dan_passage_berbeda(self, e5, encoder):
        e5.embed_query("kapan KRS")
        e5.embed_documents(["Pengisian KRS dibuka ..."])
        assert encoder.masukan == [["query: kapan KRS"], ["passage: Pengisian KRS dibuka ..."]]

    def test_model_bukan_e5_tanpa_awalan(self, encoder):
        emb = LocalEmbeddings("sentence-transformers/LaBSE", 8, encoder=encoder)
        emb.embed_query("kapan KRS")
        assert encoder.masukan == [["kapan KRS"]]

    @pytest.mark.parametrize(
        "model,harap",
        [
            ("intfloat/multilingual-e5-small", True),
            ("intfloat/multilingual-e5-large-instruct", True),
            ("BAAI/bge-m3", False),
        ],
    )
    def test_deteksi_e5(self, model, harap):
        assert pakai_awalan_e5(model) is harap


class TestPadding:
    def test_diisi_nol_sampai_dimensi_kolom(self, e5):
        v = e5.embed_query("kapan KRS")
        assert len(v) == 8
        assert v[4:] == [0.0] * 4

    def test_cosine_tidak_berubah_oleh_padding(self):
        a, b = [0.6, 0.8, 0.0], [0.0, 0.6, 0.8]
        assert cosine(pad(a, 1024), pad(b, 1024)) == pytest.approx(cosine(a, b))

    def test_vektor_lebih_panjang_dari_kolom_ditolak(self):
        with pytest.raises(ValueError, match="melebihi"):
            pad([0.1] * 5, 4)


class TestJalurBiaya:
    async def test_query_lewat_awalan_query(self, e5, encoder):
        hasil = await embed_with_usage(e5, ["kapan KRS"], sebagai_query=True)
        assert encoder.masukan == [["query: kapan KRS"]]
        assert hasil.biaya_usd == 0.0

    async def test_dokumen_lewat_awalan_passage(self, e5, encoder):
        await embed_with_usage(e5, ["isi"])
        assert encoder.masukan == [["passage: isi"]]

    async def test_embed_query_mencatat_biaya_nol_bukan_tidak_diketahui(self, e5):
        """None di AD-5 berarti "berbiaya tetapi tak terhitung" -- salah untuk model lokal."""
        embed = EmbedQuery(e5, "intfloat/multilingual-e5-small")
        v = await embed("kapan KRS")
        assert len(v) == 8
        assert embed.panggilan == 1
        assert embed.biaya_usd == 0.0
        assert embed.tokens is None


class TestBuildEmbeddings:
    def test_provider_local(self):
        s = Settings(
            _env_file=None,
            embed_provider="local",
            embed_model="intfloat/multilingual-e5-small",
        )
        emb = build_embeddings(s)
        assert isinstance(emb, LocalEmbeddings)
        assert emb.dim == 1024
