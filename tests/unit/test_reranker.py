"""Reranker setelah RRF: urutan, pemotongan, fail-open, dan pengaruhnya pada FR-3."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.rag.chain import _hits_from_documents
from app.rag.reranker import (
    RERANK_SCORE_KEY,
    ApiReranker,
    LocalReranker,
    build_reranker,
    rerank_documents,
    skor_dari_respons,
)
from app.rag.threshold import Decision, ThresholdPolicy, evaluate
from tests.fixtures.fakes import make_document
from tests.unit.test_retriever import PabrikSesiPalsu, baris, retriever_dengan


class RerankerPalsu:
    model = "palsu"

    def __init__(self, skor: dict[str, float] | None = None, galat: Exception | None = None):
        self.skor = skor or {}
        self.galat = galat
        self.panggilan: list[tuple[str, list[str]]] = []

    async def score(self, query, texts):
        self.panggilan.append((query, list(texts)))
        if self.galat is not None:
            raise self.galat
        return [self.skor.get(t, 0.0) for t in texts]


def dok(chunk_id: str, **kw):
    return make_document(chunk_id, konten=f"isi {chunk_id}", **kw)


class TestSkorDariRespons:
    def test_diurutkan_ulang_menurut_indeks_masukan(self):
        data = {
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ]
        }
        assert skor_dari_respons(data, 3) == [0.2, 0.0, 0.9]

    def test_tanpa_results_ditolak(self):
        with pytest.raises(ValueError, match="results"):
            skor_dari_respons({"data": []}, 1)

    def test_indeks_di_luar_jangkauan_ditolak(self):
        with pytest.raises(ValueError, match="jangkauan"):
            skor_dari_respons({"results": [{"index": 5, "relevance_score": 1}]}, 2)


class TestRerankDocuments:
    async def test_urutan_mengikuti_skor_reranker_lalu_dipotong(self):
        docs = [dok("a"), dok("b"), dok("c")]
        reranker = RerankerPalsu({"isi a": 0.1, "isi b": 0.3, "isi c": 0.9})
        hasil = await rerank_documents("q", docs, reranker, top_n=2)
        assert [d.metadata["chunk_id"] for d in hasil] == ["c", "b"]
        assert hasil[0].metadata[RERANK_SCORE_KEY] == 0.9

    async def test_skor_sama_mempertahankan_urutan_rrf(self):
        docs = [dok("a"), dok("b"), dok("c")]
        hasil = await rerank_documents("q", docs, RerankerPalsu(), top_n=3)
        assert [d.metadata["chunk_id"] for d in hasil] == ["a", "b", "c"]

    async def test_tanpa_reranker_hanya_dipotong(self):
        docs = [dok("a"), dok("b"), dok("c")]
        hasil = await rerank_documents("q", docs, None, top_n=2)
        assert [d.metadata["chunk_id"] for d in hasil] == ["a", "b"]
        assert RERANK_SCORE_KEY not in hasil[0].metadata

    async def test_reranker_gagal_jatuh_ke_urutan_rrf(self):
        """Reranker memperbaiki mutu; kegagalannya tidak boleh menggagalkan jawaban."""
        docs = [dok("a"), dok("b"), dok("c")]
        reranker = RerankerPalsu(galat=httpx.ConnectError("mati"))
        hasil = await rerank_documents("q", docs, reranker, top_n=2)
        assert [d.metadata["chunk_id"] for d in hasil] == ["a", "b"]
        assert all(RERANK_SCORE_KEY not in d.metadata for d in hasil)

    async def test_jumlah_skor_tidak_cocok_jatuh_ke_urutan_rrf(self):
        class Rusak(RerankerPalsu):
            async def score(self, query, texts):
                return [0.5]

        hasil = await rerank_documents("q", [dok("a"), dok("b")], Rusak(), top_n=2)
        assert [d.metadata["chunk_id"] for d in hasil] == ["a", "b"]


class TestApiReranker:
    async def test_bentuk_permintaan_dan_urutan_skor(self):
        diterima: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            diterima.append(request)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 1, "relevance_score": 0.8},
                        {"index": 0, "relevance_score": 0.1},
                    ]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = ApiReranker(
                base_url="https://gw.test/v1/", api_key="k", model="rr", client=client
            )
            skor = await reranker.score("kapan KRS", ["x", "y"])

        assert skor == [0.1, 0.8]
        req = diterima[0]
        assert str(req.url) == "https://gw.test/v1/rerank"
        assert req.headers["Authorization"] == "Bearer k"
        body = json.loads(req.content)
        assert body == {
            "model": "rr",
            "query": "kapan KRS",
            "documents": ["x", "y"],
            "top_n": 2,
        }

    async def test_status_galat_dilempar(self):
        transport = httpx.MockTransport(lambda r: httpx.Response(500))
        async with httpx.AsyncClient(transport=transport) as client:
            reranker = ApiReranker(
                base_url="https://gw.test", api_key=None, model="rr", client=client
            )
            with pytest.raises(httpx.HTTPStatusError):
                await reranker.score("q", ["x"])


class TestRetrieverDenganReranker:
    async def test_kandidat_rrf_lebih_banyak_lalu_dipotong_ke_top_n(self):
        pabrik = PabrikSesiPalsu(
            vector_rows=[baris("a", 0.9), baris("b", 0.8), baris("c", 0.7)],
            fulltext_rows=[],
        )
        reranker = RerankerPalsu({"isi c": 0.95, "isi a": 0.2, "isi b": 0.1})
        retriever = retriever_dengan(pabrik, top_n=1, reranker=reranker, rerank_candidates=3)
        docs = await retriever.ainvoke("q")
        assert [d.metadata["chunk_id"] for d in docs] == ["c"]
        assert len(reranker.panggilan[0][1]) == 3

    async def test_tanpa_reranker_perilaku_lama(self):
        pabrik = PabrikSesiPalsu(
            vector_rows=[baris("a", 0.9), baris("b", 0.8), baris("c", 0.7)],
            fulltext_rows=[],
        )
        docs = await retriever_dengan(pabrik, top_n=2).ainvoke("q")
        assert [d.metadata["chunk_id"] for d in docs] == ["a", "b"]

    async def test_top_n_lebih_besar_dari_kandidat_rerank(self):
        """RETRIEVAL_TOP_N dapat dinaikkan dari dashboard melewati RERANK_CANDIDATES."""
        pabrik = PabrikSesiPalsu(
            vector_rows=[baris(x, 0.9 - i / 10) for i, x in enumerate("abcd")],
            fulltext_rows=[],
        )
        retriever = retriever_dengan(
            pabrik, top_n=3, reranker=RerankerPalsu(), rerank_candidates=2
        )
        assert len(await retriever.ainvoke("q")) == 3


class TestThresholdReranker:
    def hits(self, rerank: float | None, vector: float = 0.9):
        doc = dok("a", vector_score=vector)
        if rerank is not None:
            doc.metadata[RERANK_SCORE_KEY] = rerank
        return _hits_from_documents([doc])

    def test_skor_reranker_rendah_menolak_walau_vektor_kuat(self):
        policy = ThresholdPolicy(rerank_threshold=0.5)
        hasil = evaluate(self.hits(rerank=0.2, vector=0.9), policy)
        assert hasil.decision is Decision.REFUSE
        assert hasil.top_rerank_score == 0.2

    def test_skor_reranker_tinggi_meloloskan_walau_vektor_lemah(self):
        policy = ThresholdPolicy(rerank_threshold=0.5)
        assert evaluate(self.hits(rerank=0.7, vector=0.1), policy).decision is Decision.PROCEED

    def test_tanpa_skor_reranker_ambang_lama_berlaku(self):
        """Reranker gagal -> tidak ada skor -> ambang vector/leksikal jadi cadangan."""
        policy = ThresholdPolicy(rerank_threshold=0.5)
        assert (
            evaluate(self.hits(rerank=None, vector=0.9), policy).decision is Decision.PROCEED
        )
        assert evaluate(self.hits(rerank=None, vector=0.1), policy).decision is Decision.REFUSE

    def test_ambang_reranker_kosong_mengabaikan_skornya(self):
        hasil = evaluate(self.hits(rerank=0.01, vector=0.9), ThresholdPolicy())
        assert hasil.decision is Decision.PROCEED

    def test_ambang_di_luar_jangkauan_ditolak(self):
        with pytest.raises(ValueError):
            ThresholdPolicy(rerank_threshold=1.5)


class TestBuildReranker:
    def test_mati_secara_bawaan(self):
        assert build_reranker(Settings(_env_file=None)) is None

    def test_api_memakai_base_url_dan_kunci_utama_bila_kosong(self):
        s = Settings(
            _env_file=None,
            base_url="https://gw.test/v1",
            api_key="utama",
            rerank_provider="api",
            rerank_model="rr",
        )
        reranker = build_reranker(s)
        assert isinstance(reranker, ApiReranker)
        assert reranker.url == "https://gw.test/v1/rerank"
        assert reranker._headers["Authorization"] == "Bearer utama"

    def test_local(self):
        s = Settings(
            _env_file=None, rerank_provider="local", rerank_model="BAAI/bge-reranker-v2-m3"
        )
        assert isinstance(build_reranker(s), LocalReranker)

    def test_provider_tanpa_model_ditolak_saat_start(self):
        with pytest.raises(ValueError, match="RERANK_MODEL"):
            Settings(_env_file=None, rerank_provider="local")

    def test_api_tanpa_url_ditolak_saat_start(self):
        with pytest.raises(ValueError, match="BASE_URL"):
            Settings(_env_file=None, rerank_provider="api", rerank_model="rr")
