"""Reranker setelah RRF: urutan, pemotongan, fail-open, dan pengaruhnya pada FR-3."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.rag.chain import _hits_from_documents
from app.rag.reranker import (
    RERANK_SCORE_KEY,
    TEI_MAX_BATCH,
    ApiReranker,
    LocalReranker,
    TeiReranker,
    build_reranker,
    rerank_documents,
    skor_dari_respons,
)
from app.rag.threshold import Decision, ThresholdPolicy, evaluate
from app.routers.chat import policy_from
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

    def test_bentuk_daftar_tei(self):
        data = [{"index": 1, "score": 0.7}, {"index": 0, "score": 0.2}]
        assert skor_dari_respons(data, 2) == [0.2, 0.7]

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


class TestTeiReranker:
    async def test_bentuk_permintaan_dan_urutan_skor(self):
        diterima: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            diterima.append(request)
            return httpx.Response(
                200, json=[{"index": 1, "score": 0.8}, {"index": 0, "score": 0.1}]
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = TeiReranker(
                base_url="http://tei.test:8081/", api_key="k", model="gte", client=client
            )
            skor = await reranker.score("kapan KRS", ["x", "y"])

        assert skor == [0.1, 0.8]
        req = diterima[0]
        assert str(req.url) == "http://tei.test:8081/rerank"
        assert req.headers["Authorization"] == "Bearer k"
        # Satu server TEI = satu model: nama model tidak dikirim.
        assert json.loads(req.content) == {
            "query": "kapan KRS",
            "texts": ["x", "y"],
            "raw_scores": False,
            "truncate": True,
        }

    async def test_kandidat_melebihi_batas_tei_dipecah(self):
        """TEI menolak lebih dari 32 teks per permintaan; indeks tiap kelompok mulai dari 0."""
        ukuran: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            teks = json.loads(request.content)["texts"]
            ukuran.append(len(teks))
            # Terurut menurut skor, seperti TEI; skor = angka di dalam teks.
            hasil = [{"index": i, "score": float(t)} for i, t in enumerate(teks)]
            return httpx.Response(200, json=sorted(hasil, key=lambda h: -h["score"]))

        teks = [str(n / 100) for n in range(TEI_MAX_BATCH + 8)]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = TeiReranker(
                base_url="http://tei.test", api_key=None, model="gte", client=client
            )
            skor = await reranker.score("q", teks)

        assert ukuran == [TEI_MAX_BATCH, 8]
        assert skor == [float(t) for t in teks]

    async def test_tanpa_kunci_tanpa_authorization(self):
        diterima: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            diterima.append(request)
            return httpx.Response(200, json=[{"index": 0, "score": 0.5}])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = TeiReranker(
                base_url="http://tei.test", api_key=None, model="gte", client=client
            )
            await reranker.score("q", ["x"])

        assert "Authorization" not in diterima[0].headers

    async def test_status_galat_dilempar(self):
        """Mis. 401 karena RERANK_API_KEY salah; rerank_documents lalu jatuh ke urutan RRF."""
        transport = httpx.MockTransport(lambda r: httpx.Response(401))
        async with httpx.AsyncClient(transport=transport) as client:
            reranker = TeiReranker(
                base_url="http://tei.test", api_key="salah", model="gte", client=client
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


def setelan(**kw) -> Settings:
    """Reranker menyala ke server TEI; `kw` menimpa sebagian."""
    nilai = {
        "rerank_enabled": True,
        "rerank_base_url": "http://localhost:8081",
        "rerank_model": "Alibaba-NLP/gte-multilingual-reranker-base",
        **kw,
    }
    return Settings(_env_file=None, **nilai)


class TestBuildReranker:
    def test_mati_secara_bawaan(self):
        assert build_reranker(Settings(_env_file=None)) is None

    def test_mati_mengabaikan_setelan_lain(self):
        """Seperti JEV_ENABLED: URL/model boleh tetap terisi saat dimatikan."""
        assert build_reranker(setelan(rerank_enabled=False)) is None

    def test_tei_bawaan_memakai_url_dan_kunci_sendiri(self):
        s = setelan(base_url="https://gw.test/v1", api_key="utama", rerank_api_key="rr")
        reranker = build_reranker(s)
        assert isinstance(reranker, TeiReranker)
        assert reranker.url == "http://localhost:8081/rerank"
        assert reranker._headers["Authorization"] == "Bearer rr"
        assert reranker.model == "Alibaba-NLP/gte-multilingual-reranker-base"

    def test_ganti_model_cukup_url_kunci_dan_nama(self):
        s = setelan(
            rerank_base_url="http://localhost:8082",
            rerank_api_key="kunci-bge",
            rerank_model="BAAI/bge-reranker-v2-m3",
        )
        reranker = build_reranker(s)
        assert isinstance(reranker, TeiReranker)
        assert reranker.url == "http://localhost:8082/rerank"
        assert reranker._headers["Authorization"] == "Bearer kunci-bge"

    def test_kunci_kosong_tidak_jatuh_ke_api_key(self):
        """Kunci gateway tidak boleh ikut terkirim ke server reranker."""
        reranker = build_reranker(setelan(api_key="utama"))
        assert "Authorization" not in reranker._headers

    def test_api_gaya_cohere(self):
        reranker = build_reranker(setelan(rerank_provider="api", rerank_model="rr"))
        assert isinstance(reranker, ApiReranker)
        assert reranker.url == "http://localhost:8081/rerank"

    def test_local_tanpa_url(self):
        s = setelan(rerank_provider="local", rerank_base_url=None)
        assert isinstance(build_reranker(s), LocalReranker)

    def test_menyala_tanpa_model_ditolak_saat_start(self):
        with pytest.raises(ValueError, match="RERANK_MODEL"):
            setelan(rerank_model="")

    def test_menyala_tanpa_url_ditolak_walau_base_url_ada(self):
        """Gateway BASE_URL tidak punya /rerank, jadi tidak dipakai sebagai cadangan."""
        with pytest.raises(ValueError, match="RERANK_BASE_URL"):
            setelan(rerank_base_url=None, base_url="https://gw.test/v1")

    def test_url_tanpa_skema_ditolak_saat_start(self):
        with pytest.raises(ValueError, match="http://"):
            setelan(rerank_base_url="localhost:8081")

    def test_provider_none_lama_diarahkan_ke_rerank_enabled(self):
        with pytest.raises(ValueError, match="RERANK_ENABLED=false"):
            Settings(_env_file=None, rerank_provider="none")


class TestPolicyFrom:
    def test_ambang_reranker_hanya_berlaku_bila_menyala(self):
        assert policy_from(setelan(rerank_threshold=0.4)).rerank_threshold == 0.4
        mati = setelan(rerank_enabled=False, rerank_threshold=0.4)
        assert policy_from(mati).rerank_threshold is None
