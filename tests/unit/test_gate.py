"""Gerbang JEV: bentuk permintaan, keputusan per ambang, fail-open, dan posisinya di alur."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from app.config import Settings
from app.rag.chain import OutcomeKind, run_pipeline
from app.rag.gate import (
    QUESTION_KEY,
    GateLabel,
    GatePolicy,
    GateVerdict,
    JevGate,
    build_gate,
    build_request,
    lolos,
    parse_response,
)
from app.rag.rewriter import Turn
from app.rag.threshold import ThresholdPolicy
from tests.fixtures.fakes import FakeRetriever, make_document

POLICY = ThresholdPolicy(vector_threshold=0.35, lexical_threshold=0.05)


def respons(label: str, confidence: float, cost: float | None = 0.00002) -> dict:
    data = {
        "id": "dec_1",
        "model": "typesafe/jev-1.13-20260801",
        "provider": "TypeSafe",
        "answers": {
            QUESTION_KEY: {
                "type": "choice",
                "choice": label,
                "confidence": confidence,
                "probabilities": {label: confidence},
            }
        },
        "usage": {"input_tokens": 300, "output_tokens": 4},
    }
    if cost is not None:
        data["usage"]["cost"] = cost
    return data


def respons_laya(label: str, confidence: float, answer_confidence: float) -> dict:
    """Bentuk `laya-serve`: `confidence` = entropi ternormalisasi, bukan probabilitas."""
    data = respons(label, answer_confidence, cost=None)
    jawaban = data["answers"][QUESTION_KEY]
    jawaban["confidence"] = confidence
    jawaban["answer_confidence"] = answer_confidence
    return data


class GerbangPalsu:
    def __init__(self, verdict: GateVerdict) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    async def __call__(self, question, history=(), unit=None):
        self.calls.append((question, list(history)))
        return self.verdict

    @property
    def called(self) -> bool:
        return bool(self.calls)


def vonis(label: GateLabel, blocked: bool, confidence: float = 0.95) -> GateVerdict:
    return GateVerdict(label, confidence, blocked=blocked)


class TestBuildRequest:
    def test_bentuk_decisions_api(self):
        body = build_request("typesafe/jev-1.13", "kapan KRS?")
        assert body["model"] == "typesafe/jev-1.13"
        assert body["state"] == {"pesan_terbaru": "kapan KRS?"}
        pertanyaan = body["questions"][QUESTION_KEY]
        assert pertanyaan["type"] == "choice"
        assert set(pertanyaan["criteria"]) == {label.value for label in GateLabel}
        json.dumps(body)  # harus dapat diserialisasi apa adanya

    def test_riwayat_ikut_sebagai_konteks(self):
        body = build_request(
            "m", "yang kedua?", [("user", "syarat cuti?"), ("assistant", "...")]
        )
        assert body["state"]["riwayat"][0] == {"peran": "user", "isi": "syarat cuti?"}


class TestParseResponse:
    policy = GatePolicy(block_threshold=0.8, out_of_scope_threshold=0.9)

    @pytest.mark.parametrize("label", ["nonsense", "malicious", "smalltalk"])
    def test_label_blokir_di_atas_ambang(self, label):
        v = parse_response(respons(label, 0.85), self.policy)
        assert v.blocked and v.label == label and v.reply

    @pytest.mark.parametrize("label", ["nonsense", "malicious"])
    def test_ragu_berarti_diteruskan(self, label):
        assert not parse_response(respons(label, 0.6), self.policy).blocked

    def test_out_of_scope_memakai_ambang_lebih_ketat(self):
        assert not parse_response(respons("out_of_scope", 0.85), self.policy).blocked
        assert parse_response(respons("out_of_scope", 0.95), self.policy).blocked

    def test_academic_tidak_pernah_memblokir(self):
        v = parse_response(respons("academic", 0.99), self.policy)
        assert not v.blocked and v.reply == ""

    def test_nilai_tepat_di_ambang_memblokir(self):
        assert parse_response(respons("nonsense", 0.8), self.policy).blocked

    def test_biaya_dibaca_dari_usage(self):
        assert (
            parse_response(respons("academic", 0.9, cost=0.0001), self.policy).cost_usd
            == 0.0001
        )
        assert (
            parse_response(respons("academic", 0.9, cost=None), self.policy).cost_usd is None
        )

    def test_label_tak_dikenal_melempar(self):
        with pytest.raises(ValueError):
            parse_response(respons("olahraga", 0.9), self.policy)

    def test_answer_confidence_diutamakan(self):
        """Laya: "resep rendang padang" p 0,95 tetapi entropi 0,86 (2026-10-02).
        Membaca `confidence` meloloskannya di bawah ambang 0,9."""
        v = parse_response(respons_laya("out_of_scope", 0.86, 0.951), self.policy)
        assert v.blocked and v.confidence == 0.951

    def test_answer_confidence_rendah_tetap_diteruskan(self):
        """Arah sebaliknya: entropi tinggi tidak boleh mengalahkan p yang rendah."""
        assert not parse_response(respons_laya("nonsense", 0.95, 0.6), self.policy).blocked


class TestAmbangNonsense:
    def test_bawaan_mengikuti_block_threshold(self):
        policy = GatePolicy(block_threshold=0.8)
        assert policy.threshold_for(GateLabel.NONSENSE) == 0.8
        assert parse_response(respons("nonsense", 0.85), policy).blocked

    def test_ambang_sendiri_hanya_untuk_nonsense(self):
        """Laya: "toeic brp" -> nonsense 0,81. Label lain tetap memakai 0,7."""
        policy = GatePolicy(block_threshold=0.7, nonsense_threshold=0.95)
        assert not parse_response(respons_laya("nonsense", 0.6, 0.81), policy).blocked
        assert parse_response(respons_laya("nonsense", 0.9, 0.97), policy).blocked
        assert parse_response(respons_laya("malicious", 0.6, 0.81), policy).blocked
        assert parse_response(respons_laya("smalltalk", 0.6, 0.81), policy).blocked


class TestJevGate:
    async def panggil(self, handler, **kw) -> tuple[GateVerdict, list[httpx.Request]]:
        diterima: list[httpx.Request] = []

        def rekam(request):
            diterima.append(request)
            return handler(request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(rekam)) as client:
            gate = JevGate(
                url="https://openrouter.test/api/alpha/decisions",
                api_key="or-key",
                model="typesafe/jev-1.13",
                policy=GatePolicy(),
                client=client,
            )
            return await gate("asdfgh", **kw), diterima

    async def test_permintaan_dan_vonis(self):
        v, req = await self.panggil(
            lambda r: httpx.Response(200, json=respons("nonsense", 0.97))
        )
        assert v.blocked and v.label is GateLabel.NONSENSE
        assert req[0].headers["Authorization"] == "Bearer or-key"
        assert json.loads(req[0].content)["state"]["pesan_terbaru"] == "asdfgh"

    @pytest.mark.parametrize(
        "handler",
        [
            lambda r: httpx.Response(500),
            lambda r: httpx.Response(402, json={"error": "kredit habis"}),
            lambda r: httpx.Response(200, json={"tidak": "sesuai"}),
            lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("lambat")),
        ],
        ids=["500", "402", "bentuk-salah", "timeout"],
    )
    async def test_galat_apa_pun_berarti_diteruskan(self, handler):
        """Fail-open: JEV yang mati tidak boleh mematikan chatbot."""
        v, _ = await self.panggil(handler)
        assert not v.blocked
        assert v.error

    async def test_log_galat_menyebut_jenisnya(self, caplog):
        """T57: `str(httpx.ReadTimeout(""))` kosong, dan baris log dulu berakhir di
        titik dua tanpa sebab."""
        with caplog.at_level(logging.WARNING, logger="app.rag.gate"):
            await self.panggil(lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("")))
        assert "Gerbang JEV gagal, pesan diteruskan: ReadTimeout" in caplog.text


class TestBuildGate:
    def test_mati_secara_bawaan(self):
        assert build_gate(Settings(_env_file=None)) is None

    def test_hidup_tanpa_kunci_ditolak_saat_start(self, monkeypatch):
        monkeypatch.delenv("API_KEY", raising=False)
        with pytest.raises(ValueError, match="JEV_API_KEY"):
            Settings(_env_file=None, jev_enabled=True)

    def test_bawaan_gateway_base_url(self):
        """Tanpa JEV_URL, JEV lewat gateway BASE_URL -- bukan OpenRouter langsung."""
        s = Settings(
            _env_file=None, jev_enabled=True, api_key="kunci-gateway", base_url="https://gw.test/v1/"
        )
        gate = build_gate(s)
        assert gate._api_key == "kunci-gateway"
        assert gate.url == "https://gw.test/v1/systemone"
        assert gate.model == "openrouter/typesafe/jev-1.13"

    def test_jev_url_eksplisit_menang(self):
        s = Settings(
            _env_file=None,
            jev_enabled=True,
            api_key="k",
            base_url="https://gw.test/v1",
            jev_url="https://lain.test/systemone",
        )
        assert build_gate(s).url == "https://lain.test/systemone"

    def test_hidup_tanpa_base_url_ditolak_saat_start(self, monkeypatch):
        monkeypatch.delenv("BASE_URL", raising=False)
        with pytest.raises(ValueError, match="BASE_URL"):
            Settings(_env_file=None, jev_enabled=True, api_key="k")

    def test_hidup(self):
        s = Settings(
            _env_file=None,
            jev_enabled=True,
            jev_api_key="k",
            base_url="https://gw.test/v1",
            jev_out_of_scope_threshold=0.95,
        )
        gate = build_gate(s)
        assert isinstance(gate, JevGate)
        assert gate.policy.out_of_scope_threshold == 0.95
        assert gate.policy.nonsense_threshold is None

    def test_ambang_nonsense_diteruskan_ke_policy(self):
        s = Settings(
            _env_file=None,
            jev_enabled=True,
            jev_api_key="k",
            base_url="https://gw.test/v1",
            jev_nonsense_threshold=0.95,
        )
        assert build_gate(s).policy.threshold_for(GateLabel.NONSENSE) == 0.95

    @pytest.mark.parametrize("nilai", [0.0, 1.5])
    def test_ambang_nonsense_di_luar_rentang_ditolak(self, nilai):
        with pytest.raises(ValueError, match="jev_nonsense_threshold"):
            Settings(_env_file=None, jev_nonsense_threshold=nilai)

    @pytest.mark.parametrize(
        "url",
        [
            "100.111.178.48:8001/v1/systemone",  # kejadian 2026-10-02
            "localhost:8001/v1/systemone",
            "ftp://laya.test/v1/systemone",
        ],
    )
    def test_jev_url_tanpa_skema_ditolak_saat_start(self, url):
        """Bukan fail-open diam-diam saat mahasiswa bertanya."""
        with pytest.raises(ValueError, match="JEV_URL harus diawali"):
            Settings(_env_file=None, jev_enabled=True, api_key="k", jev_url=url)

    def test_jev_url_lokal_diterima(self):
        s = Settings(
            _env_file=None,
            jev_enabled=True,
            api_key="k",
            jev_url="http://127.0.0.1:8001/v1/systemone",
        )
        assert build_gate(s).url == "http://127.0.0.1:8001/v1/systemone"


class TestGerbangDiAlur:
    @pytest.mark.parametrize(
        "label", [GateLabel.NONSENSE, GateLabel.MALICIOUS, GateLabel.OUT_OF_SCOPE]
    )
    async def test_diblokir_tanpa_llm_dan_tanpa_sitasi(self, label, strong_retriever, llm):
        """Pencarian sempat berjalan (paralel dengan gerbang), tetapi hasilnya
        dibuang: LLM penjawab tidak dipanggil dan tidak ada dokumen yang ikut."""
        hasil = await run_pipeline(
            "resep rendang padang",
            retriever=strong_retriever,
            llm_call=llm,
            gate_call=GerbangPalsu(vonis(label, blocked=True)),
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.REJECTED
        assert hasil.text == vonis(label, True).reply
        assert not llm.called
        assert hasil.llm_called is False
        assert hasil.documents == ()
        assert hasil.decision is None
        assert hasil.gate.label is label

    async def test_smalltalk_dari_jev_menjadi_smalltalk(self, strong_retriever, llm):
        hasil = await run_pipeline(
            "apa kabar bot?",
            retriever=strong_retriever,
            llm_call=llm,
            gate_call=GerbangPalsu(vonis(GateLabel.SMALLTALK, blocked=True)),
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.SMALLTALK
        assert not llm.called
        assert hasil.documents == ()

    async def test_smalltalk_pamit_dari_jev_dibalas_penutup(self, strong_retriever, llm):
        """Balasan JEV bawaan untuk smalltalk adalah sapaan pembuka; pesan pamit
        yang lolos aturan sapaan tidak boleh disapa "Halo!" (T12)."""
        from app.rag.smalltalk import REPLIES, SmallTalkKind

        hasil = await run_pipeline(
            "oke deh kalau begitu, nanti saya ke kampus aja",
            retriever=strong_retriever,
            llm_call=llm,
            gate_call=GerbangPalsu(vonis(GateLabel.SMALLTALK, blocked=True)),
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.SMALLTALK
        assert hasil.text == REPLIES[SmallTalkKind.CLOSING]

    async def test_tidak_diblokir_berjalan_seperti_biasa(self, strong_retriever, llm):
        gerbang = GerbangPalsu(vonis(GateLabel.ACADEMIC, blocked=False))
        hasil = await run_pipeline(
            "kapan KRS?",
            retriever=strong_retriever,
            llm_call=llm,
            gate_call=gerbang,
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.ANSWER
        assert hasil.gate.label is GateLabel.ACADEMIC

    async def test_ragu_tetap_diteruskan(self, strong_retriever, llm):
        gerbang = GerbangPalsu(vonis(GateLabel.NONSENSE, blocked=False, confidence=0.5))
        hasil = await run_pipeline(
            "krs smt 3",
            retriever=strong_retriever,
            llm_call=llm,
            gate_call=gerbang,
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.ANSWER

    async def test_jev_gagal_tetap_menjawab(self, strong_retriever, llm):
        gerbang = GerbangPalsu(lolos("ReadTimeout: lambat"))
        hasil = await run_pipeline(
            "kapan KRS?",
            retriever=strong_retriever,
            llm_call=llm,
            gate_call=gerbang,
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.ANSWER
        assert hasil.gate.error

    async def test_sensitif_mendahului_gerbang(self, strong_retriever, llm):
        """FR-7 tidak boleh bergantung pada model luar."""
        gerbang = GerbangPalsu(vonis(GateLabel.NONSENSE, blocked=True))
        hasil = await run_pipeline(
            "saya stres berat, takut di-DO",
            retriever=strong_retriever,
            llm_call=llm,
            gate_call=gerbang,
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.SUPPORT
        assert not gerbang.called

    async def test_sapaan_aturan_tidak_membayar_jev(self, strong_retriever, llm):
        gerbang = GerbangPalsu(vonis(GateLabel.ACADEMIC, blocked=False))
        hasil = await run_pipeline(
            "halo min",
            retriever=strong_retriever,
            llm_call=llm,
            gate_call=gerbang,
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.SMALLTALK
        assert not gerbang.called

    async def test_gerbang_berjalan_bersamaan_dengan_pencarian(self, llm, rewriter):
        """Gerbang menunggu sampai pencarian SUDAH dimulai. Bila keduanya masih
        berurutan (gerbang dulu), pencarian tidak pernah mulai dan gerbang
        kehabisan waktu -- bukti tanpa bergantung pada ukuran waktu."""
        pencarian_mulai = asyncio.Event()

        class RetrieverPenanda(FakeRetriever):
            async def ainvoke(self, query, *, unit=None, original_query=None):
                pencarian_mulai.set()
                return await super().ainvoke(query, unit=unit, original_query=original_query)

        async def gerbang(question, history=(), unit=None):
            await asyncio.wait_for(pencarian_mulai.wait(), timeout=2)
            return vonis(GateLabel.ACADEMIC, blocked=False)

        hasil = await run_pipeline(
            "syaratnya apa saja?",
            retriever=RetrieverPenanda([make_document("c1", vector_score=0.8)]),
            llm_call=llm,
            rewrite_call=rewriter,
            gate_call=gerbang,
            history=[Turn("user", "syarat cuti?"), Turn("assistant", "Syaratnya ...")],
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.ANSWER
        assert rewriter.called

    async def test_diblokir_membuang_hasil_rewrite_dan_pencarian(
        self, strong_retriever, llm, rewriter
    ):
        """Harga paralelisme: pesan yang diblokir sempat membayar rewrite dan
        pencarian, tetapi hasilnya tidak ikut ke balasan maupun ke log sitasi."""
        hasil = await run_pipeline(
            "resep rendang padang",
            retriever=strong_retriever,
            llm_call=llm,
            rewrite_call=rewriter,
            gate_call=GerbangPalsu(vonis(GateLabel.NONSENSE, blocked=True)),
            history=[Turn("user", "syarat cuti?"), Turn("assistant", "Syaratnya ...")],
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.REJECTED
        assert hasil.documents == ()
        assert hasil.rewritten_query is None
        assert not llm.called

    async def test_riwayat_terakhir_dikirim_ke_gerbang(self, strong_retriever, llm, rewriter):
        gerbang = GerbangPalsu(vonis(GateLabel.ACADEMIC, blocked=False))
        riwayat = [Turn("user", f"pesan {i}") for i in range(5)]
        await run_pipeline(
            "yang kedua?",
            retriever=strong_retriever,
            llm_call=llm,
            rewrite_call=rewriter,
            gate_call=gerbang,
            history=riwayat,
            policy=POLICY,
        )
        pertanyaan, dikirim = gerbang.calls[0]
        assert pertanyaan == "yang kedua?"
        assert dikirim == [("user", "pesan 2"), ("user", "pesan 3"), ("user", "pesan 4")]


class TestTenggatMengikutiPencarian:
    """JEV ditunggu selama pencarian paralel berjalan, plus `grace_seconds` (T13).

    Batas tetap 3 dtk dulu memutus JEV di tengah pencarian 5-10 dtk: pesan
    diteruskan tanpa diperiksa padahal menunggunya tidak menambah waktu.
    Waktu di sini diperkecil (detik -> ratusan milidetik).
    """

    @staticmethod
    def gerbang(tunda: float, grace: float, vonis_: GateVerdict, dibatalkan: list):
        async def panggil(question, history=(), unit=None):
            try:
                await asyncio.sleep(tunda)
            except asyncio.CancelledError:
                dibatalkan.append(True)
                raise
            return vonis_

        panggil.grace_seconds = grace
        return panggil

    @staticmethod
    def retriever_lambat(tunda: float) -> FakeRetriever:
        class Lambat(FakeRetriever):
            async def ainvoke(self, query, *, unit=None, original_query=None):
                await asyncio.sleep(tunda)
                return await super().ainvoke(query, unit=unit, original_query=original_query)

        return Lambat([make_document("c1", vector_score=0.8)])

    async def test_jev_lambat_tetap_dipakai_selama_pencarian_berjalan(self, llm):
        dibatalkan: list = []
        hasil = await run_pipeline(
            "resep rendang padang",
            retriever=self.retriever_lambat(0.4),
            llm_call=llm,
            gate_call=self.gerbang(0.25, 0.0, vonis(GateLabel.NONSENSE, True), dibatalkan),
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.REJECTED
        assert not dibatalkan

    async def test_jev_masih_ditunggu_sebentar_setelah_pencarian(self, llm):
        dibatalkan: list = []
        hasil = await run_pipeline(
            "resep rendang padang",
            retriever=self.retriever_lambat(0.05),
            llm_call=llm,
            gate_call=self.gerbang(0.15, 0.5, vonis(GateLabel.NONSENSE, True), dibatalkan),
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.REJECTED

    async def test_lewat_tenggat_diteruskan_dan_panggilan_dibatalkan(self, llm):
        dibatalkan: list = []
        hasil = await run_pipeline(
            "kapan KRS dibuka?",
            retriever=self.retriever_lambat(0.05),
            llm_call=llm,
            gate_call=self.gerbang(5.0, 0.1, vonis(GateLabel.NONSENSE, True), dibatalkan),
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.ANSWER
        assert hasil.gate.error.startswith("Tenggat")
        assert dibatalkan == [True]

    async def test_tenggat_tidak_menunda_giliran_lebih_dari_grace(self, llm):
        """Waktu total mengikuti pencarian + grace, bukan lama JEV macet."""
        loop = asyncio.get_running_loop()
        mulai = loop.time()
        await run_pipeline(
            "kapan KRS dibuka?",
            retriever=self.retriever_lambat(0.05),
            llm_call=llm,
            gate_call=self.gerbang(5.0, 0.1, vonis(GateLabel.NONSENSE, True), []),
            policy=POLICY,
        )
        assert loop.time() - mulai < 1.5


class TestBlokirMenghentikanPencarian:
    """Pesan yang diblokir JEV tidak menunggu -- dan tidak ikut gagal bersama --
    cabang pencarian yang hasilnya akan dibuang.

    Kejadian 2026-09-28: "resep nasi goreng dong" diblokir JEV dalam 2 dtk,
    tetapi rewrite LLM gagal (token gateway dicabut) setelah 127 dtk, dan
    mahasiswa melihat "Koneksi terputus" alih-alih balasan penolakan.
    """

    @staticmethod
    def rewriter(tunda: float, dibatalkan: list, gagal: bool = False):
        async def panggil(question, history, unit=None):
            try:
                await asyncio.sleep(tunda)
            except asyncio.CancelledError:
                dibatalkan.append("rewrite")
                raise
            if gagal:
                raise RuntimeError("503 token_revoked")
            return "pertanyaan mandiri"

        return panggil

    RIWAYAT = [Turn("user", "resep?"), Turn("assistant", "Maaf ...")]

    async def test_rewrite_lambat_dibatalkan_dan_giliran_cepat(self, strong_retriever, llm):
        dibatalkan: list = []
        mulai = asyncio.get_running_loop().time()
        hasil = await run_pipeline(
            "resep nasi goreng dong",
            retriever=strong_retriever,
            llm_call=llm,
            rewrite_call=self.rewriter(5.0, dibatalkan),
            gate_call=GerbangPalsu(vonis(GateLabel.OUT_OF_SCOPE, blocked=True)),
            history=self.RIWAYAT,
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.REJECTED
        assert dibatalkan == ["rewrite"]
        assert strong_retriever.queries == []
        assert asyncio.get_running_loop().time() - mulai < 1.0

    async def test_galat_rewrite_tidak_menggagalkan_pesan_yang_diblokir(
        self, strong_retriever, llm
    ):
        async def gerbang_lambat(question, history=(), unit=None):
            await asyncio.sleep(0.05)
            return vonis(GateLabel.OUT_OF_SCOPE, blocked=True)

        hasil = await run_pipeline(
            "resep nasi goreng dong",
            retriever=strong_retriever,
            llm_call=llm,
            rewrite_call=self.rewriter(0.3, [], gagal=True),
            gate_call=gerbang_lambat,
            history=self.RIWAYAT,
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.REJECTED

    async def test_pencarian_lambat_dibatalkan(self, llm):
        dibatalkan: list = []

        class RetrieverLambat(FakeRetriever):
            async def ainvoke(self, query, *, unit=None, original_query=None):
                try:
                    await asyncio.sleep(5.0)
                except asyncio.CancelledError:
                    dibatalkan.append("retrieve")
                    raise
                return []

        async def gerbang_lambat(question, history=(), unit=None):
            # Vonis tiba saat pencarian sudah berjalan, bukan sebelum dimulai.
            await asyncio.sleep(0.1)
            return vonis(GateLabel.NONSENSE, blocked=True)

        mulai = asyncio.get_running_loop().time()
        hasil = await run_pipeline(
            "resep rendang padang",
            retriever=RetrieverLambat(),
            llm_call=llm,
            gate_call=gerbang_lambat,
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.REJECTED
        assert dibatalkan == ["retrieve"]
        assert asyncio.get_running_loop().time() - mulai < 1.0

    async def test_galat_rewrite_tetap_diteruskan_bila_tidak_diblokir(
        self, strong_retriever, llm
    ):
        """Hanya pesan yang diblokir yang dilindungi; galat biasa tetap terlihat."""
        with pytest.raises(RuntimeError, match="token_revoked"):
            await run_pipeline(
                "syaratnya apa?",
                retriever=strong_retriever,
                llm_call=llm,
                rewrite_call=self.rewriter(0.0, [], gagal=True),
                gate_call=GerbangPalsu(vonis(GateLabel.ACADEMIC, blocked=False)),
                history=self.RIWAYAT,
                policy=POLICY,
            )
