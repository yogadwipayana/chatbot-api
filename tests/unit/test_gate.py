"""Gerbang JEV: bentuk permintaan, keputusan per ambang, fail-open, dan posisinya di alur."""

from __future__ import annotations

import json

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


class GerbangPalsu:
    def __init__(self, verdict: GateVerdict) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    async def __call__(self, question, history=()):
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


class TestGerbangDiAlur:
    @pytest.mark.parametrize(
        "label", [GateLabel.NONSENSE, GateLabel.MALICIOUS, GateLabel.OUT_OF_SCOPE]
    )
    async def test_diblokir_tanpa_retrieval_dan_tanpa_llm(self, label, strong_retriever, llm):
        hasil = await run_pipeline(
            "asdf qwer",
            retriever=strong_retriever,
            llm_call=llm,
            gate_call=GerbangPalsu(vonis(label, blocked=True)),
            policy=POLICY,
        )
        assert hasil.kind is OutcomeKind.REJECTED
        assert hasil.text == vonis(label, True).reply
        assert strong_retriever.queries == []
        assert not llm.called
        assert hasil.documents == ()
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
        assert strong_retriever.queries == []

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

    async def test_gerbang_mendahului_rewrite(self, strong_retriever, llm, rewriter):
        """Pesan nonsense tidak layak dibayar satu panggilan LLM rewrite."""
        await run_pipeline(
            "asdf",
            retriever=strong_retriever,
            llm_call=llm,
            rewrite_call=rewriter,
            gate_call=GerbangPalsu(vonis(GateLabel.NONSENSE, blocked=True)),
            history=[Turn("user", "syarat cuti?"), Turn("assistant", "Syaratnya ...")],
            policy=POLICY,
        )
        assert not rewriter.called

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
