"""Penanda `[DI_LUAR_TOPIK]` dari LLM penjawab dan parameter gerbang JEV (T18).

Penanda ini cadangan gerbang JEV untuk pesan di luar urusan kampus: dengan
`JEV_ENABLED=false`, atau saat JEV meloloskan pesan di bawah ambangnya, LLM
penjawab yang memutuskan. Hasilnya harus sama dengan blokir JEV `out_of_scope`:
`rejected`, tanpa sitasi, dan TIDAK masuk AD-4.
"""

from __future__ import annotations

import json

from app.observability.chatlog import ChatLogEntry, build_meta
from app.rag.chain import OutcomeKind, is_off_topic, run_pipeline, strip_markers
from app.rag.gate import (
    REPLIES,
    GateLabel,
    GatePolicy,
    GateSource,
    GateVerdict,
    build_request,
)
from app.rag.prompts import OFF_TOPIC_MARKER, SYSTEM_PROMPT
from app.rag.smalltalk import SmallTalkKind, detect
from app.routers.chat import to_response
from tests.fixtures.fakes import RecordingLLM

JAWABAN = "Harga TOEIC Rp675.000 [Panduan Akademik 2025, hal. 12]."


async def tanya(retriever, reply: str, **kw):
    return await run_pipeline(
        "resep rendang padang", retriever=retriever, llm_call=RecordingLLM(reply), **kw
    )


class TestPrompt:
    def test_prompt_menyebut_penanda_yang_dideteksi_kode(self):
        assert OFF_TOPIC_MARKER in SYSTEM_PROMPT

    def test_pembayaran_kampus_disebut_bukan_di_luar_topik(self):
        """T18: pertanyaan pembayaran lewat bank pernah dinilai di luar topik."""
        assert "membayar biaya kuliah lewat bank" in SYSTEM_PROMPT


class TestDeteksi:
    def test_penanda_saja(self):
        assert is_off_topic(OFF_TOPIC_MARKER)
        assert is_off_topic(f"\n {OFF_TOPIC_MARKER} \n")

    def test_jawaban_bersitasi_bukan_di_luar_topik(self):
        assert not is_off_topic(f"{JAWABAN} {OFF_TOPIC_MARKER}")

    def test_penanda_dibuang_dari_jawaban_bersitasi(self):
        assert strip_markers(f"{JAWABAN}\n{OFF_TOPIC_MARKER}") == JAWABAN


class TestPipeline:
    async def test_menjadi_rejected_seperti_blokir_jev(self, strong_retriever):
        hasil = await tanya(strong_retriever, OFF_TOPIC_MARKER)
        assert hasil.kind is OutcomeKind.REJECTED
        assert hasil.text == REPLIES[GateLabel.OUT_OF_SCOPE]
        assert hasil.llm_called is True

    async def test_tanpa_sitasi_dan_tanpa_kontak(self, strong_retriever):
        respons = to_response(await tanya(strong_retriever, OFF_TOPIC_MARKER))
        assert respons.kind == "rejected"
        assert respons.citations == []
        assert respons.contacts == []

    async def test_vonis_jev_yang_meloloskan_tetap_dibawa(self, strong_retriever):
        """Pesan yang lolos JEV lalu ditolak LLM adalah bahan kalibrasi ambang JEV."""

        async def jev(question, history=(), unit=None):
            return GateVerdict(GateLabel.OUT_OF_SCOPE, 0.6, blocked=False)

        hasil = await tanya(strong_retriever, OFF_TOPIC_MARKER, gate_call=jev)
        assert hasil.gate is not None
        assert hasil.gate.confidence == 0.6
        assert hasil.gate.source is GateSource.JEV

    async def test_penanda_tidak_pernah_dialirkan(self, strong_retriever):
        potongan: list[str] = []

        async def on_token(teks: str) -> None:
            potongan.append(teks)

        hasil = await tanya(strong_retriever, OFF_TOPIC_MARKER, on_token=on_token)
        assert potongan == []
        assert hasil.kind is OutcomeKind.REJECTED


class TestLog:
    def meta(self, outcome) -> dict:
        entry = ChatLogEntry(
            session_id="sesi-uji-12345", question="q", outcome=outcome, latency_ms=1
        )
        return build_meta(entry)

    async def test_bersumber_llm_bukan_refusal(self, strong_retriever):
        meta = self.meta(await tanya(strong_retriever, OFF_TOPIC_MARKER))
        assert meta["kind"] == "rejected"
        assert meta["rejection_source"] == "llm"
        assert meta["refusal_source"] is None

    async def test_jawaban_biasa_tanpa_sumber_penolakan(self, strong_retriever):
        meta = self.meta(await tanya(strong_retriever, JAWABAN))
        assert meta["rejection_source"] is None
        assert meta["gate_source"] is None


class TestParameterJev:
    def test_topik_pilihan_ikut_dikirim(self):
        """T18: tanpa topik, "cara bayar VA BNI" dinilai urusan perbankan umum."""
        body = build_request("m", "cara bayar VA BNI lewat SMS?", (), "Keuangan")
        assert body["state"]["topik_dipilih"] == "Keuangan"

    def test_tanpa_unit_tanpa_topik(self):
        assert "topik_dipilih" not in build_request("m", "halo")["state"]

    def test_kriteria_menyebut_pembayaran_lewat_bank(self):
        kriteria = build_request("m", "q")["questions"]["kategori"]["criteria"]
        assert "virtual accounts" in kriteria["academic"]
        assert "not out of scope" in kriteria["out_of_scope"]
        json.dumps(kriteria)  # harus dapat dikirim sebagai JSON

    def test_ambang_bawaan_hasil_kalibrasi(self):
        policy = GatePolicy()
        assert policy.block_threshold == 0.7
        assert policy.out_of_scope_threshold == 0.9


class TestBasaBasiAturan:
    def test_tawa_diabaikan_di_ucapan_terima_kasih(self):
        assert detect("wkwk oke makasih").kind is SmallTalkKind.THANKS

    def test_doa_penutup_sapaan(self):
        assert detect("selamat pagi min, semoga harimu menyenangkan").handled

    def test_sapaan_hi_bukan_tawa(self):
        assert detect("hi").kind is SmallTalkKind.GREETING
