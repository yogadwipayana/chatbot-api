"""FR-8 -- isi log percakapan yang menjadi sumber AD-4 dan AD-5."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.dialects import postgresql

from app.observability.chatlog import (
    _PESAN_SQL,
    ChatLogEntry,
    build_meta,
    retrieved_chunk_ids,
)
from app.observability.costs import estimate_cost
from app.rag.chain import OutcomeKind, PipelineOutcome, run_pipeline
from tests.fixtures.fakes import make_document

USAGE = {"input_tokens": 1000, "output_tokens": 200}
CHAT_MODEL = "cx/gpt-5.5"


def entri(outcome: PipelineOutcome, **kw) -> ChatLogEntry:
    return ChatLogEntry(
        session_id="sesi-uji-12345", question="q", outcome=outcome, latency_ms=10, **kw
    )


class TestMeta:
    async def test_jawaban_mencatat_topik_model_dan_biaya(self, strong_retriever, llm):
        outcome = await run_pipeline(
            "kapan deadline pembayaran UKT?", retriever=strong_retriever, llm_call=llm
        )
        meta = build_meta(entri(outcome, model=CHAT_MODEL, usage=USAGE))
        assert meta["kind"] == "answer"
        assert meta["llm_dipanggil"] is True
        assert {"deadline", "pembayaran"} <= set(meta["topik"])
        assert meta["escalated"] is True
        assert meta["model"] == CHAT_MODEL
        assert meta["biaya_usd"] == pytest.approx(estimate_cost(CHAT_MODEL, 1000, 200).usd)

    async def test_model_tanpa_tarif_biayanya_none_bukan_nol(self, strong_retriever, llm):
        """AD-5 menghitung pesan seperti ini sebagai peringatan."""
        outcome = await run_pipeline("kapan KRS?", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, model="penyedia/model-belum-terdaftar", usage=USAGE))
        assert meta["llm_dipanggil"] is True
        assert meta["biaya_usd"] is None

    async def test_tanpa_data_token_biayanya_none(self, strong_retriever, llm):
        outcome = await run_pipeline("kapan KRS?", retriever=strong_retriever, llm_call=llm)
        assert build_meta(entri(outcome, model=CHAT_MODEL))["biaya_usd"] is None

    async def test_unit_pilihan_mahasiswa_tercatat(self, weak_retriever, llm):
        """Membedakan penolakan karena salah pilih unit dari dokumen yang memang
        belum ada -- dua masalah dengan perbaikan yang berbeda."""
        outcome = await run_pipeline(
            "kapan KRS?", retriever=weak_retriever, llm_call=llm, unit="Prodi"
        )
        assert build_meta(entri(outcome, unit="Prodi"))["unit"] == "Prodi"
        assert build_meta(entri(outcome))["unit"] is None

    async def test_penolakan_tidak_berbiaya_dan_tanpa_model(self, weak_retriever, llm):
        outcome = await run_pipeline("kapan KRS?", retriever=weak_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, model=CHAT_MODEL, usage=USAGE))
        assert meta["kind"] == "refusal"
        assert meta["llm_dipanggil"] is False
        assert meta["model"] is None
        assert meta["biaya_usd"] is None

    async def test_sensitif_tanpa_topik(self, strong_retriever, llm):
        outcome = await run_pipeline(
            "saya depresi takut bayar UKT", retriever=strong_retriever, llm_call=llm
        )
        meta = build_meta(entri(outcome))
        assert meta["kind"] == OutcomeKind.SUPPORT
        assert meta["sensitivitas"] == "distress"
        assert meta["topik"] == []


class TestSapaan:
    async def test_tanpa_model_dan_tanpa_biaya(self, strong_retriever, llm):
        outcome = await run_pipeline("halo", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, model=CHAT_MODEL, usage=USAGE))
        assert meta["kind"] == OutcomeKind.SMALLTALK
        assert meta["llm_dipanggil"] is False
        assert meta["biaya_usd"] is None
        assert meta["escalated"] is False


class TestBiayaEmbedding:
    """Biaya meng-embed pertanyaan mahasiswa (AD-5).

    Kecil per putaran -- belasan token -- tetapi tetap biaya, dan yang tidak
    tercatat tidak bisa diawasi.
    """

    EMBED = {
        "embed_dipanggil": True,
        "embed_model": "openrouter/openai/text-embedding-3-small",
        "embed_tokens": 9,
        "embed_biaya_usd": 1.8e-07,
        "embed_biaya_sumber": "provider",
    }

    async def test_jawaban_mencatat_biaya_embedding(self, strong_retriever, llm):
        outcome = await run_pipeline("kapan KRS?", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, **self.EMBED))
        assert meta["embed_dipanggil"] is True
        assert meta["embed_tokens"] == 9
        assert meta["embed_biaya_usd"] == pytest.approx(1.8e-07)
        assert meta["embed_biaya_sumber"] == "provider"

    async def test_biaya_embedding_tidak_dilebur_ke_biaya_usd(self, strong_retriever, llm):
        """`biaya_usd` sudah berarti "biaya LLM" di seluruh baris lama dan di
        `app/admin/stats.py`; menjumlahkan embedding ke dalamnya membuat baris
        sebelum dan sesudah hari ini tidak sebanding."""
        outcome = await run_pipeline("kapan KRS?", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, model="cx/gpt-5.5", usage=USAGE, **self.EMBED))
        assert meta["biaya_usd"] != meta["embed_biaya_usd"]
        assert meta["embed_biaya_usd"] == pytest.approx(1.8e-07)

    async def test_penolakan_tetap_berbiaya_embedding(self, weak_retriever, llm):
        """Jebakan utamanya: FR-3 menolak SETELAH retrieval, jadi pertanyaannya
        sudah terlanjur di-embed. Menyaring biaya embedding dengan
        `llm_dipanggil = true` akan menghapus seluruh penolakan dari laporan."""
        outcome = await run_pipeline("kapan KRS?", retriever=weak_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, **self.EMBED))
        assert meta["kind"] == "refusal"
        assert meta["llm_dipanggil"] is False
        assert meta["biaya_usd"] is None
        assert meta["embed_dipanggil"] is True
        assert meta["embed_biaya_usd"] == pytest.approx(1.8e-07)

    async def test_sensitif_tidak_pernah_di_embed(self, strong_retriever, llm):
        """FR-7 berhenti sebelum retrieval -- nol di sini berarti benar-benar
        tidak ada panggilan, bukan angka yang hilang."""
        outcome = await run_pipeline(
            "saya depresi takut bayar UKT", retriever=strong_retriever, llm_call=llm
        )
        meta = build_meta(entri(outcome))
        assert meta["kind"] == OutcomeKind.SUPPORT
        assert meta["embed_dipanggil"] is False
        assert meta["embed_tokens"] is None
        assert meta["embed_biaya_usd"] is None

    async def test_sapaan_tidak_pernah_di_embed(self, strong_retriever, llm):
        outcome = await run_pipeline("halo", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome))
        assert meta["kind"] == OutcomeKind.SMALLTALK
        assert meta["embed_dipanggil"] is False
        assert meta["embed_biaya_usd"] is None

    async def test_model_embedding_dikosongkan_bila_tak_dipanggil(self, strong_retriever, llm):
        """Sama seperti `model` pada jalur LLM: nama model tanpa panggilan yang
        menyertainya membuat laporan per model menghitung putaran yang tidak
        pernah memakai model itu."""
        outcome = await run_pipeline("halo", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(
            entri(outcome, embed_model="openrouter/openai/text-embedding-3-small")
        )
        assert meta["embed_dipanggil"] is False
        assert meta["embed_model"] is None

    async def test_dipanggil_tanpa_laporan_pemakaian(self, strong_retriever, llm):
        """Endpoint yang tidak mengirim `usage`: panggilannya nyata dan berbiaya,
        hanya angkanya tidak diketahui. Harus terlihat sebagai "tak terhitung",
        bukan sebagai gratis."""
        outcome = await run_pipeline("kapan KRS?", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, embed_dipanggil=True, embed_model="m"))
        assert meta["embed_dipanggil"] is True
        assert meta["embed_tokens"] is None
        assert meta["embed_biaya_usd"] is None


class TestChunkIds:
    def test_id_bukan_uuid_dilewati(self):
        outcome = PipelineOutcome(
            kind=OutcomeKind.ANSWER, text="x", documents=(make_document("c1"),)
        )
        assert retrieved_chunk_ids(outcome) is None

    def test_uuid_diambil_berurutan(self):
        a, b = uuid.uuid4(), uuid.uuid4()
        outcome = PipelineOutcome(
            kind=OutcomeKind.ANSWER,
            text="x",
            documents=(make_document(str(a)), make_document(str(b))),
        )
        assert retrieved_chunk_ids(outcome) == [a, b]


def test_insert_message_menyimpan_langsmith_run_id():
    sql = str(_PESAN_SQL.compile(dialect=postgresql.dialect()))
    assert "langsmith_run_id" in sql
    assert "%(langsmith_run_id)s" in sql
