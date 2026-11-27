"""FR-8 -- isi log percakapan yang menjadi sumber AD-4 dan AD-5."""

from __future__ import annotations

import uuid

from sqlalchemy.dialects import postgresql

from app.observability.chatlog import (
    _PESAN_SQL,
    ChatLogEntry,
    build_meta,
    retrieved_chunk_ids,
)
from app.prodi import ProfilMahasiswa, cari_prodi
from app.rag.chain import OutcomeKind, PipelineOutcome, run_pipeline
from tests.fixtures.fakes import make_document

USAGE = {"input_tokens": 1000, "output_tokens": 200}
CHAT_MODEL = "vl/qwen3.6-27b"
PROFIL = ProfilMahasiswa(prodi=cari_prodi("1010"), angkatan=2024)

KUNCI_BIAYA = {"llm_cost_usd", "embed_cost_usd", "embed_cost_source", "gate_cost_usd"}
"""Dihapus 2026-11-27: semua model self-hosted, tidak ada biaya per token.
Baris lama di database masih memuatnya; baris baru tidak boleh."""


def entri(outcome: PipelineOutcome, **kw) -> ChatLogEntry:
    return ChatLogEntry(
        session_id="sesi-uji-12345", question="q", outcome=outcome, latency_ms=10, **kw
    )


class TestMeta:
    async def test_jawaban_mencatat_topik_model_dan_token(self, strong_retriever, llm):
        outcome = await run_pipeline(
            "kapan deadline pembayaran UKT?", retriever=strong_retriever, llm_call=llm
        )
        meta = build_meta(entri(outcome, model=CHAT_MODEL, usage=USAGE))
        assert meta["kind"] == "answer"
        assert meta["llm_called"] is True
        assert {"deadline", "pembayaran"} <= set(meta["topics"])
        assert meta["escalated"] is True
        assert meta["model"] == CHAT_MODEL
        assert (meta["input_tokens"], meta["output_tokens"]) == (1000, 200)

    async def test_tidak_ada_kolom_biaya(self, strong_retriever, llm):
        outcome = await run_pipeline("kapan KRS?", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(
            entri(outcome, model=CHAT_MODEL, usage=USAGE, embed_dipanggil=True, embed_tokens=9)
        )
        assert not KUNCI_BIAYA & meta.keys()

    async def test_tanpa_data_token_tokennya_none(self, strong_retriever, llm):
        outcome = await run_pipeline("kapan KRS?", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, model=CHAT_MODEL))
        assert (meta["input_tokens"], meta["output_tokens"]) == (None, None)

    async def test_unit_pilihan_mahasiswa_tercatat(self, weak_retriever, llm):
        """Membedakan penolakan karena salah pilih unit dari dokumen yang memang
        belum ada -- dua masalah dengan perbaikan yang berbeda."""
        outcome = await run_pipeline(
            "kapan KRS?", retriever=weak_retriever, llm_call=llm, unit="Prodi"
        )
        assert build_meta(entri(outcome, unit="Prodi"))["unit"] == "Prodi"
        assert build_meta(entri(outcome))["unit"] is None

    async def test_penolakan_tanpa_model(self, weak_retriever, llm):
        outcome = await run_pipeline("kapan KRS?", retriever=weak_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, model=CHAT_MODEL, usage=USAGE))
        assert meta["kind"] == "refusal"
        assert meta["llm_called"] is False
        assert meta["model"] is None

    async def test_sensitif_tanpa_topik(self, strong_retriever, llm):
        outcome = await run_pipeline(
            "saya depresi takut bayar UKT", retriever=strong_retriever, llm_call=llm
        )
        meta = build_meta(entri(outcome))
        assert meta["kind"] == OutcomeKind.SUPPORT
        assert meta["sensitivity"] == "distress"
        assert meta["topics"] == []

    async def test_nim_dan_profil_penanya_tercatat(self, strong_retriever, llm):
        outcome = await run_pipeline("kapan KRS?", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, profile=PROFIL, nim="2401010101"))
        assert (meta["program_code"], meta["intake_year"]) == ("1010", 2024)
        assert meta["nim"] == "2401010101"
        tanpa = build_meta(entri(outcome))
        assert (tanpa["program_code"], tanpa["intake_year"], tanpa["nim"]) == (None,) * 3

    async def test_pesan_sensitif_tetap_mencatat_nim_tanpa_profil(self, strong_retriever, llm):
        """Pesan konseling tetap dapat ditelusuri ke penanyanya, tetapi tidak
        ikut rincian per prodi dan angkatan di statistik."""
        outcome = await run_pipeline(
            "saya depresi takut bayar UKT", retriever=strong_retriever, llm_call=llm
        )
        meta = build_meta(entri(outcome, profile=PROFIL, nim="2401010101"))
        assert meta["kind"] == OutcomeKind.SUPPORT
        assert meta["nim"] == "2401010101"
        assert (meta["program_code"], meta["intake_year"]) == (None, None)


class TestSapaan:
    async def test_tanpa_model(self, strong_retriever, llm):
        outcome = await run_pipeline("halo", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, model=CHAT_MODEL, usage=USAGE))
        assert meta["kind"] == OutcomeKind.SMALLTALK
        assert meta["llm_called"] is False
        assert meta["model"] is None
        assert meta["escalated"] is False


class TestPemakaianEmbedding:
    """Pemakaian embedding pertanyaan mahasiswa: dipanggil atau tidak, dan tokennya."""

    EMBED = {
        "embed_dipanggil": True,
        "embed_model": "intfloat/multilingual-e5-small",
        "embed_tokens": 9,
    }

    async def test_jawaban_mencatat_pemakaian_embedding(self, strong_retriever, llm):
        outcome = await run_pipeline("kapan KRS?", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, **self.EMBED))
        assert meta["embed_called"] is True
        assert meta["embed_model"] == "intfloat/multilingual-e5-small"
        assert meta["embed_tokens"] == 9

    async def test_penolakan_tetap_tercatat_di_embed(self, weak_retriever, llm):
        """FR-3 menolak SETELAH retrieval, jadi pertanyaannya sudah terlanjur
        di-embed walau `llm_called = false`."""
        outcome = await run_pipeline("kapan KRS?", retriever=weak_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, **self.EMBED))
        assert meta["kind"] == "refusal"
        assert meta["llm_called"] is False
        assert meta["embed_called"] is True
        assert meta["embed_tokens"] == 9

    async def test_sensitif_tidak_pernah_di_embed(self, strong_retriever, llm):
        """FR-7 berhenti sebelum retrieval -- nol di sini berarti benar-benar
        tidak ada panggilan, bukan angka yang hilang."""
        outcome = await run_pipeline(
            "saya depresi takut bayar UKT", retriever=strong_retriever, llm_call=llm
        )
        meta = build_meta(entri(outcome))
        assert meta["kind"] == OutcomeKind.SUPPORT
        assert meta["embed_called"] is False
        assert meta["embed_tokens"] is None

    async def test_sapaan_tidak_pernah_di_embed(self, strong_retriever, llm):
        outcome = await run_pipeline("halo", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome))
        assert meta["kind"] == OutcomeKind.SMALLTALK
        assert meta["embed_called"] is False

    async def test_model_embedding_dikosongkan_bila_tak_dipanggil(self, strong_retriever, llm):
        """Sama seperti `model` pada jalur LLM: nama model tanpa panggilan yang
        menyertainya membuat hitungan per model memuat putaran yang tidak
        pernah memakai model itu."""
        outcome = await run_pipeline("halo", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, embed_model="intfloat/multilingual-e5-small"))
        assert meta["embed_called"] is False
        assert meta["embed_model"] is None

    async def test_dipanggil_tanpa_laporan_pemakaian(self, strong_retriever, llm):
        """Endpoint yang tidak mengirim `usage`: panggilannya nyata, hanya
        jumlah tokennya tidak diketahui."""
        outcome = await run_pipeline("kapan KRS?", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, embed_dipanggil=True, embed_model="m"))
        assert meta["embed_called"] is True
        assert meta["embed_tokens"] is None


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
