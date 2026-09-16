"""FR-8 -- isi log percakapan yang menjadi sumber AD-4 dan AD-5."""

from __future__ import annotations

import uuid

import pytest

from app.observability.chatlog import ChatLogEntry, build_meta, retrieved_chunk_ids
from app.observability.costs import estimate_cost
from app.rag.chain import OutcomeKind, PipelineOutcome, run_pipeline
from tests.fixtures.fakes import make_document

USAGE = {"input_tokens": 1000, "output_tokens": 200}


def entri(outcome: PipelineOutcome, **kw) -> ChatLogEntry:
    return ChatLogEntry(
        session_id="sesi-uji-12345", question="q", outcome=outcome, latency_ms=10, **kw
    )


class TestMeta:
    async def test_jawaban_mencatat_topik_model_dan_biaya(self, strong_retriever, llm):
        outcome = await run_pipeline(
            "kapan deadline pembayaran UKT?", retriever=strong_retriever, llm_call=llm
        )
        meta = build_meta(entri(outcome, model="gpt-4o-mini", usage=USAGE))
        assert meta["kind"] == "answer"
        assert meta["llm_dipanggil"] is True
        assert {"deadline", "pembayaran"} <= set(meta["topik"])
        assert meta["escalated"] is True
        assert meta["model"] == "gpt-4o-mini"
        assert meta["biaya_usd"] == pytest.approx(estimate_cost("gpt-4o-mini", 1000, 200).usd)

    async def test_model_tanpa_tarif_biayanya_none_bukan_nol(self, strong_retriever, llm):
        """AD-5 menghitung pesan seperti ini sebagai peringatan."""
        outcome = await run_pipeline("kapan KRS?", retriever=strong_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, model="penyedia/model-belum-terdaftar", usage=USAGE))
        assert meta["llm_dipanggil"] is True
        assert meta["biaya_usd"] is None

    async def test_tanpa_data_token_biayanya_none(self, strong_retriever, llm):
        outcome = await run_pipeline("kapan KRS?", retriever=strong_retriever, llm_call=llm)
        assert build_meta(entri(outcome, model="gpt-4o-mini"))["biaya_usd"] is None

    async def test_penolakan_tidak_berbiaya_dan_tanpa_model(self, weak_retriever, llm):
        outcome = await run_pipeline("kapan KRS?", retriever=weak_retriever, llm_call=llm)
        meta = build_meta(entri(outcome, model="gpt-4o-mini", usage=USAGE))
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
        meta = build_meta(entri(outcome, model="gpt-4o-mini", usage=USAGE))
        assert meta["kind"] == OutcomeKind.SMALLTALK
        assert meta["llm_dipanggil"] is False
        assert meta["biaya_usd"] is None
        assert meta["escalated"] is False


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
