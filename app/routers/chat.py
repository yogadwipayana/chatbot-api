"""Endpoint chat mahasiswa (FE-1..FE-5)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from app.deps import (
    SessionDep,
    SettingsDep,
    build_llm_call,
    build_retriever,
    get_chat_logger,
    guard_kill_switch,
)
from app.observability.chatlog import ChatLogEntry
from app.rag.chain import OutcomeKind, PipelineOutcome, run_pipeline
from app.rag.citations import extract_citations
from app.rag.rewriter import Turn
from app.rag.threshold import ThresholdPolicy
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CitationOut,
    ContactOut,
    FeedbackRequest,
)
from app.schemas.common import Error
from app.security.sanitize import sanitize_question

router = APIRouter(prefix="/api", tags=["chat"], dependencies=[Depends(guard_kill_switch)])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    # Tanpa ini reverse proxy boleh menahan aliran sampai selesai, sehingga
    # indikator "mencari dokumen..." (FE-1) muncul bersamaan dengan jawabannya.
    "X-Accel-Buffering": "no",
}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    settings: SettingsDep,
    retriever: Any = Depends(build_retriever),
    llm_call: Any = Depends(build_llm_call),
    chat_logger: Any = Depends(get_chat_logger),
) -> ChatResponse:
    """Jawaban sekali kirim. Dipakai kotak uji coba admin (AD-6) dan test."""
    mulai = time.perf_counter()
    outcome = await run_pipeline(
        payload.question,
        retriever=retriever,
        llm_call=llm_call,
        history=[Turn(t.role, t.konten) for t in payload.history],
        policy=policy_from(settings),
    )
    response = to_response(outcome)
    response.message_id = await catat(chat_logger, payload, outcome, mulai, llm_call)
    return response


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    settings: SettingsDep,
    retriever: Any = Depends(build_retriever),
    llm_call: Any = Depends(build_llm_call),
    chat_logger: Any = Depends(get_chat_logger),
) -> StreamingResponse:
    """Server-Sent Events untuk streaming token (FE-1)."""
    # Divalidasi sebelum aliran dimulai. Galat di dalam generator terjadi setelah
    # status 200 terkirim, sehingga klien hanya melihat aliran yang terputus
    # alih-alih 422 yang jelas.
    sanitize_question(payload.question)
    mulai = time.perf_counter()

    async def event_stream() -> AsyncIterator[str]:
        yield sse("status", {"stage": "mencari dokumen"})
        outcome = await run_pipeline(
            payload.question,
            retriever=retriever,
            llm_call=llm_call,
            history=[Turn(t.role, t.konten) for t in payload.history],
            policy=policy_from(settings),
        )
        response = to_response(outcome)
        response.message_id = await catat(chat_logger, payload, outcome, mulai, llm_call)
        yield sse("message", response.model_dump(mode="json"))
        yield sse("done", {})

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=SSE_HEADERS
    )


@router.post(
    "/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={404: {"model": Error}},
)
async def submit_feedback(payload: FeedbackRequest, session: SessionDep) -> Response:
    """FE-5: satu klik, tanpa modal.

    Klik ulang pada pesan yang sama mengganti umpan balik sebelumnya: mahasiswa
    yang berubah pikiran dari 👍 ke 👎 tidak boleh terhitung dua kali pada rasio
    AD-5.
    """
    ada = (
        await session.execute(
            text("SELECT 1 FROM messages WHERE id = :id AND role = 'assistant'"),
            {"id": payload.message_id},
        )
    ).scalar()
    if not ada:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Pesan tidak ditemukan.")

    await session.execute(
        text("DELETE FROM feedback WHERE message_id = :id"), {"id": payload.message_id}
    )
    await session.execute(
        text(
            "INSERT INTO feedback (id, message_id, helpful, catatan)"
            " VALUES (:id, :message_id, :helpful, :catatan)"
        ),
        {
            "id": uuid.uuid4(),
            "message_id": payload.message_id,
            "helpful": payload.helpful,
            "catatan": (payload.catatan or "").strip() or None,
        },
    )
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def to_response(outcome: PipelineOutcome) -> ChatResponse:
    """Ubah hasil pipeline menjadi payload frontend.

    Sitasi hanya diisi untuk jawaban sungguhan. Penolakan dan balasan empatik
    tidak boleh membawa kartu sitasi -- FE-4 menuntut keduanya terlihat jelas
    berbeda dari jawaban normal, dan kartu sitasi pada layar penolakan justru
    memberi kesan jawabannya bersumber.
    """
    citations = citations_for(outcome) if outcome.kind is OutcomeKind.ANSWER else []

    return ChatResponse(
        kind=outcome.kind,
        text=outcome.text,
        citations=citations,
        contacts=[
            ContactOut(unit=c.unit, jam_layanan=c.jam_layanan, kontak=c.kontak)
            for c in outcome.contacts
        ],
        escalated=bool(outcome.contacts),
        top_score=outcome.decision.top_score if outcome.decision else None,
    )


def citations_for(outcome: PipelineOutcome) -> list[CitationOut]:
    """Kartu sitasi FE-2: hanya dokumen yang benar-benar dikutip jawaban.

    Retrieval sengaja mengambil beberapa chunk sekaligus agar LLM punya konteks,
    dan sebagian besar di antaranya tidak ikut dipakai menjawab. Menampilkan
    semuanya sebagai "Sumber" memaksa mahasiswa menebak kartu mana yang relevan
    -- padahal FE-2 justru dinilai dari verifikasi yang semudah satu klik.

    Sitasi ke dokumen di luar konteks (sumber karangan) tidak pernah menjadi
    kartu: tautannya buntu, dan kartu yang tampak sah lebih berbahaya daripada
    tidak ada kartu sama sekali.

    Jawaban tanpa satu pun penanda sitasi -- LLM melanggar FR-5 -- jatuh kembali
    ke seluruh chunk terambil, supaya mahasiswa tetap punya jalan verifikasi.
    """
    tersedia: dict[tuple[str, int], CitationOut] = {}
    for doc in outcome.documents:
        meta = doc.metadata
        kartu = CitationOut(
            judul=meta.get("judul", ""),
            halaman=meta.get("halaman", 0),
            document_id=str(meta.get("document_id", "")),
            file_path=meta.get("file_path", ""),
        )
        # Chunk berbeda dari halaman yang sama menghasilkan kartu yang sama.
        tersedia.setdefault((kartu.judul.casefold(), kartu.halaman), kartu)

    # Urutan mengikuti kemunculan di jawaban, bukan peringkat retrieval: itu
    # urutan yang dibaca mahasiswa.
    dikutip: list[CitationOut] = []
    for citation in extract_citations(outcome.text):
        kartu = tersedia.get((citation.judul.casefold(), citation.halaman))
        if kartu is not None and kartu not in dikutip:
            dikutip.append(kartu)

    return dikutip or list(tersedia.values())


async def catat(
    chat_logger: Any,
    payload: ChatRequest,
    outcome: PipelineOutcome,
    mulai: float,
    llm_call: Any,
) -> str | None:
    """Catat putaran ini (FR-8). None bila pencatatan gagal; jawaban tetap terkirim."""
    return await chat_logger.log(
        ChatLogEntry(
            session_id=payload.session_id,
            question=sanitize_question(payload.question),
            outcome=outcome,
            latency_ms=round((time.perf_counter() - mulai) * 1000),
            model=getattr(llm_call, "model", None),
            usage=getattr(llm_call, "usage", None),
        )
    )


def policy_from(settings) -> ThresholdPolicy:
    return ThresholdPolicy(
        vector_threshold=settings.vector_threshold,
        lexical_threshold=settings.lexical_threshold,
    )


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
