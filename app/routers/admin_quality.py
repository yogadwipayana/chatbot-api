"""Pertanyaan tak terjawab, umpan balik mahasiswa, dan uji coba retrieval (AD-4, AD-6)."""

from __future__ import annotations

import time
import uuid
from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.admin.feedback import fetch_feedback
from app.admin.grouping import group_questions
from app.admin.permissions import AdminRole
from app.admin.unanswered import fetch_items, set_resolved
from app.db.models import DocumentType
from app.deps import (
    BaseSettingsDep,
    SessionDep,
    SettingsDep,
    UnitDirectoryDep,
    build_gate_call,
    build_llm_call,
    build_retriever,
    build_rewrite_call,
    build_tool_registry,
    get_log_sink,
    require_admin,
    require_role,
    unit_terdaftar,
)
from app.observability.applog import catat_giliran
from app.observability.logstore import UJI_COBA
from app.observability.tracing import akhiri_jejak, id_giliran, jejak_giliran
from app.rag.chain import refusal_source, rejection_source, run_pipeline
from app.rag.threshold import ThresholdPolicy
from app.routers.chat import to_response
from app.routers.common import LAYANAN_AI_BERMASALAH, terjemahkan_galat_ai
from app.schemas.admin import (
    FeedbackPage,
    GateVerdictOut,
    RetrievedChunk,
    TestQueryRequest,
    TestQueryResponse,
    ThresholdDecisionOut,
    ThresholdValues,
    UnansweredGroup,
    UnansweredPage,
    UnansweredUpdate,
)
from app.schemas.common import Error

router = APIRouter(
    prefix="/api/admin",
    tags=["admin-quality"],
    dependencies=[Depends(require_admin)],
    responses={401: {"model": Error}},
)


# Semua level boleh melihat (staf perlu tahu dokumen apa yang dicari mahasiswa);
# hanya admin ke atas yang menandai selesai.
@router.get("/unanswered", response_model=UnansweredPage)
async def list_unanswered(
    session: SessionDep,
    settings: BaseSettingsDep,
    resolved: Annotated[
        bool | None, Query(description="Kosongkan untuk menampilkan keduanya.")
    ] = None,
    since: date | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> UnansweredPage:
    """AD-4. `limit` dan `offset` berlaku atas kelompok, bukan atas baris."""
    items = await fetch_items(
        session, resolved=resolved, sejak=since, timezone=settings.timezone
    )
    kelompok = group_questions(items)
    return UnansweredPage(
        items=[
            UnansweredGroup(
                ids=g.ids,
                sample_question=g.representative.pertanyaan,
                count=g.jumlah,
                avg_top_score=g.top_score_rata2,
                last_asked_at=g.terakhir_ditanyakan,
                resolved=g.resolved,
                unit=g.unit,
            )
            for g in kelompok[offset : offset + limit]
        ],
        total=len(kelompok),
        question_count=sum(g.jumlah for g in kelompok),
        max_count=max((g.jumlah for g in kelompok), default=0),
    )


@router.patch(
    "/unanswered/{unanswered_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[Depends(require_role(AdminRole.ADMIN))],
    responses={403: {"model": Error}, 404: {"model": Error}},
)
async def resolve_unanswered(
    unanswered_id: uuid.UUID, payload: UnansweredUpdate, session: SessionDep
) -> Response:
    if not await set_resolved(session, unanswered_id, payload.resolved):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Pertanyaan tidak ditemukan.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/feedback",
    response_model=FeedbackPage,
    dependencies=[Depends(require_role(AdminRole.ADMIN))],
    responses={403: {"model": Error}},
)
async def list_feedback(
    session: SessionDep,
    settings: BaseSettingsDep,
    helpful: Annotated[
        bool | None, Query(description="Kosongkan untuk menampilkan keduanya.")
    ] = None,
    since: date | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> FeedbackPage:
    """Umpan balik FE-5, terbaru lebih dulu, beserta pertanyaan yang dinilai.

    Sama seperti statistik AD-5: isi percakapan, jadi minimal level admin.
    Jempol ke bawah tanpa catatan pun berguna -- pertanyaan dan jawabannya
    ikut tampil, sehingga dokumen yang keliru dipakai tetap dapat ditelusuri.
    """
    return FeedbackPage(
        **await fetch_feedback(
            session,
            helpful=helpful,
            sejak=since,
            timezone=settings.timezone,
            limit=limit,
            offset=offset,
        )
    )


@router.post(
    "/test-query",
    response_model=TestQueryResponse,
    responses={422: {"model": Error}, 502: {"model": Error}},
)
async def admin_test_query(
    payload: TestQueryRequest,
    settings: SettingsDep,
    units: UnitDirectoryDep,
    retriever: Any = Depends(build_retriever),
    llm_call: Any = Depends(build_llm_call),
    rewrite_call: Any = Depends(build_rewrite_call),
    gate_call: Any = Depends(build_gate_call),
    tool_registry: Any = Depends(build_tool_registry),
    log_sink: Any = Depends(get_log_sink),
) -> TestQueryResponse:
    """AD-6. Alur yang sama persis dengan `/api/chat`, ditambah rincian retrieval.

    Termasuk gerbang JEV: tanpanya "yang dilihat mahasiswa" di halaman ini
    keliru untuk pesan yang diblokir (admin melihat "tidak ditemukan",
    mahasiswa melihat penolakan JEV). Uji coba tanpa riwayat, jadi penulisan
    ulang query (FR-4) hanya berjalan untuk menerjemahkan pertanyaan berbahasa
    Inggris, persis seperti pesan pertama mahasiswa.

    Termasuk tool-calling dengan alasan yang sama (docs/tool-call.md): tanpanya
    pertanyaan seperti "siapa dosen pengampu mata kuliah X" tampil `refusal` di
    sini padahal mahasiswa menerima jawaban dari data SADS. `retrieved` tetap
    hanya memuat chunk hasil retrieval -- sumber tool bukan chunk dan tidak punya
    skor RRF, jadi ia tidak dipalsukan menjadi baris di tabel itu.

    Tidak tunduk pada kill switch (admin perlu mendiagnosis justru saat
    layanan dimatikan) dan tidak dicatat ke log percakapan, supaya uji coba
    admin tidak mencemari statistik AD-5 maupun daftar AD-4. Ia tetap dicatat ke
    log SQLite sebagai jalur `uji_coba`, supaya langkah dan panggilan LLM/tool-nya
    dapat dibuka di tab Graf halaman Log (`turn_id`); tab Performa tidak
    menghitungnya.

    Galat layanan AI (LLM, gateway, embedding pertanyaan) dibalas 502 berisi
    kalimat siap tampil; rinciannya masuk log server.
    """
    policy = ThresholdPolicy(
        vector_threshold=(
            payload.vector_threshold
            if payload.vector_threshold is not None
            else settings.vector_threshold
        ),
        lexical_threshold=settings.lexical_threshold,
    )

    unit = await unit_terdaftar(units, payload.unit) if payload.unit else None

    mulai = time.perf_counter()
    run_id = id_giliran()
    # Pencatat giliran di DALAM penerjemah galat: ia harus melihat galat aslinya
    # (dicatat beserta traceback), bukan 502 hasil terjemahan.
    with (
        terjemahkan_galat_ai("uji coba jawaban", pesan=LAYANAN_AI_BERMASALAH),
        catat_giliran(
            log_sink,
            endpoint=UJI_COBA,
            session_id=None,
            unit=unit,
            pertanyaan=payload.question,
            langsmith_run_id=run_id,
            rekam_io=settings.log_node_io,
        ) as giliran,
    ):
        async with jejak_giliran(
            run_id=run_id, pertanyaan=payload.question, nama="uji_coba_admin"
        ) as akar:
            outcome = await run_pipeline(
                payload.question,
                retriever=retriever,
                llm_call=llm_call,
                rewrite_call=rewrite_call,
                gate_call=gate_call,
                policy=policy,
                unit=unit,
                callbacks=[giliran.recorder],
                tool_registry=tool_registry,
                tool_max_rounds=settings.tools_max_rounds,
            )
            akhiri_jejak(akar, kind=str(outcome.kind), text=outcome.text)
        giliran.selesai(
            hasil=outcome.kind,
            message_id=None,
            langsmith_run_id=run_id,
            llm_call=llm_call,
            jawaban=outcome.text,
        )
    latency_ms = round((time.perf_counter() - mulai) * 1000)

    respons = to_response(outcome)
    decision = outcome.decision
    return TestQueryResponse(
        kind=outcome.kind,
        text=outcome.text,
        rewritten_query=outcome.rewritten_query,
        retrieved=[
            RetrievedChunk(
                chunk_id=str(doc.metadata.get("chunk_id", "")),
                document_id=(
                    str(doc.metadata["document_id"])
                    if doc.metadata.get("document_id")
                    else None
                ),
                title=doc.metadata.get("judul", ""),
                type=doc.metadata.get("jenis") or DocumentType.PDF,
                page=doc.metadata.get("halaman", 0),
                content=doc.page_content,
                rrf_score=float(doc.metadata.get("rrf_score", 0.0)),
                raw_scores=dict(doc.metadata.get("raw_scores", {})),
                ranks=dict(doc.metadata.get("ranks", {})),
                neighbor_of=doc.metadata.get("neighbor_of"),
            )
            # Sumber tool (Document semu, tanpa `chunk_id`) dilewati: tabel ini
            # menjanjikan chunk hasil retrieval beserta skornya.
            for doc in outcome.documents
            if doc.metadata.get("chunk_id")
        ],
        decision=(
            ThresholdDecisionOut(
                decision=decision.decision,
                reason=decision.reason,
                top_vector_score=decision.top_vector_score,
                top_lexical_score=decision.top_lexical_score,
            )
            if decision
            else None
        ),
        thresholds=ThresholdValues(
            vector=policy.vector_threshold, fulltext=policy.lexical_threshold
        ),
        llm_called=outcome.llm_called,
        refusal_source=refusal_source(outcome),
        rejection_source=rejection_source(outcome),
        gate=(
            GateVerdictOut(
                label=str(outcome.gate.label),
                confidence=outcome.gate.confidence,
                blocked=outcome.gate.blocked,
                error=outcome.gate.error,
                source=outcome.gate.source.value,
            )
            if outcome.gate
            else None
        ),
        contacts=respons.contacts,
        escalated=respons.escalated,
        attachments=respons.attachments,
        latency_ms=latency_ms,
        langsmith_run_id=run_id,
        turn_id=giliran.turn_id if log_sink is not None else None,
    )
