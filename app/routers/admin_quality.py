"""Pertanyaan tak terjawab dan uji coba retrieval (AD-4, AD-6)."""

from __future__ import annotations

import time
import uuid
from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.admin.grouping import group_questions
from app.admin.permissions import AdminRole
from app.admin.unanswered import fetch_items, set_resolved
from app.deps import (
    SessionDep,
    SettingsDep,
    build_llm_call,
    build_retriever,
    require_admin,
    require_role,
)
from app.rag.chain import run_pipeline
from app.rag.threshold import ThresholdPolicy
from app.routers.chat import to_response
from app.schemas.admin import (
    RetrievedChunk,
    TestQueryRequest,
    TestQueryResponse,
    ThresholdDecisionOut,
    ThresholdValues,
    UnansweredGroup,
    UnansweredUpdate,
)
from app.schemas.common import Error

router = APIRouter(
    prefix="/api/admin",
    tags=["admin-kualitas"],
    dependencies=[Depends(require_admin)],
    responses={401: {"model": Error}},
)


# Semua level boleh melihat (staf perlu tahu dokumen apa yang dicari mahasiswa);
# hanya admin ke atas yang menandai selesai.
@router.get("/unanswered", response_model=list[UnansweredGroup])
async def list_unanswered(
    session: SessionDep,
    settings: SettingsDep,
    resolved: Annotated[
        bool | None, Query(description="Kosongkan untuk menampilkan keduanya.")
    ] = None,
    sejak: date | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[UnansweredGroup]:
    """AD-4. `limit` dan `offset` berlaku atas kelompok, bukan atas baris."""
    items = await fetch_items(
        session, resolved=resolved, sejak=sejak, timezone=settings.timezone
    )
    kelompok = group_questions(items)[offset : offset + limit]
    return [
        UnansweredGroup(
            ids=g.ids,
            contoh_pertanyaan=g.representative.pertanyaan,
            jumlah=g.jumlah,
            top_score_rata2=g.top_score_rata2,
            terakhir_ditanyakan=g.terakhir_ditanyakan,
            resolved=g.resolved,
        )
        for g in kelompok
    ]


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


@router.post("/test-query", response_model=TestQueryResponse)
async def admin_test_query(
    payload: TestQueryRequest,
    settings: SettingsDep,
    retriever: Any = Depends(build_retriever),
    llm_call: Any = Depends(build_llm_call),
) -> TestQueryResponse:
    """AD-6. Alur yang sama persis dengan `/api/chat`, ditambah rincian retrieval.

    Tidak tunduk pada kill switch (admin perlu mendiagnosis justru saat
    layanan dimatikan) dan tidak dicatat ke log percakapan, supaya uji coba
    admin tidak mencemari statistik AD-5 maupun daftar AD-4.
    """
    policy = ThresholdPolicy(
        vector_threshold=(
            payload.vector_threshold
            if payload.vector_threshold is not None
            else settings.vector_threshold
        ),
        lexical_threshold=settings.lexical_threshold,
    )

    mulai = time.perf_counter()
    outcome = await run_pipeline(
        payload.question, retriever=retriever, llm_call=llm_call, policy=policy
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
                judul=doc.metadata.get("judul", ""),
                halaman=doc.metadata.get("halaman", 0),
                konten=doc.page_content,
                rrf_score=float(doc.metadata.get("rrf_score", 0.0)),
                raw_scores=dict(doc.metadata.get("raw_scores", {})),
                ranks=dict(doc.metadata.get("ranks", {})),
            )
            for doc in outcome.documents
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
        ambang=ThresholdValues(
            vector=policy.vector_threshold, fulltext=policy.lexical_threshold
        ),
        contacts=respons.contacts,
        escalated=respons.escalated,
        latency_ms=latency_ms,
    )
