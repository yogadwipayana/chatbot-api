"""Endpoint chat mahasiswa (FE-1..FE-5)."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from app.deps import (
    EmbedKeyDep,
    SessionDep,
    SettingsDep,
    UnitDirectoryDep,
    batas_harian,
    batasi_penilaian,
    batasi_pertanyaan,
    build_gate_call,
    build_llm_call,
    build_retriever,
    build_rewrite_call,
    build_tool_registry,
    get_chat_logger,
    get_log_sink,
    guard_kill_switch,
    profil_dari_nim,
    unit_terdaftar,
)
from app.observability.applog import catat_giliran
from app.observability.chatlog import ChatLogEntry
from app.observability.tracing import (
    akhiri_jejak,
    id_giliran,
    jejak_giliran,
    tandai_sesi,
)
from app.prodi import ProfilMahasiswa
from app.rag.chain import OutcomeKind, PipelineOutcome, run_pipeline
from app.rag.citations import extract_citations
from app.rag.rewriter import Turn
from app.rag.threshold import ThresholdPolicy
from app.schemas.chat import (
    AttachmentOut,
    ChatRequest,
    ChatResponse,
    CitationOut,
    CitationType,
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


# Urutannya: kill switch (router), batas laju, lalu batas harian -- permintaan
# yang ditolak 429 tidak ikut menghabiskan jatah harian.
BATAS_PERTANYAAN = [Depends(batasi_pertanyaan), Depends(batas_harian)]
ERROR_PERTANYAAN = {
    403: {"model": Error},
    422: {"model": Error},
    429: {"model": Error},
    503: {"model": Error},
}


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses=ERROR_PERTANYAAN,
    dependencies=BATAS_PERTANYAAN,
)
async def chat(
    payload: ChatRequest,
    settings: SettingsDep,
    units: UnitDirectoryDep,
    embed_key: EmbedKeyDep,
    retriever: Any = Depends(build_retriever),
    llm_call: Any = Depends(build_llm_call),
    rewrite_call: Any = Depends(build_rewrite_call),
    gate_call: Any = Depends(build_gate_call),
    tool_registry: Any = Depends(build_tool_registry),
    chat_logger: Any = Depends(get_chat_logger),
    log_sink: Any = Depends(get_log_sink),
) -> ChatResponse:
    """Jawaban sekali kirim. Dipakai kotak uji coba admin (AD-6) dan test."""
    unit = await unit_terdaftar(units, payload.unit) if payload.unit else None
    profil = profil_dari_nim(payload.nim)
    mulai = time.perf_counter()
    run_id = id_giliran()
    tandai_sesi(payload.session_id, llm_call, rewrite_call)
    with catat_giliran(
        log_sink, endpoint="chat", session_id=payload.session_id, unit=unit
    ) as giliran:
        async with jejak_giliran(
            run_id=run_id, session_id=payload.session_id, pertanyaan=payload.question
        ) as akar:
            outcome = await run_pipeline(
                payload.question,
                retriever=retriever,
                llm_call=llm_call,
                rewrite_call=rewrite_call,
                gate_call=gate_call,
                history=[Turn(t.role, t.content) for t in payload.history],
                policy=policy_from(settings),
                unit=unit,
                profile=profil,
                callbacks=[giliran.recorder],
                tool_registry=tool_registry,
                tool_max_rounds=settings.tools_max_rounds,
            )
            akhiri_jejak(akar, kind=str(outcome.kind), text=outcome.text)

        response = to_response(outcome)
        response.message_id = await catat(
            chat_logger,
            payload,
            outcome,
            mulai,
            llm_call,
            retriever,
            run_id,
            unit=unit,
            profile=profil,
            embed_key=embed_key,
        )
        giliran.selesai(
            hasil=outcome.kind,
            message_id=response.message_id,
            langsmith_run_id=run_id,
            llm_call=llm_call,
        )
    return response


@router.post("/chat/stream", responses=ERROR_PERTANYAAN, dependencies=BATAS_PERTANYAAN)
async def chat_stream(
    payload: ChatRequest,
    settings: SettingsDep,
    units: UnitDirectoryDep,
    embed_key: EmbedKeyDep,
    retriever: Any = Depends(build_retriever),
    llm_call: Any = Depends(build_llm_call),
    rewrite_call: Any = Depends(build_rewrite_call),
    gate_call: Any = Depends(build_gate_call),
    tool_registry: Any = Depends(build_tool_registry),
    chat_logger: Any = Depends(get_chat_logger),
    log_sink: Any = Depends(get_log_sink),
) -> StreamingResponse:
    """Server-Sent Events untuk streaming token (FE-1)."""
    # Divalidasi sebelum aliran dimulai. Galat di dalam generator terjadi setelah
    # status 200 terkirim, sehingga klien hanya melihat aliran yang terputus
    # alih-alih 422 yang jelas.
    sanitize_question(payload.question)
    unit = await unit_terdaftar(units, payload.unit) if payload.unit else None
    profil = profil_dari_nim(payload.nim)
    mulai = time.perf_counter()
    run_id = id_giliran()
    tandai_sesi(payload.session_id, llm_call, rewrite_call)

    async def event_stream() -> AsyncIterator[str]:
        yield sse("status", {"stage": "mencari dokumen"})

        # Pipeline berjalan sebagai task tersendiri dan menitipkan potongan
        # jawaban lewat antrean; generator ini hanya meneruskannya. Alternatifnya
        # adalah menjadikan `run_pipeline` generator, padahal `/api/chat` dan
        # seluruh test pipeline memakainya sebagai fungsi biasa.
        antrean: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

        async def jalankan() -> ChatResponse:
            try:
                with catat_giliran(
                    log_sink,
                    endpoint="chat_stream",
                    session_id=payload.session_id,
                    unit=unit,
                ) as giliran:

                    def token(teks: str):
                        giliran.token_pertama()
                        return antrean.put(("token", {"text": teks}))

                    async with jejak_giliran(
                        run_id=run_id,
                        session_id=payload.session_id,
                        pertanyaan=payload.question,
                    ) as akar:
                        outcome = await run_pipeline(
                            payload.question,
                            retriever=retriever,
                            llm_call=llm_call,
                            rewrite_call=rewrite_call,
                            gate_call=gate_call,
                            history=[Turn(t.role, t.content) for t in payload.history],
                            policy=policy_from(settings),
                            on_token=token,
                            on_stage=lambda stage: antrean.put(("status", {"stage": stage})),
                            unit=unit,
                            profile=profil,
                            callbacks=[giliran.recorder],
                            tool_registry=tool_registry,
                            tool_max_rounds=settings.tools_max_rounds,
                        )
                        akhiri_jejak(akar, kind=str(outcome.kind), text=outcome.text)

                    response = to_response(outcome)
                    response.message_id = await catat(
                        chat_logger,
                        payload,
                        outcome,
                        mulai,
                        llm_call,
                        retriever,
                        run_id,
                        unit=unit,
                        profile=profil,
                        embed_key=embed_key,
                    )
                    giliran.selesai(
                        hasil=outcome.kind,
                        message_id=response.message_id,
                        langsmith_run_id=run_id,
                        llm_call=llm_call,
                    )
                return response
            finally:
                # Penutup antrean, juga saat pipeline gagal: tanpa ini penerus di
                # bawah menunggu potongan yang tidak akan pernah datang.
                await antrean.put(None)

        tugas = asyncio.create_task(jalankan())
        try:
            while (item := await antrean.get()) is not None:
                yield sse(*item)
            response = await tugas
            yield sse("message", response.model_dump(mode="json"))
            yield sse("done", {})
        finally:
            # Mahasiswa menutup panel di tengah jawaban: generator ditutup di
            # `yield`, dan tanpa ini pipeline-nya berjalan terus sampai selesai.
            tugas.cancel()

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=SSE_HEADERS
    )


@router.post(
    "/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={403: {"model": Error}, 404: {"model": Error}, 429: {"model": Error}},
    dependencies=[Depends(batasi_penilaian)],
)
async def submit_feedback(
    payload: FeedbackRequest, session: SessionDep, _embed_key: EmbedKeyDep
) -> Response:
    """FE-5: satu klik, tanpa modal.

    Klik ulang pada pesan yang sama mengganti umpan balik sebelumnya: mahasiswa
    yang berubah pikiran dari 👍 ke 👎 tidak boleh terhitung dua kali pada rasio
    AD-5. Kunci sematan hanya diperiksa masih berlaku; situs yang sudah dicabut
    tidak lagi mengisi statistik.
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
            "INSERT INTO feedback (id, message_id, helpful, comment)"
            " VALUES (:id, :message_id, :helpful, :comment)"
        ),
        {
            "id": uuid.uuid4(),
            "message_id": payload.message_id,
            "helpful": payload.helpful,
            "comment": (payload.comment or "").strip() or None,
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
            ContactOut(unit=c.unit, service_hours=c.jam_layanan, contact=c.kontak)
            for c in outcome.contacts
        ],
        escalated=bool(outcome.contacts),
        # Sudah disaring `generate`: hanya `answer` dan hanya yang sumbernya dikutip.
        attachments=[
            AttachmentOut(title=a.title, source=a.source, items=list(a.items))
            for a in outcome.attachments
        ],
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

    Jawaban tanpa satu pun penanda sitasi juga tidak membawa kartu. Dulu ia
    jatuh kembali ke seluruh chunk terambil, tetapi yang tidak dikutip justru
    bukan sumber jawabannya: balasan "Selamat pagi!" dari LLM tampil dengan
    tujuh kartu dokumen yang tidak berkaitan, seolah sapaan itu bersumber.
    """
    tersedia: dict[tuple[str, int], CitationOut] = {}
    for doc in outcome.documents:
        meta = doc.metadata
        kartu = CitationOut(
            title=meta.get("judul", ""),
            page=meta.get("halaman", 0),
            document_id=str(meta.get("document_id", "")),
            # Entri tanya jawab tidak punya berkas: `file_path` NULL dari database
            # menjadi string kosong, dan `type` memberi tahu frontend agar
            # kartunya tidak dibuat sebagai tautan yang buntu.
            file_path=meta.get("file_path") or "",
            # Hasil tool menumpang `jenis` tanya_jawab di pipeline, tetapi
            # kartunya bukan tanya jawab admin (T44).
            type=(
                CitationType.DATA
                if meta.get("dari_tool")
                else meta.get("jenis") or CitationType.PDF
            ),
        )
        # Chunk berbeda dari halaman yang sama menghasilkan kartu yang sama.
        tersedia.setdefault((kartu.title.casefold(), kartu.page), kartu)

    # Entri tanya jawab dan kartu tool dikutip `[Judul]` tanpa halaman (T25).
    tanpa_halaman = {
        kartu.title: kartu.page
        for kartu in tersedia.values()
        if kartu.type != CitationType.PDF
    }

    # Urutan mengikuti kemunculan di jawaban, bukan peringkat retrieval: itu
    # urutan yang dibaca mahasiswa.
    dikutip: list[CitationOut] = []
    for citation in extract_citations(outcome.text, tanpa_halaman):
        kartu = tersedia.get((citation.judul.casefold(), citation.halaman))
        if kartu is not None and kartu not in dikutip:
            dikutip.append(kartu)

    return dikutip


async def catat(
    chat_logger: Any,
    payload: ChatRequest,
    outcome: PipelineOutcome,
    mulai: float,
    llm_call: Any,
    retriever: Any = None,
    run_id: str | None = None,
    *,
    unit: str | None = None,
    profile: ProfilMahasiswa | None = None,
    embed_key: str | None = None,
) -> str | None:
    """Catat putaran ini (FR-8). None bila pencatatan gagal; jawaban tetap terkirim.

    `run_id` adalah akar trace giliran ini, bukan ID salah satu panggilan LLM di
    dalamnya: yang dibuka admin dari AD-4 harus giliran utuh -- rewrite,
    retrieval, dan jawaban sekaligus. Ia None saat tracing mati, dan None itu
    ikut tersimpan apa adanya; mencatat ID saat tidak ada trace yang dikirim
    hanya menghasilkan tautan yang berujung pada halaman kosong.
    """
    embed = getattr(retriever, "embed_query", None)
    return await chat_logger.log(
        ChatLogEntry(
            session_id=payload.session_id,
            question=sanitize_question(payload.question),
            outcome=outcome,
            latency_ms=round((time.perf_counter() - mulai) * 1000),
            model=getattr(llm_call, "model", None),
            usage=getattr(llm_call, "usage", None),
            tool_calls=getattr(llm_call, "tool_calls", None),
            langsmith_run_id=run_id,
            unit=unit,
            profile=profile,
            nim=payload.nim,
            embed_key=embed_key,
            # `getattr` berlapis, sama seperti `llm_call` di atas: test menyuntikkan
            # retriever palsu tanpa alat ukur, dan pencatatan tidak boleh menuntut
            # jenis retriever tertentu.
            embed_dipanggil=bool(getattr(embed, "panggilan", 0)),
            embed_model=getattr(embed, "model", None),
            embed_tokens=getattr(embed, "tokens", None),
            embed_biaya_usd=getattr(embed, "biaya_usd", None),
            embed_biaya_sumber=getattr(embed, "biaya_sumber", None),
        )
    )


def policy_from(settings) -> ThresholdPolicy:
    return ThresholdPolicy(
        vector_threshold=settings.vector_threshold,
        lexical_threshold=settings.lexical_threshold,
        # Hanya berarti bila reranker hidup; tanpa skor reranker, ambang lama
        # yang berlaku (lihat `ThresholdPolicy.rerank_threshold`).
        rerank_threshold=settings.rerank_threshold if settings.rerank_enabled else None,
    )


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
