"""Orkestrasi alur tanya-jawab (PRD §7 'Struktur Chain').

Alurnya punya beberapa jalan keluar lebih awal (sensitif, penolakan) yang
harus dapat dibuktikan lewat test -- terutama invarian "LLM tidak dipanggil
saat ditolak". Karena itu urutannya ditulis eksplisit di sini, sementara
LangChain tetap memegang bagian prompt + pemanggilan model.

Urutan sengaja: pemeriksaan sensitif (FR-7) mendahului segalanya, termasuk
retrieval. Mahasiswa yang menulis "saya stres, takut di-DO" tidak boleh
dibalas kutipan pasal tata cara DO.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.rag import risk as risk_module
from app.rag import sensitive as sensitive_module
from app.rag import smalltalk as smalltalk_module
from app.rag.rewriter import Turn, format_history, needs_rewrite
from app.rag.threshold import Decision, ThresholdDecision, ThresholdPolicy, evaluate
from app.security.sanitize import sanitize_question, wrap_user_input


class OutcomeKind(StrEnum):
    ANSWER = "answer"
    REFUSAL = "refusal"
    SUPPORT = "support"
    """Balasan empatik FR-7; bukan jawaban administrasi."""
    SMALLTALK = "smalltalk"
    """Sapaan atau basa-basi. Dibalas singkat tanpa retrieval maupun LLM, dan
    tidak pernah membawa sitasi -- tidak ada dokumen yang menjawab "hai"."""


@dataclass(frozen=True)
class PipelineOutcome:
    kind: OutcomeKind
    text: str
    documents: tuple[Any, ...] = ()
    decision: ThresholdDecision | None = None
    risk: risk_module.RiskAssessment | None = None
    sensitivity: sensitive_module.SensitivityAssessment | None = None
    rewritten_query: str | None = None
    llm_called: bool = False
    contacts: tuple[Any, ...] = field(default_factory=tuple)


REFUSAL_TEMPLATE = (
    "Maaf, saya tidak menemukan informasi ini di dokumen resmi yang saya miliki. "
    "Supaya Anda tidak mendapat jawaban yang keliru, silakan tanyakan langsung ke:"
    "\n\n{contacts}"
)

DEFAULT_FALLBACK_CONTACT = risk_module.UnitContact(
    unit="Biro Administrasi Akademik",
    jam_layanan="Senin-Jumat, 08.00-15.00",
    kontak="akademik@kampus.ac.id",
)

SUPPORT_TEMPLATE = (
    "Terima kasih sudah menyampaikan ini. Yang Anda rasakan wajar dan Anda tidak "
    "sendirian. Saya tidak bisa membantu urusan seperti ini lewat dokumen "
    "administrasi, tetapi ada orang yang siap mendengarkan:\n\n{contacts}"
)


def render_contacts(contacts) -> str:
    """Susun daftar kontak menjadi teks siap tampil (FE-3, FE-4)."""
    return "\n".join(f"- {c.unit} ({c.jam_layanan}): {c.kontak}" for c in contacts)


async def run_pipeline(
    question: str,
    *,
    retriever: Any,
    llm_call: Callable[[str, Sequence[Any]], Awaitable[str]],
    history: list[Turn] | None = None,
    rewrite_call: Callable[[str, str], Awaitable[str]] | None = None,
    policy: ThresholdPolicy | None = None,
    on_token: Callable[[str], Awaitable[None]] | None = None,
    on_stage: Callable[[str], Awaitable[None]] | None = None,
) -> PipelineOutcome:
    """Jalankan satu putaran tanya-jawab.

    Args:
        question: pertanyaan mentah dari mahasiswa.
        retriever: apa pun yang punya `ainvoke(str) -> list[Document]`.
        llm_call: (pertanyaan_terbungkus, dokumen) -> teks jawaban.
        history: riwayat percakapan; kosong berarti pesan pertama (FR-4).
        rewrite_call: (pertanyaan, riwayat_terformat) -> pertanyaan mandiri.
        policy: ambang penolakan FR-3.
        on_token: dipanggil untuk setiap potongan jawaban LLM (FE-1). Tanpa ini
            jawaban tetap dirakit utuh dulu, seperti `/api/chat`.
        on_stage: dipanggil saat tahap yang terlihat mahasiswa berganti, supaya
            indikator FE-1 tidak tertinggal di "mencari dokumen" sepanjang LLM
            menyusun kalimat pertamanya.

    `llm_call` dan `rewrite_call` disuntikkan agar test dapat membuktikan
    kapan LLM dipanggil dan kapan tidak, tanpa memanggil API sungguhan.
    """
    history = history or []
    clean = sanitize_question(question)

    # FR-7 -- mendahului retrieval dan LLM.
    sensitivity = sensitive_module.detect(clean)
    if sensitivity.bypasses_rag:
        contacts_text = render_contacts(sensitivity.contacts)
        return PipelineOutcome(
            kind=OutcomeKind.SUPPORT,
            text=SUPPORT_TEMPLATE.format(contacts=contacts_text),
            sensitivity=sensitivity,
            contacts=sensitivity.contacts,
            llm_called=False,
        )

    # Sapaan dan basa-basi, setelah FR-7 supaya "halo, saya stres" tetap
    # ditangani sebagai pertanyaan sensitif. Tidak ada dokumen resmi yang
    # menjawab "hai": menjalankannya lewat retrieval hanya menghasilkan
    # penolakan FR-3 yang kaku dan satu baris palsu di AD-4.
    chitchat = smalltalk_module.detect(clean)
    if chitchat.handled:
        return PipelineOutcome(
            kind=OutcomeKind.SMALLTALK,
            text=chitchat.reply,
            sensitivity=sensitivity,
            llm_called=False,
        )

    # FR-4 -- dilewati bila pesan pertama.
    search_query = clean
    rewritten: str | None = None
    if rewrite_call is not None and needs_rewrite(history):
        rewritten = (await rewrite_call(clean, format_history(history))).strip()
        if rewritten:
            search_query = rewritten

    # FR-2
    documents = await retriever.ainvoke(search_query)

    # FR-3 -- LLM tidak dipanggil bila ditolak.
    hits = _hits_from_documents(documents)
    decision = evaluate(hits, policy)
    if decision.decision is Decision.REFUSE:
        assessment = risk_module.detect(clean)
        # Semua unit terkait ditampilkan, bukan hanya yang pertama: pertanyaan
        # "deadline pembayaran UKT" menyangkut akademik DAN keuangan sekaligus,
        # dan mahasiswa yang ditolak tidak boleh dikirim ke loket yang salah.
        contacts = assessment.contacts or (DEFAULT_FALLBACK_CONTACT,)
        return PipelineOutcome(
            kind=OutcomeKind.REFUSAL,
            text=REFUSAL_TEMPLATE.format(contacts=render_contacts(contacts)),
            documents=tuple(documents),
            decision=decision,
            risk=assessment,
            sensitivity=sensitivity,
            rewritten_query=rewritten,
            llm_called=False,
            contacts=contacts,
        )

    # FR-6
    assessment = risk_module.detect(clean)

    if on_stage is not None:
        await on_stage("menyusun jawaban")

    # FR-5
    answer = await _jawab(llm_call, wrap_user_input(clean), documents, on_token)

    return PipelineOutcome(
        kind=OutcomeKind.ANSWER,
        text=answer,
        documents=tuple(documents),
        decision=decision,
        risk=assessment,
        sensitivity=sensitivity,
        rewritten_query=rewritten,
        llm_called=True,
        contacts=assessment.contacts,
    )


async def _jawab(
    llm_call: Any,
    wrapped_question: str,
    documents: Sequence[Any],
    on_token: Callable[[str], Awaitable[None]] | None,
) -> str:
    """Panggil LLM, alirkan potongannya bila pemanggil memintanya.

    Teks yang dikembalikan tetap jawaban utuh, bukan sisa potongan: sitasi
    (FE-2), penanda `[Judul, hal. N]`, dan baris log FR-8 semuanya dibaca dari
    satu nilai yang sama, entah jawabannya dialirkan atau tidak.

    `llm_call` tanpa metode `stream` dilayani lewat jalur biasa. Test menyuntik
    callable polos, dan tidak ada gunanya memaksa setiap pengganti LLM ikut
    menyediakan versi streaming hanya agar pipeline-nya berjalan.
    """
    stream = getattr(llm_call, "stream", None)
    if on_token is None or stream is None:
        return await llm_call(wrapped_question, documents)

    bagian: list[str] = []
    async for potongan in stream(wrapped_question, documents):
        bagian.append(potongan)
        await on_token(potongan)
    return "".join(bagian)


def _hits_from_documents(documents: Sequence[Any]):
    """Ambil kembali `FusedHit` dari metadata Document yang dikeluarkan retriever."""
    from app.rag.fusion import FusedHit

    hits = []
    for doc in documents:
        meta = getattr(doc, "metadata", {}) or {}
        hits.append(
            FusedHit(
                chunk_id=str(meta.get("chunk_id", "")),
                rrf_score=float(meta.get("rrf_score", 0.0)),
                ranks=dict(meta.get("ranks", {})),
                raw_scores=dict(meta.get("raw_scores", {})),
            )
        )
    return hits
