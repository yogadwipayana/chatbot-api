"""Orkestrasi alur tanya-jawab (PRD §7 'Struktur Chain').

Alurnya punya beberapa jalan keluar lebih awal (sensitif, sapaan, gerbang
JEV, penolakan) yang harus dapat dibuktikan lewat test -- terutama invarian
"LLM tidak dipanggil saat ditolak". Urutannya dipegang graf LangGraph di
`app/rag/graph.py`; modul ini menyimpan bentuk hasilnya dan titik masuknya,
sementara LangChain tetap memegang bagian prompt + pemanggilan model.

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
from app.rag.gate import GateVerdict
from app.rag.rewriter import Turn
from app.rag.threshold import ThresholdDecision, ThresholdPolicy


class OutcomeKind(StrEnum):
    ANSWER = "answer"
    REFUSAL = "refusal"
    SUPPORT = "support"
    """Balasan empatik FR-7; bukan jawaban administrasi."""
    SMALLTALK = "smalltalk"
    """Sapaan atau basa-basi. Dibalas singkat tanpa retrieval maupun LLM, dan
    tidak pernah membawa sitasi -- tidak ada dokumen yang menjawab "hai"."""
    REJECTED = "rejected"
    """Dihentikan gerbang JEV: nonsense, upaya manipulasi, atau di luar topik
    kampus. Tanpa retrieval dan LLM, tanpa sitasi, dan tidak masuk AD-4 --
    pesan seperti ini bukan celah dokumen yang perlu ditambal admin."""


@dataclass(frozen=True)
class PipelineOutcome:
    kind: OutcomeKind
    text: str
    documents: tuple[Any, ...] = ()
    decision: ThresholdDecision | None = None
    risk: risk_module.RiskAssessment | None = None
    sensitivity: sensitive_module.SensitivityAssessment | None = None
    gate: GateVerdict | None = None
    """Vonis gerbang JEV; None bila gerbang mati atau tidak sempat berjalan."""
    rewritten_query: str | None = None
    llm_called: bool = False
    contacts: tuple[Any, ...] = field(default_factory=tuple)


REFUSAL_TEMPLATE = (
    "Maaf, saya tidak menemukan informasi ini di dokumen resmi yang saya miliki. "
    "Supaya Anda tidak mendapat jawaban yang keliru, silakan tanyakan langsung ke:"
    "\n\n{contacts}"
)

REFUSAL_UNIT_HINT = (
    "\n\nPencarian tadi hanya di dokumen unit {unit}. Bila pertanyaan Anda "
    "ditangani unit lain, pilih unit tersebut atau semua unit lalu tanyakan lagi."
)
"""Tanpa ini, mahasiswa yang salah memilih unit hanya melihat "tidak menemukan"
dan menyimpulkan informasinya memang tidak ada -- padahal yang membatasi adalah
pilihannya sendiri."""

DEFAULT_FALLBACK_CONTACT = risk_module.UnitContact(
    unit="Biro Administrasi Akademik",
    jam_layanan="Senin-Jumat, 08.00-15.00",
    kontak="akademik@instiki.ac.id",
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
    gate_call: Callable[[str, Sequence[tuple[str, str]]], Awaitable[GateVerdict]]
    | None = None,
    policy: ThresholdPolicy | None = None,
    on_token: Callable[[str], Awaitable[None]] | None = None,
    on_stage: Callable[[str], Awaitable[None]] | None = None,
    unit: str | None = None,
    callbacks: Sequence[Any] = (),
) -> PipelineOutcome:
    """Jalankan satu putaran tanya-jawab lewat graf `app.rag.graph`.

    Args:
        question: pertanyaan mentah dari mahasiswa.
        retriever: apa pun yang punya `ainvoke(str, *, unit) -> list[Document]`.
        llm_call: (pertanyaan_terbungkus, dokumen) -> teks jawaban.
        history: riwayat percakapan; kosong berarti pesan pertama (FR-4).
        rewrite_call: (pertanyaan, riwayat_terformat) -> pertanyaan mandiri.
        gate_call: (pertanyaan, [(peran, isi)]) -> vonis gerbang JEV. None
            berarti gerbang mati dan setiap pesan diteruskan ke retrieval.
        policy: ambang penolakan FR-3.
        on_token: dipanggil untuk setiap potongan jawaban LLM (FE-1). Tanpa ini
            jawaban tetap dirakit utuh dulu, seperti `/api/chat`.
        on_stage: dipanggil saat tahap yang terlihat mahasiswa berganti, supaya
            indikator FE-1 tidak tertinggal di "mencari dokumen" sepanjang LLM
            menyusun kalimat pertamanya.
        unit: nama resmi unit pilihan mahasiswa; retrieval hanya mencari di
            dokumen unit itu. None berarti semua unit.
        callbacks: callback LangChain untuk seluruh graf, mis. perekam durasi
            per node (`app.observability.applog.NodeRecorder`).

    `llm_call`, `rewrite_call`, dan `gate_call` disuntikkan agar test dapat
    membuktikan kapan layanan luar dipanggil dan kapan tidak, tanpa memanggil
    API sungguhan.
    """
    from app.rag.graph import PipelineDeps, run_graph

    deps = PipelineDeps(
        retriever=retriever,
        llm_call=llm_call,
        rewrite_call=rewrite_call,
        gate_call=gate_call,
        policy=policy,
        on_token=on_token,
        on_stage=on_stage,
    )
    return await run_graph(
        question, deps, history=list(history or []), unit=unit, callbacks=callbacks
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
                rerank_score=meta.get("rerank_score"),
            )
        )
    return hits
