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

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.prodi import ProfilMahasiswa
from app.rag import risk as risk_module
from app.rag import sensitive as sensitive_module
from app.rag.citations import extract_citations
from app.rag.gate import GateSource, GateVerdict
from app.rag.prompts import NOT_FOUND_MARKER, OFF_TOPIC_MARKER
from app.rag.rewriter import Turn
from app.rag.threshold import ThresholdDecision, ThresholdPolicy


class OutcomeKind(StrEnum):
    ANSWER = "answer"
    REFUSAL = "refusal"
    """Dokumen resmi tidak menjawab. Diputuskan threshold FR-3 sebelum LLM
    (`llm_called=False`), atau oleh LLM sendiri lewat `NOT_FOUND_MARKER` saat
    konteksnya mirip tetapi tidak menjawab (`llm_called=True`). Keduanya
    tampil dan tercatat sama: tanpa sitasi, dengan kontak, masuk AD-4."""
    SUPPORT = "support"
    """Balasan empatik FR-7; bukan jawaban administrasi."""
    SMALLTALK = "smalltalk"
    """Sapaan atau basa-basi. Dibalas singkat tanpa retrieval maupun LLM, dan
    tidak pernah membawa sitasi -- tidak ada dokumen yang menjawab "hai"."""
    REJECTED = "rejected"
    """Bukan pertanyaan administrasi: nonsense, upaya manipulasi, atau di luar
    topik kampus. Dihentikan gerbang JEV atau saringan aturan sebelum LLM, atau
    oleh LLM penjawab sendiri lewat `OFF_TOPIC_MARKER` (`rejection_source`).
    Tanpa sitasi dan tidak masuk AD-4 -- pesan seperti ini bukan celah dokumen
    yang perlu ditambal admin. Pencarian yang berjalan paralel dengan gerbang
    dihentikan begitu vonis blokir tiba."""


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
    attachments: tuple[Any, ...] = field(default_factory=tuple)
    """Lampiran tool (`app.rag.tools.base.Lampiran`) yang tampil di bawah jawaban.
    Hanya untuk `answer`, dan hanya yang penanda sumbernya dikutip (docs/tool-call.md §10a)."""


REFUSAL_TEMPLATE = (
    "Maaf, saya tidak menemukan informasi ini di dokumen resmi yang saya miliki. "
    "Supaya Anda tidak mendapat jawaban yang keliru, silakan tanyakan langsung ke:"
    "\n\n{contacts}"
)

REFUSAL_UNIT_HINT = (
    "\n\nPencarian tadi hanya di dokumen unit {unit}. Bila pertanyaan Anda "
    "ditangani unit lain, ganti topik ke unit tersebut lalu tanyakan lagi."
)
"""Tanpa ini, mahasiswa yang salah memilih unit hanya melihat "tidak menemukan"
dan menyimpulkan informasinya memang tidak ada -- padahal yang membatasi adalah
pilihannya sendiri.

Jangan menyarankan "semua unit": widget mahasiswa -- satu-satunya pengirim
`unit` -- mewajibkan satu topik dipilih dan tidak punya pilihan semua unit."""

DEFAULT_FALLBACK_CONTACT = risk_module.FRONT_OFFICE

SUPPORT_TEMPLATE = (
    "Terima kasih sudah menyampaikan ini. Yang Anda rasakan wajar dan Anda tidak "
    "sendirian. Saya tidak bisa membantu urusan seperti ini lewat dokumen "
    "administrasi, tetapi ada orang yang siap mendengarkan:\n\n{contacts}"
)


def render_contacts(contacts) -> str:
    """Susun daftar kontak menjadi teks siap tampil (FE-3, FE-4)."""
    return "\n".join(f"- {c.unit} ({c.jam_layanan}): {c.kontak}" for c in contacts)


_NOT_FOUND_PROSE = re.compile(
    r"\btidak\s+(?:menemukan|ditemukan|tercantum)\b|\btidak\s+ada\s+informasi\b",
    re.IGNORECASE,
)


def is_not_found(answer: str, tanpa_halaman: dict[str, int] | None = None) -> bool:
    """Apakah jawaban LLM sebenarnya penolakan (aturan 3 `SYSTEM_PROMPT`).

    Jalur utamanya `NOT_FOUND_MARKER`. Kalimat "tidak menemukan ..." tanpa satu
    pun sitasi adalah jaring pengaman untuk model yang lupa memakai penanda.
    Jawaban yang bersitasi tidak pernah dianggap penolakan, walau menyebut ada
    bagian yang tidak ditemukan: itu jawaban parsial yang tetap berguna, dan
    kartu sitasinya justru yang membuatnya dapat diverifikasi.
    """
    if extract_citations(answer, tanpa_halaman):
        return False
    return NOT_FOUND_MARKER in answer or bool(_NOT_FOUND_PROSE.search(answer))


def is_off_topic(answer: str, tanpa_halaman: dict[str, int] | None = None) -> bool:
    """Apakah LLM menilai pertanyaannya di luar urusan kampus (aturan 4 `SYSTEM_PROMPT`).

    Jawaban bersitasi tidak pernah dianggap di luar topik: sitasi berarti
    dokumen kampus menjawabnya, dan penandanya cukup dibuang (`strip_markers`).
    """
    # `tanpa_halaman`: sumber tak berhalaman (tanya jawab, kartu tool) yang dikutip
    # `[Judul]`; tanpa daftar ini kutipannya tidak terbaca sebagai sitasi.
    return OFF_TOPIC_MARKER in answer and not extract_citations(answer, tanpa_halaman)


def rejection_source(outcome: PipelineOutcome) -> str | None:
    """Siapa yang menghentikan pesan `rejected`: `jev`, `rules`, atau `llm`.
    None bila bukan `rejected`. Satu sumber untuk log dan uji coba admin."""
    if outcome.kind is not OutcomeKind.REJECTED:
        return None
    if outcome.llm_called:
        return GateSource.LLM.value
    return outcome.gate.source.value if outcome.gate else None


def refusal_source(outcome: PipelineOutcome) -> str | None:
    """Asal penolakan: `threshold` (LLM tidak dipanggil) atau `llm` (lolos ambang,
    tetapi LLM membalas `NOT_FOUND_MARKER`). None bila bukan penolakan.

    Satu sumber untuk log (`messages.meta`) dan uji coba admin, supaya keduanya
    tidak pernah menjelaskan penolakan yang sama dengan cara berbeda."""
    if outcome.kind is not OutcomeKind.REFUSAL:
        return None
    return "llm" if outcome.llm_called else "threshold"


MARKERS = (NOT_FOUND_MARKER, OFF_TOPIC_MARKER)


def strip_markers(answer: str) -> str:
    """Buang penanda yang tersisa di jawaban parsial; mahasiswa tidak perlu melihatnya."""
    if not any(marker in answer for marker in MARKERS):
        return answer
    for marker in MARKERS:
        answer = answer.replace(marker, "")
    return answer.strip()


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
    profile: ProfilMahasiswa | None = None,
    callbacks: Sequence[Any] = (),
    tool_registry: Any = None,
    tool_max_rounds: int = 2,
) -> PipelineOutcome:
    """Jalankan satu putaran tanya-jawab lewat graf `app.rag.graph`.

    Args:
        question: pertanyaan mentah dari mahasiswa.
        retriever: apa pun yang punya `ainvoke(str, *, unit) -> list[Document]`;
            bila pertanyaan ditulis ulang, juga menerima `original_query`.
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
        profile: prodi dan angkatan penanya. Bukan filter retrieval: hanya
            diteruskan ke LLM (`app.prodi`). None berarti tanpa penyesuaian.
        callbacks: callback LangChain untuk seluruh graf, mis. perekam durasi
            per node (`app.observability.applog.NodeRecorder`).
        tool_registry: registry tool-calling, atau None bila TOOLS_ENABLED=false
            (docs/tool-call.md). Dipakai sebagai gerbang kelayakan sebelum FR-3
            dan di `generate`.
        tool_max_rounds: batas putaran loop agentik tool.

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
        tool_registry=tool_registry,
        tool_max_rounds=tool_max_rounds,
    )
    return await run_graph(
        question,
        deps,
        history=list(history or []),
        unit=unit,
        profile=profile,
        callbacks=callbacks,
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

    penahan = _PenahanPenanda(on_token)
    bagian: list[str] = []
    async for potongan in stream(wrapped_question, documents):
        bagian.append(potongan)
        await penahan(potongan)
    return "".join(bagian)


class _PenahanPenanda:
    """Menahan awal aliran selama ia masih mungkin berupa salah satu `MARKERS`.

    Tanpa ini mahasiswa sempat melihat "[TIDAK_DITEMUKAN]" atau
    "[DI_LUAR_TOPIK]" terketik di layar sebelum event `message` menggantinya
    dengan kartu penolakan. Jawaban biasa hanya tertahan beberapa karakter
    pertamanya: begitu awalnya menyimpang dari semua penanda -- termasuk sitasi
    `[Judul, hal. N]` di awal kalimat -- semua yang tertahan dilepas sekaligus
    dan sisanya diteruskan apa adanya.
    """

    def __init__(self, on_token: Callable[[str], Awaitable[None]]) -> None:
        self.on_token = on_token
        self.tertahan = ""
        self.lepas = False

    async def __call__(self, potongan: str) -> None:
        if self.lepas:
            await self.on_token(potongan)
            return
        self.tertahan += potongan
        awal = self.tertahan.lstrip()
        if any(m.startswith(awal) or awal.startswith(m) for m in MARKERS):
            return
        self.lepas = True
        await self.on_token(self.tertahan)
        self.tertahan = ""


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
