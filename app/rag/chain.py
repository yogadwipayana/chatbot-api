"""Orkestrasi alur tanya-jawab (PRD §7 'Struktur Chain').

Alurnya punya beberapa jalan keluar lebih awal (sensitif, sapaan, gerbang
Laya, penolakan) yang harus dapat dibuktikan lewat test -- terutama invarian
"LLM tidak dipanggil saat ditolak". Urutannya dipegang graf LangGraph di
`app/rag/graph.py`; modul ini menyimpan bentuk hasilnya dan titik masuknya,
sementara LangChain tetap memegang bagian prompt + pemanggilan model.

Urutan sengaja: pemeriksaan sensitif (FR-7) mendahului segalanya, termasuk
retrieval. Mahasiswa yang menulis "saya stres, takut di-DO" tidak boleh
dibalas kutipan pasal tata cara DO.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.db.models import DocumentType
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
    topik kampus. Dihentikan gerbang Laya atau saringan aturan sebelum LLM, atau
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
    """Vonis gerbang Laya; None bila gerbang mati atau tidak sempat berjalan."""
    rewritten_query: str | None = None
    llm_called: bool = False
    contacts: tuple[Any, ...] = field(default_factory=tuple)
    attachments: tuple[Any, ...] = field(default_factory=tuple)
    """Lampiran tool (`app.rag.tools.base.Lampiran`) yang tampil di bawah jawaban.
    Hanya untuk `answer`, dan hanya yang penanda sumbernya dikutip (docs/tool-call.md §10a)."""


REFUSAL_TEMPLATE = (
    "Maaf, saya belum menemukan jawabannya di {cakupan}. "
    "Coba tuliskan pertanyaan Anda dengan lebih rinci, atau agar jawabannya "
    "pasti, tanyakan langsung ke:"
    "\n\n{contacts}"
)
"""`{cakupan}` diisi `refusal_text`: "dokumen yang saya miliki", atau "dokumen
unit X" bila mahasiswa memilih unit. Cakupan disebut di kalimat pertama, bukan
setelah daftar kontak -- kontak satu unit bisa empat baris, dan kalimat di
bawahnya jatuh ke luar layar widget. Sengaja tidak menyebut "dokumen resmi",
begitu pula label bubble-nya di client (keputusan copy 2026-11-23).

"Belum menemukan", bukan "belum membahas hal ini", dan ajakan menulis lebih
rinci (keputusan user, 2026-11-23): sebagian penolakan terjadi karena cara
bertanya, bukan karena informasinya tidak ada. "Matakuliah ... syaratnya apa
supaya bisa diambil?" ditolak, lalu dijawab begitu ditulis "mata kuliah"
(log 2026-11-12). Kalimat yang menyatakan informasinya tidak ada membuat
mahasiswa berhenti bertanya."""

REFUSAL_UNIT_HINT = (
    "\n\nBila pertanyaan Anda ditangani unit lain, ganti topik lalu tanyakan lagi."
)
"""Tanpa ini, mahasiswa yang salah memilih unit menyimpulkan informasinya memang
tidak ada -- padahal yang membatasi adalah pilihannya sendiri.

Jangan menyarankan "semua unit": widget mahasiswa -- satu-satunya pengirim
`unit` -- mewajibkan satu topik dipilih dan tidak punya pilihan semua unit."""


ALASAN_TOOL_TEMPLATE = (
    "Maaf, {alasan} Agar jawabannya pasti, tanyakan langsung ke:\n\n{contacts}"
)
"""Penolakan jalur tool yang gagal karena sebab yang diketahui (T77), mis. NIM
tidak terdaftar di SADS: alasannya disebut, bukan "belum menemukan jawabannya di
dokumen". Ajakan menulis lebih rinci dan petunjuk ganti topik sengaja tidak ada:
bertanya ulang atau pindah topik tidak mengubah hasil tool data pribadi."""


def refusal_text(contacts, unit: str | None = None, alasan: Sequence[str] = ()) -> str:
    """Teks penolakan FR-3 lengkap dengan kontak, dan petunjuk ganti topik bila
    mahasiswa memilih unit. Satu-satunya penyusun teks penolakan.

    `alasan` (`ToolResult.alasan_mahasiswa`) mengganti kalimat pembukanya dengan
    `ALASAN_TOOL_TEMPLATE`."""
    if alasan:
        return ALASAN_TOOL_TEMPLATE.format(
            alasan=" ".join(alasan), contacts=render_contacts(contacts)
        )
    cakupan = f"dokumen unit {unit}" if unit else "dokumen yang saya miliki"
    teks = REFUSAL_TEMPLATE.format(cakupan=cakupan, contacts=render_contacts(contacts))
    return teks + REFUSAL_UNIT_HINT if unit else teks


DEFAULT_FALLBACK_CONTACT = risk_module.FRONT_OFFICE

SUPPORT_TEMPLATE = (
    "Terima kasih sudah menyampaikan ini. Yang Anda rasakan wajar dan Anda tidak "
    "sendirian. Saya tidak bisa membantu urusan seperti ini lewat dokumen "
    "administrasi, tetapi ada orang yang siap mendengarkan:\n\n{contacts}"
)


def render_contacts(contacts) -> str:
    """Susun daftar kontak menjadi teks siap tampil (FE-3, FE-4)."""
    baris = []
    for c in contacts:
        jam = f" ({c.jam_layanan})" if c.jam_layanan else ""
        baris.append(f"- {c.unit}{jam}: {c.kontak}")
    return "\n".join(baris)


_HOMOGLIF = str.maketrans(
    {
        # Huruf Kiril dan Yunani yang tampak sama dengan huruf Latin penanda.
        # Model pernah membalas "[TIDAK_DITEMUKAN]" dengan T, A, K Kiril (T72,
        # 2026-11-18); tanpa pemetaan ini teks itu lolos sebagai jawaban dan
        # tampil apa adanya ke mahasiswa. Ditulis sebagai kode titik supaya
        # sumbernya tetap ASCII.
        0x0410: "A", 0x0412: "B", 0x0415: "E", 0x041A: "K", 0x041C: "M",
        0x041D: "H", 0x041E: "O", 0x0420: "P", 0x0421: "C", 0x0422: "T",
        0x0423: "Y", 0x0425: "X", 0x0406: "I", 0x0408: "J", 0x0405: "S",
        0x0500: "D", 0x04AE: "Y", 0x04C0: "I",
        0x0391: "A", 0x0392: "B", 0x0395: "E", 0x0396: "Z", 0x0397: "H",
        0x0399: "I", 0x039A: "K", 0x039C: "M", 0x039D: "N", 0x039F: "O",
        0x03A1: "P", 0x03A4: "T", 0x03A5: "Y", 0x03A7: "X",
        # Karakter tak terlihat yang dapat menyelip di tengah penanda.
        0x200B: None, 0x200C: None, 0x200D: None, 0x2060: None, 0xFEFF: None,
    }
)  # fmt: skip


def _seragam(teks: str) -> str:
    """Bentuk pembanding penanda: NFKC (kurung dan huruf lebar penuh) lalu
    homoglif ke Latin. Hanya untuk deteksi; teks ini tidak pernah ditampilkan."""
    return unicodedata.normalize("NFKC", teks).translate(_HOMOGLIF)


_KURUNG_BUKA = "[" + chr(0xFF3B)
_KURUNG_TUTUP = "]" + chr(0xFF3D)
_CALON_PENANDA = re.compile(
    f"[{re.escape(_KURUNG_BUKA)}][^{re.escape(_KURUNG_BUKA + _KURUNG_TUTUP)}\n]{{1,40}}"
    f"[{re.escape(_KURUNG_TUTUP)}]"
)
"""Teks berkurung yang mungkin penanda, termasuk yang ditulis dengan homoglif."""


_NOT_FOUND_PROSE = re.compile(
    r"\btidak\s+(?:menemukan|ditemukan|tercantum)\b|\btidak\s+ada\s+informasi\b",
    re.IGNORECASE,
)


def sumber_tanpa_halaman(documents: Sequence[Any]) -> dict[str, int]:
    """Judul sumber tak berhalaman (entri tanya jawab) -> halamannya, untuk
    `extract_citations`: kutipan `[Judul]` baru terbaca sebagai sitasi dengan ini."""
    return {
        doc.metadata.get("judul", ""): doc.metadata.get("halaman", 1)
        for doc in documents
        if doc.metadata.get("jenis") == DocumentType.TANYA_JAWAB
    }


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
    return NOT_FOUND_MARKER in _seragam(answer) or bool(_NOT_FOUND_PROSE.search(answer))


def is_off_topic(answer: str, tanpa_halaman: dict[str, int] | None = None) -> bool:
    """Apakah LLM menilai pertanyaannya di luar urusan kampus (aturan 4 `SYSTEM_PROMPT`).

    Jawaban bersitasi tidak pernah dianggap di luar topik: sitasi berarti
    dokumen kampus menjawabnya, dan penandanya cukup dibuang (`strip_markers`).
    """
    # `tanpa_halaman`: sumber tak berhalaman (tanya jawab, kartu tool) yang dikutip
    # `[Judul]`; tanpa daftar ini kutipannya tidak terbaca sebagai sitasi.
    return OFF_TOPIC_MARKER in _seragam(answer) and not extract_citations(
        answer, tanpa_halaman
    )


def rejection_source(outcome: PipelineOutcome) -> str | None:
    """Siapa yang menghentikan pesan `rejected`: `laya`, `rules`, atau `llm`.
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
    """Buang penanda yang tersisa di jawaban parsial; mahasiswa tidak perlu melihatnya.

    Penanda berhomoglif ikut dibuang (`_seragam`), dari teks aslinya."""
    if not any(marker in _seragam(answer) for marker in MARKERS):
        return answer
    for cocok in reversed(list(_CALON_PENANDA.finditer(answer))):
        if _seragam(cocok.group()) in MARKERS:
            answer = answer[: cocok.start()] + answer[cocok.end() :]
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
    nim: str | None = None,
    unit_contacts: Sequence[Any] = (),
) -> PipelineOutcome:
    """Jalankan satu putaran tanya-jawab lewat graf `app.rag.graph`.

    Args:
        question: pertanyaan mentah dari mahasiswa.
        retriever: apa pun yang punya `ainvoke(str, *, unit) -> list[Document]`;
            bila pertanyaan ditulis ulang, juga menerima `original_query`.
        llm_call: (pertanyaan_terbungkus, dokumen) -> teks jawaban.
        history: riwayat percakapan; kosong berarti pesan pertama (FR-4).
        rewrite_call: (pertanyaan, riwayat_terformat) -> pertanyaan mandiri.
        gate_call: (pertanyaan, [(peran, isi)]) -> vonis gerbang Laya. None
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
        nim: NIM penanya, hanya untuk tool data pribadi (docs/tool-call.md §11).
            Dibawa `PipelineDeps`, bukan state graf, supaya tidak ikut terekam
            sebagai masukan node atau trace. None = tanpa NIM (uji coba admin).
        unit_contacts: kontak PIC `unit` (`risk.UnitContact`), dari
            `SqlUnitDirectory.kontak_tampil`. Kosong = kontak per topik risiko,
            lalu Front Office.

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
        nim=nim,
        unit_contacts=tuple(unit_contacts),
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
        awal = _seragam(self.tertahan.lstrip())
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
            )
        )
    return hits
