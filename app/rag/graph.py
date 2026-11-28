"""Alur tanya-jawab sebagai graf LangGraph (flow.md §4).

Setiap langkah di `chain.py` versi lama kini menjadi satu node, dan setiap
jalan keluar lebih awal menjadi satu sisi bersyarat ke END. Isi langkahnya
tidak berubah; yang berubah hanya siapa yang memegang urutan:

    START -> sanitize -> sensitive ─┬─> END                        (FR-7)
                                    └─> smalltalk ─┬─> END
                                                   └─> rule_gate ─┬─> END  (acak, manipulasi)
                                                                  ├─> laya_gate ─────────┐
                                                                  └─> cari: rewrite     │
                                                                       -> retrieve ─────┤
      validate_context (FR-3) <── menunggu keduanya ────────────────────────────────────┘
           ├─> END                  (Laya memblokir: hasil pencarian dibuang)
           ├─> refuse   -> END      (LLM tidak dipanggil)
           └─> generate -> END      (FR-6 + FR-5; penolakan bila LLM membalas
                                     NOT_FOUND_MARKER, `rejected` bila
                                     OFF_TOPIC_MARKER)

`rule_gate` dan `OFF_TOPIC_MARKER` adalah cadangan Laya: dengan
`LAYA_ENABLED=false` keduanya menghasilkan vonis yang mirip tanpa biaya per
pesan (`app.rag.rule_gate`).

Gerbang Laya dan pencarian berjalan paralel: keduanya hanya butuh pertanyaan
yang sudah bersih, dan menunggu Laya dulu menambah ~3 detik ke setiap giliran
demi menghemat pencarian pada ~5% pesan yang diblokir. `rewrite -> retrieve`
dibungkus subgraph (`cari`) karena LangGraph berjalan per superstep: sebagai
dua node terpisah, `retrieve` baru dapat mulai setelah Laya selesai.

Ketergantungan per permintaan (retriever, LLM, callback streaming) masuk
lewat `context` LangGraph, bukan lewat state: state hanya berisi data yang
mengalir antar node, dan graf dikompilasi sekali per proses.

Node yang mengakhiri alur mengisi `outcome`; sisi bersyarat hanya memeriksa
apakah `outcome` sudah ada. Satu aturan itu yang menjamin tidak ada node
sesudahnya -- termasuk LLM -- yang berjalan setelah keputusan berhenti diambil.

    python -m app.rag.graph   # cetak diagram Mermaid untuk dokumentasi
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.db.models import DocumentType
from app.prodi import ProfilMahasiswa
from app.rag import risk as risk_module
from app.rag import rule_gate as rule_gate_module
from app.rag import sensitive as sensitive_module
from app.rag import smalltalk as smalltalk_module
from app.rag.chain import (
    DEFAULT_FALLBACK_CONTACT,
    SUPPORT_TEMPLATE,
    OutcomeKind,
    PipelineOutcome,
    _hits_from_documents,
    _jawab,
    is_not_found,
    is_off_topic,
    refusal_text,
    render_contacts,
    strip_markers,
    sumber_tanpa_halaman,
)
from app.rag.citations import extract_citations, ringkas_sitasi_tanpa_halaman
from app.rag.gate import GateLabel, GateVerdict, lolos
from app.rag.gate import balasan as balasan_gerbang
from app.rag.prompts import pesan_mahasiswa
from app.rag.rewriter import HISTORY_WINDOW, Turn, format_history, needs_rewrite
from app.rag.threshold import Decision, ThresholdDecision, ThresholdPolicy, evaluate
from app.rag.tools.base import Penanya
from app.security.sanitize import sanitize_question, wrap_user_input

logger = logging.getLogger(__name__)

DEFAULT_LAYA_GRACE_SECONDS = 1.5
"""Dipakai bila `gate_call` tidak membawa `grace_seconds` (pengganti di test)."""


@dataclass
class VonisTertunda:
    """Panggilan Laya yang masih berjalan saat tenggatnya lewat (T79).

    Laya dilayani satu per satu di CPU, jadi saat banyak mahasiswa bertanya
    bersamaan vonisnya mengantre dan tiba setelah pencarian. Memutusnya di
    tenggat menghilangkan gerbang justru saat ramai, padahal pada saat itu
    panggilan LLM penjawab juga mengantre lama di vLLM. Maka lewat tenggat
    panggilannya dibiarkan berjalan: `generate` sudah memanggil LLM, dan selama
    LLM belum menghasilkan apa pun vonisnya ditunggu tanpa biaya (hanya dibatasi
    `LAYA_TIMEOUT_SECONDS`). Begitu keluaran siap dikirim -- token pertama,
    jawaban utuh, atau penolakan `refuse` -- vonisnya ditunggu paling lama
    `tahan` detik lagi (`akhir()`), lalu pesan diteruskan (fail-open).
    """

    tugas: asyncio.Future[GateVerdict] | None = None
    tahan: float = 0.0
    """Lama vonis masih ditunggu setelah keluaran siap dikirim."""
    _akhir: asyncio.Future[GateVerdict] | None = None

    @property
    def ada(self) -> bool:
        return self.tugas is not None

    @property
    def tiba(self) -> bool:
        """Laya sudah menjawab (atau gagal); `vonis()` tidak lagi menunggu."""
        return self.tugas is not None and self.tugas.done()

    def vonis(self) -> GateVerdict:
        """Vonis panggilan yang sudah selesai; galat apa pun berarti fail-open."""
        assert self.tugas is not None and self.tugas.done()
        if self.tugas.cancelled():
            if self._akhir is not None and self._akhir.done() and not self._akhir.cancelled():
                return self._akhir.result()  # dibatalkan karena jendela `tahan` habis
            return lolos("Tenggat: panggilan Laya dibatalkan")
        try:
            return self.tugas.result()
        except Exception as exc:  # `LayaGate` tidak melempar; pengganti di test bisa
            return lolos(f"{type(exc).__name__}: {exc}"[:200])

    def akhir(self) -> asyncio.Future[GateVerdict]:
        """Vonis akhir, dipanggil saat keluaran siap dikirim. Jendela `tahan`
        dihitung sejak panggilan pertama; semua penunggu berbagi future yang sama."""
        if self._akhir is None:
            self._akhir = asyncio.ensure_future(self._tunggu())
        return self._akhir

    async def _tunggu(self) -> GateVerdict:
        assert self.tugas is not None
        # asyncio.wait, bukan wait_for: batalnya penunggu tidak membatalkan Laya.
        selesai, _ = await asyncio.wait({self.tugas}, timeout=self.tahan)
        if self.tugas in selesai:
            return self.vonis()
        self.tugas.cancel()
        logger.warning(
            "Gerbang Laya belum menjawab %.1f dtk setelah jawaban siap, pesan diteruskan",
            self.tahan,
        )
        return lolos(f"Tenggat: Laya belum menjawab {self.tahan:g} dtk setelah jawaban siap")

    def batalkan(self) -> None:
        """Giliran selesai lebih dulu (mis. galat): jangan tinggalkan panggilan Laya."""
        for f in (self._akhir, self.tugas):
            if f is not None and not f.done():
                f.cancel()


@dataclass(frozen=True)
class PipelineDeps:
    """Ketergantungan satu giliran. Lihat `chain.run_pipeline` untuk artinya."""

    retriever: Any
    llm_call: Callable[[str, Sequence[Any]], Awaitable[str]]
    rewrite_call: Callable[[str, str], Awaitable[str]] | None = None
    gate_call: Callable[[str, Sequence[tuple[str, str]]], Awaitable[GateVerdict]] | None = None
    policy: ThresholdPolicy | None = None
    on_token: Callable[[str], Awaitable[None]] | None = None
    on_stage: Callable[[str], Awaitable[None]] | None = None
    search_done: asyncio.Event = field(default_factory=asyncio.Event)
    """Dinyalakan `retrieve`; tenggat gerbang Laya dihitung darinya. Satu per
    giliran -- `PipelineDeps` dibuat baru di setiap `run_pipeline`."""
    gate_blocked: asyncio.Event = field(default_factory=asyncio.Event)
    """Dinyalakan `laya_gate` saat memblokir; `rewrite` dan `retrieve` berhenti
    begitu melihatnya. Tanpa ini pesan yang sudah diblokir tetap menunggu --
    dan ikut gagal bersama -- cabang pencarian yang hasilnya akan dibuang."""
    gerbang_tertunda: VonisTertunda = field(default_factory=VonisTertunda)
    """Diisi `laya_gate` bila vonis Laya belum tiba di tenggat (T79). Di sini,
    bukan di state: isinya future, dan state terekam sebagai masukan node."""
    tool_registry: Any = None
    """Registry tool-calling, atau None bila TOOLS_ENABLED=false (docs/tool-call.md).
    Dipakai sebagai gerbang kelayakan di `validate_context` dan di `generate`."""
    tool_max_rounds: int = 2
    nim: str | None = None
    """NIM penanya untuk tool data pribadi. Di sini, bukan di `PipelineState`:
    state terekam sebagai masukan/keluaran node (tab Graf, LangSmith), dan NIM
    tidak boleh ikut (`app.prodi`)."""
    unit_contacts: tuple[Any, ...] = ()
    """Kontak PIC unit pilihan mahasiswa (`risk.UnitContact`), dimuat pemanggil
    dari tabel `unit_contacts`. Graf tidak membaca database sendiri; kosong
    berarti kontak per topik risiko atau Front Office. Yang tampil disaring
    per pertanyaan (`risk.pilih_kontak`): kata kunci kontak dan dosen PA."""


class PipelineState(TypedDict, total=False):
    question: str
    history: list[Turn]
    unit: str | None
    profile: ProfilMahasiswa | None
    clean: str
    sensitivity: sensitive_module.SensitivityAssessment
    gate: GateVerdict | None
    search_query: str
    rewritten: str | None
    documents: list[Any]
    decision: ThresholdDecision
    tool_eligible: bool
    outcome: PipelineOutcome


Rt = Runtime[PipelineDeps]


# --- Node ---------------------------------------------------------------


async def sanitize(state: PipelineState) -> dict:
    return {"clean": sanitize_question(state["question"])}


async def sensitive(state: PipelineState) -> dict:
    """FR-7 -- mendahului retrieval dan LLM."""
    sensitivity = sensitive_module.detect(state["clean"])
    if not sensitivity.bypasses_rag:
        return {"sensitivity": sensitivity}
    return {
        "sensitivity": sensitivity,
        "outcome": PipelineOutcome(
            kind=OutcomeKind.SUPPORT,
            text=SUPPORT_TEMPLATE.format(contacts=render_contacts(sensitivity.contacts)),
            sensitivity=sensitivity,
            contacts=sensitivity.contacts,
            llm_called=False,
        ),
    }


async def smalltalk(state: PipelineState) -> dict:
    """Sapaan berbasis aturan, setelah FR-7 supaya "halo, saya stres" tetap sensitif.

    Tidak ada dokumen resmi yang menjawab "hai": menjalankannya lewat retrieval
    hanya menghasilkan penolakan FR-3 yang kaku dan satu baris palsu di AD-4.
    """
    chitchat = smalltalk_module.detect(state["clean"])
    if not chitchat.handled:
        return {}
    return {
        "outcome": PipelineOutcome(
            kind=OutcomeKind.SMALLTALK,
            text=chitchat.reply,
            sensitivity=state["sensitivity"],
            llm_called=False,
        )
    }


async def rule_gate(state: PipelineState) -> dict:
    """Saringan aturan (`app.rag.rule_gate`): pesan acak, tawa, basa-basi tentang
    PANDU, dan upaya manipulasi. Selalu berjalan, Laya hidup atau mati -- pesan
    yang jelas bukan pertanyaan tidak perlu dibayar satu panggilan Laya."""
    verdict = rule_gate_module.detect(state["clean"])
    if verdict is None:
        return {}
    return {"gate": verdict, "outcome": _hasil_blokir(state, verdict)}


async def laya_gate(state: PipelineState, runtime: Rt) -> dict:
    """Gerbang semantik Laya (`app.rag.gate`). Dilewati bila LAYA_ENABLED=false."""
    gate_call = runtime.context.gate_call
    if gate_call is None:
        return {"gate": None}

    riwayat = [(t.role, t.konten) for t in state["history"][-HISTORY_WINDOW:]]
    verdict = await _vonis_sebelum_tenggat(
        gate_call(state["clean"], riwayat, unit=state.get("unit")),
        runtime.context.search_done,
        grace=getattr(gate_call, "grace_seconds", DEFAULT_LAYA_GRACE_SECONDS),
        tahan=getattr(gate_call, "hold_seconds", 0.0),
        tertunda=runtime.context.gerbang_tertunda,
    )
    if not verdict.blocked:
        return {"gate": verdict}
    runtime.context.gate_blocked.set()
    return {"gate": verdict, "outcome": _hasil_blokir(state, verdict)}


def _hasil_blokir(state: PipelineState, verdict: GateVerdict) -> PipelineOutcome:
    """Balasan untuk pesan yang dihentikan gerbang, Laya maupun aturan."""
    kind = (
        OutcomeKind.SMALLTALK if verdict.label is GateLabel.SMALLTALK else OutcomeKind.REJECTED
    )
    text = balasan_gerbang(verdict.label, state.get("unit"))
    if kind is OutcomeKind.SMALLTALK:
        # Gerbang hanya tahu "basa-basi"; nadanya dibaca dari pesannya sendiri,
        # supaya "sip, itu saja dulu" tidak dibalas sapaan pembuka.
        text = smalltalk_module.reply_for(state["clean"]) or text
    return PipelineOutcome(
        kind=kind,
        text=text,
        sensitivity=state["sensitivity"],
        gate=verdict,
        llm_called=False,
    )


async def _vonis_sebelum_tenggat(
    panggilan: Awaitable[GateVerdict],
    pencarian_selesai: asyncio.Event,
    *,
    grace: float,
    tahan: float = 0.0,
    tertunda: VonisTertunda | None = None,
) -> GateVerdict:
    """Tunggu vonis Laya selama pencarian paralel masih berjalan, plus `grace`.

    `validate_context` menunggu kedua cabang, jadi selama pencarian belum
    selesai menunggu Laya tidak menambah waktu apa pun. Batas tetap (dulu 3 dtk)
    justru memutus gerbang di tengah pencarian 5-10 dtk -- perlindungan hilang
    tanpa ada waktu yang dihemat. Lewat tenggat, pesan diteruskan (fail-open)
    seperti galat Laya lainnya. Dengan `tahan` > 0 panggilannya dibiarkan
    berjalan di `tertunda` (T79): vonis sementaranya lolos, vonis akhirnya
    ditunggu `generate`/`refuse`. Tanpa itu panggilannya dibatalkan.
    """
    tugas = asyncio.ensure_future(panggilan)
    tunggu_cari = asyncio.ensure_future(pencarian_selesai.wait())
    try:
        selesai, _ = await asyncio.wait(
            {tugas, tunggu_cari}, return_when=asyncio.FIRST_COMPLETED
        )
        if tugas not in selesai:
            selesai, _ = await asyncio.wait({tugas}, timeout=grace)
        if tugas in selesai:
            return tugas.result()
        if tahan > 0 and tertunda is not None:
            tertunda.tugas = tugas
            tertunda.tahan = tahan
            logger.info(
                "Gerbang Laya melewati tenggat (pencarian selesai + %.1f dtk), "
                "LLM dimulai dan token pertamanya ditahan sampai vonis tiba",
                grace,
            )
            return lolos(
                f"Tenggat: pencarian selesai lebih dulu (+{grace:g} dtk), "
                "vonis ditunggu sambil LLM berjalan"
            )
        tugas.cancel()
        logger.warning(
            "Gerbang Laya melewati tenggat (pencarian selesai + %.1f dtk), pesan diteruskan",
            grace,
        )
        return lolos(f"Tenggat: pencarian selesai lebih dulu (+{grace:g} dtk)")
    finally:
        tunggu_cari.cancel()


async def _kecuali_diblokir(
    kerja: Callable[[], Awaitable[Any]], diblokir: asyncio.Event
) -> tuple[bool, Any]:
    """Jalankan `kerja`, tetapi hentikan begitu gerbang Laya memblokir pesan.

    Kembali `(True, hasil)` bila selesai, `(False, None)` bila dihentikan.
    Galat `kerja` tetap diteruskan selama pesan tidak diblokir.
    """
    if diblokir.is_set():
        return False, None
    tugas = asyncio.ensure_future(kerja())
    tunggu_blokir = asyncio.ensure_future(diblokir.wait())
    try:
        selesai, _ = await asyncio.wait(
            {tugas, tunggu_blokir}, return_when=asyncio.FIRST_COMPLETED
        )
        if tugas in selesai:
            return True, tugas.result()
        tugas.cancel()
        return False, None
    finally:
        tunggu_blokir.cancel()


def _tahan_token(
    on_token: Callable[[str], Awaitable[None]] | None, tertunda: VonisTertunda
) -> Callable[[str], Awaitable[None]] | None:
    """`on_token` yang baru meneruskan potongan setelah vonis Laya tiba (T79).

    Potongan pertama membuka jendela `tahan` (`VonisTertunda.akhir`); bila Laya
    memblokir, potongan dibuang dan panggilan LLM-nya dibatalkan
    `_kecuali_vonis_memblokir`.
    """
    if on_token is None:
        return None

    async def tahan(teks: str) -> None:
        # shield: batalnya panggilan LLM tidak boleh ikut membatalkan vonis
        # yang juga ditunggu `_kecuali_vonis_memblokir`.
        verdict = await asyncio.shield(tertunda.akhir())
        if not verdict.blocked:
            await on_token(teks)

    return tahan


async def _kecuali_vonis_memblokir(
    kerja: Callable[[], Awaitable[Any]], tertunda: VonisTertunda
) -> tuple[GateVerdict, Any]:
    """Jalankan panggilan LLM sambil menunggu vonis Laya yang tertunda (T79).

    Kembali `(vonis, hasil)`. Bila vonisnya memblokir, `hasil` None dan
    panggilannya dibatalkan -- saat ramai biasanya masih mengantre di vLLM,
    jadi pesan yang diblokir tidak sempat memakai GPU. Galat LLM diteruskan
    seketika, tanpa menunggu vonis: menahannya hanya memperlama galat yang
    sama (panggilan Laya-nya dibatalkan `run_graph`).
    """
    assert tertunda.tugas is not None
    tugas = asyncio.ensure_future(kerja())
    try:
        # Selama LLM belum selesai, vonis Laya ditunggu tanpa batas sendiri:
        # token pertamanya sudah ditahan `_tahan_token`, yang membuka jendela
        # `tahan` begitu potongan pertama siap.
        await asyncio.wait({tugas, tertunda.tugas}, return_when=asyncio.FIRST_COMPLETED)
        galat = tugas.exception() if tugas.done() and not tugas.cancelled() else None
        if tertunda.tiba and tertunda.vonis().blocked:
            return tertunda.vonis(), None
        if galat is not None:
            raise galat
        if not tugas.done():
            # Laya meloloskan pesan lebih dulu: biarkan LLM selesai.
            return tertunda.vonis(), await tugas
        # LLM selesai lebih dulu (tanpa streaming, atau tanpa potongan): jawaban
        # utuhnya siap, jadi vonisnya ditunggu paling lama `tahan` lagi.
        verdict = await asyncio.shield(tertunda.akhir())
        return verdict, None if verdict.blocked else tugas.result()
    finally:
        if not tugas.done():
            tugas.cancel()


async def rewrite(state: PipelineState, runtime: Rt) -> dict:
    """FR-4 -- dilewati bila pesan pertama berbahasa Indonesia, atau Laya sudah memblokir."""
    clean = state["clean"]
    rewrite_call = runtime.context.rewrite_call
    if rewrite_call is None or not needs_rewrite(state["history"], clean):
        return {"search_query": clean, "rewritten": None}
    jalan, hasil = await _kecuali_diblokir(
        lambda: rewrite_call(clean, format_history(state["history"])),
        runtime.context.gate_blocked,
    )
    rewritten = hasil.strip() if jalan else ""
    return {"search_query": rewritten or clean, "rewritten": rewritten or None}


async def retrieve(state: PipelineState, runtime: Rt) -> dict:
    """FR-2: vector + fulltext paralel lalu RRF -- semuanya di dalam retriever.

    Bila pertanyaan ditulis ulang, pertanyaan aslinya ikut dikirim dan dicari
    juga (T40, lihat `PostgresHybridRetriever`). Hanya saat itu argumennya
    dikirim, jadi retriever tanpa dukungan multi-query tetap berjalan untuk
    pesan pertama.

    Dihentikan bila Laya memblokir: hasilnya toh dibuang di `validate_context`."""
    opsi: dict[str, Any] = {"unit": state.get("unit")}
    if state.get("rewritten"):
        opsi["original_query"] = state["clean"]
    try:
        jalan, documents = await _kecuali_diblokir(
            lambda: runtime.context.retriever.ainvoke(state["search_query"], **opsi),
            runtime.context.gate_blocked,
        )
    finally:
        # Juga saat gagal: gerbang Laya tidak perlu menunggu pencarian yang sudah berhenti.
        runtime.context.search_done.set()
    return {"documents": list(documents) if jalan else []}


async def validate_context(state: PipelineState, runtime: Rt) -> dict:
    """FR-3 -- memutuskan apakah LLM boleh dipanggil.

    Juga titik temu gerbang Laya dan pencarian yang berjalan paralel. Bila Laya
    sudah mengisi `outcome`, hasil pencarian tidak dinilai sama sekali.
    """
    if "outcome" in state:
        return {}
    deps = runtime.context
    hits = _hits_from_documents(state["documents"])
    hasil: dict[str, Any] = {"decision": evaluate(hits, deps.policy)}
    # Gerbang kelayakan tool (docs/tool-call.md §8): pertanyaan yang cocok pemicu
    # tool boleh maju ke `generate` walau konteks retrieval lemah, tanpa itu
    # "siapa dosen Web Programming" ditolak FR-3 sebelum tool sempat dipanggil.
    if deps.tool_registry is not None and _tool_eligible(deps.tool_registry, state):
        hasil["tool_eligible"] = True
    return hasil


def _tool_eligible(registry: Any, state: PipelineState) -> bool:
    """Pemicu dicocokkan pada pertanyaan asli DAN hasil rewrite (FR-4).

    Pertanyaan lanjutan ("kalau Basis Data?") tidak memuat kata pemicu, tetapi
    versi mandiri hasil rewrite-nya memuat ("Siapa dosen pengampu mata kuliah
    Basis Data?"). Rewrite hanya berjalan bila ada riwayat atau pertanyaannya
    terdeteksi berbahasa Inggris (`needs_rewrite`). Pertanyaan Inggris yang
    pendek ("who teaches X?") tidak terdeteksi, jadi ditangani pemicu Inggris
    di `ToolSpec.triggers`, bukan di sini."""
    return any(t and registry.eligible(t) for t in (state["clean"], state.get("rewritten")))


def _pesan_tool(state: PipelineState) -> str:
    """Pertanyaan untuk loop tool, ditambah versi mandirinya bila ada.

    Loop tool tidak menerima riwayat percakapan, dan berbeda dari jalur RAG ia
    tidak punya konteks hasil pencarian yang membawa maksud pertanyaan lanjutan.
    Tanpa versi mandiri hasil rewrite, model membaca "kalau Basis Data?" tanpa
    tahu bahwa yang ditanyakan adalah dosen pengampunya. Keduanya tetap di dalam
    tag `<pertanyaan_mahasiswa>` (FR-5): hasil rewrite juga berasal dari teks
    mahasiswa, jadi diperlakukan sebagai data, bukan instruksi."""
    teks = state["clean"]
    rewritten = (state.get("rewritten") or "").strip()
    if rewritten and rewritten.casefold() != teks.strip().casefold():
        teks = (
            f"{teks}\n\nMaksud lengkap pertanyaan di atas, ditulis ulang dari "
            f"riwayat percakapan: {rewritten}"
        )
    return pesan_mahasiswa(wrap_user_input(teks), state.get("unit"), state.get("profile"))


async def _vonis_akhir(state: PipelineState, tertunda: VonisTertunda) -> PipelineState:
    """Salinan `state` dengan vonis Laya akhir, bila tadinya tertunda (T79)."""
    if not tertunda.ada:
        return state
    baru = state.copy()
    baru["gate"] = await asyncio.shield(tertunda.akhir())
    return baru


async def refuse(state: PipelineState, runtime: Rt) -> dict:
    """Penolakan FR-3. Tidak memanggil LLM.

    Vonis Laya yang tertunda tetap ditunggu: pesan di luar topik yang ternyata
    diblokir tidak boleh tercatat sebagai celah dokumen di AD-4."""
    tertunda = runtime.context.gerbang_tertunda
    state = await _vonis_akhir(state, tertunda)
    verdict = state.get("gate")
    hasil: dict[str, Any] = {"gate": verdict} if tertunda.ada else {}
    if verdict is not None and verdict.blocked:
        return {**hasil, "outcome": _hasil_blokir(state, verdict)}
    assessment = risk_module.detect(state["clean"])
    hasil["outcome"] = _penolakan(state, runtime.context, assessment, llm_called=False)
    return hasil


def _kontak(
    state: PipelineState, deps: PipelineDeps, assessment: risk_module.RiskAssessment
) -> tuple[Any, ...]:
    """Kontak PIC unit yang dipilih mahasiswa yang relevan dengan pertanyaannya;
    tanpa itu, kontak per topik risiko (`risk.pilih_kontak`)."""
    teks = state["clean"]
    rewritten = (state.get("rewritten") or "").strip()
    if rewritten:
        teks = f"{teks}\n{rewritten}"
    return risk_module.pilih_kontak(deps.unit_contacts, assessment, teks)


def _penolakan(
    state: PipelineState,
    deps: PipelineDeps,
    assessment: risk_module.RiskAssessment,
    *,
    llm_called: bool,
    alasan: Sequence[str] = (),
) -> PipelineOutcome:
    """Satu bentuk penolakan, entah diputuskan threshold atau oleh LLM.

    `alasan`: sebab kegagalan tool untuk mahasiswa (T77, `refusal_text`)."""
    # Tanpa unit yang dipilih, semua unit terkait topik ditampilkan, bukan hanya
    # yang pertama: pertanyaan "deadline pembayaran UKT" menyangkut akademik DAN
    # keuangan sekaligus, dan mahasiswa yang ditolak tidak boleh dikirim ke
    # loket yang salah.
    contacts = _kontak(state, deps, assessment) or (DEFAULT_FALLBACK_CONTACT,)
    text = refusal_text(contacts, state.get("unit"), alasan)
    return PipelineOutcome(
        kind=OutcomeKind.REFUSAL,
        text=text,
        documents=tuple(state["documents"]),
        decision=state["decision"],
        risk=assessment,
        sensitivity=state["sensitivity"],
        gate=state.get("gate"),
        rewritten_query=state.get("rewritten"),
        llm_called=llm_called,
        contacts=contacts,
    )


async def generate(state: PipelineState, runtime: Rt) -> dict:
    """FR-6 (eskalasi) lalu FR-5 (jawaban bersumber, pertanyaan terbungkus).

    Pertanyaan tool-eligible (TOOLS_ENABLED) disusun lewat loop agentik
    tool-calling (docs/tool-call.md): LLM boleh memanggil tool, dan hasilnya
    menjadi kartu sumber sintetis yang digabung ke `documents`."""
    deps = runtime.context
    assessment = risk_module.detect(state["clean"])
    wrapped = pesan_mahasiswa(
        wrap_user_input(state["clean"]), state.get("unit"), state.get("profile")
    )
    documents = list(state["documents"])
    pakai_tool = (
        deps.tool_registry is not None
        and bool(state.get("tool_eligible"))
        and hasattr(deps.llm_call, "run_tools")
    )
    # Sampai di `generate` dengan vonis REFUSE hanya mungkin karena pertanyaannya
    # tool-eligible (`route_context`). Bila tool ternyata TIDAK dapat dijalankan
    # -- mis. `llm_call` pengganti tanpa `run_tools` -- konteksnya tetap lemah,
    # jadi kembalikan penolakan FR-3 tanpa memanggil LLM. Tanpa penjagaan ini
    # LLM dipanggil dengan konteks kosong hanya untuk membalas
    # [TIDAK_DITEMUKAN]: satu panggilan berbayar, dan penolakannya tercatat
    # `refusal_source = llm` padahal yang menolak adalah ambang.
    if not pakai_tool and state["decision"].decision is Decision.REFUSE:
        return {"outcome": _penolakan(state, deps, assessment, llm_called=False)}
    if deps.on_stage is not None:
        await deps.on_stage("menyusun jawaban")
    tertunda = deps.gerbang_tertunda
    on_token = _tahan_token(deps.on_token, tertunda) if tertunda.ada else deps.on_token
    pesan_tool = _pesan_tool(state) if pakai_tool else ""
    penanya = Penanya(nim=deps.nim, profile=state.get("profile"))

    async def kerja() -> tuple[str, list[Any], list[Any], tuple[str, ...]]:
        if not pakai_tool:
            return await _jawab(deps.llm_call, wrapped, documents, on_token), [], [], ()
        # Kelayakan sudah diputuskan `validate_context`; yang di-bind seluruh
        # registry, karena memilih tool adalah tugas model (ToolRegistry.eligible).
        jawaban, tool_docs = await deps.llm_call.run_tools(
            pesan_tool,
            documents,
            specs=deps.tool_registry.specs,
            on_token=on_token,
            on_stage=deps.on_stage,
            max_rounds=deps.tool_max_rounds,
            penanya=penanya,
        )
        return (
            jawaban,
            list(tool_docs),
            list(getattr(deps.llm_call, "attachments", None) or []),
            tuple(getattr(deps.llm_call, "alasan_gagal", None) or ()),
        )

    # Vonis Laya yang tertunda (T79): LLM sudah berjalan, token pertamanya ditahan
    # `_tahan_token`, dan panggilannya dibatalkan bila Laya ternyata memblokir.
    tambahan: dict[str, Any] = {}
    if tertunda.ada:
        verdict, hasil = await _kecuali_vonis_memblokir(kerja, tertunda)
        state = state.copy()
        state["gate"] = verdict
        tambahan["gate"] = verdict
        if verdict.blocked:
            return {**tambahan, "outcome": _hasil_blokir(state, verdict)}
    else:
        hasil = await kerja()
    answer, tool_docs, lampiran, alasan_gagal = hasil
    documents = [*documents, *tool_docs]
    # Sumber tak berhalaman (entri tanya jawab, kartu tool) dikutip `[Judul]`.
    # Tanpa daftar ini jawaban parsial bersumber tool yang menyebut "tidak
    # ditemukan" untuk sebagian pertanyaan dianggap penolakan dan dibuang.
    tanpa_hal = sumber_tanpa_halaman(documents)
    # Pertanyaan di luar urusan kampus yang lolos gerbang (atau Laya mati):
    # dibalas dan dicatat seperti blokir Laya `out_of_scope` -- bukan celah
    # dokumen, jadi tidak masuk AD-4.
    if is_off_topic(answer, tanpa_hal):
        return {
            **tambahan,
            "outcome": PipelineOutcome(
                kind=OutcomeKind.REJECTED,
                text=balasan_gerbang(GateLabel.OUT_OF_SCOPE, state.get("unit")),
                documents=tuple(documents),
                decision=state["decision"],
                sensitivity=state["sensitivity"],
                gate=state.get("gate"),
                rewritten_query=state.get("rewritten"),
                llm_called=True,
            ),
        }
    # Tool gagal karena sebab yang diketahui (mis. NIM tidak terdaftar) dan
    # jawabannya tanpa sumber: tolak dengan sebab itu (T77). Apa pun yang ditulis
    # model -- penanda saja, penjelasan plus penanda, atau penjelasan berkurung
    # siku -- mahasiswa membaca alasan dan kontak yang sama. Jawaban yang
    # mengutip sumber (bagian yang dijawab dokumen) tetap ditampilkan.
    if alasan_gagal and not extract_citations(answer, tanpa_hal):
        return {
            **tambahan,
            "outcome": _penolakan(
                state, deps, assessment, llm_called=True, alasan=alasan_gagal
            ),
        }
    # Konteks lolos threshold tetapi tidak menjawab: tampilkan dan catat sebagai
    # penolakan, bukan sebagai jawaban berisi "tidak menemukan" yang membawa
    # kartu sitasi dokumen yang tidak relevan dan tidak pernah sampai ke AD-4.
    if is_not_found(answer, tanpa_hal):
        return {**tambahan, "outcome": _penolakan(state, deps, assessment, llm_called=True)}
    tanya_jawab = [
        doc.metadata.get("judul", "")
        for doc in documents
        if doc.metadata.get("jenis") == DocumentType.TANYA_JAWAB
    ]
    text = ringkas_sitasi_tanpa_halaman(strip_markers(answer), tanya_jawab)
    # Lampiran ikut aturan kartu sumber (`citations_for`): hanya bila jawabannya
    # mengutip sumbernya. Model yang memanggil tool lalu tidak memakai hasilnya
    # tidak boleh menempelkan daftar 220 dosen di bawah jawaban lain.
    dikutip = {c.judul.casefold() for c in extract_citations(text, tanpa_hal)}
    return {
        **tambahan,
        "outcome": PipelineOutcome(
            kind=OutcomeKind.ANSWER,
            text=text,
            documents=tuple(documents),
            decision=state["decision"],
            risk=assessment,
            sensitivity=state["sensitivity"],
            gate=state.get("gate"),
            rewritten_query=state.get("rewritten"),
            llm_called=True,
            contacts=_kontak(state, deps, assessment) if assessment.needs_escalation else (),
            attachments=tuple(a for a in lampiran if a.source.casefold() in dikutip),
        ),
    }


# --- Sisi bersyarat -------------------------------------------------------


def _selesai_atau(*berikutnya: str) -> Callable[[PipelineState], str | list[str]]:
    """Ke END bila `outcome` sudah ada; selain itu ke semua node `berikutnya`
    sekaligus -- lebih dari satu berarti berjalan paralel."""

    def route(state: PipelineState) -> str | list[str]:
        if "outcome" in state:
            return "selesai"
        return berikutnya[0] if len(berikutnya) == 1 else list(berikutnya)

    route.__name__ = "selesai_atau_" + "_".join(berikutnya)
    return route


def route_context(state: PipelineState) -> str:
    if "outcome" in state:
        return "selesai"
    # Konteks lemah tetap ke `generate` bila pertanyaannya tool-eligible: tool
    # yang akan menyediakan datanya, bukan retrieval (docs/tool-call.md §8).
    if state["decision"].decision is Decision.REFUSE and not state.get("tool_eligible"):
        return "refuse"
    return "generate"


SEARCH_NODE = "cari"
"""Subgraph `rewrite -> retrieve` yang berjalan sejajar dengan `laya_gate`.

Hanya pembungkus: perekam durasi melewatinya (`applog.WRAPPER_NODES`) dan
mencatat `rewrite` serta `retrieve` di dalamnya sebagai langkah biasa."""


def _search_graph() -> Any:
    g = StateGraph(PipelineState, context_schema=PipelineDeps)
    g.add_node("rewrite", rewrite)
    g.add_node("retrieve", retrieve)
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", END)
    return g.compile(name="pandu_cari")


@lru_cache(maxsize=1)
def build_graph() -> Any:
    """Kompilasi graf sekali per proses; ketergantungan masuk lewat `context`."""
    g = StateGraph(PipelineState, context_schema=PipelineDeps)
    for node in (
        sanitize,
        sensitive,
        smalltalk,
        rule_gate,
        laya_gate,
        validate_context,
        refuse,
        generate,
    ):
        g.add_node(node.__name__, node)
    g.add_node(SEARCH_NODE, _search_graph())

    g.add_edge(START, "sanitize")
    g.add_edge("sanitize", "sensitive")
    g.add_conditional_edges(
        "sensitive", _selesai_atau("smalltalk"), {"selesai": END, "smalltalk": "smalltalk"}
    )
    g.add_conditional_edges(
        "smalltalk", _selesai_atau("rule_gate"), {"selesai": END, "rule_gate": "rule_gate"}
    )
    g.add_conditional_edges(
        "rule_gate",
        _selesai_atau("laya_gate", SEARCH_NODE),
        {"selesai": END, "laya_gate": "laya_gate", SEARCH_NODE: SEARCH_NODE},
    )
    # Titik temu: validate_context baru berjalan setelah KEDUA cabang selesai.
    g.add_edge(["laya_gate", SEARCH_NODE], "validate_context")
    g.add_conditional_edges(
        "validate_context",
        route_context,
        {"selesai": END, "refuse": "refuse", "generate": "generate"},
    )
    g.add_edge("refuse", END)
    g.add_edge("generate", END)
    return g.compile(name="pandu_chat")


async def run_graph(
    question: str,
    deps: PipelineDeps,
    *,
    history: list[Turn],
    unit: str | None,
    profile: ProfilMahasiswa | None = None,
    callbacks: Sequence[Any] = (),
) -> PipelineOutcome:
    """`callbacks` menerima event per node, mis. `applog.NodeRecorder`."""
    config = {"callbacks": list(callbacks)} if callbacks else None
    try:
        state = await build_graph().ainvoke(
            {"question": question, "history": history, "unit": unit, "profile": profile},
            config=config,
            context=deps,
        )
    finally:
        # Galat di tengah jalan, atau mahasiswa menutup panel: vonis Laya yang
        # tertunda (T79) tidak lagi ditunggu siapa pun.
        deps.gerbang_tertunda.batalkan()
    return state["outcome"]


if __name__ == "__main__":  # pragma: no cover
    print(build_graph().get_graph().draw_mermaid())
