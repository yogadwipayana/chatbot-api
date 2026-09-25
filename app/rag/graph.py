"""Alur tanya-jawab sebagai graf LangGraph (flow.md §4).

Setiap langkah di `chain.py` versi lama kini menjadi satu node, dan setiap
jalan keluar lebih awal menjadi satu sisi bersyarat ke END. Isi langkahnya
tidak berubah; yang berubah hanya siapa yang memegang urutan:

    START -> sanitize -> sensitive ─┬─> END                        (FR-7)
                                    └─> smalltalk ─┬─> END
                                                   └─> jev_gate ─┬─> END
                                                                 └─> rewrite (FR-4)
      -> retrieve (hybrid + RRF + rerank, filter unit) -> validate_context (FR-3)
           ├─> refuse   -> END      (LLM tidak dipanggil)
           └─> generate -> END      (FR-6 + FR-5)

Ketergantungan per permintaan (retriever, LLM, callback streaming) masuk
lewat `context` LangGraph, bukan lewat state: state hanya berisi data yang
mengalir antar node, dan graf dikompilasi sekali per proses.

Node yang mengakhiri alur mengisi `outcome`; sisi bersyarat hanya memeriksa
apakah `outcome` sudah ada. Satu aturan itu yang menjamin tidak ada node
sesudahnya -- termasuk LLM -- yang berjalan setelah keputusan berhenti diambil.

    python -m app.rag.graph   # cetak diagram Mermaid untuk dokumentasi
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.rag import risk as risk_module
from app.rag import sensitive as sensitive_module
from app.rag import smalltalk as smalltalk_module
from app.rag.chain import (
    DEFAULT_FALLBACK_CONTACT,
    REFUSAL_TEMPLATE,
    REFUSAL_UNIT_HINT,
    SUPPORT_TEMPLATE,
    OutcomeKind,
    PipelineOutcome,
    _hits_from_documents,
    _jawab,
    render_contacts,
)
from app.rag.gate import GateLabel, GateVerdict
from app.rag.rewriter import HISTORY_WINDOW, Turn, format_history, needs_rewrite
from app.rag.threshold import Decision, ThresholdDecision, ThresholdPolicy, evaluate
from app.security.sanitize import sanitize_question, wrap_user_input


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


class PipelineState(TypedDict, total=False):
    question: str
    history: list[Turn]
    unit: str | None
    clean: str
    sensitivity: sensitive_module.SensitivityAssessment
    gate: GateVerdict | None
    search_query: str
    rewritten: str | None
    documents: list[Any]
    decision: ThresholdDecision
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


async def jev_gate(state: PipelineState, runtime: Rt) -> dict:
    """Gerbang semantik JEV (`app.rag.gate`). Dilewati bila JEV_ENABLED=false."""
    gate_call = runtime.context.gate_call
    if gate_call is None:
        return {"gate": None}

    riwayat = [(t.role, t.konten) for t in state["history"][-HISTORY_WINDOW:]]
    verdict = await gate_call(state["clean"], riwayat)
    if not verdict.blocked:
        return {"gate": verdict}

    kind = (
        OutcomeKind.SMALLTALK if verdict.label is GateLabel.SMALLTALK else OutcomeKind.REJECTED
    )
    return {
        "gate": verdict,
        "outcome": PipelineOutcome(
            kind=kind,
            text=verdict.reply,
            sensitivity=state["sensitivity"],
            gate=verdict,
            llm_called=False,
        ),
    }


async def rewrite(state: PipelineState, runtime: Rt) -> dict:
    """FR-4 -- dilewati bila pesan pertama."""
    clean = state["clean"]
    rewrite_call = runtime.context.rewrite_call
    if rewrite_call is None or not needs_rewrite(state["history"]):
        return {"search_query": clean, "rewritten": None}
    rewritten = (await rewrite_call(clean, format_history(state["history"]))).strip()
    return {"search_query": rewritten or clean, "rewritten": rewritten or None}


async def retrieve(state: PipelineState, runtime: Rt) -> dict:
    """FR-2: vector + fulltext paralel, RRF, rerank -- semuanya di dalam retriever."""
    documents = await runtime.context.retriever.ainvoke(
        state["search_query"], unit=state.get("unit")
    )
    return {"documents": list(documents)}


async def validate_context(state: PipelineState, runtime: Rt) -> dict:
    """FR-3 -- memutuskan apakah LLM boleh dipanggil."""
    hits = _hits_from_documents(state["documents"])
    return {"decision": evaluate(hits, runtime.context.policy)}


async def refuse(state: PipelineState) -> dict:
    """Penolakan FR-3. Tidak memanggil LLM."""
    assessment = risk_module.detect(state["clean"])
    # Semua unit terkait ditampilkan, bukan hanya yang pertama: pertanyaan
    # "deadline pembayaran UKT" menyangkut akademik DAN keuangan sekaligus,
    # dan mahasiswa yang ditolak tidak boleh dikirim ke loket yang salah.
    contacts = assessment.contacts or (DEFAULT_FALLBACK_CONTACT,)
    text = REFUSAL_TEMPLATE.format(contacts=render_contacts(contacts))
    unit = state.get("unit")
    if unit is not None:
        text += REFUSAL_UNIT_HINT.format(unit=unit)
    return {
        "outcome": PipelineOutcome(
            kind=OutcomeKind.REFUSAL,
            text=text,
            documents=tuple(state["documents"]),
            decision=state["decision"],
            risk=assessment,
            sensitivity=state["sensitivity"],
            gate=state.get("gate"),
            rewritten_query=state.get("rewritten"),
            llm_called=False,
            contacts=contacts,
        )
    }


async def generate(state: PipelineState, runtime: Rt) -> dict:
    """FR-6 (eskalasi) lalu FR-5 (jawaban bersumber, pertanyaan terbungkus)."""
    deps = runtime.context
    assessment = risk_module.detect(state["clean"])
    if deps.on_stage is not None:
        await deps.on_stage("menyusun jawaban")
    answer = await _jawab(
        deps.llm_call, wrap_user_input(state["clean"]), state["documents"], deps.on_token
    )
    return {
        "outcome": PipelineOutcome(
            kind=OutcomeKind.ANSWER,
            text=answer,
            documents=tuple(state["documents"]),
            decision=state["decision"],
            risk=assessment,
            sensitivity=state["sensitivity"],
            gate=state.get("gate"),
            rewritten_query=state.get("rewritten"),
            llm_called=True,
            contacts=assessment.contacts,
        )
    }


# --- Sisi bersyarat -------------------------------------------------------


def _selesai_atau(berikutnya: str) -> Callable[[PipelineState], str]:
    def route(state: PipelineState) -> str:
        return "selesai" if "outcome" in state else berikutnya

    route.__name__ = f"selesai_atau_{berikutnya}"
    return route


def route_context(state: PipelineState) -> str:
    return "refuse" if state["decision"].decision is Decision.REFUSE else "generate"


@lru_cache(maxsize=1)
def build_graph() -> Any:
    """Kompilasi graf sekali per proses; ketergantungan masuk lewat `context`."""
    g = StateGraph(PipelineState, context_schema=PipelineDeps)
    for node in (
        sanitize,
        sensitive,
        smalltalk,
        jev_gate,
        rewrite,
        retrieve,
        validate_context,
        refuse,
        generate,
    ):
        g.add_node(node.__name__, node)

    g.add_edge(START, "sanitize")
    g.add_edge("sanitize", "sensitive")
    for asal, lanjut in (
        ("sensitive", "smalltalk"),
        ("smalltalk", "jev_gate"),
        ("jev_gate", "rewrite"),
    ):
        g.add_conditional_edges(asal, _selesai_atau(lanjut), {"selesai": END, lanjut: lanjut})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "validate_context")
    g.add_conditional_edges(
        "validate_context", route_context, {"refuse": "refuse", "generate": "generate"}
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
    callbacks: Sequence[Any] = (),
) -> PipelineOutcome:
    """`callbacks` menerima event per node, mis. `applog.NodeRecorder`."""
    config = {"callbacks": list(callbacks)} if callbacks else None
    state = await build_graph().ainvoke(
        {"question": question, "history": history, "unit": unit},
        config=config,
        context=deps,
    )
    return state["outcome"]


if __name__ == "__main__":  # pragma: no cover
    print(build_graph().get_graph().draw_mermaid())
