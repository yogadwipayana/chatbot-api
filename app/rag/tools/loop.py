"""Loop agentik tool-calling (docs/tool-call.md §5, §9).

Mengganti satu panggilan LLM `generate` dengan loop: LLM memutuskan memanggil
tool, hasil dikembalikan sebagai pesan `role:"tool"`, lalu LLM dipanggil lagi
sampai menghasilkan jawaban (atau batas putaran tercapai).

Streaming: giliran tool tidak membawa konten (content=null di gateway proyek
ini), jadi tidak ada token yang bocor ke mahasiswa untuknya; hanya giliran
jawaban final yang mengalir. Penyaring `[Error]` gateway dan penahan penanda
`[TIDAK_DITEMUKAN]`/`[DI_LUAR_TOPIK]` dipakai ulang dari jalur jawaban biasa.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.rag.chain import _PenahanPenanda
from app.rag.prompts import TOOL_SYSTEM_PROMPT, format_context
from app.rag.providers import PenyaringGalatGateway
from app.rag.tools.base import (
    ToolArgumentError,
    ToolResult,
    ToolSpec,
    hasil_tool_ke_dokumen,
    validasi_argumen,
)

logger = logging.getLogger(__name__)

OnToken = Callable[[str], Awaitable[None]]
OnStage = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class ToolLoopResult:
    text: str
    documents: list[Document]
    usage: dict[str, Any] | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """Satu entri per panggilan tool: nama, argumen (mentah dari model), ok,
    latency_ms. Untuk `messages.meta` (observability, docs/tool-call.md §13)."""


def _chunk_text(chunk: Any) -> str:
    isi = getattr(chunk, "content", "") or ""
    if isinstance(isi, str):
        return isi
    if isinstance(isi, list):  # beberapa provider mengirim konten sebagai blok
        return "".join(p.get("text", "") for p in isi if isinstance(p, dict))
    return str(isi)


def _tambah_usage(akun: dict | None, baru: Any) -> dict | None:
    """Jumlahkan usage antar giliran LLM. Akunting gateway tak presisi (docs §2)."""
    if not baru:
        return akun
    if akun is None:
        return dict(baru)
    gabung = dict(akun)
    for k, v in baru.items():
        if isinstance(v, (int, float)) and isinstance(gabung.get(k), (int, float)):
            gabung[k] = gabung[k] + v
        else:
            gabung.setdefault(k, v)
    return gabung


async def _satu_giliran(
    llm: Any, messages: list[Any], *, on_token: OnToken | None, config: Any
) -> tuple[Any, str]:
    """Stream satu giliran LLM. Kembalikan (pesan_terkumpul, teks_tersaring).

    Konten dipancarkan ke `on_token` begitu tiba (lewat penahan penanda +
    penyaring galat). Giliran tool tidak berkonten, jadi tidak memancarkan apa
    pun. `teks` adalah jawaban utuh untuk diproses pemanggil (penanda, sitasi)."""
    gathered: Any = None
    penahan = _PenahanPenanda(on_token) if on_token is not None else None
    penyaring = PenyaringGalatGateway()
    bagian: list[str] = []
    async for chunk in llm.astream(messages, config=config):
        gathered = chunk if gathered is None else gathered + chunk
        delta = _chunk_text(chunk)
        if delta:
            aman = penyaring.terima(delta)
            if aman:
                bagian.append(aman)
                if penahan is not None:
                    await penahan(aman)
    sisa = penyaring.sisa()
    if sisa:
        bagian.append(sisa)
        if penahan is not None:
            await penahan(sisa)
    return gathered, "".join(bagian)


async def _jalankan_tool(tc: dict, by_name: dict[str, ToolSpec]) -> ToolResult:
    spec = by_name.get(tc.get("name", ""))
    if spec is None:
        return ToolResult(name=str(tc.get("name")), label="", text="", ok=False)
    try:
        args = validasi_argumen(spec, tc.get("args") or {})
        return await spec.handler(**args)
    except ToolArgumentError as e:
        logger.warning("argumen tool %s tidak valid: %s", spec.name, e)
    except httpx.HTTPError as e:
        logger.warning("tool %s gagal memanggil layanan: %s", spec.name, e)
    except Exception:  # pragma: no cover - jaring pengaman, jangan jatuhkan giliran
        logger.exception("tool %s galat tak terduga", spec.name)
    return ToolResult(name=spec.name, label=spec.citation_label, text="", ok=False)


async def run_tool_loop(
    *,
    llm_tools: Any,
    llm_plain: Any,
    wrapped_question: str,
    documents: Sequence[Any],
    specs: Sequence[ToolSpec],
    on_token: OnToken | None = None,
    on_stage: OnStage | None = None,
    max_rounds: int = 2,
    config: Any = None,
) -> ToolLoopResult:
    """Jalankan loop. `llm_tools` sudah di-bind dengan skema tool; `llm_plain`
    tanpa tool, dipakai memaksa jawaban bila batas putaran tercapai."""
    by_name = {s.name: s for s in specs}
    messages: list[Any] = [
        SystemMessage(content=TOOL_SYSTEM_PROMPT.format(context=format_context(documents))),
        HumanMessage(content=wrapped_question),
    ]
    hasil: list[ToolResult] = []
    jejak: list[dict[str, Any]] = []
    usage: dict | None = None

    for _ in range(max_rounds):
        gathered, teks = await _satu_giliran(
            llm_tools, messages, on_token=on_token, config=config
        )
        usage = _tambah_usage(usage, getattr(gathered, "usage_metadata", None))
        panggilan = list(getattr(gathered, "tool_calls", None) or [])
        if not panggilan:
            return ToolLoopResult(teks, hasil_tool_ke_dokumen(hasil), usage, jejak)
        if on_stage is not None:
            await on_stage("mengambil data akademik")
        messages.append(gathered)
        for tc in panggilan:
            mulai = time.perf_counter()
            r = await _jalankan_tool(tc, by_name)
            jejak.append(
                {
                    "name": tc.get("name"),
                    "args": tc.get("args") or {},
                    "ok": r.ok,
                    "latency_ms": round((time.perf_counter() - mulai) * 1000),
                }
            )
            hasil.append(r)
            messages.append(
                ToolMessage(content=r.pesan_untuk_model(), tool_call_id=tc.get("id", ""))
            )

    # Batas putaran tercapai dan model masih meminta tool: paksa jawaban akhir
    # tanpa tool, supaya loop selalu berujung pada teks.
    gathered, teks = await _satu_giliran(
        llm_plain, messages, on_token=on_token, config=config
    )
    usage = _tambah_usage(usage, getattr(gathered, "usage_metadata", None))
    return ToolLoopResult(teks, hasil_tool_ke_dokumen(hasil), usage, jejak)
