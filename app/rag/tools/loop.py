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

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.rag.chain import _PenahanPenanda
from app.rag.prompts import NOT_FOUND_MARKER, TOOL_SYSTEM_PROMPT, format_context
from app.rag.providers import PenyaringGalatGateway
from app.rag.tools.base import (
    MAKS_PANJANG_ARGUMEN,
    Lampiran,
    ToolArgumentError,
    ToolResult,
    ToolSpec,
    hasil_tool_ke_dokumen,
    validasi_argumen,
)

logger = logging.getLogger(__name__)

OnToken = Callable[[str], Awaitable[None]]
OnStage = Callable[[str], Awaitable[None]]

MAKS_TOOL_PER_GILIRAN = 4
"""Panggilan tool yang dijalankan dari satu giliran model (docs/tool-call.md §11).

Model boleh meminta beberapa tool sekaligus. Tanpa batas, satu pertanyaan bisa
memicu puluhan permintaan ke SADS. Panggilan di atas batas tetap dibalas
(`PESAN_BATAS`), tetapi tidak dijalankan."""

PESAN_BATAS = "DATA_TIDAK_TERSEDIA: batas panggilan alat per giliran tercapai."
PESAN_ARGUMEN_RUSAK = "DATA_TIDAK_TERSEDIA: argumen panggilan alat tidak dapat dibaca."


@dataclass(frozen=True)
class ToolLoopResult:
    text: str
    documents: list[Document]
    usage: dict[str, Any] | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """Satu entri per panggilan tool: nama, argumen (mentah dari model), ok,
    latency_ms. Untuk `messages.meta` (observability, docs/tool-call.md §13)."""
    attachments: list[Lampiran] = field(default_factory=list)
    """Lampiran dari tool yang berhasil, urut panggilan (docs/tool-call.md §10a).
    Belum disaring: `generate` hanya meneruskan yang sumbernya dikutip jawaban."""


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


def _args_untuk_log(args: Any) -> dict[str, Any]:
    """Argumen mentah model, dipotong untuk `messages.meta`.

    Mentah (bukan hasil validasi) supaya argumen yang dihalusinasi tetap terlihat
    di log, tetapi dibatasi panjangnya: tanpa ini satu string raksasa dari model
    ikut tersimpan utuh di kolom `meta` setiap giliran."""
    return {
        k: (v[:MAKS_PANJANG_ARGUMEN] if isinstance(v, str) else v)
        for k, v in (args or {}).items()
    }


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


async def _jalankan_terukur(tc: dict, by_name: dict[str, ToolSpec]) -> tuple[ToolResult, int]:
    """`_jalankan_tool` beserta latensinya (ms) untuk `meta.tool_calls`."""
    mulai = time.perf_counter()
    r = await _jalankan_tool(tc, by_name)
    return r, round((time.perf_counter() - mulai) * 1000)


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
    buat_config: Callable[[], Any] | None = None,
) -> ToolLoopResult:
    """Jalankan loop. `llm_tools` sudah di-bind dengan skema tool; `llm_plain`
    tanpa tool, dipakai memaksa jawaban bila batas putaran tercapai.

    `buat_config` dipanggil sekali per giliran LLM. Setiap giliran butuh `run_id`
    sendiri: LangSmith menolak run kedua dengan ID yang sama, sehingga satu
    config yang dipakai ulang membuat giliran sesudah tool -- yang berisi
    jawabannya -- hilang dari trace."""
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
            llm_tools, messages, on_token=on_token, config=_config(buat_config)
        )
        usage = _tambah_usage(usage, getattr(gathered, "usage_metadata", None))
        panggilan = list(getattr(gathered, "tool_calls", None) or [])
        # Argumen yang bukan JSON sah tidak masuk `tool_calls`, melainkan
        # `invalid_tool_calls`. Tanpa ini giliran itu dikira jawaban final
        # yang kosong.
        rusak = list(getattr(gathered, "invalid_tool_calls", None) or [])
        if not panggilan and not rusak:
            return _hasil_akhir(teks, hasil, usage, jejak)
        if on_stage is not None:
            await on_stage("mengambil data akademik")
        messages.append(gathered)

        dijalankan = panggilan[:MAKS_TOOL_PER_GILIRAN]
        # Bersamaan, seperti FTS ∥ pgvector: setiap handler memakai klien HTTP
        # sendiri, dan `gather` mengembalikan hasil menurut urutan panggilan.
        keluaran = await asyncio.gather(
            *(_jalankan_terukur(tc, by_name) for tc in dijalankan)
        )
        for tc, (r, latency_ms) in zip(dijalankan, keluaran, strict=True):
            jejak.append(
                {
                    "name": tc.get("name"),
                    "args": _args_untuk_log(tc.get("args")),
                    "ok": r.ok,
                    "latency_ms": latency_ms,
                }
            )
            hasil.append(r)
            messages.append(
                ToolMessage(content=r.pesan_untuk_model(), tool_call_id=tc.get("id") or "")
            )
        # Setiap `tool_call_id` wajib dibalas satu pesan `role:"tool"`; tanpa itu
        # API menolak giliran berikutnya. Yang tidak dijalankan dibalas penanda.
        for tc in panggilan[MAKS_TOOL_PER_GILIRAN:]:
            args = _args_untuk_log(tc.get("args"))
            jejak.append(_jejak_lewat(tc, args, "batas_per_giliran"))
            messages.append(ToolMessage(content=PESAN_BATAS, tool_call_id=tc.get("id") or ""))
        for tc in rusak:
            mentah = {"_mentah": str(tc.get("args") or "")[:MAKS_PANJANG_ARGUMEN]}
            jejak.append(_jejak_lewat(tc, mentah, "argumen_rusak"))
            messages.append(
                ToolMessage(content=PESAN_ARGUMEN_RUSAK, tool_call_id=tc.get("id") or "")
            )
        if on_stage is not None:
            await on_stage("menyusun jawaban")

    # Batas putaran tercapai dan model masih meminta tool: paksa jawaban akhir
    # tanpa tool, supaya loop selalu berujung pada teks.
    gathered, teks = await _satu_giliran(
        llm_plain, messages, on_token=on_token, config=_config(buat_config)
    )
    usage = _tambah_usage(usage, getattr(gathered, "usage_metadata", None))
    return _hasil_akhir(teks, hasil, usage, jejak)


def _config(buat_config: Callable[[], Any] | None) -> Any:
    return buat_config() if buat_config is not None else None


def _hasil_akhir(
    teks: str, hasil: list[ToolResult], usage: dict | None, jejak: list[dict[str, Any]]
) -> ToolLoopResult:
    """Bungkus jawaban final.

    Giliran final tanpa teks sama sekali tidak boleh tampil sebagai `answer`
    kosong. Ia diganti `NOT_FOUND_MARKER`, sehingga `generate` membalas
    penolakan resmi beserta kontak unit."""
    if not teks.strip():
        teks = NOT_FOUND_MARKER
    lampiran = [r.attachment for r in hasil if r.ok and r.attachment is not None]
    return ToolLoopResult(teks, hasil_tool_ke_dokumen(hasil), usage, jejak, lampiran)


def _jejak_lewat(tc: dict, args: dict[str, Any], alasan: str) -> dict[str, Any]:
    """Entri `meta.tool_calls` untuk panggilan yang sengaja tidak dijalankan."""
    return {
        "name": tc.get("name"),
        "args": args,
        "ok": False,
        "latency_ms": 0,
        "error": alasan,
    }
