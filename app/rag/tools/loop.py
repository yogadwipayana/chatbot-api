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

from app.observability.rekaman import catat_panggilan
from app.rag.chain import (
    _PenahanPenanda,
    _seragam,
    is_not_found,
    strip_markers,
    sumber_tanpa_halaman,
)
from app.rag.prompts import (
    NOT_FOUND_MARKER,
    OFF_TOPIC_MARKER,
    TOOL_SYSTEM_PROMPT,
    format_context,
)
from app.rag.providers import PenyaringGalatGateway
from app.rag.tools.base import (
    MAKS_PANJANG_ARGUMEN,
    Lampiran,
    Penanya,
    ToolArgumentError,
    ToolResult,
    ToolSpec,
    hasil_tool_ke_dokumen,
    pilih_lampiran,
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

DORONGAN_T1 = (
    "Catatan sistem, bukan dari mahasiswa: Anda menyatakan jawabannya tidak "
    "ditemukan atau tidak tercantum tanpa mencoba alat. Periksa lagi aturan T1: "
    "bila salah satu alat dapat mengambil data yang ditanyakan, termasuk untuk "
    "prodi, semester, atau mata kuliah yang tidak dimuat KONTEKS, panggil alat "
    "itu sekarang. Bila tidak ada alat yang relevan, atau aturan T8 melarang "
    "memanggilnya, balas HANYA dengan [TIDAK_DITEMUKAN]."
)
"""Dikirim sekali bila model menolak tanpa mencoba satu alat pun (T1).

T1 sudah melarang `[TIDAK_DITEMUKAN]` sebelum mencoba alat, tetapi model tetap
melakukannya: "apa 9 sks pilihan yang perlu saya penuhi?" ditolak tanpa
panggilan tool, lalu "apa saja 9 sks pilihan yang belum saya ambil" dijawab
`get_nilai_saya` (log 2026-11-12). Dalam 30 hari sampai 2026-11-23, 13 dari 35
penolakan LLM terjadi saat tool ter-bind, dan tidak satu pun mencoba tool.

Sejak 2026-11-27 juga untuk penolakan berupa kalimat ("... tidak tercantum di
dokumen resmi") tanpa satu sitasi pun (T76): Qwen3.6-27B menjawab "mata kuliah
semester 2 prodi Magister Informatika apa saja?" dengan "dokumen resmi hanya
memuat kurikulum Informatika ..." tanpa memanggil `get_matkul_kurikulum` (2 dari
2). T1 di prompt yang diperluas untuk kasus prodi lain tidak menolong: Magister
Informatika dan "Sistem Informasi" tetap ditolak tanpa tool 6 dari 6. Dengan
dorongan ini keduanya dijawab dari SADS 6 dari 6 (docs/tool-call.md §18).

Berupa pesan pengguna, bukan pesan sistem kedua, karena tidak semua penyedia di
balik gateway menerima pesan sistem di tengah percakapan. Teks mahasiswa selalu
terbungkus `<pertanyaan_mahasiswa>` (FR-5), jadi catatan ini tetap terbedakan."""


@dataclass(frozen=True)
class ToolLoopResult:
    text: str
    documents: list[Document]
    usage: dict[str, Any] | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """Satu entri per panggilan tool: nama, argumen (mentah dari model), ok,
    latency_ms. Untuk `messages.meta` (observability, docs/tool-call.md §13)."""
    attachments: list[Lampiran] = field(default_factory=list)
    """Lampiran dari tool yang berhasil, urut panggilan (docs/tool-call.md §10a),
    sudah dipilih menurut nama yang ditulis jawaban (`pilih_lampiran`). Sumbernya
    belum diperiksa: `generate` hanya meneruskan yang sumbernya dikutip jawaban."""
    alasan_gagal: tuple[str, ...] = ()
    """`alasan_mahasiswa` tool yang gagal, tanpa duplikat, urut panggilan (T77)."""


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


async def _jalankan_tool(
    tc: dict, by_name: dict[str, ToolSpec], penanya: Penanya | None
) -> ToolResult:
    spec = by_name.get(tc.get("name", ""))
    if spec is None:
        return ToolResult(name=str(tc.get("name")), label="", text="", ok=False)
    try:
        args = validasi_argumen(spec, tc.get("args") or {})
        if spec.pakai_penanya:
            # Identitas dari request, di luar argumen model: model tidak bisa
            # memilih NIM siapa yang diambil datanya (docs/tool-call.md §11).
            return await spec.handler(**args, penanya=penanya or Penanya())
        return await spec.handler(**args)
    except ToolArgumentError as e:
        logger.warning("argumen tool %s tidak valid: %s", spec.name, e)
    # Tanpa teks galatnya: pesan httpx memuat URL, dan URL endpoint data pribadi
    # memuat NIM, yang tidak boleh masuk log aplikasi (`app.prodi`).
    except httpx.HTTPStatusError as e:
        logger.warning("tool %s: layanan membalas HTTP %s", spec.name, e.response.status_code)
    except httpx.HTTPError as e:
        logger.warning("tool %s gagal memanggil layanan: %s", spec.name, type(e).__name__)
    except Exception:  # pragma: no cover - jaring pengaman, jangan jatuhkan giliran
        logger.exception("tool %s galat tak terduga", spec.name)
    return ToolResult(name=spec.name, label=spec.citation_label, text="", ok=False)


async def _jalankan_terukur(
    tc: dict, by_name: dict[str, ToolSpec], penanya: Penanya | None = None
) -> tuple[ToolResult, int]:
    """`_jalankan_tool` beserta latensinya (ms) untuk `meta.tool_calls`.

    Juga tercatat sebagai panggilan di bawah node `generate` di tab Graf
    (`rekaman.py`): argumen dari model, dan teks yang dikembalikan ke model --
    persis yang dibaca model, termasuk sisipan `DAFTAR_DITAMPILKAN`."""
    mulai = time.perf_counter()
    r = await _jalankan_tool(tc, by_name, penanya)
    latency_ms = round((time.perf_counter() - mulai) * 1000)
    await catat_panggilan(
        jenis="tool",
        nama=str(tc.get("name") or "tool"),
        masukan=tc.get("args") or {},
        keluaran={
            "ok": r.ok,
            "content": r.pesan_untuk_model(),
            "attachments": [{"title": a.title, "items": len(a.items)} for a in r.attachments],
        },
        mulai=mulai,
        galat=None if r.ok else "DATA_TIDAK_TERSEDIA",
    )
    return r, latency_ms


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
    penanya: Penanya | None = None,
) -> ToolLoopResult:
    """Jalankan loop. `llm_tools` sudah di-bind dengan skema tool; `llm_plain`
    tanpa tool, dipakai memaksa jawaban bila batas putaran tercapai.

    `buat_config` dipanggil sekali per giliran LLM. Setiap giliran butuh `run_id`
    sendiri: LangSmith menolak run kedua dengan ID yang sama, sehingga satu
    config yang dipakai ulang membuat giliran sesudah tool -- yang berisi
    jawabannya -- hilang dari trace.

    `penanya` hanya diteruskan ke tool ber-`pakai_penanya`; None = tanpa NIM
    (uji coba admin), dan tool data pribadi membalas bahwa NIM tidak diketahui.

    Bila model menolak sebelum mencoba satu tool pun, ia didorong sekali
    (`DORONGAN_T1`) dengan tool tetap ter-bind; dorongan tidak dihitung putaran."""
    by_name = {s.name: s for s in specs}
    tanpa_hal = sumber_tanpa_halaman(documents)
    messages: list[Any] = [
        SystemMessage(content=TOOL_SYSTEM_PROMPT.format(context=format_context(documents))),
        HumanMessage(content=wrapped_question),
    ]
    hasil: list[ToolResult] = []
    jejak: list[dict[str, Any]] = []
    usage: dict | None = None
    didorong = False
    putaran = 0

    while putaran < max_rounds:
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
            if not didorong and not jejak and _menolak_tanpa_mencoba(teks, tanpa_hal):
                # Tidak menghabiskan putaran: dengan `max_rounds=1` dorongan
                # yang menghabiskan putaran hanya menyisakan giliran tanpa
                # tool. Penahan penanda tidak memancarkan apa pun dari balasan
                # yang hanya berisi penanda, jadi mahasiswa tidak melihatnya.
                didorong = True
                logger.info("model menolak tanpa mencoba tool; didorong sekali (T1)")
                messages.extend([gathered, HumanMessage(content=DORONGAN_T1)])
                continue
            return _hasil_akhir(teks, hasil, usage, jejak)
        putaran += 1
        if on_stage is not None:
            await on_stage("mengambil data akademik")
        messages.append(gathered)

        dijalankan = panggilan[:MAKS_TOOL_PER_GILIRAN]
        # Bersamaan, seperti FTS ∥ pgvector: setiap handler memakai klien HTTP
        # sendiri, dan `gather` mengembalikan hasil menurut urutan panggilan.
        keluaran = await asyncio.gather(
            *(_jalankan_terukur(tc, by_name, penanya) for tc in dijalankan)
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


def _menolak_tanpa_mencoba(teks: str, tanpa_halaman: dict[str, int] | None = None) -> bool:
    """Balasan yang akan dijadikan penolakan oleh `generate` (`is_not_found`).

    Yaitu `NOT_FOUND_MARKER` saja seperti diminta aturan 3, atau kalimat "tidak
    ditemukan/tercantum" tanpa satu sitasi pun (T76). Jawaban parsial yang
    bersitasi tidak didorong: yang tertulis tetap berguna, dan dorongan bisa
    membuat model membuangnya. Balasan kosong juga tidak (`_hasil_akhir` tetap
    menjadikannya penolakan), begitu pula `OFF_TOPIC_MARKER`: pertanyaan di luar
    urusan kampus tidak punya alat.

    Kalimat penolakan, berbeda dari penanda, sudah sempat mengalir ke mahasiswa;
    jawaban sesudah dorongan menyusul di gelembung yang sama sampai event
    `message` menggantinya dengan jawaban akhir. Tanpa dorongan, kalimat itu
    toh diganti penolakan resmi."""
    seragam = _seragam(teks)
    if OFF_TOPIC_MARKER in seragam:
        return False
    if NOT_FOUND_MARKER in seragam:
        return not strip_markers(teks)
    return is_not_found(teks, tanpa_halaman)


def _hasil_akhir(
    teks: str, hasil: list[ToolResult], usage: dict | None, jejak: list[dict[str, Any]]
) -> ToolLoopResult:
    """Bungkus jawaban final.

    Giliran final tanpa teks sama sekali tidak boleh tampil sebagai `answer`
    kosong. Ia diganti `NOT_FOUND_MARKER`, sehingga `generate` membalas
    penolakan resmi beserta kontak unit."""
    if not teks.strip():
        teks = NOT_FOUND_MARKER
    lampiran = [a for r in hasil if r.ok for a in pilih_lampiran(r.attachments, teks)]
    alasan = tuple(
        dict.fromkeys(r.alasan_mahasiswa for r in hasil if not r.ok and r.alasan_mahasiswa)
    )
    return ToolLoopResult(teks, hasil_tool_ke_dokumen(hasil), usage, jejak, lampiran, alasan)


def _jejak_lewat(tc: dict, args: dict[str, Any], alasan: str) -> dict[str, Any]:
    """Entri `meta.tool_calls` untuk panggilan yang sengaja tidak dijalankan."""
    return {
        "name": tc.get("name"),
        "args": args,
        "ok": False,
        "latency_ms": 0,
        "error": alasan,
    }
