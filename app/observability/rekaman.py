"""Rekaman input/output giliran chat untuk tab Graf halaman Log (`logs.md`, tahap 4).

Pengganti LangSmith di sisi aplikasi. Setiap node LangGraph mencatat state yang
diterimanya, perubahan yang dikembalikannya, dan rute yang diambil sesudahnya;
panggilan di dalam node (LLM, retriever, tool, JEV) dicatat sebagai pohon di
bawahnya. Dengan ini tracing LangSmith boleh dimatikan tanpa kehilangan
kemampuan membaca prompt dan jawaban model saat menelusuri jawaban buruk.

Satu giliran disimpan sebagai SATU blob di tabel `traces`:

- **Serialisasi** (`jadikan_json`): dataclass, pydantic, `Document`, pesan
  LangChain, dan enum menjadi JSON. String dipotong `MAKS_TEKS` karakter dan
  daftar `MAKS_BUTIR` butir, supaya satu keluaran raksasa tidak membengkakkan
  berkas log.
- **Deduplikasi** (`_padatkan`): state LangGraph bersifat kumulatif, jadi daftar
  dokumen yang sama muncul di input `validate_context`, input `generate`, dan
  `outcome`, dan konteks yang sama dikirim ke setiap putaran loop tool. Subpohon
  yang panjang disimpan sekali dan dirujuk lewat `{"$ref": hash}`.
- **Penyamaran** (`samarkan`): teks pertanyaan sensitif (FR-7) diganti penanda
  yang sama dengan di Postgres, di mana pun ia muncul -- state, prompt LLM,
  permintaan JEV.
- **Kompresi**: zlib. Satu giliran ber-tool sekitar 60 KB JSON mentah.

Semua langkah ini berjalan di thread penulis log (`LogStore.tulis`), bukan di
event loop yang melayani mahasiswa.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import time
import zlib
from collections.abc import Mapping
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

VERSI_REKAMAN = 1

MAKS_TEKS = 32_000
"""Karakter per string. Prompt jawaban ber-tool (sistem + konteks + hasil SADS)
sekitar 20 ribu karakter; batas ini menyisakan ruang tanpa membiarkan satu
string tak wajar memenuhi berkas."""

MAKS_BUTIR = 300
"""Butir per daftar. Daftar dosen SADS terpanjang 220 nama."""

MAKS_KEDALAMAN = 12

MIN_DEDUP = 300
"""Subpohon (atau string) yang JSON-nya lebih panjang dari ini disimpan sekali.
Di bawah itu rujukan `{"$ref": "<16 heks>"}` justru lebih panjang daripada isinya."""

ACARA_PANGGILAN = "pandu_panggilan"
"""Nama custom event LangChain untuk panggilan yang bukan runnable (JEV, tool)."""


def _potong(teks: str, batas: int = MAKS_TEKS) -> str:
    if len(teks) <= batas:
        return teks
    return f"{teks[:batas]}… [dipotong, {len(teks) - batas} karakter lagi]"


def _isi_pesan(isi: Any) -> Any:
    if isinstance(isi, str):
        return _potong(isi)
    if isinstance(isi, list):  # blok konten (teks/gambar) dari sebagian provider
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in isi)
    return jadikan_json(isi)


def _pesan(m: BaseMessage) -> dict[str, Any]:
    """Pesan LangChain sebagai dict ringkas, mirip tampilan pesan di LangSmith."""
    hasil: dict[str, Any] = {"role": m.type, "content": _isi_pesan(m.content)}
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls:
        hasil["tool_calls"] = [
            {"name": tc.get("name"), "args": jadikan_json(tc.get("args")), "id": tc.get("id")}
            for tc in tool_calls
        ]
    invalid = getattr(m, "invalid_tool_calls", None)
    if invalid:
        hasil["invalid_tool_calls"] = jadikan_json(invalid)
    tool_call_id = getattr(m, "tool_call_id", None)
    if tool_call_id:
        hasil["tool_call_id"] = tool_call_id
    if m.name:
        hasil["name"] = m.name
    return hasil


def jadikan_json(x: Any, kedalaman: int = 0) -> Any:
    """Nilai Python apa pun -> struktur JSON. Tidak pernah melempar."""
    try:
        return _jadikan_json(x, kedalaman)
    except Exception:  # pragma: no cover - jaring pengaman untuk objek asing
        return _potong(repr(x), 500)


def _jadikan_json(x: Any, kedalaman: int) -> Any:
    if x is None or isinstance(x, bool | int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, str):
        return _potong(x)
    if isinstance(x, Enum):
        return jadikan_json(x.value, kedalaman)
    if isinstance(x, datetime | date):
        return x.isoformat()
    if isinstance(x, UUID | Path):
        return str(x)
    if isinstance(x, bytes):
        return f"<{len(x)} byte>"
    if kedalaman >= MAKS_KEDALAMAN:
        return _potong(repr(x), 200)
    k = kedalaman + 1
    if isinstance(x, BaseMessage):
        return _pesan(x)
    if isinstance(x, Document):
        return {
            "page_content": _potong(x.page_content),
            "metadata": jadikan_json(x.metadata, k),
        }
    if isinstance(x, BaseModel):
        return jadikan_json(x.model_dump(), k)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: jadikan_json(getattr(x, f.name), k) for f in dataclasses.fields(x)}
    if isinstance(x, Mapping):
        butir = list(x.items())
        hasil = {str(kunci): jadikan_json(v, k) for kunci, v in butir[:MAKS_BUTIR]}
        if len(butir) > MAKS_BUTIR:
            hasil["…"] = f"{len(butir) - MAKS_BUTIR} kunci lagi"
        return hasil
    if isinstance(x, list | tuple | set | frozenset):
        butir = list(x)
        hasil_daftar = [jadikan_json(v, k) for v in butir[:MAKS_BUTIR]]
        if len(butir) > MAKS_BUTIR:
            hasil_daftar.append(f"… {len(butir) - MAKS_BUTIR} butir lagi")
        return hasil_daftar
    return _potong(repr(x), 500)


def samarkan(nilai: Any, teks: list[str], pengganti: str) -> Any:
    """Ganti setiap kemunculan `teks` di semua string dalam `nilai`.

    Yang lebih panjang diganti lebih dulu, supaya pertanyaan mentah tidak
    tersisa sebagian setelah versi bersihnya (yang lebih pendek) diganti."""
    urut = sorted({t for t in teks if t and t.strip()}, key=len, reverse=True)
    if not urut:
        return nilai

    def ganti(v: Any) -> Any:
        if isinstance(v, str):
            for t in urut:
                if t in v:
                    v = v.replace(t, pengganti)
            return v
        if isinstance(v, dict):
            return {kunci: ganti(isi) for kunci, isi in v.items()}
        if isinstance(v, list):
            return [ganti(isi) for isi in v]
        return v

    return ganti(nilai)


def _hash(teks: str) -> str:
    return hashlib.sha1(teks.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _padatkan(nilai: Any, kamus: dict[str, Any]) -> Any:
    """Ganti subpohon panjang dengan `{"$ref": h}`; isinya masuk `kamus`."""
    if isinstance(nilai, dict):
        padat: Any = {k: _padatkan(v, kamus) for k, v in nilai.items()}
    elif isinstance(nilai, list):
        padat = [_padatkan(v, kamus) for v in nilai]
    elif isinstance(nilai, str):
        padat = nilai
    else:
        return nilai
    teks = json.dumps(padat, ensure_ascii=False, separators=(",", ":"))
    if len(teks) < MIN_DEDUP:
        return padat
    h = _hash(teks)
    kamus.setdefault(h, padat)
    return {"$ref": h}


def _bentangkan(nilai: Any, kamus: dict[str, Any]) -> Any:
    if isinstance(nilai, dict):
        if len(nilai) == 1 and "$ref" in nilai and nilai["$ref"] in kamus:
            return _bentangkan(kamus[nilai["$ref"]], kamus)
        return {k: _bentangkan(v, kamus) for k, v in nilai.items()}
    if isinstance(nilai, list):
        return [_bentangkan(v, kamus) for v in nilai]
    return nilai


def kemas(isi: Any, *, sensitif: list[str] | None = None, pengganti: str = "") -> bytes:
    """Rekaman mentah satu giliran -> blob untuk kolom `traces.data`."""
    data = jadikan_json(isi)
    if sensitif:
        data = samarkan(data, sensitif, pengganti)
    kamus: dict[str, Any] = {}
    akar = _padatkan(data, kamus)
    teks = json.dumps(
        {"v": VERSI_REKAMAN, "values": kamus, "root": akar},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return zlib.compress(teks.encode("utf-8"), 6)


def buka(blob: bytes) -> Any:
    """Kebalikan `kemas`: blob -> rekaman utuh, rujukan sudah dibentangkan."""
    data = json.loads(zlib.decompress(blob).decode("utf-8"))
    return _bentangkan(data["root"], data.get("values") or {})


async def catat_panggilan(
    *,
    jenis: str,
    nama: str,
    masukan: Any,
    keluaran: Any = None,
    mulai: float,
    galat: BaseException | str | None = None,
) -> None:
    """Catat satu panggilan yang bukan runnable LangChain (JEV, handler tool).

    Dikirim sebagai custom event: `NodeRecorder` menerimanya dengan `run_id`
    milik runnable yang sedang berjalan -- node, atau panggilan LLM di dalamnya
    -- sehingga panggilan ini tergantung di tempat yang benar di pohon. Di luar
    graf (skrip, test fungsi tunggal) tidak ada induk; event itu dibuang diam-diam.

    `mulai` adalah `time.perf_counter()` saat panggilan dimulai.
    """
    from langchain_core.callbacks.manager import adispatch_custom_event

    durasi = (time.perf_counter() - mulai) * 1000
    if isinstance(galat, BaseException):
        galat = f"{type(galat).__name__}: {galat}"
    try:
        await adispatch_custom_event(
            ACARA_PANGGILAN,
            {
                "kind": jenis,
                "name": nama,
                "input": masukan,
                "output": keluaran,
                "ended_at": datetime.now(UTC),
                "duration_ms": round(durasi, 2),
                "error": galat,
            },
        )
    except Exception:
        # Tanpa induk (di luar graf) langchain-core melempar RuntimeError.
        logger.debug("Panggilan %s tidak tercatat: tidak ada run induk", nama)
