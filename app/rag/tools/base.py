"""Primitif tool-calling: spesifikasi, hasil, validasi argumen, kartu sitasi.

Murni data + fungsi; tidak mengimpor LangGraph maupun pipeline, supaya mudah
diuji dan dipakai ulang antar layanan. Lihat `docs/tool-call.md` (§6, §10, §11).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.db.models import DocumentType

MAKS_PANJANG_ARGUMEN = 200
"""Batas panjang nilai argumen string dari model sebelum diteruskan ke layanan.

Argumen yang dihalusinasi model paling buruk menghasilkan hasil kosong, bukan
beban atau panggilan tak terduga ke layanan hulu (docs/tool-call.md §11)."""


class ToolArgumentError(ValueError):
    """Argumen dari model tidak memenuhi skema tool."""


@dataclass(frozen=True)
class ToolResult:
    """Hasil satu panggilan tool: teks untuk model + label kartu sitasi."""

    name: str
    label: str
    """Penanda sitasi, mis. "Data akademik SADS". Kosong bila gagal."""
    text: str
    ok: bool = True

    def pesan_untuk_model(self) -> str:
        """Isi pesan `role:"tool"` yang dibalikkan ke model.

        Diberi awalan penanda sitasi supaya model tahu persis cara mengutipnya
        (aturan T3). Kegagalan dibalas penanda datar: model menjawab apa adanya
        atau mengakui data tidak tersedia, bukan mengarang (docs/tool-call.md §12)."""
        if not self.ok:
            return "DATA_TIDAK_TERSEDIA: data tidak dapat diambil saat ini."
        return f"(Kutip data berikut dengan menyalin penanda [{self.label}].)\n\n{self.text}"


@dataclass(frozen=True)
class ToolSpec:
    """Satu tool: skema untuk LLM, pemicu kelayakan, handler, label sitasi."""

    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema argumen (gaya OpenAI function)."""
    handler: Callable[..., Awaitable[ToolResult]]
    """Async, menerima argumen bernama yang sudah divalidasi, mengembalikan ToolResult."""
    triggers: tuple[str, ...] = ()
    """Kata kunci rute kelayakan (docs/tool-call.md §8). Kosong = tak pernah eligible."""
    citation_label: str = ""
    unit: str | None = None

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def cocok(self, pertanyaan: str) -> bool:
        teks = pertanyaan.casefold()
        return any(pemicu.casefold() in teks for pemicu in self.triggers)


def validasi_argumen(spec: ToolSpec, mentah: dict[str, Any] | None) -> dict[str, Any]:
    """Tegakkan skema sebelum handler dipanggil (docs/tool-call.md §11).

    Argumen tak dikenal dibuang, string dibersihkan dari karakter non-cetak dan
    dibatasi panjangnya, dan argumen wajib yang kosong menolak panggilan. Handler
    tidak pernah menerima URL -- hanya nilai argumen bernama dari skema ini.
    """
    props = spec.parameters.get("properties", {})
    wajib = spec.parameters.get("required", [])
    bersih: dict[str, Any] = {}
    for nama, nilai in (mentah or {}).items():
        if nama not in props:
            continue
        if props[nama].get("type") == "string":
            teks = "".join(ch for ch in str(nilai) if ch.isprintable())
            bersih[nama] = teks[:MAKS_PANJANG_ARGUMEN].strip()
        else:
            bersih[nama] = nilai
    for nama in wajib:
        if not bersih.get(nama):
            raise ToolArgumentError(f"argumen wajib '{nama}' tidak ada atau kosong")
    return bersih


def hasil_tool_ke_dokumen(hasil: Iterable[ToolResult]) -> list[Document]:
    """Kartu sitasi sintetis: satu Document semu per label (docs/tool-call.md §10).

    Hasil tool bukan baris di tabel `documents`; Document semu ini hanya
    menumpang mesin sitasi yang ada (`citations_for`, `ringkas_sitasi_tanpa_halaman`).
    Ditandai `tanya_jawab` agar frontend menampilkannya tanpa tautan dan tanpa
    "hal. N" (lihat `CitationOut.type`). Gabungan per label supaya "Data akademik
    SADS" menjadi satu kartu walau beberapa tool dipanggil.
    """
    per_label: dict[str, list[str]] = {}
    for r in hasil:
        if r.ok and r.text.strip():
            per_label.setdefault(r.label, []).append(r.text)
    return [
        Document(
            page_content="\n\n".join(teks),
            metadata={
                "judul": label,
                "jenis": DocumentType.TANYA_JAWAB,
                "halaman": 1,
                "file_path": "",
                "document_id": "",
            },
        )
        for label, teks in per_label.items()
    ]
