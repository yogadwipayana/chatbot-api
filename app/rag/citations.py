"""Format, ekstraksi, dan validasi sitasi (FR-5, FE-2).

Format wajib: `[Panduan Akademik 2025, hal. 12]`

FE-2 disebut PRD §8 sebagai komponen paling kritis: verifikasi harus semudah
satu klik. Karena itu sitasi bukan sekadar teks -- ia harus dapat diurai
kembali menjadi (dokumen, halaman) agar frontend bisa membuka PDF tepat di
halaman tersebut.

Validasi di sini menutup dua kegagalan yang tidak terlihat oleh mata:
1. Jawaban tanpa sitasi sama sekali (melanggar FR-5).
2. Sitasi ke dokumen/halaman yang TIDAK ada dalam konteks -- LLM mengarang
   sumber. Ini lebih berbahaya daripada tidak menjawab, karena tampak sah.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

CITATION_RE = re.compile(
    r"\[\s*(?:sumber\s*:\s*)?(?P<judul>[^\[\]]+?)\s*,\s*hal\.?\s*"
    r"(?P<halaman>\d+(?:\s*(?:[-–—,&]|dan)\s*\d+)*)\s*\]",
    re.IGNORECASE,
)
"""Satu penanda boleh menyebut beberapa halaman: `hal. 8–9`, `hal. 4, 6`,
`hal. 4 dan 6`. Prompt meminta satu halaman per sitasi, tetapi daftar yang
bersambung ke halaman berikutnya (potongan lanjutan, `RETRIEVAL_NEIGHBORS`)
membuat LLM menulis rentang. Tanpa ini penanda itu tidak terbaca sama sekali
dan jawabannya tampil tanpa kartu sumber (T19).

Awalan `Sumber:` ([Sumber: Judul, hal. 2]) juga sesekali ditulis LLM. Tanpa
dibuang, awalan itu ikut menjadi judul, tidak cocok dengan dokumen mana pun,
dan kartunya hilang dengan cara yang sama."""

_BAGIAN_HALAMAN_RE = re.compile(r"(\d+)(?:\s*[-–—]\s*(\d+))?")

RENTANG_MAKS = 10
"""Rentang yang lebih lebar dari ini (atau terbalik) hanya diambil kedua
ujungnya: `hal. 1-300` hampir pasti salah tulis, bukan 300 sumber."""


@dataclass(frozen=True)
class Citation:
    judul: str
    halaman: int

    def render(self) -> str:
        return format_citation(self.judul, self.halaman)


@dataclass(frozen=True)
class CitationValidation:
    citations: tuple[Citation, ...]
    unknown: tuple[Citation, ...]
    """Sitasi yang tidak ada padanannya di konteks -- sumber karangan."""

    @property
    def has_citation(self) -> bool:
        return bool(self.citations)

    @property
    def is_valid(self) -> bool:
        return self.has_citation and not self.unknown


def format_citation(judul: str, halaman: int) -> str:
    """Bentuk satu penanda sitasi.

    Args:
        judul: judul dokumen apa adanya, mis. "Panduan Akademik 2025".
        halaman: nomor halaman 1-based dari `chunks.page`.
    """
    judul = judul.strip()
    if not judul:
        raise ValueError("judul dokumen kosong")
    if halaman < 1:
        raise ValueError(f"halaman harus >= 1, diberi {halaman}")
    return f"[{judul}, hal. {halaman}]"


def _halaman(teks: str) -> list[int]:
    """`"8–9"` -> [8, 9]; `"4, 6"` dan `"4 dan 6"` -> [4, 6]."""
    hasil: list[int] = []
    for bagian in _BAGIAN_HALAMAN_RE.finditer(teks):
        awal = int(bagian.group(1))
        akhir = int(bagian.group(2) or awal)
        if awal <= akhir <= awal + RENTANG_MAKS:
            hasil.extend(range(awal, akhir + 1))
        else:
            hasil.extend((awal, akhir))
    return hasil


def extract_citations(
    answer: str, tanpa_halaman: Mapping[str, int] | None = None
) -> tuple[Citation, ...]:
    """Ambil semua sitasi dari jawaban, urut kemunculan, tanpa duplikat.

    Penanda dengan beberapa halaman menjadi satu sitasi per halaman.

    `tanpa_halaman`: judul sumber yang tidak berhalaman (entri tanya jawab)
    beserta halamannya di database. Penandanya cukup `[Judul]` -- judulnya
    dicocokkan persis, sehingga teks berkurung lain ("TRANSFER[SPASI]...")
    tidak pernah terbaca sebagai sitasi.
    """
    kemunculan: list[tuple[int, Citation]] = []
    for match in CITATION_RE.finditer(answer):
        judul = match.group("judul").strip()
        for halaman in _halaman(match.group("halaman")):
            kemunculan.append((match.start(), Citation(judul=judul, halaman=halaman)))
    for judul, halaman in (tanpa_halaman or {}).items():
        pola = re.compile(r"\[\s*" + re.escape(judul) + r"\s*\]", re.IGNORECASE)
        for match in pola.finditer(answer):
            kemunculan.append((match.start(), Citation(judul=judul, halaman=halaman)))

    found: list[Citation] = []
    # Pengurutan stabil: halaman dalam satu penanda tetap berurutan.
    for _, citation in sorted(kemunculan, key=lambda item: item[0]):
        if citation not in found:
            found.append(citation)
    return tuple(found)


def ringkas_sitasi_tanpa_halaman(answer: str, judul: Iterable[str]) -> str:
    """`[Judul, hal. 1]` menjadi `[Judul]` untuk sumber yang tidak berhalaman.

    Entri tanya jawab tidak punya halaman, tetapi LLM kadang tetap menulis
    "hal. 1" mengikuti format sitasi PDF (T25). Judul lain tidak disentuh.
    """
    kunci = {j.casefold() for j in judul}
    if not kunci:
        return answer

    def ganti(match: re.Match[str]) -> str:
        nama = match.group("judul").strip()
        return f"[{nama}]" if nama.casefold() in kunci else match.group(0)

    return CITATION_RE.sub(ganti, answer)


def validate_answer(
    answer: str,
    allowed: set[tuple[str, int]],
) -> CitationValidation:
    """Periksa jawaban terhadap daftar (judul, halaman) yang benar-benar diambil.

    Args:
        answer: teks jawaban LLM.
        allowed: pasangan (judul, halaman) dari chunk yang masuk konteks.
                 Perbandingan judul case-insensitive agar variasi kapitalisasi
                 dari LLM tidak dianggap sumber karangan.
    """
    normalised = {(judul.casefold(), halaman) for judul, halaman in allowed}
    citations = extract_citations(answer)
    unknown = tuple(
        c for c in citations if (c.judul.casefold(), c.halaman) not in normalised
    )
    return CitationValidation(citations=citations, unknown=unknown)
