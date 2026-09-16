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
from dataclasses import dataclass

CITATION_RE = re.compile(
    r"\[\s*(?P<judul>[^\[\]]+?)\s*,\s*hal\.?\s*(?P<halaman>\d+)\s*\]",
    re.IGNORECASE,
)


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
        halaman: nomor halaman 1-based dari `chunks.halaman`.
    """
    judul = judul.strip()
    if not judul:
        raise ValueError("judul dokumen kosong")
    if halaman < 1:
        raise ValueError(f"halaman harus >= 1, diberi {halaman}")
    return f"[{judul}, hal. {halaman}]"


def extract_citations(answer: str) -> tuple[Citation, ...]:
    """Ambil semua sitasi dari jawaban, urut kemunculan, tanpa duplikat."""
    found: list[Citation] = []
    for match in CITATION_RE.finditer(answer):
        citation = Citation(
            judul=match.group("judul").strip(),
            halaman=int(match.group("halaman")),
        )
        if citation not in found:
            found.append(citation)
    return tuple(found)


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
