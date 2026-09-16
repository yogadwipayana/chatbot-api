"""Pemuatan dokumen PDF (FR-1).

`PyMuPDFLoader` dipakai apa adanya dari LangChain -- ekstraksi PDF bukan
kontribusi proyek ini. Yang ditulis sendiri adalah deteksi PDF hasil scan,
karena PDF scan menghasilkan chunk kosong yang diam-diam merusak retrieval:
dokumen terlihat masuk indeks, tetapi tidak pernah terambil.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MIN_CHARS_PER_PAGE = 100
"""Di bawah ini, satu halaman dianggap tanpa lapisan teks."""

MAX_EMPTY_PAGE_RATIO = 0.30
"""Bila lebih dari 30% halaman kosong teks, dokumen ditolak sebagai hasil scan."""


class ScannedPdfError(ValueError):
    """PDF tanpa lapisan teks yang memadai. FR-1: tolak dengan pesan jelas."""


class UnreadablePdfError(ValueError):
    """Berkas tidak dapat dibuka sebagai PDF: rusak, terpotong, atau bukan PDF."""


@dataclass(frozen=True)
class LoadedPage:
    halaman: int
    konten: str


def empty_page_ratio(
    pages: Sequence[str], min_chars: int = MIN_CHARS_PER_PAGE
) -> float:
    """Proporsi halaman yang praktis tidak punya teks."""
    if not pages:
        return 1.0
    empty = sum(1 for page in pages if len(page.strip()) < min_chars)
    return empty / len(pages)


def is_probably_scanned(
    pages: Sequence[str],
    *,
    min_chars: int = MIN_CHARS_PER_PAGE,
    max_empty_ratio: float = MAX_EMPTY_PAGE_RATIO,
) -> bool:
    """Tebakan apakah PDF merupakan hasil scan tanpa OCR."""
    return empty_page_ratio(pages, min_chars) > max_empty_ratio


def load_pdf(path: str | Path) -> list[LoadedPage]:
    """Muat PDF menjadi daftar halaman, mempertahankan nomor halaman (FR-1).

    Raises:
        ScannedPdfError: bila dokumen terdeteksi hasil scan.
        UnreadablePdfError: bila berkas tidak dapat dibuka sebagai PDF.
    """
    from langchain_community.document_loaders import PyMuPDFLoader

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dokumen tidak ditemukan: {path}")

    try:
        documents = PyMuPDFLoader(str(path)).load()
    except Exception as exc:
        raise UnreadablePdfError(
            f"'{path.name}' tidak dapat dibaca sebagai PDF. Pastikan berkasnya tidak "
            "rusak dan tidak dilindungi kata sandi."
        ) from exc
    pages = [doc.page_content for doc in documents]

    if is_probably_scanned(pages):
        raise ScannedPdfError(
            f"'{path.name}' tampaknya hasil scan tanpa lapisan teks "
            f"({empty_page_ratio(pages):.0%} halaman kosong). "
            "Jalankan OCR terlebih dahulu, atau unggah versi digital aslinya."
        )

    return [
        LoadedPage(halaman=doc.metadata.get("page", index) + 1, konten=doc.page_content)
        for index, doc in enumerate(documents)
    ]
