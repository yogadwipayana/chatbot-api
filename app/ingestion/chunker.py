"""Pemecahan dokumen menjadi chunk (FR-1).

`RecursiveCharacterTextSplitter` dipakai dari LangChain -- sudah teruji, dan
bukan bagian yang ingin dibuktikan proyek ini (PRD §6). Yang ditambahkan di
sini adalah metadata per chunk: dokumen asal, halaman, unit, tahun berlaku.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.loader import LoadedPage

DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 105


@dataclass(frozen=True)
class PreparedChunk:
    konten: str
    halaman: int
    urutan: int
    metadata: dict[str, Any] = field(default_factory=dict)


def split_pages(
    pages: Sequence[LoadedPage],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    metadata: dict[str, Any] | None = None,
) -> list[PreparedChunk]:
    """Pecah tiap halaman menjadi chunk, nomor halaman ikut terbawa.

    Pemecahan dilakukan per halaman, bukan atas seluruh dokumen yang
    disambung, supaya satu chunk tidak pernah mencakup dua halaman -- sitasi
    FE-2 harus menunjuk ke satu halaman yang pasti.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) harus lebih kecil dari "
            f"chunk_size ({chunk_size})"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[PreparedChunk] = []
    urutan = 0
    for page in pages:
        for piece in splitter.split_text(page.konten):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                PreparedChunk(
                    konten=piece,
                    halaman=page.halaman,
                    urutan=urutan,
                    metadata={**(metadata or {}), "halaman": page.halaman},
                )
            )
            urutan += 1
    return chunks
