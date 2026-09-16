"""Query rewriting (FR-4).

Chain terpisah: 3 pesan terakhir + pertanyaan baru -> pertanyaan mandiri.
Dilewati bila ini pesan pertama, karena tidak ada konteks untuk diserap dan
pemanggilan LLM tambahan hanya menambah latency serta biaya.
"""

from __future__ import annotations

from dataclasses import dataclass

HISTORY_WINDOW = 3
"""Jumlah pesan terakhir yang dipertimbangkan, sesuai FR-4."""


@dataclass(frozen=True)
class Turn:
    role: str
    konten: str


def needs_rewrite(history: list[Turn]) -> bool:
    """FR-4: lewati penulisan ulang pada pesan pertama."""
    return bool(history)


def format_history(history: list[Turn], window: int = HISTORY_WINDOW) -> str:
    """Ambil `window` pesan terakhir sebagai teks untuk prompt penulisan ulang."""
    recent = history[-window:] if window > 0 else []
    return "\n".join(f"{turn.role}: {turn.konten}" for turn in recent)
