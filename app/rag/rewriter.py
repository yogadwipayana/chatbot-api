"""Query rewriting (FR-4).

Chain terpisah: 3 pesan terakhir + pertanyaan baru -> pertanyaan mandiri.
Dilewati bila ini pesan pertama, karena tidak ada konteks untuk diserap dan
pemanggilan LLM tambahan hanya menambah latency serta biaya -- kecuali pesan
pertama itu berbahasa Inggris, yang harus diterjemahkan dulu agar pencarian
fulltext (kamus bahasa Indonesia) ikut menemukannya.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HISTORY_WINDOW = 3
"""Jumlah pesan terakhir yang dipertimbangkan, sesuai FR-4."""

# fmt: off
_KATA_INGGRIS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
        "do", "does", "did", "can", "could", "should", "would", "will", "may", "must",
        "what", "how", "when", "where", "which", "who", "whom", "whose", "why",
        "i", "me", "my", "you", "your", "we", "our", "they", "their",
        "it", "its", "this", "that", "these", "those", "there",
        "to", "of", "for", "in", "on", "at", "with", "from", "about", "into", "by",
        "and", "or", "but", "not", "if", "have", "has", "had", "get", "need", "want",
    }
)
_KATA_INDONESIA = frozenset(
    {
        "apa", "bagaimana", "berapa", "kapan", "mana", "siapa", "mengapa", "kenapa",
        "apakah", "yang", "dan", "atau", "di", "ke", "dari", "untuk", "dengan", "pada",
        "dalam", "ini", "itu", "ada", "tidak", "bisa", "saya", "aku", "kami", "kita",
        "cara", "bagi", "jika", "kalau", "sudah", "belum", "akan", "harus", "boleh",
    }
)
# fmt: on


@dataclass(frozen=True)
class Turn:
    role: str
    konten: str


def looks_english(question: str) -> bool:
    """True bila pertanyaan tampak berbahasa Inggris.

    Dihitung dari kata tugas saja ("what", "the", "for" lawan "apa", "yang",
    "untuk"), bukan kata benda: pertanyaan Indonesia yang menyelipkan istilah
    Inggris ("cara reset password SIAKAD") tetap dianggap Indonesia dan tidak
    memicu panggilan LLM tambahan. Pertanyaan tanpa kata tugas ("TOEIC?")
    juga tidak -- pencarian vektor multibahasa cukup untuknya.
    """
    kata = re.findall(r"[a-z]+", question.lower())
    inggris = sum(k in _KATA_INGGRIS for k in kata)
    indonesia = sum(k in _KATA_INDONESIA for k in kata)
    return inggris >= 2 and inggris > indonesia


def needs_rewrite(history: list[Turn], question: str = "") -> bool:
    """FR-4: lewati penulisan ulang pada pesan pertama, kecuali berbahasa Inggris (T22)."""
    return bool(history) or looks_english(question)


def format_history(history: list[Turn], window: int = HISTORY_WINDOW) -> str:
    """Ambil `window` pesan terakhir sebagai teks untuk prompt penulisan ulang."""
    recent = history[-window:] if window > 0 else []
    return "\n".join(f"{turn.role}: {turn.konten}" for turn in recent)
