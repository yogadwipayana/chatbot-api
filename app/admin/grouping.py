"""Pengelompokan pertanyaan tak terjawab (AD-4).

"12 mahasiswa menanyakan ini" jauh lebih mudah ditindaklanjuti daripada 12 baris
terpisah yang kalimatnya sedikit berbeda. Kemiripan diukur dengan kata yang sama
(Jaccard) setelah kata tanya dan kata sapaan dibuang -- bukan embedding -- supaya
hasilnya dapat dijelaskan ke admin non-teknis, deterministik, dan membuka
halaman AD-4 tidak memanggil API berbayar.

Sengaja tanpa impor pihak ketiga, sama seperti `fusion` dan `threshold`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

SIMILARITY_THRESHOLD = 0.5
"""Jaccard minimum terhadap anggota pertama kelompok."""

STOPWORDS = frozenset(
    {
        # kata tanya
        "apa",
        "apakah",
        "bagaimana",
        "gimana",
        "gmn",
        "bgmn",
        "kapan",
        "dimana",
        "mana",
        "siapa",
        "berapa",
        "kenapa",
        "mengapa",
        "cara",
        # kata sambung dan depan
        "yang",
        "dan",
        "atau",
        "di",
        "ke",
        "dari",
        "untuk",
        "utk",
        "buat",
        "dengan",
        "dgn",
        "pada",
        "dalam",
        "tentang",
        "mengenai",
        "terkait",
        "jika",
        "kalau",
        "kalo",
        "klo",
        "agar",
        "supaya",
        "seperti",
        "juga",
        "lagi",
        "saja",
        "aja",
        # kata ganti dan penunjuk
        "ini",
        "itu",
        "saya",
        "aku",
        "gue",
        "gw",
        "kami",
        "kita",
        # modalitas
        "ada",
        "bisa",
        "bisakah",
        "boleh",
        "mau",
        "ingin",
        "pengen",
        "harus",
        "perlu",
        "sudah",
        "udah",
        "belum",
        "blm",
        "tidak",
        "gak",
        "nggak",
        "ngga",
        "enggak",
        "tak",
        # sapaan dan partikel
        "tolong",
        "mohon",
        "min",
        "admin",
        "kak",
        "kakak",
        "pak",
        "bu",
        "dong",
        "sih",
        "ya",
        "yah",
        "deh",
        "nih",
        "kah",
        "halo",
        "hai",
        "info",
        "informasi",
        "terima",
        "kasih",
        "makasih",
    }
)

_TOKEN_RE = re.compile(r"[^\W_]+")


@dataclass(frozen=True)
class UnansweredItem:
    id: str
    pertanyaan: str
    top_score: float | None
    created_at: datetime
    resolved: bool


@dataclass
class QuestionGroup:
    items: list[UnansweredItem] = field(default_factory=list)
    """Terurut dari yang terbaru."""

    @property
    def representative(self) -> UnansweredItem:
        """Pertanyaan terbaru, apa adanya seperti diketik mahasiswa."""
        return self.items[0]

    @property
    def ids(self) -> list[str]:
        return [item.id for item in self.items]

    @property
    def jumlah(self) -> int:
        return len(self.items)

    @property
    def resolved(self) -> bool:
        return self.items[0].resolved

    @property
    def terakhir_ditanyakan(self) -> datetime:
        return max(item.created_at for item in self.items)

    @property
    def top_score_rata2(self) -> float | None:
        skor = [item.top_score for item in self.items if item.top_score is not None]
        return sum(skor) / len(skor) if skor else None


def key_terms(text: str) -> frozenset[str]:
    """Kata bermakna dalam pertanyaan: huruf kecil, tanpa tanda baca, tanpa kata tanya.

    Akhiran "-nya" dilepas ("wisudanya" -> "wisuda"), karena mahasiswa menulis
    keduanya bergantian untuk maksud yang sama.
    """
    normal = unicodedata.normalize("NFKC", text).casefold()
    terms: set[str] = set()
    for token in _TOKEN_RE.findall(normal):
        if len(token) > 5 and token.endswith("nya"):
            token = token[:-3]
        if len(token) < 2 or token in STOPWORDS:
            continue
        terms.add(token)
    return frozenset(terms)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def group_questions(
    items: Sequence[UnansweredItem],
    *,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[QuestionGroup]:
    """Kelompokkan pertanyaan yang mirip.

    Pertanyaan diproses dari yang terbaru. Masing-masing masuk ke kelompok yang
    anggota pertamanya paling mirip (minimal `threshold`), atau membuka kelompok
    baru. Pembanding selalu anggota pertama, bukan gabungan kata seluruh
    anggota, supaya kelompok tidak melebar berantai: A mirip B dan B mirip C
    tidak berarti A dan C membahas hal yang sama.

    Pertanyaan yang sudah dan belum ditindaklanjuti tidak pernah dicampur.
    Pertanyaan tanpa kata bermakna (mis. "??") hanya digabung dengan yang
    teksnya persis sama.

    Urutan hasil: kelompok terbesar dulu, lalu yang paling baru ditanyakan.
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold harus di (0, 1], diberi {threshold}")

    kelompok: list[tuple[frozenset[str], str, QuestionGroup]] = []
    for item in sorted(items, key=lambda i: (i.created_at, i.id), reverse=True):
        terms = key_terms(item.pertanyaan)
        persis = " ".join(item.pertanyaan.casefold().split())

        tujuan: QuestionGroup | None = None
        terbaik = 0.0
        for g_terms, g_persis, group in kelompok:
            if group.resolved != item.resolved:
                continue
            if not terms or not g_terms:
                if persis == g_persis:
                    tujuan = group
                    break
                continue
            skor = jaccard(terms, g_terms)
            if skor >= threshold and skor > terbaik:
                tujuan, terbaik = group, skor

        if tujuan is None:
            tujuan = QuestionGroup()
            kelompok.append((terms, persis, tujuan))
        tujuan.items.append(item)

    hasil = [group for _, _, group in kelompok]
    hasil.sort(key=lambda g: (-g.jumlah, -g.terakhir_ditanyakan.timestamp()))
    return hasil
