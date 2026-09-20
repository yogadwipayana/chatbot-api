"""Deteksi topik berisiko tinggi + eskalasi ke kontak manusia (FR-6, FE-3).

Jawaban pada topik ini SELALU disertai kontak unit resmi. Deteksi berbasis
kata kunci, bukan LLM: hasilnya harus dapat diaudit dan tidak boleh berubah
diam-diam saat model diganti.

Daftar kontak di bawah adalah *placeholder*. PRD §13 Fase 0 menetapkan daftar
topik berisiko tinggi dan kontak resminya sebagai prasyarat blocking -- ganti
`DEFAULT_CONTACTS` dengan hasil kesepakatan biro akademik sebelum rilis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class RiskTopic(StrEnum):
    DEADLINE = "deadline"
    SYARAT_KELULUSAN = "syarat_kelulusan"
    PEMBAYARAN = "pembayaran"
    SANKSI = "sanksi"
    DROP_OUT = "drop_out"


@dataclass(frozen=True)
class UnitContact:
    """Isi banner eskalasi FE-3."""

    unit: str
    jam_layanan: str
    kontak: str


# Pola per topik. Tuple ke-2 menandai apakah pola case-sensitive.
# Pola case-sensitive dipakai untuk akronim pendek seperti "DO", yang kalau
# dicocokkan case-insensitive akan ikut kena di "dosen", "kado", "Indonesia".
_PATTERNS: dict[RiskTopic, tuple[tuple[str, bool], ...]] = {
    RiskTopic.DEADLINE: (
        (r"\bdead\s?-?line\b", False),
        (r"\bbatas\s+(?:waktu|akhir|pengumpulan)\b", False),
        (r"\btenggat\b", False),
        (r"\bkapan\s+terakhir\b", False),
        (r"\bpaling\s+lambat\b", False),
    ),
    RiskTopic.SYARAT_KELULUSAN: (
        (r"\bsyarat\s+(?:kelulusan|lulus|yudisium|wisuda)\b", False),
        (r"\bketentuan\s+lulus\b", False),
        (r"\bsks\s+minimal\b", False),
        (r"\bipk\s+minim(?:al|um)\b", False),
    ),
    RiskTopic.PEMBAYARAN: (
        (r"\bpembayaran\b", False),
        (r"\bbayar\b", False),
        (r"\bukt\b", False),
        (r"\bspp\b", False),
        (r"\bdenda\b", False),
        (r"\btagihan\b", False),
        (r"\bbiaya\b", False),
    ),
    RiskTopic.SANKSI: (
        (r"\bsanksi\b", False),
        (r"\bskors(?:ing)?\b", False),
        (r"\bpelanggaran\b", False),
        (r"\bhukuman\b", False),
    ),
    RiskTopic.DROP_OUT: (
        (r"\bDO\b", True),
        (r"\bdrop\s*-?\s*out\b", False),
        (r"\b(?:kena|di|terancam|ke)\s*-?\s*do\b", False),
        (r"\bdikeluarkan\s+dari\s+kampus\b", False),
    ),
}

_COMPILED: dict[RiskTopic, tuple[re.Pattern[str], ...]] = {
    topic: tuple(
        re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        for pattern, case_sensitive in patterns
    )
    for topic, patterns in _PATTERNS.items()
}

DEFAULT_CONTACTS: dict[RiskTopic, UnitContact] = {
    RiskTopic.DEADLINE: UnitContact(
        unit="Biro Administrasi Akademik",
        jam_layanan="Senin-Jumat, 08.00-15.00",
        kontak="akademik@instiki.ac.id",
    ),
    RiskTopic.SYARAT_KELULUSAN: UnitContact(
        unit="Biro Administrasi Akademik",
        jam_layanan="Senin-Jumat, 08.00-15.00",
        kontak="akademik@instiki.ac.id",
    ),
    RiskTopic.PEMBAYARAN: UnitContact(
        unit="Biro Keuangan",
        jam_layanan="Senin-Jumat, 08.00-14.00",
        kontak="keuangan@instiki.ac.id",
    ),
    RiskTopic.SANKSI: UnitContact(
        unit="Bagian Kemahasiswaan",
        jam_layanan="Senin-Jumat, 08.00-15.00",
        kontak="kemahasiswaan@instiki.ac.id",
    ),
    RiskTopic.DROP_OUT: UnitContact(
        unit="Dosen Wali / Bagian Kemahasiswaan",
        jam_layanan="Senin-Jumat, 08.00-15.00",
        kontak="kemahasiswaan@instiki.ac.id",
    ),
}


@dataclass(frozen=True)
class RiskAssessment:
    topics: tuple[RiskTopic, ...]
    contacts: tuple[UnitContact, ...]

    @property
    def needs_escalation(self) -> bool:
        return bool(self.topics)


def detect(
    text: str,
    contacts: dict[RiskTopic, UnitContact] | None = None,
) -> RiskAssessment:
    """Deteksi topik berisiko tinggi pada teks pertanyaan.

    Topik dikembalikan dalam urutan deklarasi `RiskTopic` agar deterministik.
    Kontak dideduplikasi -- dua topik yang ditangani unit yang sama hanya
    memunculkan satu banner.
    """
    contacts = DEFAULT_CONTACTS if contacts is None else contacts
    matched = tuple(
        topic
        for topic in RiskTopic
        if any(pattern.search(text) for pattern in _COMPILED[topic])
    )

    seen: list[UnitContact] = []
    for topic in matched:
        contact = contacts.get(topic)
        if contact is not None and contact not in seen:
            seen.append(contact)

    return RiskAssessment(topics=matched, contacts=tuple(seen))
