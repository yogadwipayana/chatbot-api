"""Penanganan pertanyaan sensitif (FR-7).

Pertanyaan bernuansa tekanan mental, konflik personal, atau ketakutan di-DO
TIDAK diperlakukan sebagai pertanyaan administrasi. Sistem mengarahkan ke unit
bimbingan konseling / dosen wali dengan nada empatik.

Pemeriksaan ini berjalan paling awal dan menang atas seluruh alur lain --
termasuk atas eskalasi FR-6. Mahasiswa yang menulis "saya stres, takut di-DO"
tidak boleh dibalas dengan kutipan pasal tentang tata cara DO.

Nomor layanan di bawah adalah placeholder; verifikasi pada Fase 0 (PRD §13).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class SensitivityLevel(StrEnum):
    NONE = "none"
    DISTRESS = "distress"
    """Tertekan, cemas, konflik personal -> arahkan ke konseling."""
    CRISIS = "crisis"
    """Indikasi menyakiti diri sendiri -> respons krisis, prioritas tertinggi."""


@dataclass(frozen=True)
class SupportContact:
    unit: str
    jam_layanan: str
    kontak: str


CRISIS_CONTACTS: tuple[SupportContact, ...] = (
    SupportContact(
        unit="Layanan SEJIWA (Kemenkes)",
        jam_layanan="24 jam",
        kontak="119 ext. 8",
    ),
    SupportContact(
        unit="Unit Bimbingan & Konseling Kampus",
        jam_layanan="Senin-Jumat, 08.00-15.00",
        kontak="konseling@instiki.ac.id",
    ),
)

DISTRESS_CONTACTS: tuple[SupportContact, ...] = (
    SupportContact(
        unit="Unit Bimbingan & Konseling Kampus",
        jam_layanan="Senin-Jumat, 08.00-15.00",
        kontak="konseling@instiki.ac.id",
    ),
    SupportContact(
        unit="Dosen Wali",
        jam_layanan="Sesuai jadwal bimbingan",
        kontak="Hubungi melalui program studi",
    ),
)

_CRISIS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bbunuh\s+diri\b",
        r"\bmengakhiri\s+hidup\b",
        r"\bakhiri\s+hidup\b",
        r"\b(?:nggak|ngga|gak|tidak)\s+mau\s+hidup\s+lagi\b",
        r"\bmenyakiti\s+diri\b",
        r"\bself\s*-?\s*harm\b",
        r"\btidak\s+ingin\s+hidup\b",
    )
)

_DISTRESS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bstres+\b",
        r"\bdepresi\b",
        r"\bputus\s+asa\b",
        r"\bfrustrasi\b",
        r"\bcemas\s+(?:berlebihan|terus)\b",
        r"\b(?:nggak|ngga|gak|tidak)\s+kuat\s+lagi\b",
        r"\bmau\s+menyerah\b",
        r"\bingin\s+menyerah\b",
        r"\bburn\s*-?\s*out\b",
        r"\bdi\s*-?\s*bully\b",
        r"\bdiintimidasi\b",
        r"\bdilecehkan\b",
        r"\bkonflik\s+dengan\s+(?:dosen|pembimbing|teman)\b",
        r"\bmasalah\s+dengan\s+(?:dosen|pembimbing)\s+pembimbing\b",
        r"\btakut\s+(?:di\s*-?\s*DO|drop\s*out)\b",
        r"\bterancam\s+di\s*-?\s*DO\b",
    )
)


@dataclass(frozen=True)
class SensitivityAssessment:
    level: SensitivityLevel
    contacts: tuple[SupportContact, ...]

    @property
    def bypasses_rag(self) -> bool:
        """True berarti retrieval dan LLM dilewati; balasan empatik yang dikirim."""
        return self.level is not SensitivityLevel.NONE


def detect(text: str) -> SensitivityAssessment:
    """Klasifikasikan tingkat sensitivitas pertanyaan.

    CRISIS diperiksa lebih dulu dan menang atas DISTRESS.
    """
    if any(pattern.search(text) for pattern in _CRISIS_PATTERNS):
        return SensitivityAssessment(SensitivityLevel.CRISIS, CRISIS_CONTACTS)
    if any(pattern.search(text) for pattern in _DISTRESS_PATTERNS):
        return SensitivityAssessment(SensitivityLevel.DISTRESS, DISTRESS_CONTACTS)
    return SensitivityAssessment(SensitivityLevel.NONE, ())
