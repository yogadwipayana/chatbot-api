"""Deteksi topik berisiko tinggi + eskalasi ke kontak manusia (FR-6, FE-3).

Jawaban pada topik ini SELALU disertai kontak unit resmi. Deteksi berbasis
kata kunci, bukan LLM: hasilnya harus dapat diaudit dan tidak boleh berubah
diam-diam saat model diganti.

Kontak yang ditampilkan mengikuti unit (topik) yang dipilih mahasiswa:
`UNIT_CONTACTS` memuat PIC per unit dari daftar "Kontak PIC per bidang"
(2026-10-12).
Unit yang belum punya PIC di daftar itu (Prodi, Fakultas, Akademik) dan
pertanyaan tanpa unit (Uji coba admin "Semua unit") jatuh ke `DEFAULT_CONTACTS`
per topik, yang semuanya masih Front Office INSTIKI.

Alamat email per unit yang dulu ada di sini (akademik@, keuangan@,
kemahasiswaan@) tidak pernah tercantum di sumber mana pun; banner yang menyebut
alamat yang tidak ada lebih buruk daripada tanpa banner. Jangan menambah kontak
yang tidak berasal dari kampus.
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
    """Kosong bila kampus belum menetapkannya; tampilan melewatinya."""
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

KONTAK_FRONT_OFFICE = "Telepon (0361) 256995 / WhatsApp +62 813-3982-2383"
"""Telepon dari halaman Contact Us dan Brand Guideline; WhatsApp dari daftar
kontak per unit (2026-10-12), menggantikan 0813-3896-9832. Alamat: Jl. Tukad
Pakerisan No. 97, Denpasar."""

JAM_FRONT_OFFICE = "Senin-Sabtu, 08.00-22.00 WITA"
"""Jam layanan di halaman Contact Us instiki.ac.id. Artikel resmi "Perdana!
INSTIKI Terapkan Hybrid Learning" (6 Maret 2022) menyebut jam operasional yang
sama untuk BAAK, luring maupun daring."""

FRONT_OFFICE = UnitContact(
    unit="Front Office INSTIKI",
    jam_layanan=JAM_FRONT_OFFICE,
    kontak=KONTAK_FRONT_OFFICE,
)


def _pic(nama: str, nomor: str) -> UnitContact:
    """Nomor di daftar PIC adalah nomor ponsel; label "WhatsApp" yang membuat
    widget membukanya di WhatsApp, bukan di aplikasi telepon."""
    return UnitContact(unit=nama, jam_layanan="", kontak=f"WhatsApp {nomor}")


UNIT_CONTACTS: dict[str, tuple[UnitContact, ...]] = {
    # Kuncinya nama unit di tabel `units` (pilihan topik di widget). Unit yang
    # diganti namanya dari dashboard harus diganti juga di sini; kalau tidak,
    # kontaknya jatuh ke Front Office.
    "Kemahasiswaan": (
        _pic("PIC Kemahasiswaan", "+62 813-3936-5103"),
        _pic("Bidang Pengembangan Karakter Mahasiswa (Ibu Asti)", "+62 812-3673-3230"),
        _pic("Bidang Prestasi Mahasiswa (Pak Redy)", "+62 819-9843-9707"),
        _pic("Beasiswa (Ibu Suci Ariantini)", "+62 812-3608-8993"),
    ),
    "BAAK": (_pic("PIC BAAK", "+62 813-5346-2032"),),
    "UPS": (_pic("Layanan Pendidikan (UPS)", "+62 813-3734-8654"),),
    "PLK": (_pic("Admin PLK INSTIKI", "+62 851-1931-5346"),),
    "Keuangan": (_pic("Keuangan INSTIKI", "+62 851-7426-1765"),),
    "FO": (FRONT_OFFICE,),
}
"""Kontak PIC per unit, dalam urutan tampil."""


def contacts_for_unit(unit: str | None) -> tuple[UnitContact, ...]:
    """Kontak PIC unit yang dipilih mahasiswa; kosong bila unit itu belum punya."""
    if not unit:
        return ()
    kunci = unit.strip().casefold()
    for nama, kontak in UNIT_CONTACTS.items():
        if nama.casefold() == kunci:
            return kontak
    return ()


DEFAULT_CONTACTS: dict[RiskTopic, UnitContact] = {
    # Dipakai bila unit yang dipilih tidak punya entri di `UNIT_CONTACTS`.
    # Satu entri per topik walau isinya sama: begitu kampus menetapkan unit
    # penanggung jawab per topik, cukup ganti entri topik itu.
    RiskTopic.DEADLINE: FRONT_OFFICE,
    RiskTopic.SYARAT_KELULUSAN: FRONT_OFFICE,
    RiskTopic.PEMBAYARAN: FRONT_OFFICE,
    RiskTopic.SANKSI: FRONT_OFFICE,
    RiskTopic.DROP_OUT: FRONT_OFFICE,
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
