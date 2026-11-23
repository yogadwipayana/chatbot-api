"""Deteksi topik berisiko tinggi + eskalasi ke kontak manusia (FR-6, FE-3).

Jawaban pada topik ini SELALU disertai kontak unit resmi. Deteksi berbasis
kata kunci, bukan LLM: hasilnya harus dapat diaudit dan tidak boleh berubah
diam-diam saat model diganti.

Kontak yang ditampilkan mengikuti unit (topik) yang dipilih mahasiswa: PIC per
unit disimpan di tabel `unit_contacts` dan dikelola superadmin dari halaman Unit
di dashboard (`app.units`; isi awalnya daftar "Kontak PIC per bidang",
2026-10-12). Router chat memuatnya sebelum pipeline berjalan.
Unit yang belum punya PIC (Prodi, Fakultas) dan pertanyaan tanpa unit (Uji coba
admin "Semua unit") jatuh ke `DEFAULT_CONTACTS` per topik, yang semuanya masih
Front Office INSTIKI. Cadangan ini sengaja tetap di kode: ia tidak boleh ikut
hilang bila tabel kontak kosong.

Tidak semua kontak unit cocok untuk setiap pertanyaan (`pilih_kontak`, 2026-11-23):

- Kontak berkata kunci hanya tampil bila pertanyaannya menyebut salah satu kata
  kuncinya. Admin PLK di unit Akademik hanya untuk urusan PLK; urusan surat
  dan administrasi lain cukup ke BAAK sebagai pintu pertama, supaya mahasiswa
  tidak salah kontak.
- Pengambilan mata kuliah diarahkan ke dosen Pembimbing Akademik (`DOSEN_PA`):
  mata kuliah di KRS ditentukan dengan persetujuan dosen PA (Buku Pedoman
  Akademik FTI §4.7). Dosen PA berbeda untuk tiap mahasiswa, jadi tidak ada
  nomor yang dapat disimpan di tabel kontak.

Alamat email per unit yang dulu ada di sini (akademik@, keuangan@,
kemahasiswaan@) tidak pernah tercantum di sumber mana pun; banner yang menyebut
alamat yang tidak ada lebih buruk daripada tanpa banner. Jangan menambah kontak
yang tidak berasal dari kampus.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache


class RiskTopic(StrEnum):
    DEADLINE = "deadline"
    SYARAT_KELULUSAN = "syarat_kelulusan"
    PEMBAYARAN = "pembayaran"
    SANKSI = "sanksi"
    DROP_OUT = "drop_out"


@dataclass(frozen=True)
class UnitContact:
    """Isi banner eskalasi FE-3. Kontak PIC dari database memakai bentuk yang
    sama (`app.units.ContactRecord.untuk_tampil`)."""

    unit: str
    jam_layanan: str
    """Kosong bila kampus belum menetapkannya; tampilan melewatinya."""
    kontak: str
    kata_kunci: tuple[str, ...] = ()
    """Kosong = selalu tampil untuk unitnya. Terisi = hanya bila pertanyaan
    menyebut salah satunya (`pilih_kontak`). Tidak pernah tampil ke mahasiswa."""


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


DEFAULT_CONTACTS: dict[RiskTopic, UnitContact] = {
    # Dipakai bila unit yang dipilih tidak punya kontak di `unit_contacts`.
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


# --- Kontak mana yang relevan untuk pertanyaan ini ------------------------------

DOSEN_PA = UnitContact(
    unit="Dosen Pembimbing Akademik (PA) Anda",
    jam_layanan="",
    kontak="hubungi langsung, karena mata kuliah di KRS diambil dengan persetujuan dosen PA",
)
"""Tanpa nomor: dosen PA berbeda untuk tiap mahasiswa, dan SADS tidak
menyediakannya. Dasarnya Buku Pedoman Akademik FTI §4.7 ("Penentuan mata
kuliah dalam KRS ... dilakukan oleh mahasiswa dengan persetujuan dosen
Pembimbing Akademik") dan bagian registrasi akademiknya (pengesahan KRS oleh
dosen PA), halaman 14 di indeks."""

_AMBIL = r"(?:ambil|mengambil|diambil|ngambil|pengambilan|kontrak|mengontrak|dikontrak)"
"""Kata kerja yang boleh berjarak beberapa kata dari mata kuliahnya: "ambil
berapa sks", "mata kuliah apa yang bisa saya ambil"."""

_UBAH = (
    r"(?:tambah|menambah|nambah|penambahan|drop|ngedrop|batal|membatalkan|pembatalan"
    r"|ganti|mengganti|penggantian|mengulang|pengulangan)"
)
"""Kata kerja yang harus langsung diikuti mata kuliahnya. "ganti jadwal mata
kuliah" atau "ujian ulang" bukan urusan dosen PA."""

_MK = r"(?:mata\s*kuliah|matkul|makul|mk|sks)(?:nya)?"

_POLA_DOSEN_PA: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pola, re.IGNORECASE)
    for pola in (
        rf"\b{_AMBIL}(?:nya)?(?:\s+[\w-]+){{0,3}}?\s+{_MK}\b",
        rf"\b{_MK}(?:\s+[\w-]+){{0,6}}?\s+{_AMBIL}(?:nya)?\b",
        rf"\b{_UBAH}\s+{_MK}\b",
        r"\bperwalian\b",
        r"\b(?:konsultasi|persetujuan|pengesahan|acc)\s+(?:krs|rencana\s+studi)\b",
        r"\bkrs(?:\s+[\w-]+){0,3}?\s+(?:di)?(?:acc|setujui|disetujui|sahkan|disahkan)\b",
        r"\bsks\s+maks(?:imal|imum)?\b",
        r"\bmaks(?:imal|imum)?\s+sks\b",
        r"\bbeban\s+studi\b",
    )
)


def butuh_dosen_pa(teks: str) -> bool:
    """Pertanyaan soal pengambilan mata kuliah (KRS, perwalian, beban SKS)."""
    return any(pola.search(teks) for pola in _POLA_DOSEN_PA)


@lru_cache(maxsize=256)
def _pola_kata_kunci(kata: str) -> re.Pattern[str]:
    """Utuh per kata, tanpa membedakan huruf besar, spasi di dalamnya longgar:
    "PLK" tidak cocok di "kaplk", "luar  kampus" sama dengan "luar kampus"."""
    isi = r"\s+".join(re.escape(bagian) for bagian in kata.split())
    return re.compile(rf"(?<!\w){isi}(?!\w)", re.IGNORECASE)


def menyebut(teks: str, kata_kunci: Iterable[str]) -> bool:
    return any(kata.strip() and _pola_kata_kunci(kata).search(teks) for kata in kata_kunci)


def pilih_kontak(
    kontak_unit: Iterable[UnitContact],
    assessment: RiskAssessment,
    teks: str,
) -> tuple[UnitContact, ...]:
    """Kontak untuk penolakan dan banner eskalasi; kosong = pemanggil memakai
    Front Office.

    1. Kontak unit berkata kunci hanya ikut bila `teks` menyebut salah satunya.
    2. Pengambilan mata kuliah: dosen PA lebih dulu, menggantikan kontak umum
       unit (mis. BAAK). Kontak berkata kunci yang cocok tetap ikut -- ia lebih
       spesifik. Bila pertanyaannya juga topik berisiko (tenggat, biaya, sanksi,
       DO, syarat lulus), kontak umum tetap ikut: dosen PA tidak memutuskan
       batas KRS atau KRS yang terkunci karena tagihan.
    3. Selain itu kontak unit yang relevan, atau kontak per topik risiko bila
       unit tidak punya.

    `teks` = pertanyaan, ditambah tulisan ulang dari riwayat bila ada, supaya
    "syaratnya apa?" sesudah bertanya soal PLK tetap dianggap urusan PLK.
    """
    relevan = tuple(k for k in kontak_unit if not k.kata_kunci or menyebut(teks, k.kata_kunci))
    if not butuh_dosen_pa(teks):
        return relevan or assessment.contacts
    hasil = [DOSEN_PA, *(k for k in relevan if k.kata_kunci)]
    if assessment.needs_escalation:
        umum = tuple(k for k in relevan if not k.kata_kunci) or assessment.contacts
        hasil.extend(k for k in umum if k not in hasil)
    return tuple(hasil)
