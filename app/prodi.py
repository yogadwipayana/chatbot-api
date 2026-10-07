"""Program studi INSTIKI dan profil penanya yang diurai dari NIM.

NIM INSTIKI berformat `aaabbddccc`: tiga digit angkatan ("240" = 2024), dua
digit fakultas, dua digit prodi, lalu nomor urut. NIM wajib diisi di widget dan
dikirim utuh (`ChatRequest.nim`, PRD §11, 2026-10-07); API mengurainya sendiri
di sini, jadi profil yang dicatat dan diteruskan ke LLM selalu sesuai NIM-nya.
NIM-nya sendiri hanya masuk `messages.meta` -- tidak pernah ke LLM, ke trace
LangSmith, atau ke log aplikasi.

Profil BUKAN filter retrieval seperti unit. Tidak ada dokumen yang khusus satu
prodi: ketentuan per prodi dan angkatan tertulis sebagai baris di dalam dokumen
umum -- harga sertifikasi bertanda "PRODI: TI, RSK, BD", kurikulum OBE untuk
angkatan 2025 dan 2026 (uji korpus 2026-10-05). Filter akan membuang justru
potongan yang menjawab; profil diteruskan ke LLM supaya ia memilih baris yang
berlaku (`app.rag.prompts.PROFIL_PENANYA`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prodi:
    code: str
    """`bbdd` dari NIM: dua digit fakultas lalu dua digit prodi."""
    name: str
    level: str
    faculty: str
    aliases: tuple[str, ...] = ()
    """Sebutan lain di dokumen resmi: singkatan dan nama lama prodi. Tanpa ini
    LLM tidak tahu bahwa baris "PRODI: TI, RSK" berlaku untuk mahasiswa
    Informatika. Selaras dengan nama lama di `app.rag.glossary`."""


DAFTAR_PRODI: tuple[Prodi, ...] = (
    Prodi(
        "1010",
        "Informatika",
        "S1",
        "Fakultas Teknik Informatika",
        ("Teknik Informatika", "TI"),
    ),
    Prodi(
        "1020",
        "Rekayasa Sistem Komputer",
        "S1",
        "Fakultas Teknik Informatika",
        ("Sistem Komputer", "RSK"),
    ),
    Prodi(
        "2010",
        "Desain Komunikasi Visual",
        "S1",
        "Fakultas Bisnis dan Desain Kreatif",
        ("DKV",),
    ),
    Prodi("2020", "Bisnis Digital", "S1", "Fakultas Bisnis dan Desain Kreatif", ("BD",)),
    Prodi(
        "0301",
        "Magister Informatika",
        "S2",
        "Pascasarjana",
        ("S2 Informatika", "Magister Teknik Informatika", "MTI"),
    ),
)
"""Urutan tampil di `GET /api/programs`. Prodi baru cukup ditambahkan di sini."""

ANGKATAN_PERTAMA = 2000
"""NIM hanya membawa dua digit tahun, dibaca sebagai 20xx."""

POLA_NIM = r"^\d{10}$"


def cari_prodi(code: str) -> Prodi | None:
    return next((p for p in DAFTAR_PRODI if p.code == code), None)


def urai_nim(nim: str) -> tuple[str, int]:
    """Kode prodi (digit 4-7) dan tahun angkatan (dua digit pertama) dari NIM
    yang sudah cocok `POLA_NIM`. Digit ketiga tidak dipakai. Sama dengan
    `uraiNim` di widget (`client/src/lib/nim.ts`)."""
    return nim[3:7], ANGKATAN_PERTAMA + int(nim[:2])


@dataclass(frozen=True)
class ProfilMahasiswa:
    prodi: Prodi
    angkatan: int

    def keterangan(self) -> str:
        """Satu baris untuk LLM, mis. "Informatika (S1; disebut juga Teknik
        Informatika, TI), Fakultas Teknik Informatika, angkatan 2024"."""
        sebutan = "; ".join(
            bagian
            for bagian in (
                self.prodi.level,
                "disebut juga " + ", ".join(self.prodi.aliases) if self.prodi.aliases else "",
            )
            if bagian
        )
        return f"{self.prodi.name} ({sebutan}), {self.prodi.faculty}, angkatan {self.angkatan}"
