"""Daftar bab dokumen untuk konteks LLM (T9).

Pertanyaan daftar ("jenis beasiswa apa saja?") menuntut gambaran seluruh
dokumen, sedangkan retrieval hanya mengambil beberapa potongan. Pedoman
beasiswa memuat enam jenis, satu per bab (BAB II-VII), dan TRANSKRIP
Kemahasiswaan hanya menyebut empat di antaranya ("antara lain"). Terukur
2026-10-09: 10 dari 10 jawaban menyebut empat jenis itu saja, padahal potongan
BAB VI dan BAB VII ikut di konteks. Model mengikuti daftar yang tertulis, dan
aturan 3 melarangnya menggabungkan sendiri. Memperbesar top-N atau menambah
keragaman bab tidak menolong: babnya sudah terambil.

Daftar bab disusun dari jejak judul di baris pertama setiap potongan
(`app.ingestion.chunker.PEMISAH_JEJAK`), jadi tidak butuh kolom atau migrasi
baru, dan dokumen lama ikut terlayani tanpa dipecah ulang.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from app.ingestion.chunker import LANJUTAN, PEMISAH_JEJAK

MIN_BAB = 3
"""Dokumen dengan kurang dari tiga bab tidak butuh daftar: potongannya sudah
hampir menggambarkan seluruh isi."""

MAKS_BAB = 40
"""Lebih dari ini hampir pasti bukan daftar bab, melainkan baris isi yang
terbaca sebagai judul, dan hanya memperpanjang konteks."""


@dataclass(frozen=True)
class Bab:
    judul: str
    halaman: int
    """Halaman potongan pertama bab ini -- halaman yang dikutip untuknya."""


@dataclass(frozen=True)
class DaftarBab:
    bab: tuple[Bab, ...]
    bab_potongan: Mapping[str, int]
    """`chunk_id` -> indeks di `bab`. Potongan di luar bab mana pun tidak tercatat."""

    def tersentuh(self, chunk_ids: Iterable[str]) -> set[int]:
        """Bab yang memuat salah satu potongan ini."""
        return {self.bab_potongan[c] for c in chunk_ids if c in self.bab_potongan}

    def teks(self, judul_dokumen: str) -> str:
        """Isi blok konteksnya. Kepalanya sendiri yang menjelaskan cara mengutip,
        bukan penanda `[Judul, hal. N]`: blok ini tidak berasal dari satu halaman."""
        return "\n".join(
            [
                f'Daftar bab dokumen "{judul_dokumen}" (hanya judul bab, isinya tidak '
                "disertakan; kutip dengan halaman bab yang tertulis):",
                *(f"- {b.judul} (hal. {b.halaman})" for b in self.bab),
            ]
        )


def _jejak(kepala: str) -> list[str]:
    baris = kepala.strip().removesuffix(LANJUTAN).rstrip()
    return [bagian.strip() for bagian in baris.split(PEMISAH_JEJAK)]


def susun_daftar_bab(rows: Sequence[Mapping[str, object]]) -> DaftarBab | None:
    """Susun daftar bab dari baris `chunk_id`, `page`, `kepala` (baris pertama
    potongan), terurut menurut posisi potongan di dokumen.

    Bab adalah tingkat teratas jejak judul, menurut urutan kemunculan pertamanya.
    Tingkat teratas yang memayungi separuh dokumen atau lebih diganti
    anak-anaknya: TRANSKRIP memayungi semua bagiannya dengan "KNOWLEDGE BASE
    CHATBOT ...", dan pedoman sertifikasi menaruh bab 1-7 di bawah satu judul.

    None bila hasilnya bukan daftar yang berguna: kurang dari `MIN_BAB` atau
    lebih dari `MAKS_BAB` bab, atau setiap potongan menjadi babnya sendiri.
    Yang terakhir tanda dokumen tanpa judul bagian -- baris pertama potongannya
    kalimat isi, mis. "1. Login pada https://sads.instiki.ac.id ...".
    """
    if not rows:
        return None
    jejak = [(str(r["chunk_id"]), int(r["page"]), _jejak(str(r["kepala"]))) for r in rows]

    jumlah: dict[str, int] = {}
    halaman: dict[str, int] = {}
    anak: dict[str, dict[str, int]] = {}
    for _, page, bagian in jejak:
        atas = bagian[0]
        jumlah[atas] = jumlah.get(atas, 0) + 1
        halaman.setdefault(atas, page)
        if len(bagian) > 1:
            anak.setdefault(atas, {}).setdefault(bagian[1], page)

    dipecah = {
        atas
        for atas, n in jumlah.items()
        if n * 2 >= len(jejak) and len(anak.get(atas, {})) >= 2
    }
    bab: list[Bab] = []
    indeks: dict[tuple[str, ...], int] = {}
    for atas in jumlah:
        if atas in dipecah:
            for judul, page in anak[atas].items():
                indeks[(atas, judul)] = len(bab)
                bab.append(Bab(judul, page))
        else:
            indeks[(atas,)] = len(bab)
            bab.append(Bab(atas, halaman[atas]))

    if not MIN_BAB <= len(bab) <= MAKS_BAB or len(bab) >= len(jejak):
        return None

    bab_potongan: dict[str, int] = {}
    for chunk_id, _, bagian in jejak:
        kunci = tuple(bagian[:2]) if bagian[0] in dipecah else (bagian[0],)
        if kunci in indeks:
            bab_potongan[chunk_id] = indeks[kunci]
    return DaftarBab(tuple(bab), bab_potongan)
