"""Kamus sinonim dan singkatan kampus untuk pencarian fulltext (FR-2).

Postgres FTS mencocokkan leksem, bukan makna: "STIKI" menjadi leksem `stiki`,
sedangkan "INSTIKI" menjadi `instik`. Mahasiswa yang bertanya "akreditasi
STIKI" tidak akan pernah menemukan dokumen yang menulis "INSTIKI" lewat jalur
fulltext, padahal keduanya kampus yang sama (STIKI berganti nama menjadi
INSTIKI lewat Kepmendikbudristek No. 55/E/O/2022). Hal yang sama berlaku untuk
singkatan yang di dokumen ditulis panjang, dan sebaliknya.

Kamus ini sengaja berupa daftar tetap, bukan tugas LLM: dapat diaudit, tanpa
biaya, dan tetap berlaku pada pesan pertama -- saat query rewriting (FR-4)
dilewati. Isinya diambil dari sumber resmi kampus (riset `instiki/`).

Sengaja TIDAK dimasukkan karena bermakna ganda di lingkungan kampus:
- "SP": Semester Pendek atau Surat Peringatan.
- "TI": Teknik Informatika atau Teknologi Informasi.
- "TA": Tugas Akhir atau Tahun Akademik.
"""

from __future__ import annotations

import itertools
import re

KELOMPOK: tuple[tuple[str, ...], ...] = (
    # Nama kampus: nama lama tetap muncul di dokumen lama dan nama UKM.
    (
        "INSTIKI",
        "Institut Bisnis dan Teknologi Indonesia",
        "STIKI",
        "STIKI Indonesia",
        "STMIK STIKOM Indonesia",
    ),
    # Prodi yang berganti nama.
    ("Informatika", "Teknik Informatika"),
    ("Rekayasa Sistem Komputer", "Sistem Komputer", "RSK"),
    ("Desain Komunikasi Visual", "DKV"),
    # Unit dan program.
    (
        "UPS",
        "Unit Pelaksana Sertifikasi",
        # Nama keliru yang pernah tercantum di tabel `units` (lihat migrasi
        # 0012); mahasiswa yang menyalinnya tetap harus menemukan dokumennya.
        "Unit Pelayanan Sertifikasi",
    ),
    ("PLK", "Pembelajaran di Luar Kampus"),
    ("KP", "Kerja Praktik", "Kerja Praktek"),
    ("RPL", "Rekognisi Pembelajaran Lampau"),
    ("KIP Kuliah", "KIP-K", "KIPK"),
    ("PKKMB", "Pengenalan Kehidupan Kampus bagi Mahasiswa Baru"),
    ("LPPM", "Lembaga Penelitian dan Pengabdian kepada Masyarakat"),
    ("LPMI", "Lembaga Penjaminan Mutu Internal"),
    ("INBIS", "Inkubator Bisnis"),
    # Sistem informasi kampus.
    ("SADS", "Sistem Akademik"),
    ("SIPUT", "Sistem Penginputan Nilai"),
    ("SISKEU", "Sistem Informasi Keuangan"),
)
"""Setiap kelompok berisi istilah yang setara; anggota pertama adalah nama
yang berlaku sekarang."""

MAKS_VARIAN = 8
"""Batas jumlah varian query, termasuk query asli. Setiap varian menambah satu
`websearch_to_tsquery` dan satu `ts_rank` pada query fulltext; pertanyaan yang
menyebut banyak istilah sekaligus cukup diwakili varian-varian pertamanya."""


def _kunci(istilah: str) -> str:
    return " ".join(istilah.split()).casefold()


_KELOMPOK_DARI: dict[str, tuple[str, ...]] = {}
for _kelompok in KELOMPOK:
    for _istilah in _kelompok:
        if _kunci(_istilah) in _KELOMPOK_DARI:
            raise ValueError(f"istilah {_istilah!r} muncul di lebih dari satu kelompok")
        _KELOMPOK_DARI[_kunci(_istilah)] = _kelompok


def _pola_istilah(istilah: str) -> str:
    # Spasi di istilah boleh berupa spasi apa pun sejumlah berapa pun.
    return r"\s+".join(re.escape(kata) for kata in istilah.split())


_POLA = re.compile(
    r"(?<!\w)(?:"
    # Terpanjang lebih dulu: "STIKI Indonesia" harus menang atas "STIKI".
    + "|".join(_pola_istilah(i) for i in sorted(_KELOMPOK_DARI, key=len, reverse=True))
    + r")(?!\w)",
    re.IGNORECASE,
)
"""Batas kata dengan lookaround, bukan `\\b`: "KIP-K" diakhiri huruf setelah
tanda hubung, dan "instiki" tidak boleh cocok sebagai "stiki"."""


def cari_istilah(teks: str) -> list[re.Match[str]]:
    """Posisi setiap istilah kamus di `teks`, terpanjang lebih dulu, tanpa tumpang tindih.

    Dipakai `app.rag.fts_query` supaya istilah multi-kata ("Unit Pelaksana
    Sertifikasi") dikirim sebagai frasa, bukan dipecah menjadi kata umum."""
    return list(_POLA.finditer(teks))


def fulltext_variants(query: str) -> list[str]:
    """Query asli ditambah varian yang istilah kampusnya diganti padanannya.

    Query asli selalu di urutan pertama, dan tanpa istilah yang dikenali
    hanya query asli yang dikembalikan -- jalur fulltext tidak berubah untuk
    pertanyaan biasa. Varian "semua istilah memakai nama sekarang" selalu ikut
    walau batas `MAKS_VARIAN` tercapai, karena dokumen terbaru memakai nama itu.
    """
    temuan = list(_POLA.finditer(query))
    if not temuan:
        return [query]

    pilihan = [_KELOMPOK_DARI[_kunci(t.group(0))] for t in temuan]

    def rakit(pengganti: tuple[str, ...]) -> str:
        bagian, awal = [], 0
        for t, ganti in zip(temuan, pengganti, strict=True):
            bagian.append(query[awal : t.start()])
            bagian.append(ganti)
            awal = t.end()
        bagian.append(query[awal:])
        return "".join(bagian)

    kandidat = itertools.chain(
        (query, rakit(tuple(k[0] for k in pilihan))),
        (rakit(kombinasi) for kombinasi in itertools.product(*pilihan)),
    )
    varian: list[str] = []
    terlihat: set[str] = set()
    for teks in kandidat:
        kunci = _kunci(teks)
        if kunci in terlihat:
            continue
        terlihat.add(kunci)
        varian.append(teks)
        if len(varian) == MAKS_VARIAN:
            break
    return varian
