"""Pemecahan dokumen menjadi chunk (FR-1).

Pemecahan mengikuti struktur dokumen, bukan hitungan karakter semata. Alasannya
terlihat pada dokumen prosedur kampus: memecah "Langkah-Langkah Pembayaran
Virtual Account BNI" per 700 karakter menghasilkan potongan yang dimulai dari
"1. Ketik alamat https://ibank.bni.co.id..." tanpa memuat judul "iBank
Personal" di mana pun. Potongan itu berisi jawaban yang benar, tetapi tidak
mengandung satu pun kata yang akan diketik mahasiswa ("iBank", "internet
banking"), sehingga nyaris tidak pernah terambil -- dan bila terambil, tidak
ada yang memberi tahu chatbot itu langkah untuk kanal yang mana.

Karena itu batas chunk diletakkan di judul bagian, dan tiap potongan lanjutan
membawa ulang judul bagiannya. `RecursiveCharacterTextSplitter` dari LangChain
tetap dipakai, tetapi hanya untuk bagian yang memang lebih panjang daripada
satu chunk (PRD §6).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.loader import LoadedLine, LoadedPage

DEFAULT_CHUNK_SIZE = 900
"""Cukup untuk memuat satu prosedur bernomor yang lazim (10-12 langkah) utuh.

Dinaikkan dari 700 setelah batas chunk menjadi struktural: yang menentukan
sekarang adalah judul bagian, dan ukuran ini hanya menjadi pagar untuk bagian
yang kepanjangan.
"""

DEFAULT_CHUNK_OVERLAP = 105

LANJUTAN = "(lanjutan)"
"""Penanda pada judul potongan kedua dan seterusnya dari satu bagian."""


@dataclass(frozen=True)
class PreparedChunk:
    konten: str
    halaman: int
    urutan: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _splitter(chunk_size: int, chunk_overlap: int) -> Any:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _baris_dari_teks(konten: str) -> list[LoadedLine]:
    """Halaman tanpa informasi struktur: tiap baris teks biasa.

    Terjadi untuk `LoadedPage` yang dibuat langsung dari teks (tes, dan jalur
    mana pun yang tidak melewati `load_pdf`).
    """
    return [LoadedLine(baris) for baris in konten.splitlines() if baris.strip()]


def _bagian(
    baris: Sequence[LoadedLine], judul_awal: str | None
) -> list[tuple[str | None, list[LoadedLine]]]:
    """Kelompokkan baris menjadi (judul bagian, isi).

    `judul_awal` adalah judul bagian terakhir dari halaman sebelumnya. Halaman
    yang dibuka oleh baris non-judul adalah lanjutan bagian itu, dan tanpa
    mewariskannya potongan pertama tiap halaman akan kehilangan identitasnya --
    persis masalah yang ingin dihindari pemecahan ini.
    """
    hasil: list[tuple[str | None, list[LoadedLine]]] = []
    for b in baris:
        if b.jenis == "judul":
            hasil.append((b.teks, []))
        else:
            if not hasil:
                hasil.append((judul_awal, []))
            hasil[-1][1].append(b)
    return hasil


def _runtun(baris: Sequence[LoadedLine]) -> Iterator[tuple[bool, list[LoadedLine]]]:
    """Pisahkan isi bagian menjadi runtun tabel dan runtun non-tabel."""
    runtun: list[LoadedLine] = []
    tabel = False
    for b in baris:
        if runtun and (b.jenis == "tabel") != tabel:
            yield tabel, runtun
            runtun = []
        tabel = b.jenis == "tabel"
        runtun.append(b)
    if runtun:
        yield tabel, runtun


def _kemas_tabel(baris: Sequence[LoadedLine], ruang: int) -> list[str]:
    """Kemas baris tabel tanpa pernah memecah satu baris.

    Baris tabel yang terpotong memisahkan nilai dari nama kolomnya, sehingga
    potongannya menjadi deret angka tanpa arti. Baris yang sendirian sudah
    melebihi `ruang` dibiarkan utuh melewati batas -- itu lebih baik daripada
    dipotong di tengah.
    """
    potongan: list[str] = []
    berjalan: list[str] = []
    panjang = 0
    for b in baris:
        tambahan = len(b.teks) + (1 if berjalan else 0)
        if berjalan and panjang + tambahan > ruang:
            potongan.append("\n".join(berjalan))
            berjalan, panjang = [], 0
            tambahan = len(b.teks)
        berjalan.append(b.teks)
        panjang += tambahan
    if berjalan:
        potongan.append("\n".join(berjalan))
    return potongan


def _gabung_kecil(potongan: Sequence[str], ruang: int) -> list[str]:
    """Satukan potongan berdampingan yang masih muat bersama.

    Tanpa ini, bagian yang diawali satu kalimat pengantar lalu disusul tabel
    akan pecah menjadi chunk sepanjang satu kalimat.
    """
    hasil: list[str] = []
    for p in potongan:
        if hasil and len(hasil[-1]) + 1 + len(p) <= ruang:
            hasil[-1] = f"{hasil[-1]}\n{p}"
        else:
            hasil.append(p)
    return hasil


def _potong_bagian(
    judul: str | None,
    isi: Sequence[LoadedLine],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Pecah satu bagian menjadi teks chunk yang siap disimpan."""
    kepala = f"{judul}\n" if judul else ""
    # Judul yang memakan lebih dari separuh chunk lebih berguna sebagai baris
    # isi biasa daripada sebagai awalan yang diulang-ulang.
    if kepala and len(kepala) >= chunk_size // 2:
        isi = [LoadedLine(judul or "", "teks"), *isi]
        judul, kepala = None, ""

    ruang = chunk_size - len(kepala)
    potongan: list[str] = []
    for tabel, runtun in _runtun(isi):
        if tabel:
            potongan.extend(_kemas_tabel(runtun, ruang))
            continue
        blok = "\n".join(b.teks for b in runtun)
        if len(blok) <= ruang:
            potongan.append(blok)
        else:
            pecahan = _splitter(ruang, min(chunk_overlap, ruang // 4)).split_text(blok)
            potongan.extend(p.strip() for p in pecahan if p.strip())

    potongan = _gabung_kecil([p for p in potongan if p.strip()], ruang)
    if not judul:
        return potongan
    return [f"{judul}{'' if i == 0 else f' {LANJUTAN}'}\n{p}" for i, p in enumerate(potongan)]


def split_pages(
    pages: Sequence[LoadedPage],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    metadata: dict[str, Any] | None = None,
) -> list[PreparedChunk]:
    """Pecah tiap halaman menjadi chunk, nomor halaman ikut terbawa.

    Pemecahan dilakukan per halaman, bukan atas seluruh dokumen yang
    disambung, supaya satu chunk tidak pernah mencakup dua halaman -- sitasi
    FE-2 harus menunjuk ke satu halaman yang pasti. Judul bagian tetap
    diwariskan lintas halaman, sehingga bagian yang menyeberang pergantian
    halaman tidak kehilangan identitasnya meski chunk-nya terpisah.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) harus lebih kecil dari chunk_size ({chunk_size})"
        )

    chunks: list[PreparedChunk] = []
    urutan = 0
    judul_berjalan: str | None = None
    for page in pages:
        baris = list(page.baris) or _baris_dari_teks(page.konten)
        for judul, isi in _bagian(baris, judul_berjalan):
            if judul:
                judul_berjalan = judul
            if not isi:
                continue
            for teks in _potong_bagian(
                judul, isi, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ):
                chunks.append(
                    PreparedChunk(
                        konten=teks,
                        halaman=page.halaman,
                        urutan=urutan,
                        metadata={**(metadata or {}), "halaman": page.halaman},
                    )
                )
                urutan += 1
    return chunks


QA_TEMPLATE = "Pertanyaan: {pertanyaan}\nJawaban: {jawaban}"
"""Bentuk chunk entri tanya jawab.

Pertanyaannya ikut disimpan, bukan hanya jawabannya, karena justru kalimat
pertanyaan itulah yang paling mirip dengan yang diketik mahasiswa -- baik bagi
pencarian vektor maupun bagi full-text search.
"""


def split_qa(
    pertanyaan: str,
    jawaban: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    metadata: dict[str, Any] | None = None,
) -> list[PreparedChunk]:
    """Susun satu entri tanya jawab menjadi chunk siap di-embed.

    Jawaban pendek -- sebagian besar entri -- menjadi satu chunk utuh. Jawaban
    panjang dipecah seperti halaman PDF, dan setiap potongannya tetap membawa
    pertanyaannya: tanpa itu, potongan kedua ("...lalu bawa ke loket 3") tidak
    akan pernah cocok dengan pertanyaan mahasiswa mana pun, sehingga separuh
    jawaban diam-diam hilang dari indeks.

    `halaman` selalu 1. Entri tanya jawab tidak berhalaman, tetapi sitasi FR-5
    berformat `[Judul, hal. N]` dan kolom `chunks.halaman` NOT NULL, jadi
    nilainya dibuat tetap alih-alih dibuat pengecualian di banyak tempat.

    Raises:
        ValueError: pertanyaan atau jawaban kosong setelah dirapikan.
    """
    pertanyaan = " ".join(pertanyaan.split())
    jawaban = jawaban.strip()
    if not pertanyaan:
        raise ValueError("pertanyaan kosong")
    if not jawaban:
        raise ValueError("jawaban kosong")

    if len(jawaban) <= chunk_size:
        bagian = [jawaban]
    else:
        pecahan = _splitter(chunk_size, chunk_overlap).split_text(jawaban)
        bagian = [p.strip() for p in pecahan if p.strip()]

    meta = {**(metadata or {}), "halaman": 1}
    return [
        PreparedChunk(
            konten=QA_TEMPLATE.format(pertanyaan=pertanyaan, jawaban=isi),
            halaman=1,
            urutan=urutan,
            metadata=dict(meta),
        )
        for urutan, isi in enumerate(bagian)
    ]
