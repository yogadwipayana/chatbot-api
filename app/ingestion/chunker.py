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

Judul yang dibawa adalah jejak bertingkat, bukan hanya judul terdekat
(`_JejakJudul`): "BAB IV BEASISWA SATU KELUARGA SATU SARJANA (SKSS) › 4.2
Cakupan Pembiayaan". Tanpa nama BAB, potongan "Cakupan Pembiayaan" tidak
menyebut beasiswa mana yang dibahas -- enam bab pedoman beasiswa memakai
sub-judul yang sama persis (Gambaran Umum, Kuota, Persyaratan), sehingga
pertanyaan tentang satu beasiswa mengambil potongan beasiswa lain.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

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

PEMISAH_JEJAK = " › "
"""Pemisah antartingkat judul pada baris pertama chunk."""

_STRUKTUR: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"BAB\s+(?:[IVXLC]+|\d+)", re.IGNORECASE), 1),
    (re.compile(r"Bagian\s+\w+", re.IGNORECASE), 2),
    (re.compile(r"Pasal\s+\d+[a-z]?", re.IGNORECASE), 3),
)
"""Penanda tingkat dokumen resmi kampus, dari yang tertinggi: BAB > Bagian > Pasal."""

_NOMOR = re.compile(r"(\d{1,3}(?:\.\d{1,3})*)\.?|([A-Z])\.")
"""Nomor bagian: "2", "2.1.", "2.2.1", atau huruf "A." (Buku SKP). Paling banyak
tiga digit, supaya judul yang diawali tahun ("2026 ...") tidak dianggap nomor."""


@dataclass(frozen=True)
class _Judul:
    teks: str
    jenis: Literal["struktur", "nomor", "polos"]
    tingkat: int = 0
    """Hanya untuk `struktur`: 1 = BAB, 2 = Bagian, 3 = Pasal."""
    nomor: str = ""
    """Hanya untuk `nomor`: "2.1", "A"."""

    @property
    def kedalaman(self) -> int:
        return self.nomor.count(".") + 1


def _batas_kata(teks: str, i: int) -> bool:
    return teks[i : i + 1] in ("", " ")


def _penanda(teks: str) -> tuple[_Judul, bool] | None:
    """(judul, hanya_penanda) bila baris diawali penanda tingkat, selain itu None.

    `hanya_penanda` = barisnya cuma penanda ("BAB II", "2.10", "Pasal 3"): judul
    bagiannya tercetak di baris berikut dan harus disambungkan ke sini.
    """
    for pola, tingkat in _STRUKTUR:
        cocok = pola.match(teks)
        if cocok and _batas_kata(teks, cocok.end()):
            return _Judul(teks, "struktur", tingkat=tingkat), cocok.end() == len(teks)
    cocok = _NOMOR.match(teks)
    if cocok and _batas_kata(teks, cocok.end()):
        nomor = cocok.group(1) or cocok.group(2)
        return _Judul(teks, "nomor", nomor=nomor), cocok.end() == len(teks)
    return None


def _kapital(teks: str) -> bool:
    huruf = [c for c in teks if c.isalpha()]
    return len(huruf) >= 2 and all(c.isupper() for c in huruf)


def _pecah_runtun_judul(runtun: Sequence[str]) -> list[_Judul]:
    """Baris judul berturut-turut (tanpa isi di antaranya) -> daftar judul.

    PDF memecah satu judul menjadi beberapa baris: "BAB II" / "BEASISWA KIP
    KULIAH", "2.10" / "Mekanisme Pendaftaran", atau judul panjang yang terlipat.
    Baris tanpa penanda disambung ke baris sebelumnya bila sebelumnya hanya
    penanda, atau bila gaya hurufnya sama (sama-sama kapital) -- tanda judul
    yang terlipat. Selain itu ia judul tersendiri.
    """
    hasil: list[tuple[_Judul, list[str], bool]] = []
    for teks in runtun:
        penanda = _penanda(teks)
        if penanda is not None:
            hasil.append((penanda[0], [teks], penanda[1]))
            continue
        if hasil:
            judul, baris, hanya_penanda = hasil[-1]
            if hanya_penanda or _kapital(teks) == _kapital(baris[-1]):
                hasil[-1] = (judul, [*baris, teks], False)
                continue
        hasil.append((_Judul(teks, "polos"), [teks], False))
    return [
        _Judul(" ".join(baris), judul.jenis, judul.tingkat, judul.nomor)
        for judul, baris, _ in hasil
    ]


@dataclass
class _JejakJudul:
    """Jejak judul yang sedang berlaku, dibawa lintas halaman.

    Aturan saat judul baru datang (`_lepas`):

    * BAB/Bagian/Pasal menggantikan semua yang setingkat atau lebih rendah.
    * Nomor menggantikan nomor sekedalaman ("2.3" menggantikan "2.2" beserta
      anaknya) dan nomor lain yang bukan awalannya ("3.7" menggantikan "2." >
      "2.3"), dan menjadi anak nomor yang menjadi awalannya ("2" > "2.1").
      Daftar bernomor di dalam sub-bagian tetap di bawahnya ("3.7" > "1.",
      lalu "3.7" > "2."), sampai "3.8" menggantikan keduanya.
    * Judul tanpa penanda ("ATM BNI", lalu "Mobile Banking") saling
      menggantikan, tetapi tetap berada di bawah BAB atau nomor di atasnya.
    * Judul KAPITAL tanpa penanda ("REKTOR INSTITUT ...", "SATUAN KREDIT
      PARTISIPASI") mengosongkan jejak. Di dokumen kampus, sub-judul di dalam
      satu BAB selalu bernomor atau berupa Pasal, jadi judul seperti itu
      menandai bagian besar baru -- blok tanda tangan, lampiran, atau buku
      panduan yang disatukan sesudah SK. Tanpa aturan ini, isi Buku SKP akan
      tercatat di bawah "BAB VIII KETENTUAN PENUTUP" milik SK-nya.

    Di dalam satu runtun, judul kedua dan seterusnya menjadi anak judul
    sebelumnya ("BAB IV ..." / "4.1 ...", "3.7 Cakupan Beasiswa" / "1. Biaya
    Pendidikan:"), kecuali BAB/Bagian/Pasal yang selalu menempati tingkatnya
    sendiri ("Lampiran SK ..." / "BAB I" -- BAB I bukan anak lampiran).
    """

    tumpukan: list[_Judul] = field(default_factory=list)
    tertunda: list[str] = field(default_factory=list)
    """Baris judul yang belum disusul isi. Bisa menyeberang halaman: "BAB II"
    di dasar halaman, judul bab dan "2.1 ..." di halaman berikutnya."""

    def tunda(self, teks: str) -> None:
        self.tertunda.append(teks)

    def judul(self) -> str | None:
        """Terapkan judul tertunda, lalu kembalikan jejak lengkapnya."""
        for i, judul in enumerate(_pecah_runtun_judul(self.tertunda)):
            if i == 0 or judul.jenis == "struktur":
                self._lepas(judul)
            self.tumpukan.append(judul)
        self.tertunda = []
        return PEMISAH_JEJAK.join(j.teks for j in self.tumpukan) or None

    def _lepas(self, baru: _Judul) -> None:
        """Buang judul di puncak tumpukan yang tidak lagi menaungi `baru`."""
        t = self.tumpukan
        if baru.jenis == "struktur":
            while t and not (t[-1].jenis == "struktur" and t[-1].tingkat < baru.tingkat):
                t.pop()
        elif baru.jenis == "polos":
            if _kapital(baru.teks):
                t.clear()
            while t and t[-1].jenis == "polos":
                t.pop()
        else:
            if any(j.jenis == "nomor" for j in t):
                while t[-1].jenis == "polos":
                    t.pop()
            saudara_lepas = False
            while t and t[-1].jenis == "nomor":
                atas = t[-1]
                if baru.nomor.startswith(atas.nomor + "."):
                    break
                if saudara_lepas and atas.kedalaman > baru.kedalaman:
                    # "3.7" > "1." lalu "2.": sesudah "1." dilepas, "3.7" adalah
                    # induk daftar bernomor itu, bukan saudara yang digantikan.
                    break
                t.pop()
                saudara_lepas = saudara_lepas or atas.kedalaman == baru.kedalaman


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
    baris: Sequence[LoadedLine], jejak: _JejakJudul
) -> list[tuple[str | None, list[LoadedLine]]]:
    """Kelompokkan baris menjadi (jejak judul, isi).

    `jejak` dibawa dari halaman sebelumnya. Halaman yang dibuka oleh baris
    non-judul adalah lanjutan bagian terakhir, dan tanpa mewariskannya potongan
    pertama tiap halaman akan kehilangan identitasnya -- persis masalah yang
    ingin dihindari pemecahan ini.

    Judul yang langsung disusul judul lain ("BAB II", lalu "2.1 Gambaran Umum")
    tidak menjadi bagian kosong yang dibuang, melainkan masuk ke jejak.
    """
    hasil: list[tuple[str | None, list[LoadedLine]]] = []
    for b in baris:
        if b.jenis == "judul":
            jejak.tunda(b.teks)
            continue
        if jejak.tertunda or not hasil:
            hasil.append((jejak.judul(), []))
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
    jejak = _JejakJudul()
    for page in pages:
        baris = list(page.baris) or _baris_dari_teks(page.konten)
        for judul, isi in _bagian(baris, jejak):
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
    berformat `[Judul, hal. N]` dan kolom `chunks.page` NOT NULL, jadi
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
