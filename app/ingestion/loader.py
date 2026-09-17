"""Pemuatan dokumen PDF (FR-1).

Ekstraksi memakai PyMuPDF langsung, bukan `PyMuPDFLoader` dari LangChain.
Loader itu hanya mengembalikan teks polos per halaman, sedangkan chunking
sadar-struktur (`chunker.split_pages`) butuh dua hal yang hilang begitu halaman
diratakan menjadi teks polos:

* **flag tebal per baris** -- di dokumen kampus, judul bagian ("ATM BNI",
  "iBank Personal") kerap tebal dengan ukuran huruf yang sama persis dengan
  teks isi, sehingga tidak terdeteksi oleh pendekatan mana pun yang menebak
  judul dari ukuran huruf;
* **tabel sebagai baris utuh** -- teks polos membacanya kolom demi kolom,
  sehingga pasangan pertanyaan-jawaban putus dan nomor urut berhamburan.

Deteksi PDF scan tetap ditulis sendiri: PDF scan menghasilkan chunk kosong yang
diam-diam merusak retrieval -- dokumen terlihat masuk indeks, tetapi tidak
pernah terambil.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

MIN_CHARS_PER_PAGE = 100
"""Di bawah ini, satu halaman dianggap tanpa lapisan teks."""

MAX_EMPTY_PAGE_RATIO = 0.30
"""Bila lebih dari 30% halaman kosong teks, dokumen ditolak sebagai hasil scan."""

MIN_CHARS_PER_PAGE_WAJAR = 400
"""Ambang kepadatan teks satu halaman panduan yang wajar.

Jauh di atas `MIN_CHARS_PER_PAGE`: yang ini bukan syarat kelulusan, hanya batas
untuk memperingatkan admin.
"""

MAX_THIN_PAGE_RATIO = 0.50
"""Bila lebih dari separuh halaman tipis teks, admin diperingatkan."""

MAKS_PANJANG_JUDUL = 120
"""Baris tebal yang lebih panjang dari ini kalimat, bukan judul bagian."""

MAKS_RASIO_JUDUL = 0.60
"""Bila lebih dari 60% baris pada satu halaman tebal, ketebalan tidak lagi
menandai apa pun -- seluruh halaman diperlakukan sebagai teks biasa."""

TEBAL = 2**4
"""Bit `flags` PyMuPDF yang menandai span tercetak tebal."""

_AKHIR_KALIMAT = (".", ";")
"""Baris tebal yang ditutup begini kalimat, bukan judul.

Nomor di awal baris sengaja TIDAK dipakai sebagai penolak. Judul bagian kerap
bernomor ("2. Biro Administrasi Akademik"), sementara langkah prosedur yang
juga bernomor ("1. Masukkan Kartu Anda.") hampir selalu tidak tebal -- dan bila
seluruh langkah kebetulan tebal, `MAKS_RASIO_JUDUL` yang menangkapnya. Tanda
tanya juga dibiarkan lolos: di dokumen tanya jawab, pertanyaan tebal justru
judul bagiannya.
"""


class ScannedPdfError(ValueError):
    """PDF tanpa lapisan teks yang memadai. FR-1: tolak dengan pesan jelas."""


class UnreadablePdfError(ValueError):
    """Berkas tidak dapat dibuka sebagai PDF: rusak, terpotong, atau bukan PDF."""


JenisBaris = Literal["judul", "teks", "tabel"]


@dataclass(frozen=True)
class LoadedLine:
    """Satu baris halaman beserta perannya.

    `tabel` menandai baris yang berasal dari satu baris tabel dan karena itu
    tidak boleh dipecah di tengah: memotongnya membuat nomor terlepas dari
    pertanyaannya.
    """

    teks: str
    jenis: JenisBaris = "teks"


@dataclass(frozen=True)
class LoadedPage:
    halaman: int
    konten: str
    baris: tuple[LoadedLine, ...] = field(default=())
    """Kosong bila halaman dibentuk tanpa informasi struktur (mis. dari tes).
    `split_pages` mundur ke pemecahan berbasis teks polos bila begitu."""


def empty_page_ratio(pages: Sequence[str], min_chars: int = MIN_CHARS_PER_PAGE) -> float:
    """Proporsi halaman yang praktis tidak punya teks."""
    if not pages:
        return 1.0
    empty = sum(1 for page in pages if len(page.strip()) < min_chars)
    return empty / len(pages)


def is_probably_scanned(
    pages: Sequence[str],
    *,
    min_chars: int = MIN_CHARS_PER_PAGE,
    max_empty_ratio: float = MAX_EMPTY_PAGE_RATIO,
) -> bool:
    """Tebakan apakah PDF merupakan hasil scan tanpa OCR."""
    return empty_page_ratio(pages, min_chars) > max_empty_ratio


def peringatan_kepadatan(
    pages: Sequence[str],
    *,
    min_chars: int = MIN_CHARS_PER_PAGE_WAJAR,
    max_thin_ratio: float = MAX_THIN_PAGE_RATIO,
) -> str | None:
    """Peringatan bila dokumen lolos deteksi scan tetapi teksnya tetap tipis.

    Panduan berbasis tangkapan layar adalah kasus yang sering lolos: tiap
    halaman punya satu-dua kalimat keterangan -- cukup untuk melewati ambang
    `MIN_CHARS_PER_PAGE` -- sementara langkah sebenarnya ada di dalam gambar,
    yang tidak terlihat chatbot. Dokumen seperti ini tetap diterima (menolaknya
    akan salah), tetapi admin perlu tahu bahwa jawaban akan tipis.
    """
    if not pages:
        return None
    tipis = sum(1 for page in pages if len(page.strip()) < min_chars)
    if tipis / len(pages) <= max_thin_ratio:
        return None
    rata = sum(len(p.strip()) for p in pages) // len(pages)
    return (
        f"Teks yang terbaca sangat sedikit (rata-rata {rata} karakter per halaman "
        f"dari {len(pages)} halaman). Bila isi dokumen sebagian besar berupa "
        "gambar atau tangkapan layar, chatbot tidak dapat membacanya. "
        "Pertimbangkan menambahkan keterangan teks pada tiap langkah."
    )


def _rapikan(teks: str) -> str:
    """Satukan spasi ganda dan sisa lipatan baris dari PDF."""
    return " ".join(teks.split())


def _sel(nilai: Any) -> str:
    return _rapikan(str(nilai or "").replace("**", ""))


MAKS_PANJANG_HEADER = 40
"""Nama kolom lebih panjang dari ini hampir pasti isi, bukan nama kolom."""


def _mungkin_header(baris: Sequence[str]) -> bool:
    """Tebakan apakah baris ini nama kolom, bukan baris data pertama.

    Tabel tanpa baris header ada -- jadwal, daftar tarif -- dan memperlakukan
    baris pertamanya sebagai nama kolom akan melabeli seluruh tabel dengan
    "Senin: Selasa". Nama kolom praktis selalu pendek, terisi, dan tanpa angka;
    angka di baris pertama menandakan itu sudah data.
    """
    return bool(baris) and all(
        sel and len(sel) <= MAKS_PANJANG_HEADER and not any(c.isdigit() for c in sel)
        for sel in baris
    )


def tabel_ke_baris(tabel: Any) -> list[str]:
    """Ubah satu tabel menjadi baris teks yang berdiri sendiri.

    Tiap baris ditulis `Kolom: nilai | Kolom: nilai` alih-alih pipa markdown,
    supaya potongannya tetap bermakna saat dibaca terpisah dari header
    tabelnya -- baik oleh pencarian vektor maupun oleh LLM yang mengutipnya.

    Sel yang di-merge membuat PyMuPDF mengulang nilai yang sama di beberapa
    kolom berturut-turut, dan pengulangan itu tidak selalu sejajar antara baris
    header dan baris isi. Karena itu perataan dilakukan per baris: sel kembar
    yang berdampingan diruntuhkan, sel kosong dibuang, lalu sisanya disejajarkan
    dari kiri. Asumsinya kolom yang kosong ada di sebelah kanan -- pola lazim
    pada borang yang belum terisi.
    """
    try:
        mentah = tabel.extract()
    except Exception:  # pragma: no cover - tabel rusak, lewati saja
        return []

    baris: list[list[str]] = []
    for row in mentah:
        padat: list[str] = []
        for sel in row:
            nilai = _sel(sel)
            if nilai and (not padat or padat[-1] != nilai):
                padat.append(nilai)
        if padat:
            baris.append(padat)

    if not baris:
        return []

    kepala = baris[0]
    punya_kepala = (
        len(baris) > 1
        # Header yang lebih sempit daripada isinya berarti tebakan perataan
        # meleset, dan label yang salah lebih buruk daripada tanpa label.
        and len(kepala) >= max(len(r) for r in baris[1:])
        and _mungkin_header(kepala)
    )

    hasil: list[str] = []
    for row in baris[1:] if punya_kepala else baris:
        if punya_kepala:
            bagian = [f"{k}: {v}" for k, v in zip(kepala, row, strict=False) if v]
        else:
            bagian = list(row)
        if bagian:
            hasil.append(" | ".join(bagian))
    return hasil


def _judul_bagian(teks: str, tebal: bool) -> bool:
    """Baris tebal yang pendek dan tidak ditutup seperti kalimat = judul bagian."""
    return tebal and len(teks) <= MAKS_PANJANG_JUDUL and not teks.endswith(_AKHIR_KALIMAT)


def _baris_halaman(page: Any) -> list[LoadedLine]:
    """Baris satu halaman menurut urutan baca, dengan tabel sudah dirakit."""
    import pymupdf

    try:
        tabel = list(page.find_tables().tables)
    except Exception:  # pragma: no cover - deteksi tabel gagal, lanjut tanpa
        tabel = []
    kotak = [pymupdf.Rect(t.bbox) for t in tabel]

    # (posisi vertikal, posisi horizontal, baris) supaya tabel dan teks biasa
    # dapat diurutkan bersama menurut tata letak halaman.
    tersusun: list[tuple[float, float, LoadedLine]] = []

    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            teks = _rapikan("".join(s["text"] for s in line["spans"]))
            if not teks:
                continue
            x0, y0, x1, y1 = line["bbox"]
            titik = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
            if any(k.contains(titik) for k in kotak):
                continue  # dirakit ulang lewat tabel_ke_baris
            spans = [s for s in line["spans"] if s["text"].strip()]
            tebal = bool(spans) and all(s["flags"] & TEBAL for s in spans)
            jenis: JenisBaris = "judul" if _judul_bagian(teks, tebal) else "teks"
            tersusun.append((y0, x0, LoadedLine(teks, jenis)))

    for t, k in zip(tabel, kotak, strict=True):
        for i, isi in enumerate(tabel_ke_baris(t)):
            # Selisih kecil menjaga urutan antarbaris dalam satu tabel tanpa
            # menggeser tabel melewati paragraf sesudahnya.
            tersusun.append((k.y0 + i * 1e-3, k.x0, LoadedLine(isi, "tabel")))

    tersusun.sort(key=lambda b: (b[0], b[1]))
    baris = [b[2] for b in tersusun]

    # Dokumen yang seluruhnya tercetak tebal tidak menandai apa pun dengan
    # ketebalan. Lebih baik kehilangan judul daripada memecah tiap baris.
    jumlah_judul = sum(1 for b in baris if b.jenis == "judul")
    if baris and jumlah_judul > MAKS_RASIO_JUDUL * len(baris):
        baris = [LoadedLine(b.teks, "teks") if b.jenis == "judul" else b for b in baris]
    return baris


def load_pdf(path: str | Path) -> list[LoadedPage]:
    """Muat PDF menjadi daftar halaman, mempertahankan nomor halaman (FR-1).

    Raises:
        ScannedPdfError: bila dokumen terdeteksi hasil scan.
        UnreadablePdfError: bila berkas tidak dapat dibuka sebagai PDF.
    """
    import pymupdf

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dokumen tidak ditemukan: {path}")

    try:
        dokumen = pymupdf.open(str(path))
    except Exception as exc:
        raise UnreadablePdfError(
            f"'{path.name}' tidak dapat dibaca sebagai PDF. Pastikan berkasnya tidak "
            "rusak dan tidak dilindungi kata sandi."
        ) from exc

    with dokumen:
        if dokumen.needs_pass:
            raise UnreadablePdfError(
                f"'{path.name}' dilindungi kata sandi. Unggah versi tanpa proteksi."
            )
        try:
            halaman = [
                LoadedPage(
                    halaman=nomor + 1,
                    konten="\n".join(b.teks for b in baris),
                    baris=tuple(baris),
                )
                for nomor, baris in (
                    (n, _baris_halaman(dokumen[n])) for n in range(dokumen.page_count)
                )
            ]
        except Exception as exc:
            raise UnreadablePdfError(
                f"'{path.name}' tidak dapat dibaca sampai selesai. Berkasnya "
                "kemungkinan rusak atau terpotong."
            ) from exc

    isi = [p.konten for p in halaman]
    if is_probably_scanned(isi):
        raise ScannedPdfError(
            f"'{path.name}' tampaknya hasil scan tanpa lapisan teks "
            f"({empty_page_ratio(isi):.0%} halaman kosong). "
            "Jalankan OCR terlebih dahulu, atau unggah versi digital aslinya."
        )

    return halaman
