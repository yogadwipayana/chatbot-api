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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, NamedTuple

MIN_CHARS_PER_PAGE = 100
"""Di bawah ini, satu halaman dianggap tanpa lapisan teks."""

MAX_EMPTY_PAGE_RATIO = 0.30
"""Bila lebih dari 30% halaman kosong teks, dokumen ditolak karena tidak memiliki
lapisan teks: hasil scan, atau dicetak dengan Print to PDF yang mengubah setiap
huruf menjadi garis gambar."""

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
    """Tebakan apakah PDF tidak memiliki lapisan teks (hasil scan tanpa OCR, atau
    hasil Print to PDF). Keduanya tampak sama dari sini: halaman tanpa teks."""
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


def _kepala_subtabel(baris: Sequence[str], kepala: Sequence[str]) -> bool:
    """Apakah baris di tengah tabel ini header subtabel, bukan data.

    Satu tabel PDF kerap memuat beberapa subtabel bertumpuk, masing-masing
    dengan header sendiri ("SERTIFIKASI DASAR | HARGA | PRODI", lalu
    "SERTIFIKASI BIDANG | HARGA | PRODI"). Tampang header saja tidak cukup --
    baris data tanpa angka ("Budi | Ketua") juga lolos `_mungkin_header` --
    jadi header subtabel harus mengulang setidaknya satu nama kolom di posisi
    yang sama. Header yang terulang persis (tabel bersambung) ikut tertangkap.
    """
    return (
        len(baris) == len(kepala)
        and _mungkin_header(baris)
        and any(a.casefold() == b.casefold() for a, b in zip(baris, kepala, strict=True))
    )


TOLERANSI_RENTANG = 1.0
"""Selisih vertikal (pt) yang masih dianggap garis yang sama saat memeriksa
apakah sel gabungan dari baris di atas masih menjangkau baris ini."""

TOLERANSI_KOLOM = 3.0
"""Selisih batas kolom (pt) yang masih dianggap kolom yang sama antara tabel di
dua halaman. Tabel Word yang bersambung bergeser 1-2 pt antarhalaman."""


@dataclass(frozen=True)
class _Sel:
    """Satu sel grid tabel beserta kotaknya di halaman."""

    teks: str
    x0: float
    y0: float
    x1: float
    y1: float

    def irisan(self, lain: _Sel) -> float:
        """Lebar tumpang-tindih horizontal dengan sel lain; <= 0 bila tidak."""
        return min(self.x1, lain.x1) - max(self.x0, lain.x0)


@dataclass(frozen=True)
class EkorTabel:
    """Header dan nilai baris data terakhir sebuah tabel.

    Dipakai bila tabel bersambung ke halaman berikutnya. Halaman baru bisa
    mengulang header-nya atau tidak, dan sel yang di-merge ke bawah terpotong
    pergantian halaman sehingga di halaman baru tampil kosong.
    """

    kepala: tuple[_Sel, ...]
    nilai: tuple[str, ...]


class TabelRakitan(NamedTuple):
    baris: list[str]
    ekor: EkorTabel | None


def _padat(isi: Sequence[str]) -> list[str]:
    """Isi baris tanpa sel kosong dan tanpa ulangan nilai sel gabungan."""
    hasil: list[str] = []
    for nilai in isi:
        if nilai and (not hasil or hasil[-1] != nilai):
            hasil.append(nilai)
    return hasil


def _sel_per_baris(tabel: Any) -> list[list[_Sel]]:
    """Sel tiap baris grid, tanpa posisi yang tertutup sel gabungan (`None`).

    PyMuPDF kadang mengulang nilai sel gabungan ke tiap kolom grid yang
    dilewatinya; sel kembar yang berdampingan disatukan menjadi satu sel
    selebar gabungannya.
    """
    hasil: list[list[_Sel]] = []
    for row, isi in zip(tabel.rows, tabel.extract(), strict=True):
        sel: list[_Sel] = []
        for kotak, nilai in zip(row.cells, isi, strict=True):
            if kotak is None:
                continue
            baru = _Sel(_sel(nilai), *kotak)
            if sel and baru.teks and sel[-1].teks == baru.teks:
                lama = sel[-1]
                sel[-1] = _Sel(
                    lama.teks, lama.x0, min(lama.y0, baru.y0), baru.x1, max(lama.y1, baru.y1)
                )
            else:
                sel.append(baru)
        hasil.append(sel)
    return hasil


def _jajarkan(sel: Sequence[_Sel], kepala: Sequence[_Sel]) -> dict[int, _Sel] | None:
    """Pasangkan tiap sel dengan kolom header yang paling lebar ditumpanginya.

    `None` bila ada sel bertulisan yang tidak berada di bawah header mana pun,
    atau dua isi berbeda jatuh ke kolom yang sama: tanda grid tabel tidak
    sejajar dengan header-nya.
    """
    hasil: dict[int, _Sel] = {}
    for s in sel:
        lebar, k = max((s.irisan(h), k) for k, h in enumerate(kepala))
        if lebar <= 0:
            if s.teks:
                return None
            continue
        lama = hasil.get(k)
        if lama is not None and lama.teks and s.teks and lama.teks != s.teks:
            return None
        if lama is None or not lama.teks:
            hasil[k] = s
    return hasil


def _kolom_sama(sel: Sequence[_Sel], kepala: Sequence[_Sel]) -> bool:
    """Apakah batas sel baris ini sama dengan batas kolom header."""
    return len(sel) == len(kepala) and all(
        abs(a.x0 - b.x0) <= TOLERANSI_KOLOM and abs(a.x1 - b.x1) <= TOLERANSI_KOLOM
        for a, b in zip(sel, kepala, strict=True)
    )


def tabel_ke_baris(tabel: Any, lanjutan: EkorTabel | None = None) -> TabelRakitan:
    """Ubah satu tabel menjadi baris teks yang berdiri sendiri.

    Tiap baris ditulis `Kolom: nilai | Kolom: nilai` alih-alih pipa markdown,
    supaya potongannya tetap bermakna saat dibaca terpisah dari header
    tabelnya -- baik oleh pencarian vektor maupun oleh LLM yang mengutipnya.

    Sel dipasangkan dengan kolom header menurut letak horizontalnya, bukan
    urutannya, karena sel kosong di tengah baris dan sel yang di-merge ke bawah
    (rowspan) membuat urutan tidak lagi sejajar. Sel yang di-merge ke bawah
    diwariskan ke tiap baris yang dijangkaunya, supaya "Anggota | 20" tetap
    membawa kegiatan dan tingkatnya (T52). Baris yang selnya tidak dapat
    dijajarkan mundur ke perataan dari kiri, dengan asumsi kolom yang kosong
    ada di sebelah kanan -- pola lazim pada borang yang belum terisi.

    `lanjutan` adalah ekor tabel di halaman sebelumnya. Tabel ini dianggap
    sambungannya bila header-nya sama, atau bila tanpa header tetapi batas
    kolomnya sama. Kolom kosong di awal baris data pertama lalu diisi dari situ.
    """
    try:
        semua = [r for r in _sel_per_baris(tabel) if r]
    except Exception:  # pragma: no cover - tabel rusak, lewati saja
        return TabelRakitan([], None)

    isi = [_padat([s.teks for s in r]) for r in semua]
    berisi = [i for i, t in enumerate(isi) if t]
    if not berisi:
        return TabelRakitan([], None)

    kepala = [s for s in semua[berisi[0]] if s.teks]
    awal = berisi[0] + 1  # baris data pertama
    sambung = None
    if lanjutan is not None:
        if [s.teks.casefold() for s in kepala] == [s.teks.casefold() for s in lanjutan.kepala]:
            sambung = lanjutan
        elif _kolom_sama(semua[berisi[0]], lanjutan.kepala):
            kepala, awal, sambung = list(lanjutan.kepala), berisi[0], lanjutan
    nama = [s.teks for s in kepala]

    punya_kepala = (
        sambung is not None
        or (
            len(berisi) > 1
            # Header yang lebih sempit daripada isinya berarti tebakan perataan
            # meleset, dan label yang salah lebih buruk daripada tanpa label.
            and len(kepala) >= max(len(isi[i]) for i in berisi[1:])
            and _mungkin_header(nama)
        )
    )
    if not punya_kepala:
        return TabelRakitan([" | ".join(isi[i]) for i in berisi], None)

    hasil: list[str] = []
    atas: dict[int, _Sel] = {}  # sel terakhir tiap kolom, untuk sel gabungan ke bawah
    terakhir: dict[int, str] = {}
    for row, teks in zip(semua[awal:], isi[awal:], strict=True):
        if teks and _kepala_subtabel(teks, nama):
            # Berlaku untuk baris sesudahnya; header sendiri bukan data.
            kepala = [s for s in row if s.teks]
            nama = [s.teks for s in kepala]
            atas, terakhir, sambung = {}, {}, None
            continue
        milik = _jajarkan(row, kepala)
        if milik is None:
            bagian = [f"{k}: {v}" for k, v in zip(nama, teks, strict=False)]
            if bagian:
                hasil.append(" | ".join(bagian))
            continue

        y0 = min(s.y0 for s in row)
        sel = {k: s for k, s in atas.items() if s.y1 > y0 + TOLERANSI_RENTANG} | milik
        atas.update(milik)
        if not teks:
            continue  # nilai warisan saja bukan baris data

        nilai = {k: s.teks for k, s in sel.items() if s.teks}
        if sambung is not None:
            for k, warisan in enumerate(sambung.nilai):
                if nilai.get(k):
                    break
                if warisan:
                    nilai[k] = warisan
                    if k in sel:  # sel kosong itu menjangkau baris sesudahnya juga
                        atas[k] = replace(sel[k], teks=warisan)
            sambung = None
        hasil.append(" | ".join(f"{nama[k]}: {v}" for k, v in sorted(nilai.items())))
        terakhir = nilai

    ekor = EkorTabel(tuple(kepala), tuple(terakhir.get(k, "") for k in range(len(kepala))))
    return TabelRakitan(hasil, ekor if terakhir else None)


def _judul_bagian(teks: str, tebal: bool) -> bool:
    """Baris tebal yang pendek dan tidak ditutup seperti kalimat = judul bagian."""
    return tebal and len(teks) <= MAKS_PANJANG_JUDUL and not teks.endswith(_AKHIR_KALIMAT)


def _tabel_halaman(page: Any) -> list[Any]:
    """Tabel di halaman menurut urutan baca, diutamakan dari garis tegas saja.

    Tabel Word yang dicetak ke PDF kerap memberi tiap sel kotak latar putih.
    Strategi bawaan PyMuPDF ikut membaca tepi kotak itu sebagai garis, sehingga
    grid pecah: satu sel "Pengurus Inti" terbelah menjadi beberapa baris dan
    kolom, dan sel yang di-merge ke bawah tidak lagi tampak sebagai satu sel
    (T52). `lines_strict` hanya memakai garis yang benar-benar digambar. Tabel
    yang hanya tertangkap strategi bawaan tetap dipakai seperti sebelumnya.
    """
    import pymupdf

    def cari(**opsi: Any) -> list[Any]:
        try:
            return list(page.find_tables(**opsi).tables)
        except Exception:  # pragma: no cover - deteksi tabel gagal, lanjut tanpa
            return []

    ketat = cari(strategy="lines_strict")
    kotak = [pymupdf.Rect(t.bbox) for t in ketat]
    tabel = ketat + [t for t in cari() if not any(k.intersects(t.bbox) for k in kotak)]
    return sorted(tabel, key=lambda t: (t.bbox[1], t.bbox[0]))


def _baris_halaman(
    page: Any, lanjutan: EkorTabel | None = None
) -> tuple[list[LoadedLine], EkorTabel | None]:
    """Baris satu halaman menurut urutan baca, dengan tabel sudah dirakit.

    `lanjutan` adalah ekor tabel terakhir halaman sebelumnya, untuk tabel
    pertama halaman ini. Ekor tabel terakhir halaman ini ikut dikembalikan.
    """
    import pymupdf

    tabel = _tabel_halaman(page)
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

    ekor = None
    for nomor, (t, k) in enumerate(zip(tabel, kotak, strict=True)):
        rakitan = tabel_ke_baris(t, lanjutan if nomor == 0 else None)
        ekor = rakitan.ekor
        for i, isi in enumerate(rakitan.baris):
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
    return baris, ekor


def load_pdf(path: str | Path) -> list[LoadedPage]:
    """Muat PDF menjadi daftar halaman, mempertahankan nomor halaman (FR-1).

    Raises:
        ScannedPdfError: bila dokumen tidak memiliki lapisan teks (hasil scan
            atau Print to PDF).
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
            halaman: list[LoadedPage] = []
            ekor = None  # tabel dapat bersambung ke halaman berikutnya
            for nomor in range(dokumen.page_count):
                baris, ekor = _baris_halaman(dokumen[nomor], ekor)
                halaman.append(
                    LoadedPage(
                        halaman=nomor + 1,
                        konten="\n".join(b.teks for b in baris),
                        baris=tuple(baris),
                    )
                )
        except Exception as exc:
            raise UnreadablePdfError(
                f"'{path.name}' tidak dapat dibaca sampai selesai. Berkasnya "
                "kemungkinan rusak atau terpotong."
            ) from exc

    isi = [p.konten for p in halaman]
    if is_probably_scanned(isi):
        # "Print to PDF" di Windows kerap menggambar ulang setiap huruf sebagai
        # garis vektor. Hasilnya tidak berisi gambar sama sekali, jadi menyebutnya
        # "hasil scan" saja menyesatkan admin yang tahu dokumennya bukan scan.
        raise ScannedPdfError(
            f"'{path.name}' tidak memiliki lapisan teks (hasil scan atau dicetak "
            f"dengan Print to PDF); {empty_page_ratio(isi):.0%} halamannya kosong. "
            "Simpan ulang dari aplikasi aslinya dengan Save as PDF, atau jalankan "
            "OCR bila dokumennya hasil scan."
        )

    return halaman
