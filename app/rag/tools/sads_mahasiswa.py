"""Tool SADS data pribadi: nilai, KRS, skripsi, biaya, pembayaran, semester aktif.

Setiap endpoint di sini menerima NIM di path. NIM itu SELALU dari request
(`Penanya`, `ChatRequest.nim`), tidak pernah dari model: tidak ada tool yang
punya argumen `nim` (`ARGUMEN_TERLARANG`), sehingga "nilai NIM 2401010001
berapa?" tetap hanya mengambil data penanya sendiri. NIM dan nama mahasiswa
tidak ikut ke teks untuk model (`app.prodi`: NIM tidak pernah ke LLM).

NIM di widget belum diverifikasi: siapa pun yang memanggil `POST /api/chat`
dengan NIM orang lain dapat membaca datanya. Risiko itu diterima pengguna untuk
dev/staging pada 2026-11-07 (docs/tool-call.md §11).

Jumlah, IPK, sisa SKS, dan selisih tanggal dihitung di sini, bukan oleh model
(T46: LLM tidak andal menghitung).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import partial
from typing import Any

import httpx

from app.prodi import POLA_NIM
from app.rag.tools.base import Lampiran, Penanya, ToolResult, ToolSpec
from app.rag.tools.client import SadsClient
from app.rag.tools.sads import LABEL_SADS, _bersih

PATH_LAYANAN = "/service/tp/chatbot"

WITA = timezone(timedelta(hours=8))
"""Zona waktu kampus. Tanpa musim panas, jadi offset tetap cukup."""

KEPALA_PRIBADI = (
    "Data pribadi milik mahasiswa yang sedang bertanya (bukan data mahasiswa lain)."
)

GALAT_TANPA_NIM = "NIM penanya tidak diketahui, jadi data pribadinya tidak dapat diambil."
GALAT_NIM_TIDAK_TERDAFTAR = "NIM penanya tidak terdaftar sebagai mahasiswa di SADS."

_BULAN = (
    "Januari",
    "Februari",
    "Maret",
    "April",
    "Mei",
    "Juni",
    "Juli",
    "Agustus",
    "September",
    "Oktober",
    "November",
    "Desember",
)

_URUTAN_HURUF = ("A", "AB", "B", "BC", "C", "CD", "D", "E")


def hari_ini() -> date:
    """Tanggal hari ini di kampus. Fungsi tersendiri supaya test dapat menggantinya."""
    return datetime.now(WITA).date()


# --- Format ------------------------------------------------------------------


def _rupiah(nilai: Any) -> str:
    if isinstance(nilai, bool) or not isinstance(nilai, (int, float)):
        return "-"
    return "Rp" + f"{round(nilai):,}".replace(",", ".")


def _angka(nilai: Any) -> str:
    """81.6 -> "81,6"; 87.0 -> "87"."""
    if isinstance(nilai, bool) or not isinstance(nilai, (int, float)):
        return "-"
    teks = f"{nilai:.2f}".rstrip("0").rstrip(".")
    return teks.replace(".", ",")


def _ya(nilai: Any) -> str:
    if nilai is True:
        return "ya"
    if nilai is False:
        return "tidak"
    return "tidak diketahui"


def _waktu(teks: Any) -> datetime | None:
    if not isinstance(teks, str):
        return None
    for pola in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(teks.strip(), pola)
        except ValueError:
            continue
    return None


def _tanggal(teks: Any) -> str:
    """ "2026-09-12 21:00:00" -> "12 September 2026 pukul 21.00"."""
    w = _waktu(teks)
    if w is None:
        return _bersih(teks) if isinstance(teks, str) else "-"
    hasil = f"{w.day} {_BULAN[w.month - 1]} {w.year}"
    if (w.hour, w.minute) != (0, 0):
        hasil += f" pukul {w.hour:02d}.{w.minute:02d}"
    return hasil


def _selisih(teks: Any, hari: date) -> str:
    """Jarak tanggal itu dari hari ini, dihitung di sini, bukan oleh model."""
    w = _waktu(teks)
    if w is None:
        return ""
    n = (w.date() - hari).days
    if n > 0:
        return f"{n} hari lagi"
    if n < 0:
        return f"{-n} hari yang lalu"
    return "hari ini"


def _dengan_selisih(teks: Any, hari: date) -> str:
    selisih = _selisih(teks, hari)
    return f"{_tanggal(teks)} ({selisih})" if selisih else _tanggal(teks)


def _hari_ini_teks(hari: date) -> str:
    return f"Tanggal hari ini: {hari.day} {_BULAN[hari.month - 1]} {hari.year}."


def _int(nilai: Any) -> int | None:
    return nilai if isinstance(nilai, int) and not isinstance(nilai, bool) else None


# --- Ambil -------------------------------------------------------------------


@dataclass(frozen=True)
class _Gagal:
    hasil: ToolResult


async def _ambil(
    client: SadsClient, endpoint: str, penanya: Penanya, nama: str
) -> dict[str, Any] | _Gagal:
    """Data satu endpoint `/{endpoint}/{nim}` untuk penanya, atau hasil gagalnya.

    404 = NIM tidak terdaftar (`{"msg": "Mahasiswa Tidak Ditemukan"}`), bukan
    gangguan layanan. Galat HTTP lain diteruskan ke loop, yang mencatatnya tanpa
    URL (URL memuat NIM)."""
    nim = penanya.nim or ""
    if not re.fullmatch(POLA_NIM, nim):
        return _Gagal(ToolResult(nama, LABEL_SADS, "", ok=False, galat=GALAT_TANPA_NIM))
    try:
        data = await client.get_json(f"{PATH_LAYANAN}/{endpoint}/{nim}")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return _Gagal(
                ToolResult(nama, LABEL_SADS, "", ok=False, galat=GALAT_NIM_TIDAK_TERDAFTAR)
            )
        raise
    if not isinstance(data, dict):
        return _Gagal(ToolResult(nama, LABEL_SADS, "", ok=False))
    return data


async def kode_mk_penanya(client: SadsClient, penanya: Penanya) -> list[str]:
    """Kode mata kuliah yang sudah bernilai bagi penanya; [] bila tidak dapat diambil.

    Dipakai tool kurikulum untuk menyaring ke kurikulum penanya sendiri
    (`sads_kurikulum.jejak_penanya`). Kegagalan apa pun berarti tanpa saringan,
    bukan galat: datanya hanya mempersempit, tidak wajib."""
    try:
        data = await _ambil(client, "krs-mahasiswa", penanya, "kode_mk_penanya")
    except (httpx.HTTPError, ValueError):
        return []
    if isinstance(data, _Gagal):
        return []
    return [n.kdmk for n in _baris_nilai(data) if n.kdmk]


def _profil(data: dict[str, Any]) -> str:
    """Satu baris dari blok `mahasiswa`, TANPA NIM dan nama."""
    m = data.get("mahasiswa")
    if not isinstance(m, dict):
        return ""
    bagian = [
        f"status {_bersih(m.get('status'))}" if m.get("status") else "",
        _bersih(m.get("prodi_nama")),
        f"angkatan {_bersih(m.get('angkatan'))}" if m.get("angkatan") else "",
        f"kelas {_bersih(m.get('jenis_kelas'))}" if m.get("jenis_kelas") else "",
    ]
    isi = ", ".join(b for b in bagian if b)
    return f"Mahasiswa penanya: {isi}." if isi else ""


def _teks(*baris: str) -> str:
    return "\n".join(b for b in baris if b)


Susun = Callable[[dict[str, Any], date], tuple[str, tuple[Lampiran, ...]]]
"""(respons SADS, hari ini) -> (teks untuk model, lampiran)."""


async def _jalankan(
    client: SadsClient,
    *,
    endpoint: str,
    nama: str,
    susun: Susun,
    penanya: Penanya,
) -> ToolResult:
    data = await _ambil(client, endpoint, penanya, nama)
    if isinstance(data, _Gagal):
        return data.hasil
    teks, lampiran = susun(data, hari_ini())
    return ToolResult(
        name=nama,
        label=LABEL_SADS,
        text=_teks(KEPALA_PRIBADI, _profil(data), teks),
        attachments=lampiran,
    )


# --- Nilai (krs-mahasiswa) -----------------------------------------------------


@dataclass(frozen=True)
class _Nilai:
    kdmk: str
    matkul: str
    jenis: str
    sks: int
    grade: str
    na: int | None
    ntot: Any


def _baris_nilai(data: dict[str, Any]) -> list[_Nilai]:
    """Nilai yang ditampilkan: `hide` = 1 dibuang. Mata kuliah yang tercatat lebih
    dari sekali (diulang) dihitung nilai terbaiknya."""
    terbaik: dict[str, _Nilai] = {}
    for b in data.get("nilai") or []:
        if not isinstance(b, dict) or b.get("hide") == 1:
            continue
        n = _Nilai(
            kdmk=_bersih(b.get("kdmk")),
            matkul=_bersih(b.get("matkul")),
            jenis=_bersih(b.get("jenis_matkul")) or "-",
            sks=_int(b.get("sks")) or 0,
            grade=_bersih(b.get("grade")),
            na=_int(b.get("na")),
            ntot=b.get("ntot"),
        )
        kunci = n.kdmk or n.matkul
        lama = terbaik.get(kunci)
        if lama is None or _bobot(n) > _bobot(lama):
            terbaik[kunci] = n
    return sorted(terbaik.values(), key=lambda n: n.matkul.casefold())


def _bobot(n: _Nilai) -> int:
    """Bobot untuk memilih percobaan terbaik; nilai E (bobot 0) tetap di atas
    "belum ada nilai"."""
    return -1 if n.na is None else n.na


def ipk(nilai: list[_Nilai]) -> float | None:
    """Jumlah bobot x SKS dibagi jumlah SKS, dari mata kuliah yang berbobot."""
    pasangan = [(n.na, n.sks) for n in nilai if n.na is not None and n.sks > 0]
    sks = sum(s for _, s in pasangan)
    return sum(na * s for na, s in pasangan) / sks if sks else None


def _sebaran(nilai: list[_Nilai]) -> str:
    hitung = Counter(n.grade or "belum ada" for n in nilai)
    urut = sorted(
        hitung, key=lambda g: (_URUTAN_HURUF.index(g) if g in _URUTAN_HURUF else 99, g)
    )
    return ", ".join(f"{g} {hitung[g]}" for g in urut)


def _sisa(lulus: Any, minimal: Any) -> str:
    a, b = _int(lulus), _int(minimal)
    if a is None or b is None:
        return f"{_angka(lulus)} dari minimal {_angka(minimal)}"
    kurang = b - a
    return f"{a} dari minimal {b} ({'terpenuhi' if kurang <= 0 else f'kurang {kurang}'})"


def _kurang(ref: dict[str, Any], jenis: str) -> int | None:
    """SKS `jenis` (wajib/pilihan) yang belum terpenuhi; None bila datanya tidak lengkap."""
    a, b = _int(ref.get(f"sks_{jenis}_lulus")), _int(ref.get(f"sks_{jenis}_minimal"))
    return None if a is None or b is None else max(0, b - a)


def _susun_nilai(data: dict[str, Any], _hari: date) -> tuple[str, tuple[Lampiran, ...]]:
    nilai = _baris_nilai(data)
    if not nilai:
        return "Belum ada nilai mata kuliah yang tercatat di SADS: 0 mata kuliah.", ()
    nilai_ipk = ipk(nilai)
    ref = data.get("referensi_sks") if isinstance(data.get("referensi_sks"), dict) else {}
    baris = [
        f"Jumlah mata kuliah bernilai: {len(nilai)} ({sum(n.sks for n in nilai)} SKS).",
        (
            "IPK, dihitung dari daftar nilai ini (jumlah bobot x SKS dibagi jumlah SKS): "
            + f"{nilai_ipk:.2f}.".replace(".", ",", 1)
            if nilai_ipk is not None
            else ""
        ),
        f"Sebaran nilai huruf: {_sebaran(nilai)}.",
    ]
    if ref:
        baris.append(
            f"SKS lulus menurut SADS: total {_angka(ref.get('total_sks_lulus'))}; "
            f"wajib {_sisa(ref.get('sks_wajib_lulus'), ref.get('sks_wajib_minimal'))}; "
            f"pilihan {_sisa(ref.get('sks_pilihan_lulus'), ref.get('sks_pilihan_minimal'))}."
        )
        wajib, pilihan = _kurang(ref, "wajib"), _kurang(ref, "pilihan")
        if wajib is not None and pilihan is not None:
            # Jumlahnya dari sini: model yang menjumlahkan sendiri (51 + 9) bisa
            # salah, dan jawabannya tetap tampak bersumber SADS (T46).
            baris.append(
                f"SKS yang masih kurang untuk kelulusan (wajib + pilihan): {wajib + pilihan}."
            )
    baris.append(
        "Daftar nilai (kode, mata kuliah, jenis, SKS: nilai huruf, bobot, nilai angka):"
    )
    baris += [
        f"- {n.kdmk} {n.matkul} ({n.jenis}, {n.sks} SKS): {n.grade or '-'}, "
        f"bobot {_angka(n.na)}, nilai {_angka(n.ntot)}"
        for n in nilai
    ]
    lampiran = Lampiran(
        # "Anda": lampiran tetap tampil di bawah penolakan yang mengutip SADS
        # ("cek IPK teman saya NIM …"), jadi pemiliknya harus jelas (s73).
        title="Daftar nilai Anda",
        source=LABEL_SADS,
        items=tuple(f"{n.matkul} ({n.kdmk}) · {n.sks} SKS · {n.grade or '-'}" for n in nilai),
    )
    return _teks(*baris), (lampiran,)


# --- KRS, revisi KRS, semester aktif -----------------------------------------


def _jadwal(d: dict[str, Any], hari: date) -> str:
    """Satu jendela KRS: periode, buka/tutup, status SADS, dan selisih hari."""
    return _teks(
        f"Periode {_bersih(d.get('periode')) or '-'}: "
        f"dibuka {_dengan_selisih(d.get('tanggal_buka'), hari)}, "
        f"ditutup {_dengan_selisih(d.get('tanggal_tutup'), hari)}.",
        f"Status menurut SADS: {_bersih(d.get('status')) or '-'}; "
        f"masih bisa mengisi: {_ya(d.get('masih_bisa_mengisi'))}.",
    )


def _susun_krs(data: dict[str, Any], hari: date) -> tuple[str, tuple[Lampiran, ...]]:
    return (
        _teks("Jadwal pengisian KRS penanya.", _jadwal(data, hari), _hari_ini_teks(hari)),
        (),
    )


def _susun_revisi(data: dict[str, Any], hari: date) -> tuple[str, tuple[Lampiran, ...]]:
    jenis = _bersih(data.get("jenis_jadwal"))
    keterangan = _bersih(data.get("keterangan"))
    return (
        _teks(
            "Jadwal revisi KRS penanya" + (f" ({jenis})." if jenis else "."),
            _jadwal(data, hari),
            f"Keterangan SADS: {keterangan}" if keterangan else "",
            _hari_ini_teks(hari),
        ),
        (),
    )


_HARI = ("Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu")


def _str(nilai: Any) -> str:
    return _bersih(nilai) if isinstance(nilai, str) else ""


def _jam(nilai: Any) -> str:
    """ "19:50 - 22:20" apa adanya; " - " (jam kosong di SADS) -> ""."""
    teks = _str(nilai)
    return teks if teks.strip(" -") else ""


def _indeks_hari(teks: str) -> int:
    """ "Jumat"/"Jum'at" -> 4; nama yang tidak dikenal diurutkan paling akhir."""
    kunci = re.sub(r"['’]", "", teks).casefold()
    return next((i for i, h in enumerate(_HARI) if h.casefold() == kunci), len(_HARI))


@dataclass(frozen=True)
class _Pertemuan:
    hari: str
    jam: str
    ruang: str

    def teks(self) -> str:
        return " ".join(
            b for b in (self.hari, self.jam, self.ruang and f"ruang {self.ruang}") if b
        )


@dataclass(frozen=True)
class _Kelas:
    kdmk: str
    matkul: str
    kelas: str
    sks: int | None
    mbkm: bool
    dosen: str
    pertemuan: tuple[_Pertemuan, ...]

    def waktu(self) -> str:
        return "; ".join(p.teks() for p in self.pertemuan) or "jadwal belum tercantum"


def _kelas_kuliah(data: dict[str, Any]) -> list[_Kelas]:
    """Jadwal kuliah penanya dari `semester-aktif` (bentuk respons SADS sejak 2026-11).

    Satu kelas bisa punya sampai tiga pertemuan (`hari`, `hari2`, `hari3`);
    pertemuan tanpa hari diabaikan, jam kosong ditulis SADS sebagai " - "."""
    baris = data.get("jadwal_matakuliah")
    if not isinstance(baris, list):
        return []
    hasil = []
    for r in baris:
        if not isinstance(r, dict):
            continue
        pertemuan = tuple(
            _Pertemuan(
                hari=_str(r.get(f"hari{k}")),
                jam=_jam(r.get(f"jam{k}")),
                ruang=_str(r.get(f"ruang{k}")),
            )
            for k in ("", "2", "3")
            if _str(r.get(f"hari{k}"))
        )
        hasil.append(
            _Kelas(
                kdmk=_str(r.get("kdmk")),
                matkul=_str(r.get("matkul")) or "-",
                kelas=_str(r.get("nama_kelas")),
                sks=_int(r.get("sks")),
                mbkm=r.get("is_mbkm") in (1, True, "1"),
                dosen=_str(r.get("nmdosen")),
                pertemuan=pertemuan,
            )
        )
    hasil.sort(
        key=lambda k: (
            min((_indeks_hari(p.hari) for p in k.pertemuan), default=len(_HARI)),
            k.pertemuan[0].jam if k.pertemuan else "",
            k.matkul,
        )
    )
    return hasil


def _baris_kelas(k: _Kelas) -> str:
    ciri = [k.kdmk] if k.kdmk else []
    if k.kelas:
        ciri.append(f"kelas {k.kelas}")
    if k.sks is not None:
        ciri.append(f"{k.sks} SKS")
    if k.mbkm:
        ciri.append("MBKM")
    dosen = f"; dosen {k.dosen}" if k.dosen else ""
    return f"- {k.matkul} ({', '.join(ciri)}): {k.waktu()}{dosen}"


def _jadwal_kuliah(kelas: list[_Kelas], hari: date) -> tuple[list[str], tuple[Lampiran, ...]]:
    if not kelas:
        return [], ()

    def kuliah_pada(label: str, tanggal: date) -> str:
        # Dihitung di sini: model sering keliru mencocokkan hari dengan tanggal.
        nama = [
            k.matkul
            for k in kelas
            if any(_indeks_hari(p.hari) == tanggal.weekday() for p in k.pertemuan)
        ]
        return (
            f"{label.capitalize()} hari {_HARI[tanggal.weekday()]}; kuliah {label}: "
            + (", ".join(nama) if nama else "tidak ada")
            + "."
        )

    sks = sum(k.sks or 0 for k in kelas)
    baris = [
        f"Jadwal kuliah penanya semester ini: {len(kelas)} kelas, {sks} SKS.",
        *(_baris_kelas(k) for k in kelas),
        kuliah_pada("hari ini", hari),
        kuliah_pada("besok", hari + timedelta(days=1)),
    ]
    lampiran = Lampiran(
        title="Jadwal kuliah Anda",
        source=LABEL_SADS,
        items=tuple(
            " · ".join(b for b in (f"{k.matkul} ({k.kdmk})", k.waktu(), k.dosen) if b)
            for k in kelas
        ),
    )
    return baris, (lampiran,)


def _susun_semester(data: dict[str, Any], hari: date) -> tuple[str, tuple[Lampiran, ...]]:
    krs = data.get("jadwal_krs")
    ke = _int(data.get("semester_ke_mahasiswa"))
    mulai, selesai = data.get("tanggal_mulai_semester"), data.get("tanggal_selesai_semester")
    if {_tanggal(mulai), _tanggal(selesai)} <= {"", "-"}:
        # Sejak 2026-11 SADS tidak lagi mengirim tanggal ini. Kalimat jelas
        # supaya model mencari di Kalender Akademik, bukan menulis "- sampai -".
        kuliah = "Tanggal mulai dan selesai perkuliahan semester ini tidak tercantum di SADS."
    else:
        kuliah = (
            f"Perkuliahan semester ini: {_dengan_selisih(mulai, hari)} "
            f"sampai {_dengan_selisih(selesai, hari)}."
        )
    jadwal, lampiran = _jadwal_kuliah(_kelas_kuliah(data), hari)
    return (
        _teks(
            f"Semester aktif: periode {_bersih(data.get('periode')) or '-'}, tahun akademik "
            f"{_bersih(data.get('tahun_akademik')) or '-'}, semester "
            f"{_bersih(data.get('semester')) or '-'}.",
            f"Bagi penanya ini semester ke-{ke}." if ke is not None else "",
            kuliah,
            ("Jadwal KRS semester ini:\n" + _jadwal(krs, hari))
            if isinstance(krs, dict)
            else "",
            *jadwal,
            _hari_ini_teks(hari),
        ),
        lampiran,
    )


# --- Syarat skripsi ----------------------------------------------------------


def _susun_skripsi(data: dict[str, Any], _hari: date) -> tuple[str, tuple[Lampiran, ...]]:
    syarat = data.get("syarat") if isinstance(data.get("syarat"), dict) else {}
    status = (
        data.get("status_mahasiswa") if isinstance(data.get("status_mahasiswa"), dict) else {}
    )
    mk = []
    for b in syarat.get("mata_kuliah_metodologi_penelitian") or []:
        if isinstance(b, dict):
            teks = f"{_bersih(b.get('matkul'))} ({_bersih(b.get('kdmk'))})"
            if teks not in mk:
                mk.append(teks)
    minimal = syarat.get("minimal_sks_lulus")
    keterangan = _bersih(status.get("keterangan"))
    return (
        _teks(
            "Syarat mengajukan seminar proposal skripsi menurut SADS:",
            f"- SKS lulus minimal {_angka(minimal)}",
            ("- lulus mata kuliah metodologi penelitian; yang diakui SADS: " + "; ".join(mk))
            if mk
            else "",
            "Status penanya:",
            f"- SKS lulus: {_sisa(status.get('sks_lulus'), minimal)}",
            f"- memenuhi minimal SKS: {_ya(status.get('memenuhi_minimal_sks'))}",
            f"- lulus metodologi penelitian: {_ya(status.get('lulus_metodologi_penelitian'))}",
            "- layak mengajukan seminar proposal: "
            + _ya(status.get("layak_mengajukan_seminar")),
            f"- keterangan SADS: {keterangan}" if keterangan else "",
        ),
        (),
    )


# --- Biaya dan pembayaran ----------------------------------------------------

_LABEL_TARIF = {
    "registrasi": "registrasi",
    "kemahasiswaan": "kemahasiswaan",
    "sks_teori": "per SKS teori",
    "sks_praktik": "per SKS praktik",
    "ujian_per_sks": "ujian per SKS",
    "admin_bank": "admin bank",
}


def _tagihan(t: dict[str, Any], status_kunci: str) -> str:
    return (
        f"total {_rupiah(t.get('total_biaya'))}; terbayar {_rupiah(t.get('terbayar'))}; "
        f"sisa {_rupiah(t.get('sisa_tagihan'))}; status {_bersih(t.get(status_kunci)) or '-'}"
    )


def _susun_biaya(data: dict[str, Any], _hari: date) -> tuple[str, tuple[Lampiran, ...]]:
    tarif = data.get("tarif") if isinstance(data.get("tarif"), dict) else {}
    tagihan = data.get("tagihan_krs")
    return (
        _teks(
            f"Biaya KRS semester aktif, periode {_bersih(data.get('periode')) or '-'}.",
            (
                "Tarif komponen: "
                + "; ".join(
                    f"{_LABEL_TARIF.get(k, k.replace('_', ' '))} {_rupiah(v)}"
                    for k, v in tarif.items()
                )
                + "."
            )
            if tarif
            else "",
            (
                f"Tagihan KRS penanya: {_tagihan(tagihan, 'status_pembayaran')}."
                if isinstance(tagihan, dict)
                else "Penanya belum memiliki KRS pada semester aktif, jadi belum ada "
                "tagihan KRS."
            ),
        ),
        (),
    )


def _susun_pembayaran(data: dict[str, Any], hari: date) -> tuple[str, tuple[Lampiran, ...]]:
    aktif = data.get("tagihan_aktif")
    tunggakan = [t for t in data.get("tunggakan") or [] if isinstance(t, dict)]
    sisa = sum(
        t["sisa_tagihan"] for t in tunggakan if isinstance(t.get("sisa_tagihan"), (int, float))
    )
    if tunggakan:
        rincian = (
            f"Tunggakan: {len(tunggakan)} tagihan, total sisa {_rupiah(sisa)}:\n"
            + "\n".join(
                f"- {_bersih(t.get('thnaka')) or '-'} {_bersih(t.get('semester')) or ''}: "
                f"total {_rupiah(t.get('total_biaya'))}; "
                f"terbayar {_rupiah(t.get('terbayar'))}; "
                f"sisa {_rupiah(t.get('sisa_tagihan'))}"
                for t in tunggakan
            )
        )
    elif data.get("memiliki_tunggakan") is True:
        rincian = "Tunggakan: ada menurut SADS, tetapi rinciannya tidak tercantum."
    else:
        rincian = "Tunggakan: tidak ada."
    return (
        _teks(
            f"Pembayaran periode {_bersih(data.get('periode')) or '-'}: pembayaran dibuka: "
            f"{_ya(data.get('pembayaran_dibuka'))}; batas pembayaran "
            f"{_dengan_selisih(data.get('batas_pembayaran'), hari)}.",
            (
                f"Tagihan aktif: {_tagihan(aktif, 'status')}."
                if isinstance(aktif, dict)
                else "Tagihan aktif: tidak ada."
            ),
            rincian,
            f"Boleh mengisi KRS menurut SADS: {_ya(data.get('boleh_mengisi_krs'))}.",
            _hari_ini_teks(hari),
        ),
        (),
    )


# --- Spesifikasi -------------------------------------------------------------

_TANPA_ARGUMEN: dict[str, Any] = {"type": "object", "properties": {}}

_CATATAN_PRIBADI = (
    " Data pribadi mahasiswa yang SEDANG BERTANYA; identitasnya sudah diketahui "
    "sistem, jadi jangan meminta NIM. Tidak dapat mengambil data mahasiswa lain."
)


def tool_specs(client: SadsClient) -> list[ToolSpec]:
    def spec(
        name: str,
        endpoint: str,
        susun: Susun,
        description: str,
        triggers: tuple[str, ...],
    ) -> ToolSpec:
        return ToolSpec(
            name=name,
            description=description + _CATATAN_PRIBADI,
            parameters=_TANPA_ARGUMEN,
            handler=partial(_jalankan, client, endpoint=endpoint, nama=name, susun=susun),
            triggers=triggers,
            citation_label=LABEL_SADS,
            pakai_penanya=True,
        )

    return [
        spec(
            "get_nilai_saya",
            "krs-mahasiswa",
            _susun_nilai,
            "Seluruh nilai mata kuliah penanya (nilai huruf, bobot, SKS), IPK yang "
            "sudah dihitung, sebaran nilai, serta SKS wajib dan pilihan yang sudah "
            "lulus dibanding minimal kelulusan. Untuk 'berapa IPK saya?', 'nilai "
            "Basis Data saya?', 'berapa SKS lagi yang harus saya ambil?'. Tidak "
            "memuat nilai per semester (IPS).",
            (
                "nilai",
                "ipk",
                "indeks prestasi",
                "transkrip",
                "khs",
                "sks",
                "lulus",
                "grade",
                "gpa",
            ),
        ),
        spec(
            "get_jadwal_krs_saya",
            "jadwal-krs",
            _susun_krs,
            "Jadwal pengisian KRS untuk angkatan dan jenis kelas penanya: periode, "
            "tanggal buka dan tutup, dan apakah KRS masih bisa diisi.",
            ("krs", "rencana studi", "perwalian"),
        ),
        spec(
            "get_revisi_krs_saya",
            "revisi-krs",
            _susun_revisi,
            "Jadwal dan batas revisi (perubahan) KRS penanya, dan apakah revisi masih "
            "bisa dilakukan.",
            ("revisi", "krs", "ubah krs", "ganti krs"),
        ),
        spec(
            "get_syarat_skripsi_saya",
            "syarat-skripsi",
            _susun_skripsi,
            "Syarat mengajukan seminar proposal skripsi (minimal SKS lulus dan mata "
            "kuliah metodologi penelitian) beserta status kelayakan penanya.",
            (
                "skripsi",
                "sempro",
                "seminar proposal",
                "tugas akhir",
                "seminar",
                "metodologi penelitian",
                "thesis",
            ),
        ),
        spec(
            "get_biaya_krs_saya",
            "biaya-krs",
            _susun_biaya,
            "Tarif komponen biaya KRS semester aktif (registrasi, kemahasiswaan, per "
            "SKS teori/praktik, ujian per SKS, admin bank) dan tagihan KRS penanya: "
            "total, terbayar, sisa, status lunas.",
            ("biaya", "tagihan", "tarif", "ukt", "spp", "uang kuliah", "bayar", "tuition"),
        ),
        spec(
            "get_jadwal_pembayaran_saya",
            "jadwal-pembayaran",
            _susun_pembayaran,
            "Status pembukaan dan batas pembayaran semester aktif, tagihan aktif, dan "
            "seluruh tunggakan penanya beserta jumlahnya.",
            ("bayar", "pembayaran", "tunggakan", "tagihan", "lunas", "payment"),
        ),
        spec(
            "get_semester_aktif_saya",
            "semester-aktif",
            _susun_semester,
            "Semester aktif (periode, tahun akademik, ganjil/genap), semester ke-berapa "
            "bagi penanya, dan jadwal kuliah penanya semester ini (mata kuliah, kelas, "
            "hari, jam, ruang, dosen, serta kuliah hari ini). Untuk 'saya semester "
            "berapa?', 'jadwal kuliah saya?', 'hari ini saya kuliah apa?'. Tanggal "
            "mulai dan selesai perkuliahan tidak selalu ada; bila tidak ada, pakai "
            "Kalender Akademik.",
            (
                "semester",
                "tahun akademik",
                "kalender akademik",
                "mulai kuliah",
                "perkuliahan",
                "jadwal kuliah",
                "jam kuliah",
                "ruang kuliah",
                "kuliah hari",
                "ada kuliah",
                "kuliah apa",
                "masuk kuliah",
                "kuliah besok",
                "jadwal kelas",
                "class schedule",
            ),
        ),
    ]
