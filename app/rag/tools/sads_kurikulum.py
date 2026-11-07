"""Tool SADS kurikulum: prasyarat mata kuliah dan daftar mata kuliah per semester.

Datanya publik (sama untuk semua mahasiswa) dan berubah per kurikulum, bukan per
hari, jadi respons `daftar-matkul-kurikulum` di-cache singkat (`TTL_KURIKULUM`).
Mencari prasyarat menurut NAMA mata kuliah butuh daftar seluruh semester: tanpa
cache satu pertanyaan memanggil SADS belasan kali (docs/tool-call.md §7).
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any
from urllib.parse import quote

from app.prodi import DAFTAR_PRODI
from app.rag.tools.base import Lampiran, Penanya, ToolResult, ToolSpec, kunci_sebutan
from app.rag.tools.client import SadsClient
from app.rag.tools.sads import LABEL_SADS, _bersih
from app.rag.tools.sads_mahasiswa import kode_mk_penanya

PATH_KURIKULUM = "/service/tp/chatbot/daftar-matkul-kurikulum"
PATH_PRASYARAT = "/service/tp/chatbot/prasyarat-matkul"

SEMESTER_TERTINGGI = 8
SEMESTER_INDEKS = range(0, SEMESTER_TERTINGGI + 1)
"""Semester yang dibaca untuk mencari mata kuliah menurut nama. Semester 0 ada
di SADS: mata kuliah tanpa semester tetap (mis. pilihan BD)."""

KURIKULUM_TAMBAHAN: tuple[int, ...] = (2,)
"""Kurikulum yang TIDAK aktif di SADS tetapi masih dipakai mahasiswa.

Tanpa `kurikulum_id`, SADS hanya mengembalikan kurikulum aktif. "Kurikulum 2022
Informatika" (id 2) tidak aktif, padahal ke-30 mata kuliah seorang mahasiswa IF
angkatan 2024 tercatat di sana dan tidak satu pun di kurikulum aktif (uji
2026-11-07). Kurikulum 2022 prodi lain (RSK, DKV, BD) aktif. Hapus id-nya dari
sini begitu SADS menandainya aktif; baris ganda tetap dibuang (`_semester`)."""

TTL_KURIKULUM = 1800.0
"""Umur cache respons kurikulum, detik."""

_CACHE: dict[tuple[str, int, int | None], tuple[float, list[dict[str, Any]]]] = {}
"""(base_url, semester, kurikulum_id) -> (waktu ambil, baris). Hanya data publik;
data pribadi (`sads_mahasiswa`) tidak pernah di-cache."""

KODE_PRODI_SADS = {"1010": "IF", "1020": "RSK", "2010": "SK", "2020": "BD", "0301": "MI"}
"""Kode prodi dari NIM (`app.prodi`) -> `prodi_kode` SADS. DKV di SADS = "SK".

Saringan `prodi` milik SADS sendiri tidak dipakai: ia mencocokkan potongan nama,
sehingga "TI" juga mengembalikan Magister Informatika (uji 2026-11-07)."""

MAKS_PRASYARAT_DIRINCI = 3
"""Mata kuliah yang prasyaratnya dirinci dalam satu panggilan. Nama pendek
("Web", "Basis Data") cocok dengan banyak mata kuliah; di atas batas ini hanya
nama-namanya yang dikirim, supaya model meminta nama lengkap."""

_POLA_KDMK = re.compile(r"^[A-Za-z]{2,6}-\d{3}[A-Za-z]?$")


def kosongkan_cache() -> None:
    """Untuk test: setiap test mulai tanpa respons kurikulum tersimpan."""
    _CACHE.clear()


async def _baris(
    client: SadsClient, semester: int, kurikulum_id: int | None
) -> list[dict[str, Any]]:
    kunci = (client.base_url, semester, kurikulum_id)
    simpan = _CACHE.get(kunci)
    if simpan is not None and time.monotonic() - simpan[0] < TTL_KURIKULUM:
        return simpan[1]
    params = {"kurikulum_id": kurikulum_id} if kurikulum_id is not None else None
    data = await client.get_json(f"{PATH_KURIKULUM}/{semester}", params=params)
    baris = data.get("data") if isinstance(data, dict) else None
    if not isinstance(baris, list):
        raise ValueError("respons daftar-matkul-kurikulum tidak berbentuk {data: [...]}")
    baris = [b for b in baris if isinstance(b, dict)]
    _CACHE[kunci] = (time.monotonic(), baris)
    return baris


async def _semester(client: SadsClient, semester: int) -> list[dict[str, Any]]:
    """Baris satu semester dari kurikulum aktif dan `KURIKULUM_TAMBAHAN`."""
    respons = await asyncio.gather(
        _baris(client, semester, None),
        *(_baris(client, semester, k) for k in KURIKULUM_TAMBAHAN),
    )
    terlihat: set[tuple[Any, str]] = set()
    hasil = []
    for baris in respons:
        for b in baris:
            kunci = (b.get("kurikulum_id"), _bersih(b.get("kdmk")))
            if kunci not in terlihat:
                terlihat.add(kunci)
                hasil.append(b)
    return hasil


def _sebutan(teks: str) -> str:
    return " ".join(teks.replace(".", " ").split()).casefold()


def kode_prodi_sads(teks: str) -> str | None:
    """ "Informatika", "TI", "DKV", "Bisnis Digital", "RSK" -> kode prodi SADS."""
    dicari = _sebutan(teks)
    for p in DAFTAR_PRODI:
        kode = KODE_PRODI_SADS.get(p.code)
        if kode and dicari in {_sebutan(s) for s in (p.name, *p.aliases, kode, p.code)}:
            return kode
    return None


def _prodi_penanya(penanya: Penanya) -> str | None:
    return KODE_PRODI_SADS.get(penanya.profile.prodi.code) if penanya.profile else None


def _daftar_prodi() -> str:
    return ", ".join(f"{p.name} ({', '.join(p.aliases)})" for p in DAFTAR_PRODI)


def _semester_sah(nilai: Any) -> int | None:
    if isinstance(nilai, bool):
        return None
    try:
        n = int(str(nilai).strip())
    except ValueError:
        return None
    return n if 1 <= n <= SEMESTER_TERTINGGI else None


def _status(b: dict[str, Any]) -> str:
    return _bersih(b.get("status_matkul")) or "-"


def _sks(b: dict[str, Any]) -> int:
    nilai = b.get("sks")
    return nilai if isinstance(nilai, int) and not isinstance(nilai, bool) else 0


# --- Kurikulum penanya ---------------------------------------------------------


def rumpun_kode(kdmk: str) -> str:
    """Rumpun kode mata kuliah: "BIFW-063" -> "BIF", "CDVG-001" -> "CDV".

    Huruf terakhir awalan 4 huruf adalah jenisnya (W wajib, P pilihan, N umum).
    "Kurikulum 2022 Informatika" mencampur beberapa rumpun (BIF, BTI, TIW, MKB,
    MKK): mata kuliah yang sama tercatat dengan beberapa kode, dan semester 5
    saja berisi 110 mata kuliah, padahal rumpun BIF mahasiswa IF 2024 hanya 16
    (uji 2026-11-07)."""
    awal = kdmk.split("-", 1)[0].upper()
    return awal[:-1] if len(awal) >= 4 else awal


@dataclass(frozen=True)
class JejakPenanya:
    """Kurikulum dan rumpun kode penanya, dari mata kuliah yang sudah ditempuhnya."""

    kurikulum: str
    rumpun: str

    def keterangan(self) -> str:
        return (
            f"Disaring ke kurikulum penanya sendiri: {self.kurikulum}, kode mata kuliah "
            f"berawalan {self.rumpun} (ditentukan dari mata kuliah yang sudah ditempuh "
            "penanya)."
        )

    def cocok(self, kurikulum: str, kdmk: str) -> bool:
        return kurikulum == self.kurikulum and rumpun_kode(kdmk) == self.rumpun


async def jejak_penanya(
    client: SadsClient, penanya: Penanya, indeks: Sequence[_MataKuliah]
) -> JejakPenanya | None:
    """None bila penanya tanpa NIM, belum punya nilai, atau datanya tidak dapat
    diambil: tool lalu menyaring menurut prodi saja."""
    if not penanya.nim:
        return None
    kode = await kode_mk_penanya(client, penanya)
    per_kode = {m.kdmk: m.kurikulum for m in indeks}
    kurikulum = Counter(per_kode[k] for k in kode if k in per_kode)
    if not kurikulum:
        return None
    rumpun = Counter(rumpun_kode(k) for k in kode)
    return JejakPenanya(kurikulum.most_common(1)[0][0], rumpun.most_common(1)[0][0])


# --- get_matkul_kurikulum ---------------------------------------------------


async def _matkul_kurikulum(
    client: SadsClient, *, semester: Any, prodi: str = "", penanya: Penanya
) -> ToolResult:
    nama = "get_matkul_kurikulum"
    smt = _semester_sah(semester)
    if smt is None:
        galat = f"semester harus angka 1 sampai {SEMESTER_TERTINGGI}."
        return ToolResult(name=nama, label=LABEL_SADS, text="", ok=False, galat=galat)
    jejak = None
    milik_penanya = _prodi_penanya(penanya)
    if prodi:
        kode = kode_prodi_sads(prodi)
        if kode is None:
            teks = f'Prodi "{prodi}" tidak dikenal. Prodi di INSTIKI: {_daftar_prodi()}.'
            return ToolResult(name=nama, label=LABEL_SADS, text=teks)
    else:
        kode = milik_penanya
    # Model sering mengisi `prodi` dengan prodi penanya sendiri dari baris profil
    # ("Informatika"); itu tetap pertanyaan tentang kurikulum penanya. Tanpa ini
    # "mata kuliah semester 5 apa saja?" dijawab "110 mata kuliah" (uji live
    # 2026-11-07), bukan 17 mata kuliah rumpun BIF penanya.
    if kode is not None and kode == milik_penanya:
        jejak = await jejak_penanya(client, penanya, await _indeks(client))
    semua = await _semester(client, smt)
    baris = [b for b in semua if not kode or b.get("prodi_kode") == kode]
    if jejak is not None:
        sempit = [
            b
            for b in semua
            if jejak.cocok(_bersih(b.get("kurikulum")), _bersih(b.get("kdmk")))
        ]
        if sempit:
            baris = sempit
        else:
            jejak = None
    if not baris:
        teks = f"Tidak ada mata kuliah semester {smt} di kurikulum SADS untuk prodi itu."
        return ToolResult(name=nama, label=LABEL_SADS, text=teks)
    per_kurikulum: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for b in baris:
        kunci = (_bersih(b.get("kurikulum")), _bersih(b.get("prodi_nama")))
        per_kurikulum.setdefault(kunci, []).append(b)
    bagian = []
    lampiran = []
    for (kurikulum, nama_prodi), isi in sorted(per_kurikulum.items()):
        isi = sorted(isi, key=lambda b: (_status(b) != "Wajib", _bersih(b.get("matkul"))))
        bagian.append(_rincian_semester(smt, kurikulum, nama_prodi, isi))
        lampiran.append(
            Lampiran(
                title=f"Mata kuliah semester {smt}, {kurikulum}",
                source=LABEL_SADS,
                items=tuple(
                    f"{_bersih(b.get('matkul'))} ({_bersih(b.get('kdmk'))}) · "
                    f"{_sks(b)} SKS · {_status(b)}"
                    for b in isi
                ),
            )
        )
    jumlah = len(per_kurikulum)
    kepala = f"Mata kuliah semester {smt} menurut kurikulum di SADS: {jumlah} kurikulum."
    if jejak is not None:
        kepala += "\n" + jejak.keterangan()
    return ToolResult(
        name=nama,
        label=LABEL_SADS,
        text="\n\n".join([kepala, *bagian]),
        attachments=tuple(lampiran),
    )


def _rincian_semester(
    smt: int, kurikulum: str, nama_prodi: str, isi: Sequence[dict[str, Any]]
) -> str:
    wajib = [b for b in isi if _status(b) == "Wajib"]
    lain = [b for b in isi if _status(b) != "Wajib"]
    ringkas = (
        f"Kurikulum {kurikulum} ({nama_prodi}), semester {smt}: {len(isi)} mata kuliah, "
        f"{sum(map(_sks, isi))} SKS (wajib {len(wajib)} mata kuliah, "
        f"{sum(map(_sks, wajib))} SKS; pilihan atau lainnya {len(lain)} mata kuliah, "
        f"{sum(map(_sks, lain))} SKS)."
    )
    daftar = "\n".join(
        f"- {_bersih(b.get('kdmk'))} {_bersih(b.get('matkul'))} ({_sks(b)} SKS, {_status(b)})"
        for b in isi
    )
    return f"{ringkas}\n{daftar}"


# --- get_prasyarat_matkul ---------------------------------------------------


@dataclass(frozen=True)
class _MataKuliah:
    kdmk: str
    nama: str = ""
    kurikulum: str = ""
    prodi: str = ""
    """`prodi_kode` SADS."""

    def judul(self) -> str:
        keterangan = ", ".join(x for x in (self.kdmk, self.kurikulum) if x)
        return f"{self.nama} ({keterangan})" if self.nama else self.kdmk


async def _indeks(client: SadsClient) -> list[_MataKuliah]:
    respons = await asyncio.gather(*(_semester(client, s) for s in SEMESTER_INDEKS))
    terlihat: set[str] = set()
    hasil = []
    for baris in respons:
        for b in baris:
            kdmk = _bersih(b.get("kdmk"))
            if kdmk and kdmk not in terlihat:
                terlihat.add(kdmk)
                hasil.append(
                    _MataKuliah(
                        kdmk=kdmk,
                        nama=_bersih(b.get("matkul")),
                        kurikulum=_bersih(b.get("kurikulum")),
                        prodi=_bersih(b.get("prodi_kode")),
                    )
                )
    return hasil


def _cari(indeks: Sequence[_MataKuliah], dicari: str, prodi: str | None) -> list[_MataKuliah]:
    """Nama persis lebih dulu, lalu potongan nama; prodi penanya diutamakan."""
    kunci = kunci_sebutan(dicari)
    calon = [m for m in indeks if kunci_sebutan(m.nama) == kunci] or [
        m for m in indeks if kunci in kunci_sebutan(m.nama)
    ]
    if prodi and any(m.prodi == prodi for m in calon):
        calon = [m for m in calon if m.prodi == prodi]
    return sorted(calon, key=lambda m: (m.prodi, m.nama, m.kdmk))


def _syarat(b: dict[str, Any]) -> str:
    jenis = _bersih(b.get("jenis"))
    nilai = _bersih(str(b.get("nilai_syarat") or ""))
    if jenis == "mata_kuliah":
        kode = _bersih(b.get("kdmk_syarat")) or nilai
        nama = _bersih(b.get("matkul_syarat"))
        return (
            f"memenuhi mata kuliah {nama} ({kode})" if nama else f"memenuhi mata kuliah {kode}"
        )
    if jenis == "sks_lulus":
        return f"SKS lulus minimal {nilai}"
    if jenis == "sks_pilihan_lulus":
        return f"SKS mata kuliah pilihan lulus minimal {nilai}"
    return f"{jenis or 'syarat'}: {nilai}"


def _kelompok(b: dict[str, Any]) -> int:
    nilai = b.get("kelompok")
    return nilai if isinstance(nilai, int) and not isinstance(nilai, bool) else 1


def _rincian_prasyarat(mk: _MataKuliah, data: Any) -> str:
    """Prasyarat satu mata kuliah, digabung antar jurusan/konsentrasi yang sama.

    SADS menulis satu baris per `kdjur` (IF-AI, IF-GD, IF-MDA), dan biasanya
    isinya sama persis. Kelompok 2 dibaca sebagai ALTERNATIF kelompok 1: dari
    144 sampel berkelompok dua, isinya mata kuliah padanan kurikulum lama
    ("Mathematics" / "Matematika 1", "Microprocessor" / "Mikroprosesor") atau
    syarat SKS (uji 2026-11-07). Apakah semua baris di dalam SATU kelompok wajib
    dipenuhi tidak dinyatakan: kelompok 2 "Basis Data Lanjut" berisi dua kode
    "Basis Data" lama, yang tidak masuk akal bila keduanya wajib. Tafsiran ini
    belum dikonfirmasi pengembang SADS (docs/tool-call.md §7)."""
    if not isinstance(data, list):
        raise ValueError("respons prasyarat-matkul bukan daftar")
    if not data:
        return f"{mk.judul()}: tidak memiliki prasyarat menurut SADS."
    per_jur: dict[str, set[tuple[int, str]]] = {}
    for b in data:
        if isinstance(b, dict):
            jur = _bersih(b.get("kdjur")) or "-"
            per_jur.setdefault(jur, set()).add((_kelompok(b), _syarat(b)))
    per_isi: dict[frozenset[tuple[int, str]], list[str]] = {}
    for jur, isi in per_jur.items():
        per_isi.setdefault(frozenset(isi), []).append(jur)
    baris = [f"Prasyarat {mk.judul()}:"]
    for isi, jur in sorted(per_isi.items(), key=lambda kv: sorted(kv[1])):
        baris.append(f"Berlaku untuk jurusan/konsentrasi {', '.join(sorted(jur))}:")
        kelompok = sorted({k for k, _ in isi})
        for i, k in enumerate(kelompok):
            if len(kelompok) > 1:
                baris.append(
                    f"Kelompok {k}:"
                    if i == 0
                    else f"Kelompok {k} (alternatif kelompok {kelompok[0]}; umumnya "
                    "mata kuliah padanan dari kurikulum lama):"
                )
            baris += [f"- {s}" for s in sorted(s for kk, s in isi if kk == k)]
    return "\n".join(baris)


async def _prasyarat(client: SadsClient, *, matkul: str, penanya: Penanya) -> ToolResult:
    nama = "get_prasyarat_matkul"
    dicari = " ".join(matkul.split())
    catatan = ""
    if _POLA_KDMK.fullmatch(dicari):
        kdmk = dicari.upper()
        dikenal = [m for m in await _indeks(client) if m.kdmk == kdmk]
        target = dikenal[:1] or [_MataKuliah(kdmk=kdmk)]
    else:
        indeks = await _indeks(client)
        target = _cari(indeks, dicari, _prodi_penanya(penanya))
        # "Kerja Praktik" tercatat 6 kali (BIFW-063, BTIW-063, TIW-063, MKB-105,
        # ... dan BIFW-295 OBE 2025): tanpa saringan ini, pertanyaan yang paling
        # umum pun berakhir di "terlalu banyak untuk dirinci".
        jejak = await jejak_penanya(client, penanya, indeks) if len(target) > 1 else None
        if jejak is not None:
            sempit = [m for m in target if jejak.cocok(m.kurikulum, m.kdmk)]
            if sempit:
                lain = len(target) - len(sempit)
                catatan = jejak.keterangan() + (
                    f" Mata kuliah bernama serupa di kurikulum atau rumpun kode lain: {lain}."
                    if lain
                    else ""
                )
                target = sempit
        if not target:
            teks = (
                f'Tidak ada mata kuliah di kurikulum SADS yang namanya memuat "{dicari}": '
                "0 mata kuliah."
            )
            return ToolResult(name=nama, label=LABEL_SADS, text=teks)
        if len(target) > MAKS_PRASYARAT_DIRINCI:
            teks = (
                f'Ada {len(target)} mata kuliah yang namanya memuat "{dicari}", terlalu '
                f"banyak untuk dirinci prasyaratnya (paling banyak {MAKS_PRASYARAT_DIRINCI}). "
                "Mata kuliah:\n" + "\n".join(f"- {m.judul()}" for m in target)
            )
            return ToolResult(name=nama, label=LABEL_SADS, text=teks)
    respons = await asyncio.gather(
        *(client.get_json(f"{PATH_PRASYARAT}/{quote(m.kdmk, safe='')}") for m in target)
    )
    teks = "\n\n".join(
        [
            *([catatan] if catatan else []),
            *(_rincian_prasyarat(m, d) for m, d in zip(target, respons, strict=True)),
        ]
    )
    return ToolResult(name=nama, label=LABEL_SADS, text=teks)


def tool_specs(client: SadsClient) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="get_prasyarat_matkul",
            description=(
                "Prasyarat (syarat mengambil) SATU mata kuliah INSTIKI menurut "
                "kurikulum di SADS: mata kuliah yang harus dipenuhi dan/atau jumlah "
                "SKS lulus minimal. Untuk pertanyaan seperti 'apa prasyarat Basis "
                "Data Lanjut?' atau 'syarat ambil Kerja Praktik apa?'. 'matkul' "
                "boleh berupa nama atau kode mata kuliah (mis. 'BIFW-242'). Sebagian "
                "nama mata kuliah di SADS berbahasa Inggris; bila hasilnya 0, coba "
                "padanan Inggris atau kata kunci yang lebih pendek. Satu mata kuliah "
                "per panggilan."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "matkul": {
                        "type": "string",
                        "description": (
                            "nama atau kode SATU mata kuliah, mis. 'Basis Data "
                            "Lanjut' atau 'BIFW-242'"
                        ),
                    }
                },
                "required": ["matkul"],
            },
            handler=partial(_prasyarat, client),
            triggers=(
                "prasyarat",
                "syarat mata kuliah",
                "syarat matkul",
                "syarat ambil",
                "syarat mengambil",
                "prerequisite",
            ),
            citation_label=LABEL_SADS,
            pakai_penanya=True,
        ),
        ToolSpec(
            name="get_matkul_kurikulum",
            description=(
                "Daftar mata kuliah pada satu semester menurut kurikulum di SADS, "
                "beserta kode, SKS, dan status wajib/pilihan, dengan jumlah mata "
                "kuliah dan SKS yang sudah dihitung. Untuk pertanyaan seperti 'mata "
                "kuliah semester 5 apa saja?' atau 'berapa SKS semester 3 prodi "
                "DKV?'. Tanpa 'prodi' = kurikulum penanya sendiri (ditentukan dari "
                "mata kuliah yang sudah ditempuhnya), atau seluruh kurikulum "
                "prodinya bila belum ada. Satu prodi dapat memiliki lebih dari satu "
                "kurikulum (mis. Kurikulum 2022 dan OBE 2025); hasilnya "
                "dikelompokkan per kurikulum."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "semester": {
                        "type": "integer",
                        "description": f"semester mata kuliah, 1 sampai {SEMESTER_TERTINGGI}",
                    },
                    "prodi": {
                        "type": "string",
                        "description": (
                            "nama atau singkatan prodi bila pertanyaan menyebut prodi "
                            "lain, mis. 'DKV', 'Bisnis Digital'. Kosongkan untuk prodi "
                            "penanya."
                        ),
                    },
                },
                "required": ["semester"],
            },
            handler=partial(_matkul_kurikulum, client),
            triggers=(
                "kurikulum",
                "mata kuliah semester",
                "matkul semester",
                "mata kuliah apa saja",
                "curriculum",
            ),
            citation_label=LABEL_SADS,
            pakai_penanya=True,
        ),
    ]
