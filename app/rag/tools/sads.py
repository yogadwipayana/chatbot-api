"""Tool SADS: daftar dosen, dosen pengampu mata kuliah, dan mata kuliah per dosen.

Handler tipis: panggil endpoint, normalisasi (data SADS punya spasi & koma di
ujung `nmdosen`), ratakan jadi teks ringkas. Menambah endpoint = menambah satu
handler + satu `ToolSpec` di sini, lalu registry memungutnya (docs/tool-call.md §16).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from app.config import Settings
from app.rag.tools.base import Lampiran, ToolResult, ToolSpec, kunci_sebutan
from app.rag.tools.client import SadsClient

LABEL_SADS = "Data akademik SADS"

PATH_MK_DIAMPU = "/service/tp/chatbot/mk-diampu-dosen"

_GELAR_DEPAN = re.compile(r"^(?:(?:prof|drs|dra|dr|ir)\b\.?\s*)+", re.IGNORECASE)
"""Gelar di depan nama SADS: "Dr. Ir. I Putu ...", "Dr.Ir. Aniek ...", "Ir.  Adi ...".

Bukan "A.A." (Anak Agung) -- itu bagian nama."""

_SINONIM_GELAR = {
    "doktor": {"dr", "phd"},
    "doctor": {"dr", "phd"},
    "profesor": {"prof"},
    "professor": {"prof"},
}
"""Kata yang mungkin dikirim model alih-alih singkatan gelar."""


def _bersih(nama: str | None) -> str:
    """Buang spasi dan koma di ujung (data SADS: "Aulia Iefan Datya, ST.,MT   ")."""
    return (nama or "").strip().rstrip(",").strip()


def _token_gelar(teks: str) -> str:
    """Samakan penulisan gelar: "Dr." -> "dr", "Ph.D." -> "phd", "M.Kom" -> "mkom".

    Titik, spasi, dan huruf besar diabaikan, karena SADS menulis gelar yang sama
    dengan banyak cara ("S.Kom.", "S.Kom", "S.Kom,")."""
    return re.sub(r"[\s.]", "", teks).casefold()


def _urai_dosen(lengkap: str) -> tuple[str, frozenset[str]]:
    """Pisahkan "Dr. Ir. I Putu Agus, S.Kom., M.T., IPM." menjadi nama inti
    ("I Putu Agus") dan himpunan token gelar ({"dr", "ir", "skom", "mt", "ipm"}).

    Gelar belakang dipisah koma atau spasi ("M.Cs. IPM"). Nama tanpa koma dianggap
    tanpa gelar belakang."""
    depan_belakang = lengkap.split(",", 1)
    kepala = depan_belakang[0].strip()
    m = _GELAR_DEPAN.match(kepala)
    depan = m.group(0) if m else ""
    nama = kepala[len(depan) :].strip()
    belakang = depan_belakang[1] if len(depan_belakang) > 1 else ""
    token = {
        _token_gelar(g) for g in re.findall(r"(?:prof|drs|dra|dr|ir)\b", depan, re.IGNORECASE)
    }
    token |= {_token_gelar(g) for g in re.split(r"[,\s]+", belakang) if _token_gelar(g)}
    return nama, frozenset(token)


def _cocok(lengkap: str, *, nama: str, gelar: str) -> bool:
    """Saringan `get_daftar_dosen`. `nama` dicocokkan sebagai bagian nama inti
    (bukan gelar: "kom" tidak boleh mencocokkan "S.Kom"); `gelar` dicocokkan
    sebagai token utuh, sehingga "Dr." tidak mencocokkan "Drs."."""
    inti, token = _urai_dosen(lengkap)
    if nama and " ".join(nama.split()).casefold() not in " ".join(inti.split()).casefold():
        return False
    if gelar:
        dicari = _SINONIM_GELAR.get(gelar.strip().casefold(), {_token_gelar(gelar)})
        if not dicari & token:
            return False
    return True


def _keterangan_saring(nama: str, gelar: str) -> str:
    bagian = []
    if gelar:
        bagian.append(f"bergelar {gelar}")
    if nama:
        bagian.append(f'dengan nama memuat "{nama}"')
    return " ".join(bagian)


async def _daftar_dosen(client: SadsClient, *, nama: str = "", gelar: str = "") -> ToolResult:
    data = await client.get_json("/service/tp/chatbot/dosen-mengajar")
    semua = sorted({_bersih(d.get("nmdosen")) for d in data if _bersih(d.get("nmdosen"))})
    if not semua:
        # SADS tidak mengembalikan satu nama pun: itu kegagalan layanan, bukan
        # jawaban "tidak ada dosen".
        return ToolResult(name="get_daftar_dosen", label=LABEL_SADS, text="", ok=False)
    cocok = [n for n in semua if _cocok(n, nama=nama, gelar=gelar)]
    saring = _keterangan_saring(nama, gelar)
    subjek = "dosen yang mengajar di INSTIKI" + (f" {saring}" if saring else "")
    if not cocok:
        # Saringan yang tidak cocok adalah jawaban sah dari SADS ("tidak ada
        # dosen bergelar Prof."), bukan DATA_TIDAK_TERSEDIA: yang terakhir
        # membuat model menolak dan menyuruh mahasiswa bertanya ke FO (T43).
        teks = f"Tidak ada {subjek}: 0 orang (dari {len(semua)} dosen yang mengajar)."
        return ToolResult(name="get_daftar_dosen", label=LABEL_SADS, text=teks)
    # Jumlahnya dihitung di sini, bukan oleh model: LLM tidak andal menghitung
    # daftar panjang. Terbukti 2026-10-07: dari 220 nama, model menjawab
    # "Terdapat 223 dosen"; dari 25 dosen bergelar Dr., "38" lalu "39" (T46).
    # Keduanya lengkap dengan kartu sumber yang membuatnya tampak sah.
    teks = f"Jumlah {subjek}: {len(cocok)} orang.\n"
    if saring:
        teks += f"(Seluruh dosen yang mengajar: {len(semua)} orang.)\n"
    teks += "Daftar dosen:\n" + "\n".join(f"- {n}" for n in cocok)
    judul = f"Dosen {saring}" if saring else "Dosen yang mengajar di INSTIKI"
    return ToolResult(
        name="get_daftar_dosen",
        label=LABEL_SADS,
        text=teks,
        attachments=(Lampiran(title=judul, source=LABEL_SADS, items=tuple(cocok)),),
    )


def _gabung_per_dosen(*respons: Any) -> dict[str, list[str]]:
    """Ratakan respons `mk-diampu-dosen` menjadi {nama dosen: [mata kuliah]}.

    Beberapa respons (varian ejaan) digabung per dosen tanpa duplikat; urutan
    mengikuti kemunculan pertama di SADS. Dosen tanpa mata kuliah dibuang."""
    per_dosen: dict[str, list[str]] = {}
    for data in respons:
        for d in data:
            nm = _bersih(d.get("nmdosen"))
            mk = [_bersih(m.get("matkul")) for m in (d.get("matkul") or [])]
            mk = [m for m in mk if m]
            if not (nm and mk):
                continue
            daftar = per_dosen.setdefault(nm, [])
            daftar += [m for m in mk if m not in daftar]
    return per_dosen


def _varian_ejaan(matkul: str) -> list[str]:
    """Kata kunci beserta varian huruf gandanya dirapatkan ("Intelligence" ->
    "Inteligence").

    SADS mencocokkan `matkul` sebagai potongan teks persis, dan ejaannya tidak
    seragam: "Artificial Intelligence" (34 dosen) dan "Artificial Inteligence"
    (1 dosen) adalah mata kuliah yang sama (T47, uji 2026-10-07)."""
    rapat = re.sub(r"(\w)\1", r"\1", matkul)
    return [matkul] if rapat == matkul else [matkul, rapat]


@dataclass
class _MataKuliah:
    ejaan: dict[str, set[str]] = field(default_factory=dict)
    """Penulisan nama di SADS -> dosen yang tercatat dengan penulisan itu."""
    dosen: list[str] = field(default_factory=list)

    @property
    def nama(self) -> str:
        """Penulisan yang paling banyak dipakai; seri = yang muncul lebih dulu."""
        return max(self.ejaan, key=lambda e: len(self.ejaan[e]))


def _gabung_per_mk(*respons: Any) -> list[_MataKuliah]:
    """Kelompokkan respons `mk-diampu-dosen` per mata kuliah, terbanyak dosennya dulu.

    Satu kata kunci cocok dengan beberapa mata kuliah ("Basis Data" dan "Basis
    Data Lanjut"), dan varian ejaan ("Artificial Inteligence") digabung ke mata
    kuliah yang sama. Dulu hasilnya diratakan per dosen, sehingga model yang
    menyaring "Lanjut", menghitung, dan menyalin daftarnya (T49)."""
    per_mk: dict[str, _MataKuliah] = {}
    for data in respons:
        for d in data:
            nm = _bersih(d.get("nmdosen"))
            if not nm:
                continue
            for m in d.get("matkul") or []:
                mk = _bersih(m.get("matkul"))
                if not mk:
                    continue
                kelompok = per_mk.setdefault(kunci_sebutan(mk), _MataKuliah())
                kelompok.ejaan.setdefault(mk, set()).add(nm)
                if nm not in kelompok.dosen:
                    kelompok.dosen.append(nm)
    return sorted(per_mk.values(), key=lambda k: -len(k.dosen))


def _rincian_mk(mk: _MataKuliah) -> str:
    lain = [e for e in mk.ejaan if e != mk.nama]
    juga = "".join(f'juga tertulis "{e}"; ' for e in lain)
    daftar = "\n".join(f"- {nm}" for nm in sorted(mk.dosen))
    return f"{mk.nama} ({juga}jumlah dosen pengampu: {len(mk.dosen)} orang):\n{daftar}"


async def _mk_diampu_dosen(client: SadsClient, *, matkul: str) -> ToolResult:
    respons = await asyncio.gather(
        *(client.get_json(PATH_MK_DIAMPU, params={"matkul": v}) for v in _varian_ejaan(matkul))
    )
    per_mk = _gabung_per_mk(*respons)
    if not per_mk:
        # Tidak ada yang cocok adalah jawaban sah dari SADS, bukan galat (T43).
        # DATA_TIDAK_TERSEDIA ("tidak dapat diambil saat ini") membuat model
        # mengira layanannya gangguan: ia tidak mencoba nama Inggris, lalu
        # menolak dengan kontak FO ("Kecerdasan Buatan" -> 0, padahal
        # "Artificial Intelligence" diampu 35 dosen; T47). Petunjuk mencoba
        # kata kunci lain ada di `description`, bukan di sini: isi pesan tool
        # adalah data (aturan T2).
        teks = (
            f'Tidak ada mata kuliah di SADS yang namanya memuat "{matkul}": 0 dosen pengampu.'
        )
        return ToolResult(name="get_mk_diampu_dosen", label=LABEL_SADS, text=teks)
    # Setiap mata kuliah berlampiran sendiri, dengan jumlah dari handler: model
    # yang menyalin 35 nama "Artificial Intelligence" kadang kehilangan satu
    # nama, atau hanya menulis 4 nama "antara lain" (T49, uji 2026-10-07).
    # Lampiran yang tampil dipilih dari nama mata kuliah yang ditulis jawaban
    # (`pilih_lampiran`), supaya "Basis Data Lanjut" tidak menempel di bawah
    # jawaban tentang "Basis Data".
    teks = (
        f'Mata kuliah di SADS yang namanya memuat "{matkul}": {len(per_mk)} mata kuliah.\n\n'
        + "\n\n".join(_rincian_mk(mk) for mk in per_mk)
    )
    lampiran = tuple(
        Lampiran(
            title=f"Dosen pengampu {mk.nama}",
            source=LABEL_SADS,
            items=tuple(sorted(mk.dosen)),
            disebut=mk.nama,
        )
        for mk in per_mk
    )
    return ToolResult(
        name="get_mk_diampu_dosen", label=LABEL_SADS, text=teks, attachments=lampiran
    )


MAKS_DOSEN_DIRINCI = 5
"""Dosen yang mata kuliahnya dirinci oleh `get_mk_dosen` dalam satu panggilan.

Satu dosen mengampu sampai sekitar 25 mata kuliah. Nama pendek seperti "Wayan"
cocok dengan 15 dosen, dan merinci semuanya hanya membebani prompt; di atas
batas ini hanya nama-namanya yang dikirim."""

_SAPAAN = re.compile(r"^(?:bapak|pak|ibu|bu)\b\.?\s*", re.IGNORECASE)


async def _mk_dosen(client: SadsClient, *, nama: str) -> ToolResult:
    # Tanpa `matkul`, SADS mengembalikan semua dosen beserta seluruh mata
    # kuliahnya (220 dosen, sekitar 98 KB; uji 2026-10-07). Endpointnya tidak
    # bisa disaring menurut dosen, jadi saringan nama dikerjakan di sini.
    per_dosen = _gabung_per_dosen(await client.get_json(PATH_MK_DIAMPU))
    if not per_dosen:
        return ToolResult(name="get_mk_dosen", label=LABEL_SADS, text="", ok=False)
    dicari = _SAPAAN.sub("", nama).strip()
    cocok = {
        nm: sorted(mk, key=str.casefold)
        for nm, mk in sorted(per_dosen.items())
        if _cocok(nm, nama=dicari, gelar="")
    }
    if not cocok:
        teks = (
            f'Tidak ada dosen pengampu dengan nama memuat "{dicari}": 0 orang '
            f"(dari {len(per_dosen)} dosen yang mengampu mata kuliah)."
        )
        return ToolResult(name="get_mk_dosen", label=LABEL_SADS, text=teks)
    if len(cocok) > MAKS_DOSEN_DIRINCI:
        teks = (
            f'Ada {len(cocok)} dosen dengan nama memuat "{dicari}", terlalu banyak '
            f"untuk dirinci mata kuliahnya (paling banyak {MAKS_DOSEN_DIRINCI}). "
            "Nama-nama dosen:\n" + "\n".join(f"- {nm}" for nm in cocok)
        )
        return ToolResult(name="get_mk_dosen", label=LABEL_SADS, text=teks)
    bagian = [
        f"Mata kuliah yang diampu {nm} (jumlah: {len(mk)} mata kuliah):\n"
        + "\n".join(f"- {m}" for m in mk)
        for nm, mk in cocok.items()
    ]
    lampiran: tuple[Lampiran, ...] = ()
    if len(cocok) == 1:
        # Satu dosen bisa mengampu 25 mata kuliah: daftarnya tampil langsung di
        # widget, model cukup merangkum (docs/tool-call.md §10a).
        [(nm, mk)] = cocok.items()
        judul = f"Mata kuliah yang diampu {nm}"
        lampiran = (Lampiran(title=judul, source=LABEL_SADS, items=tuple(mk)),)
    return ToolResult(
        name="get_mk_dosen", label=LABEL_SADS, text="\n\n".join(bagian), attachments=lampiran
    )


def tool_specs(settings: Settings) -> list[ToolSpec]:
    """Bangun ToolSpec SADS dengan handler yang sudah terikat ke satu klien."""
    secret = settings.kunci_sads()
    client = SadsClient(
        base_url=settings.sads_base_url or "",
        secret=secret.get_secret_value() if secret else "",
        timeout=settings.sads_timeout_seconds,
    )
    return [
        ToolSpec(
            name="get_daftar_dosen",
            description=(
                "Daftar nama dosen yang mengajar di kampus INSTIKI beserta "
                "jumlahnya, dapat disaring menurut nama atau gelar. Untuk "
                "pertanyaan seperti 'berapa dosen bergelar Dr.' atau 'dosen yang "
                "bernama Wayan', isi argumen saringnya; jangan menghitung atau "
                "menyaring sendiri dari daftar lengkap. Tanpa argumen = semua dosen."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "nama": {
                        "type": "string",
                        "description": (
                            "bagian nama dosen, tanpa gelar, mis. 'Wayan'. "
                            "Kosongkan bila tidak menyaring nama."
                        ),
                    },
                    "gelar": {
                        "type": "string",
                        "description": (
                            "satu gelar akademik persis seperti singkatannya, mis. "
                            "'Dr.', 'Ir.', 'Ph.D.', 'M.Kom.'. Kosongkan bila tidak "
                            "menyaring gelar."
                        ),
                    },
                },
            },
            handler=partial(_daftar_dosen, client),
            triggers=("daftar dosen", "nama dosen", "siapa saja dosen", "dosen di"),
            citation_label=LABEL_SADS,
        ),
        ToolSpec(
            name="get_mk_diampu_dosen",
            # Petunjuk "coba nama Inggris" ditaruh di sini karena isi pesan tool
            # adalah data (aturan T2). Kalimat "bukan nama dosen" ada karena
            # model pernah mengisi `matkul="Ahmad Asroni"` (T48, 2026-10-07).
            description=(
                "Daftar dosen pengampu suatu mata kuliah di INSTIKI, dicari menurut "
                "NAMA MATA KULIAH, bukan nama dosen (untuk mata kuliah yang diampu "
                "seorang dosen, pakai get_mk_dosen). 'matkul' dicocokkan sebagai "
                "potongan teks persis pada nama mata kuliah di SADS, dan sebagian "
                "nama itu berbahasa Inggris, mis. 'Artificial Intelligence' "
                "(Kecerdasan Buatan), 'Algorithms', 'Database'. Satu mata kuliah "
                "per panggilan; untuk beberapa mata kuliah, panggil sekali per mata "
                "kuliah. Bila hasilnya 0 dosen, panggil lagi dengan padanan Inggris, "
                "sinonim, atau kata kunci yang lebih pendek sebelum menyimpulkan "
                "mata kuliah itu tidak ada. Bila padanannya yang ditemukan, jawab "
                "dengan dosen dari hasil itu seperti biasa dan sebut nama yang "
                "tercatat di SADS, mis. 'Kecerdasan Buatan (tercatat sebagai "
                "Artificial Intelligence)'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "matkul": {
                        "type": "string",
                        "description": (
                            "nama atau kata kunci SATU mata kuliah, mis. "
                            "'Basis Data', 'Programming'"
                        ),
                    }
                },
                "required": ["matkul"],
            },
            handler=partial(_mk_diampu_dosen, client),
            triggers=(
                "mata kuliah",
                "matkul",
                "mengampu",
                "pengampu",
                "mengajar",
                "ngajar",
                "dosen",
                # Pertanyaan Inggris pendek tidak di-rewrite (`looks_english` butuh
                # >= 2 kata tugas): "who teaches Web Programming?" ditolak FR-3
                # sebelum pemicu ini ada (uji live 2026-10-07).
                "lecturer",
                "teach",
            ),
            citation_label=LABEL_SADS,
        ),
        ToolSpec(
            name="get_mk_dosen",
            description=(
                "Daftar mata kuliah yang diampu SEORANG dosen INSTIKI, dicari menurut "
                "NAMA DOSEN, beserta jumlahnya. Untuk pertanyaan seperti 'mata kuliah "
                "apa yang diajar Pak X' atau 'apakah Bu Y mengajar Basis Data'. Nama "
                "yang terlalu umum (cocok dengan banyak dosen) hanya mengembalikan "
                "nama-nama dosennya."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "nama": {
                        "type": "string",
                        "description": (
                            "bagian nama dosen tanpa sapaan (Pak/Bu) dan tanpa "
                            "gelar, mis. 'Ahmad Asroni' atau 'Totok'"
                        ),
                    }
                },
                "required": ["nama"],
            },
            handler=partial(_mk_dosen, client),
            # Bukan "course": "how do I register for courses (KRS)?" bukan
            # pertanyaan data dosen. "What courses does X teach?" tertangkap "teach".
            triggers=("diajar", "diampu", "mata kuliah", "matkul"),
            citation_label=LABEL_SADS,
        ),
    ]
