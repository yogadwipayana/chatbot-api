"""Tool SADS perdana: daftar dosen dan dosen pengampu mata kuliah.

Handler tipis: panggil endpoint, normalisasi (data SADS punya spasi & koma di
ujung `nmdosen`), ratakan jadi teks ringkas. Menambah endpoint = menambah satu
handler + satu `ToolSpec` di sini, lalu registry memungutnya (docs/tool-call.md §16).
"""

from __future__ import annotations

import re
from functools import partial

from app.config import Settings
from app.rag.tools.base import Lampiran, ToolResult, ToolSpec
from app.rag.tools.client import SadsClient

LABEL_SADS = "Data akademik SADS"

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
        attachment=Lampiran(title=judul, source=LABEL_SADS, items=tuple(cocok)),
    )


async def _mk_diampu_dosen(client: SadsClient, *, matkul: str) -> ToolResult:
    data = await client.get_json(
        "/service/tp/chatbot/mk-diampu-dosen", params={"matkul": matkul}
    )
    baris: list[str] = []
    for d in data:
        nm = _bersih(d.get("nmdosen"))
        mk = [_bersih(m.get("matkul")) for m in (d.get("matkul") or [])]
        mk = [m for m in mk if m]
        if nm and mk:
            baris.append(f"- {nm}: {', '.join(mk)}")
    teks = (
        f'Dosen pengampu untuk mata kuliah yang cocok dengan "{matkul}" '
        f"(jumlah: {len(baris)} orang):\n" + "\n".join(baris)
    )
    return ToolResult(name="get_mk_diampu_dosen", label=LABEL_SADS, text=teks, ok=bool(baris))


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
            description=(
                "Daftar dosen pengampu suatu mata kuliah di INSTIKI. Berikan "
                "argumen 'matkul' berisi nama atau kata kunci mata kuliah "
                "(mis. 'Programming', 'Basis Data')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "matkul": {
                        "type": "string",
                        "description": "nama atau kata kunci mata kuliah",
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
    ]
