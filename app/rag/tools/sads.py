"""Tool SADS perdana: daftar dosen dan dosen pengampu mata kuliah.

Handler tipis: panggil endpoint, normalisasi (data SADS punya spasi & koma di
ujung `nmdosen`), ratakan jadi teks ringkas. Menambah endpoint = menambah satu
handler + satu `ToolSpec` di sini, lalu registry memungutnya (docs/tool-call.md §16).
"""

from __future__ import annotations

from functools import partial

from app.config import Settings
from app.rag.tools.base import ToolResult, ToolSpec
from app.rag.tools.client import SadsClient

LABEL_SADS = "Data akademik SADS"


def _bersih(nama: str | None) -> str:
    """Buang spasi dan koma di ujung (data SADS: "Aulia Iefan Datya, ST.,MT   ")."""
    return (nama or "").strip().rstrip(",").strip()


async def _daftar_dosen(client: SadsClient) -> ToolResult:
    data = await client.get_json("/service/tp/chatbot/dosen-mengajar")
    nama = sorted({_bersih(d.get("nmdosen")) for d in data if _bersih(d.get("nmdosen"))})
    # Jumlahnya dihitung di sini, bukan oleh model: LLM tidak andal menghitung
    # daftar panjang. Terbukti 2026-10-07: dari 220 nama, model menjawab
    # "Terdapat 223 dosen", lengkap dengan kartu sumber yang membuatnya tampak sah.
    teks = (
        f"Jumlah dosen yang mengajar di INSTIKI: {len(nama)} orang.\n"
        "Daftar dosen:\n" + "\n".join(f"- {n}" for n in nama)
    )
    return ToolResult(name="get_daftar_dosen", label=LABEL_SADS, text=teks, ok=bool(nama))


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
                "Daftar nama dosen yang mengajar di kampus INSTIKI. "
                "Tidak memerlukan argumen."
            ),
            parameters={"type": "object", "properties": {}},
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
