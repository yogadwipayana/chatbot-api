"""Registry tool + rute kelayakan (docs/tool-call.md §4, §8).

Deklaratif: menambah tool tidak menyentuh loop maupun graf, cukup mendaftarkan
`ToolSpec` baru lewat `tool_specs` masing-masing layanan.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.config import Settings
from app.rag.tools.base import ToolSpec


class ToolRegistry:
    def __init__(self, specs: Sequence[ToolSpec]) -> None:
        self._specs = list(specs)
        self._by_name = {s.name: s for s in self._specs}

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    def get(self, name: str) -> ToolSpec | None:
        return self._by_name.get(name)

    def eligible(self, pertanyaan: str) -> list[ToolSpec]:
        """SEMUA tool bila ADA satu pemicu yang cocok; [] bila tidak ada.

        Sengaja semua, bukan hanya yang pemicunya cocok: pemicu kata kunci
        menentukan APAKAH pertanyaan masuk jalur tool, bukan tool MANA yang
        boleh dilihat model -- pemilihan tool adalah tugas model lewat
        `description` masing-masing.

        Tanpa ini pemicu yang tidak simetris menyembunyikan tool yang benar.
        Terbukti: "ada berapa dosen?" hanya cocok pemicu `get_mk_diampu_dosen`
        (yang mewajibkan `matkul`), sementara `get_daftar_dosen` -- satu-satunya
        tool yang bisa menjawabnya -- tidak ter-bind, sehingga model harus
        mengarang `matkul` atau menolak pertanyaan yang sebetulnya terjawab.

        Catatan skala: dengan satu layanan (SADS) membuka semuanya paling tepat.
        Begitu registry memuat banyak layanan, kelompokkan per layanan dan bind
        hanya kelompok yang cocok, supaya skema tool tidak membebani tiap prompt.
        """
        return self.specs if any(s.cocok(pertanyaan) for s in self._specs) else []


def build_registry(settings: Settings) -> ToolRegistry:
    from app.rag.tools.sads import tool_specs

    return ToolRegistry(tool_specs(settings))
