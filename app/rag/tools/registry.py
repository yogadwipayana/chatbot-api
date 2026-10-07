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
        """ToolSpec yang pemicunya cocok dengan pertanyaan (kosong = tidak eligible)."""
        return [s for s in self._specs if s.cocok(pertanyaan)]


def build_registry(settings: Settings) -> ToolRegistry:
    from app.rag.tools.sads import tool_specs

    return ToolRegistry(tool_specs(settings))
