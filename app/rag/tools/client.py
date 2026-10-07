"""Klien HTTP untuk layanan akademik SADS (docs/tool-call.md §7).

Auth lewat header `secret` (bukan Bearer). URL endpoint tetap dan terdaftar di
sini -- model tidak pernah memberi URL, hanya argumen query (anti-SSRF, §11).
"""

from __future__ import annotations

from typing import Any

import httpx

from app.rag.providers import USER_AGENT


class SadsClient:
    """GET berautentikasi ke SADS. Satu `AsyncClient` per panggilan, seperti
    klien JEV (`app.rag.gate`); timeout tegas supaya giliran mahasiswa tidak
    tertahan oleh layanan yang lambat."""

    def __init__(self, *, base_url: str, secret: str, timeout: float) -> None:
        self._base = base_url.rstrip("/")
        self._secret = secret
        self._timeout = timeout

    async def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        headers = {
            "secret": self._secret,
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self._base}{path}", params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()
