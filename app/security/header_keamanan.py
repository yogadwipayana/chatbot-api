"""Header keamanan untuk setiap balasan API (H23; OWASP A02, A04).

API ini hanya menyajikan JSON, aliran SSE, dan PDF sumber: tidak ada halaman
yang perlu dibingkai, menjalankan skrip, atau disimpan di cache peramban.
Header di sini menegaskan itu kepada peramban:

- `X-Content-Type-Options: nosniff`: jenis balasan tidak ditebak ulang, mis.
  teks yang dirender sebagai HTML.
- `X-Frame-Options: DENY`: tidak dapat dibingkai situs lain. Kartu sitasi
  membuka PDF di tab baru, bukan di iframe.
- CSP `default-src 'none'; frame-ancestors 'none'` untuk JSON: kalaupun dibuka
  sebagai dokumen, tidak ada yang dapat dimuat atau dijalankan. Tidak untuk PDF,
  karena penampil PDF peramban perlu memuat dirinya sendiri.
- `Referrer-Policy: no-referrer`.
- `Cache-Control: no-store` untuk `/api/admin/*`: daftar akun, log, dan
  statistik tidak boleh tertinggal di cache peramban komputer bersama.
- `Strict-Transport-Security` di luar `ENVIRONMENT=local`: setelah kunjungan
  pertama, peramban hanya memakai HTTPS ke domain API. Header ini diabaikan
  bila diterima lewat HTTP, jadi akses LAN `http://` tidak terpengaruh.

Ini middleware ASGI murni, sama seperti `batas_body`, supaya SSE tetap
mengalir. Header yang sudah dipasang handler tidak ditimpa, mis.
`Cache-Control: no-cache` milik SSE.
"""

from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

HSTS = "max-age=31536000; includeSubDomains"
CSP_JSON = "default-src 'none'; frame-ancestors 'none'"
AWALAN_ADMIN = "/api/admin"


class HeaderKeamanan:
    """Tambahkan header keamanan ke balasan yang belum memilikinya."""

    def __init__(self, app: ASGIApp, *, hsts: bool) -> None:
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        admin = scope["path"].startswith(AWALAN_ADMIN)

        async def kirim(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                _pasang(headers, "X-Content-Type-Options", "nosniff")
                _pasang(headers, "X-Frame-Options", "DENY")
                _pasang(headers, "Referrer-Policy", "no-referrer")
                if headers.get("content-type", "").startswith("application/json"):
                    _pasang(headers, "Content-Security-Policy", CSP_JSON)
                if admin:
                    _pasang(headers, "Cache-Control", "no-store")
                if self.hsts:
                    _pasang(headers, "Strict-Transport-Security", HSTS)
            await send(message)

        await self.app(scope, receive, kirim)


def _pasang(headers: MutableHeaders, nama: str, nilai: str) -> None:
    if nama not in headers:
        headers[nama] = nilai
