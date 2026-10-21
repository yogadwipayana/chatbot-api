"""Jaring pengaman untuk galat yang tidak ditangani handler mana pun (H33; OWASP A09, A10).

Tanpa ini, galat tak tertangani dijawab Starlette dengan teks `Internal Server
Error` dari lapisan terluarnya, yaitu di luar CORS. Dashboard tidak dapat membaca
balasan itu dan menampilkan "Tidak dapat terhubung ke server". Galatnya pun hanya
tercetak di konsol uvicorn, tidak masuk Log aplikasi dashboard.

Di sini galat dicatat lengkap dengan traceback lewat logger `app.*`, sehingga
masuk SQLite (beserta `turn_id` bila terjadi di tengah giliran chat). Lalu
galat dijawab JSON 500 berisi kalimat umum, tanpa rincian teknis (CWE-209).

Middleware ini dipasang di DALAM CORS, supaya balasan 500 tetap membawa header
CORS. Balasan yang sudah mulai terkirim, yaitu aliran SSE, tidak dapat diganti.
Galatnya diteruskan apa adanya: giliran chat sudah mencatatnya sendiri
(`applog.catat_giliran`), dan widget menampilkan pesan "terputus".
"""

from __future__ import annotations

import logging

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

GALAT_SERVER = "Terjadi gangguan di server. Coba lagi beberapa saat lagi."
"""Sama bunyinya dengan kalimat 5xx widget (`client/src/lib/api/client.ts`)."""


class TangkapGalat:
    """Catat galat tak tertangani, lalu balas 500 JSON bila balasan belum dimulai."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        dimulai = False

        async def kirim(message: Message) -> None:
            nonlocal dimulai
            if message["type"] == "http.response.start":
                dimulai = True
            await send(message)

        try:
            await self.app(scope, receive, kirim)
        except Exception:
            if dimulai:
                raise
            # Tanpa query string: isinya bisa memuat teks pencarian admin.
            logger.exception(
                "Galat tak tertangani pada %s %s", scope.get("method"), scope.get("path")
            )
            await JSONResponse({"detail": GALAT_SERVER}, status_code=500)(scope, receive, send)
