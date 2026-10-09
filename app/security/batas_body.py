"""Batas ukuran body permintaan (FR-9).

FastAPI membaca SELURUH body sebelum dependency apa pun berjalan -- termasuk
batas laju -- dan Pydantic baru memotong `TurnIn.content` setelah body selesai
diurai. Tanpa batas di depan, satu permintaan ratusan megabita (Cloudflare
meneruskan sampai 100 MB) ditampung utuh di memori sebelum ada yang sempat
menolaknya; unggahan multipart malah ditulis ke disk sementara lebih dulu.
Di server, uvicorn menerima lalu lintas langsung dari cloudflared, tanpa Caddy
yang membatasi ukuran body.

Dua lapis:

- `Content-Length` yang sudah melewati batas ditolak 413 tanpa membaca body.
- Body tanpa `Content-Length` (chunked) dihitung selagi mengalir; begitu
  melewati batas, pembacaan dihentikan dengan 413.

Middleware ASGI murni, bukan `BaseHTTPMiddleware`, supaya respons SSE
`/api/chat/stream` tetap mengalir tanpa ditampung.
"""

from __future__ import annotations

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

BATAS_UMUM = 256 * 1024
"""Body JSON terbesar yang sah jauh di bawah ini: pertanyaan 500 karakter plus
tiga giliran riwayat x 2000 karakter dari portal, atau jawaban tanya jawab
5000 karakter dari dashboard. Sisanya ruang untuk pemanggil langsung yang
mengirim 50 giliran riwayat utuh (`ChatRequest.history`)."""

CADANGAN_MULTIPART = 1024 * 1024
"""Tambahan di atas `MAX_UPLOAD_MB` untuk field judul/unit/tahun dan pembatas
multipart. Ukuran berkas yang persis tetap ditegakkan handler unggah, dengan
pesan yang sama."""

JALUR_UNGGAH = frozenset({("POST", "/api/admin/documents")})
"""Satu-satunya endpoint yang menerima berkas (`admin_documents.upload_document`)."""

TERLALU_BESAR = "Permintaan terlalu besar."


def pesan_berkas_terlalu_besar(max_upload_mb: int) -> str:
    return f"Berkas terlalu besar. Ukuran maksimum {max_upload_mb} MB."


class BatasUkuranBody:
    """Tolak 413 body yang melewati batas jalurnya, sebelum aplikasi membacanya."""

    def __init__(
        self, app: ASGIApp, *, max_upload_mb: int, batas_umum: int = BATAS_UMUM
    ) -> None:
        self.app = app
        self.batas_umum = batas_umum
        self.batas_unggah = max_upload_mb * 1024 * 1024 + CADANGAN_MULTIPART
        self.pesan_unggah = pesan_berkas_terlalu_besar(max_upload_mb)

    def _batas(self, scope: Scope) -> tuple[int, str]:
        if (scope["method"], scope["path"]) in JALUR_UNGGAH:
            return self.batas_unggah, self.pesan_unggah
        return self.batas_umum, TERLALU_BESAR

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        batas, pesan = self._batas(scope)
        panjang = _content_length(scope)
        if panjang is not None and panjang > batas:
            await JSONResponse({"detail": pesan}, status_code=413)(scope, receive, send)
            return

        diterima = 0

        async def receive_terbatas() -> Message:
            nonlocal diterima
            message = await receive()
            if message["type"] == "http.request":
                diterima += len(message.get("body", b""))
                if diterima > batas:
                    # FastAPI meneruskan HTTPException yang muncul saat membaca
                    # body apa adanya (`fastapi.routing`), jadi ini menjadi 413
                    # biasa lewat ExceptionMiddleware -- bukan 400 "error
                    # parsing the body".
                    raise HTTPException(413, pesan)
            return message

        await self.app(scope, receive_terbatas, send)


def _content_length(scope: Scope) -> int | None:
    """Nilai `Content-Length`, atau None bila tidak ada atau bukan angka.

    Nilai yang rusak dibiarkan lewat: uvicorn sendiri menolaknya, dan lapis
    penghitung di atas tetap berlaku."""
    for nama, nilai in scope["headers"]:
        if nama == b"content-length":
            try:
                return int(nilai)
            except ValueError:
                return None
    return None
