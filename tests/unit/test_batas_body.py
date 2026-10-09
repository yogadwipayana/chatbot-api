"""Batas ukuran body (`app.security.batas_body`) sebagai middleware ASGI murni."""

from __future__ import annotations

import pytest
from starlette.exceptions import HTTPException

from app.security.batas_body import CADANGAN_MULTIPART, TERLALU_BESAR, BatasUkuranBody

MB = 1024 * 1024


class AplikasiPembaca:
    """Aplikasi ASGI yang membaca seluruh body, seperti FastAPI sebelum handler."""

    def __init__(self) -> None:
        self.dipanggil = False
        self.body = b""

    async def __call__(self, scope, receive, send) -> None:
        self.dipanggil = True
        while True:
            message = await receive()
            self.body += message.get("body", b"")
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def scope_http(path="/api/chat", method="POST", content_length: int | str | None = None):
    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    return {"type": "http", "method": method, "path": path, "headers": headers}


def pengirim_potongan(*potongan: bytes):
    antrean = list(potongan)

    async def receive():
        isi = antrean.pop(0)
        return {"type": "http.request", "body": isi, "more_body": bool(antrean)}

    return receive


class Perekam:
    def __init__(self) -> None:
        self.pesan: list[dict] = []

    async def __call__(self, message) -> None:
        self.pesan.append(message)

    @property
    def status(self) -> int:
        return next(m["status"] for m in self.pesan if m["type"] == "http.response.start")

    @property
    def isi(self) -> bytes:
        return b"".join(
            m.get("body", b"") for m in self.pesan if m["type"] == "http.response.body"
        )


async def jalankan(mw, scope, receive) -> Perekam:
    kirim = Perekam()
    await mw(scope, receive, kirim)
    return kirim


class TestContentLength:
    async def test_melewati_batas_ditolak_tanpa_memanggil_aplikasi(self):
        app = AplikasiPembaca()
        mw = BatasUkuranBody(app, max_upload_mb=1, batas_umum=100)
        kirim = await jalankan(mw, scope_http(content_length=101), pengirim_potongan(b""))
        assert kirim.status == 413
        assert TERLALU_BESAR.encode() in kirim.isi
        assert not app.dipanggil

    async def test_tepat_di_batas_diteruskan(self):
        app = AplikasiPembaca()
        mw = BatasUkuranBody(app, max_upload_mb=1, batas_umum=100)
        kirim = await jalankan(
            mw, scope_http(content_length=100), pengirim_potongan(b"x" * 100)
        )
        assert kirim.status == 200
        assert app.body == b"x" * 100

    async def test_nilai_rusak_jatuh_ke_penghitung(self):
        mw = BatasUkuranBody(AplikasiPembaca(), max_upload_mb=1, batas_umum=100)
        with pytest.raises(HTTPException) as galat:
            await jalankan(
                mw, scope_http(content_length="banyak"), pengirim_potongan(b"x" * 101)
            )
        assert galat.value.status_code == 413


class TestBodyMengalir:
    """Body tanpa `Content-Length` (chunked) dihitung per potongan."""

    async def test_potongan_di_bawah_batas_diteruskan_utuh(self):
        app = AplikasiPembaca()
        mw = BatasUkuranBody(app, max_upload_mb=1, batas_umum=100)
        kirim = await jalankan(
            mw, scope_http(), pengirim_potongan(b"a" * 40, b"b" * 40, b"c" * 20)
        )
        assert kirim.status == 200
        assert app.body == b"a" * 40 + b"b" * 40 + b"c" * 20

    async def test_berhenti_begitu_jumlah_melewati_batas(self):
        app = AplikasiPembaca()
        mw = BatasUkuranBody(app, max_upload_mb=1, batas_umum=100)
        with pytest.raises(HTTPException) as galat:
            await jalankan(
                mw, scope_http(), pengirim_potongan(b"a" * 60, b"b" * 60, b"c" * 60)
            )
        assert galat.value.status_code == 413
        assert galat.value.detail == TERLALU_BESAR
        # Potongan ketiga tidak pernah diminta.
        assert app.body == b"a" * 60


class TestJalurUnggah:
    async def test_unggah_memakai_batas_berkas(self):
        """1 MB berkas + cadangan form: 1,5 MB lolos ke handler unggah."""
        app = AplikasiPembaca()
        mw = BatasUkuranBody(app, max_upload_mb=1, batas_umum=100)
        ukuran = MB + MB // 2
        kirim = await jalankan(
            mw,
            scope_http("/api/admin/documents", content_length=ukuran),
            pengirim_potongan(b"x" * ukuran),
        )
        assert kirim.status == 200

    async def test_unggah_di_atas_batas_ditolak_dengan_pesan_berkas(self):
        app = AplikasiPembaca()
        mw = BatasUkuranBody(app, max_upload_mb=1, batas_umum=100)
        kirim = await jalankan(
            mw,
            scope_http("/api/admin/documents", content_length=MB + CADANGAN_MULTIPART + 1),
            pengirim_potongan(b""),
        )
        assert kirim.status == 413
        assert b"Ukuran maksimum 1 MB" in kirim.isi
        assert not app.dipanggil

    async def test_jalur_lain_tetap_batas_umum(self):
        """Batas besar hanya untuk POST unggah, bukan PATCH dokumen yang sama."""
        mw = BatasUkuranBody(AplikasiPembaca(), max_upload_mb=1, batas_umum=100)
        kirim = await jalankan(
            mw,
            scope_http("/api/admin/documents", method="PATCH", content_length=101),
            pengirim_potongan(b""),
        )
        assert kirim.status == 413


async def test_selain_http_diteruskan_apa_adanya():
    terima = []

    async def app(scope, receive, send):
        terima.append(scope["type"])

    mw = BatasUkuranBody(app, max_upload_mb=1)
    await mw({"type": "lifespan"}, None, None)
    assert terima == ["lifespan"]
