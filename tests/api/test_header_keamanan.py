"""Konfigurasi HTTP API: header keamanan, halaman dokumentasi, dan galat tak
tertangani (H22, H23, H33; OWASP A02, A04, A09, A10)."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from app.config import Settings
from app.deps import build_llm_call
from app.main import create_app
from app.security.header_keamanan import CSP_JSON, HSTS, HeaderKeamanan
from app.security.tangkap_galat import GALAT_SERVER, TangkapGalat

ADMIN = "https://admin.dwipa.my.id"


def settings_server(**lain) -> Settings:
    return Settings(
        _env_file=None,
        environment="production",
        admin_jwt_secret="x" * 48,
        api_key="sk-uji",
        **lain,
    )


@pytest.fixture
def buat_client(monkeypatch):
    """TestClient dengan Settings yang ditentukan test, bukan dari .env mesin."""

    def factory(settings: Settings) -> TestClient:
        monkeypatch.setattr("app.main.get_settings", lambda: settings)
        return TestClient(create_app())

    return factory


class TestHeaderKeamanan:
    def test_balasan_json_membawa_header_dasar(self, buat_client):
        r = buat_client(settings_server()).get("/health")
        assert r.status_code == 200
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "no-referrer"
        assert r.headers["content-security-policy"] == CSP_JSON

    def test_hsts_hanya_di_server(self, buat_client):
        assert (
            buat_client(settings_server()).get("/health").headers["strict-transport-security"]
            == HSTS
        )
        lokal = buat_client(Settings(_env_file=None, environment="local"))
        assert "strict-transport-security" not in lokal.get("/health").headers

    def test_balasan_admin_tidak_disimpan_cache(self, buat_client):
        """Daftar akun dan log tidak boleh tertinggal di cache komputer bersama."""
        client = buat_client(settings_server())
        assert client.get("/api/admin/me").headers["cache-control"] == "no-store"
        assert "cache-control" not in client.get("/health").headers

    def test_balasan_galat_dan_preflight_juga_membawa_header(self, buat_client):
        client = buat_client(settings_server(cors_origins=ADMIN))
        r = client.options(
            "/api/admin/me",
            headers={"Origin": ADMIN, "Access-Control-Request-Method": "GET"},
        )
        assert r.status_code == 200
        assert r.headers["x-content-type-options"] == "nosniff"
        assert client.get("/api/admin/me").headers["x-frame-options"] == "DENY"

    def test_header_milik_handler_tidak_ditimpa(self, client, payload):
        """SSE memasang `Cache-Control: no-cache` sendiri; CSP JSON tidak untuknya."""
        r = client.post("/api/chat/stream", json=payload)
        assert r.headers["cache-control"] == "no-cache"
        assert "content-security-policy" not in r.headers
        assert r.headers["x-content-type-options"] == "nosniff"

    def test_pdf_tanpa_csp_supaya_penampil_pdf_tetap_jalan(self):
        async def pdf(_):
            return Response(b"%PDF-1.7", media_type="application/pdf")

        app = HeaderKeamanan(Starlette(routes=[Route("/f", pdf)]), hsts=True)
        r = TestClient(app).get("/f")
        assert "content-security-policy" not in r.headers
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["strict-transport-security"] == HSTS


class TestDokumentasiApi:
    @pytest.mark.parametrize("jalur", ["/docs", "/redoc", "/openapi.json"])
    @pytest.mark.parametrize("environment", ["staging", "production"])
    def test_tertutup_di_server(self, buat_client, jalur, environment):
        """Dulu hanya /docs yang mati; /redoc dan /openapi.json tetap terbuka (H22)."""
        settings = Settings(
            _env_file=None,
            environment=environment,
            admin_jwt_secret="x" * 48,
            api_key="sk-uji",
        )
        assert buat_client(settings).get(jalur).status_code == 404

    @pytest.mark.parametrize("jalur", ["/docs", "/redoc", "/openapi.json"])
    def test_terbuka_di_mesin_pengembang(self, buat_client, jalur):
        client = buat_client(Settings(_env_file=None, environment="local"))
        assert client.get(jalur).status_code == 200

    def test_skema_tetap_dapat_dibangkitkan_kode(self, buat_client):
        """`test_openapi_contract.py` membaca `app.openapi()`, bukan rute HTTP-nya."""
        client = buat_client(settings_server())
        assert "/api/admin/login" in client.app.openapi()["paths"]


def _app_gagal() -> Starlette:
    """Aplikasi kecil dengan penangkap galat di posisi yang sama seperti di
    `create_app`: di dalam `ServerErrorMiddleware` milik Starlette."""

    async def rusak(_):
        raise ZeroDivisionError("rahasia-internal")

    async def aliran(_):
        async def isi():
            yield b"data: satu\n\n"
            raise ZeroDivisionError("di tengah aliran")

        return StreamingResponse(isi(), media_type="text/event-stream")

    async def biasa(_):
        return JSONResponse({"ok": True})

    rute = [Route("/rusak", rusak), Route("/aliran", aliran), Route("/biasa", biasa)]
    return Starlette(routes=rute, middleware=[Middleware(TangkapGalat)])


class TestTangkapGalat:
    def test_galat_tak_tertangani_menjadi_500_json_tanpa_rincian(self, caplog):
        caplog.set_level(logging.ERROR, logger="app.security.tangkap_galat")
        r = TestClient(_app_gagal()).get("/rusak")
        assert r.status_code == 500
        assert r.json() == {"detail": GALAT_SERVER}
        assert "rahasia-internal" not in r.text
        catatan = [x for x in caplog.records if x.name == "app.security.tangkap_galat"]
        assert len(catatan) == 1
        assert catatan[0].exc_info and "GET /rusak" in catatan[0].getMessage()

    def test_balasan_normal_tidak_tersentuh(self):
        assert TestClient(_app_gagal()).get("/biasa").json() == {"ok": True}

    def test_aliran_yang_sudah_dimulai_tidak_diganti(self):
        """Status 200 sudah terkirim: galatnya diteruskan, bukan disisipi balasan kedua."""
        with pytest.raises(ZeroDivisionError):
            TestClient(_app_gagal()).get("/aliran")

    def test_500_tetap_membawa_cors_dan_header_keamanan(
        self, monkeypatch, make_client, strong_documents, admin_headers
    ):
        """Dulu 500 keluar di luar CORS: dashboard melihat "tidak dapat terhubung"."""

        class LLMRusak:
            async def __call__(self, wrapped_question, documents):
                raise ZeroDivisionError

        monkeypatch.setattr(
            "app.main.get_settings", lambda: Settings(_env_file=None, cors_origins=ADMIN)
        )
        client = make_client(strong_documents)
        client.app.dependency_overrides[build_llm_call] = LLMRusak
        r = client.post(
            "/api/admin/test-query",
            json={"question": "kapan pengisian KRS dibuka?"},
            headers={**admin_headers, "Origin": ADMIN},
        )
        assert r.status_code == 500
        assert r.json()["detail"] == GALAT_SERVER
        assert r.headers["access-control-allow-origin"] == ADMIN
        assert r.headers["cache-control"] == "no-store"
