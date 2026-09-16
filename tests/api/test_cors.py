"""Preflight CORS untuk front-end yang berada di domain terpisah.

Diuji lewat HTTP sungguhan, bukan lewat `cors_origin_list()` saja: yang rusak
di produksi adalah rangkaiannya. Daftar asal, daftar metode, dan daftar header
semuanya harus cocok dengan apa yang dikirim peramban -- satu saja meleset dan
`fetch()` gagal sebelum permintaan sebenarnya berangkat, dengan galat yang
tidak menyebut bagian mana yang salah.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

ADMIN = "https://admin.dwipa.my.id"
PORTAL = "https://sads.dwipa.my.id"
PRODUKSI = f"{ADMIN},{PORTAL}"


@pytest.fixture
def buat_client(monkeypatch):
    """TestClient dengan Settings yang ditentukan test, bukan dari .env mesin."""

    def factory(**overrides) -> TestClient:
        settings = Settings(_env_file=None, **overrides)
        monkeypatch.setattr("app.main.get_settings", lambda: settings)
        return TestClient(create_app())

    return factory


@pytest.fixture
def client_produksi(buat_client) -> TestClient:
    return buat_client(
        environment="production",
        cors_origins=PRODUKSI,
        api_key="sk-uji",
        admin_jwt_secret="x" * 48,
    )


def preflight(client: TestClient, origin: str, *, jalur: str = "/api/chat/stream"):
    """Preflight persis seperti yang dikirim peramban untuk `POST` + header kustom."""
    return client.options(
        jalur,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-session-id",
        },
    )


class TestPreflight:
    @pytest.mark.parametrize("origin", [ADMIN, PORTAL])
    def test_asal_terdaftar_diizinkan(self, client_produksi, origin):
        resp = preflight(client_produksi, origin)
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == origin

    def test_header_kustom_client_diizinkan(self, client_produksi):
        """`X-Session-Id` dan `Content-Type: application/json` membuat permintaan
        chat non-simple; keduanya harus disebut dalam balasan preflight."""
        izin = preflight(client_produksi, PORTAL).headers["access-control-allow-headers"]
        diizinkan = {h.strip().lower() for h in izin.split(",")}
        assert {"content-type", "x-session-id"} <= diizinkan

    def test_authorization_diizinkan_untuk_dashboard(self, client_produksi):
        """Dashboard mengirim bearer token di setiap permintaan."""
        resp = client_produksi.options(
            "/api/admin/documents",
            headers={
                "Origin": ADMIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == ADMIN

    @pytest.mark.parametrize("metode", ["GET", "POST", "PATCH", "DELETE"])
    def test_semua_metode_yang_dipakai_dashboard(self, client_produksi, metode):
        resp = client_produksi.options(
            "/api/admin/documents",
            headers={
                "Origin": ADMIN,
                "Access-Control-Request-Method": metode,
                "Access-Control-Request-Headers": "authorization",
            },
        )
        assert resp.status_code == 200

    def test_asal_asing_ditolak(self, client_produksi):
        """Tanpa ini daftar asal hanya dekorasi."""
        resp = preflight(client_produksi, "https://penyerang.example")
        assert "access-control-allow-origin" not in resp.headers


class TestPermintaanSebenarnya:
    def test_balasan_menyebut_asal_pemanggil(self, client_produksi):
        resp = client_produksi.get("/health", headers={"Origin": PORTAL})
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == PORTAL

    def test_retry_after_terbaca_front_end(self, client_produksi):
        """`Retry-After` (429) tidak terlihat oleh JavaScript kecuali di-expose."""
        resp = client_produksi.get("/health", headers={"Origin": PORTAL})
        assert "Retry-After" in resp.headers["access-control-expose-headers"]


class TestLokal:
    def test_dev_tetap_jalan_meski_daftar_produksi_diisi(self, buat_client):
        client = buat_client(environment="local", cors_origins=PRODUKSI)
        resp = preflight(client, "http://localhost:3001")
        assert resp.headers["access-control-allow-origin"] == "http://localhost:3001"
