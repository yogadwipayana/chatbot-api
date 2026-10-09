"""Batas ukuran body lewat aplikasi utuh (`create_app`), bukan middleware saja.

Yang diuji di sini rangkaiannya: middleware terpasang, 413 ditolak sebelum
handler (dan LLM) berjalan, dan header CORS tetap ada pada penolakan itu --
tanpanya dashboard hanya melihat galat jaringan.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.security.batas_body import BATAS_UMUM, TERLALU_BESAR

JSON = {"Content-Type": "application/json"}
ADMIN = "https://admin.dwipa.my.id"


class TestChat:
    def test_permintaan_biasa_tidak_terpengaruh(self, client, payload):
        assert client.post("/api/chat", json=payload).status_code == 200

    def test_content_length_besar_ditolak_sebelum_llm(self, client, api_llm):
        resp = client.post("/api/chat", content=b"x" * (BATAS_UMUM + 1), headers=JSON)
        assert resp.status_code == 413
        assert resp.json()["detail"] == TERLALU_BESAR
        assert api_llm.calls == []

    def test_body_chunked_tanpa_content_length_ditolak(self, client, api_llm):
        """Generator = Transfer-Encoding chunked: hanya penghitung yang menahannya."""
        resp = client.post("/api/chat", content=iter([b"x" * (BATAS_UMUM + 1)]), headers=JSON)
        assert resp.status_code == 413
        assert resp.json()["detail"] == TERLALU_BESAR
        assert api_llm.calls == []


@pytest.fixture
def client_unggah_1mb(monkeypatch) -> TestClient:
    settings = Settings(_env_file=None, cors_origins=ADMIN, max_upload_mb=1)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    return TestClient(create_app())


class TestUnggah:
    def test_berkas_di_atas_batas_ditolak_dengan_pesan_berkas(self, client_unggah_1mb):
        resp = client_unggah_1mb.post(
            "/api/admin/documents",
            files={"file": ("besar.pdf", b"%PDF-" + b"x" * (2 * 1024 * 1024 + 1))},
            data={"title": "Dokumen besar", "unit": "Keuangan"},
        )
        assert resp.status_code == 413
        assert "Ukuran maksimum 1 MB" in resp.json()["detail"]

    def test_413_tetap_membawa_header_cors(self, client_unggah_1mb):
        resp = client_unggah_1mb.post(
            "/api/chat",
            content=b"x" * (BATAS_UMUM + 1),
            headers={**JSON, "Origin": ADMIN},
        )
        assert resp.status_code == 413
        assert resp.headers["access-control-allow-origin"] == ADMIN


def test_unggah_di_bawah_batas_sampai_ke_handler(client, admin_headers):
    """Lebih besar dari batas umum, masih di bawah MAX_UPLOAD_MB: handler yang
    menilainya (di sini menolak karena isinya bukan PDF), bukan middleware."""
    resp = client.post(
        "/api/admin/documents",
        files={"file": ("bukan.pdf", b"x" * (BATAS_UMUM * 2))},
        data={"title": "Bukan PDF", "unit": "Keuangan"},
        headers=admin_headers,
    )
    assert resp.status_code == 422
    assert "bukan berkas PDF" in resp.json()["detail"]
