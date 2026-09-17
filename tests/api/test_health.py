"""Health check -- dipakai Uptime Kuma (PRD §10)."""

from __future__ import annotations

from app.routers import health


class TestHealth:
    def test_200_saat_normal(self, client):
        assert client.get("/health").status_code == 200

    def test_melaporkan_chat_aktif(self, client):
        assert client.get("/health").json()["chat_enabled"] is True

    def test_tetap_200_saat_kill_switch_aktif(self, client, kill_switch):
        """Kill switch adalah keputusan operator, bukan kegagalan sistem.

        Melaporkannya sebagai unhealthy membuat monitoring membanjiri alert
        justru saat insiden sedang ditangani manusia.
        """
        kill_switch.engage("insiden")
        assert client.get("/health").status_code == 200

    def test_melaporkan_chat_nonaktif_saat_kill_switch(self, client, kill_switch):
        kill_switch.engage("insiden jawaban salah")
        data = client.get("/health").json()
        assert data["chat_enabled"] is False
        assert data["kill_switch_reason"] == "insiden jawaban salah"

    def test_alasan_kosong_saat_normal(self, client):
        assert client.get("/health").json()["kill_switch_reason"] is None

    def test_melaporkan_tracing_mati(self, client):
        """Tracing mati tidak menjatuhkan permintaan apa pun, jadi ia tidak pernah
        muncul sebagai insiden. Satu-satunya cara operator menyadarinya adalah
        bila status itu ikut dilaporkan (FR-8)."""
        assert client.get("/health").json()["tracing_enabled"] is False

    def test_melaporkan_tracing_aktif(self, client, monkeypatch):
        monkeypatch.setattr(health, "sedang_menjejak", lambda: True)
        assert client.get("/health").json()["tracing_enabled"] is True
