"""Batas pemakaian endpoint mahasiswa: laju, jatah harian, dan panjang masukan.

API chat publik tanpa login, jadi apa pun yang dilakukan portal dapat ditiru
Postman. Yang menahannya bukan kunci atau CORS, melainkan batas di sini:
per IP untuk satu sumber, jatah harian untuk banjir dari banyak sumber, dan
batas panjang supaya satu permintaan tidak dapat dibuat mahal.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.deps import TERLALU_BANYAK_PENILAIAN, TERLALU_BANYAK_PERTANYAAN
from app.security.batas_harian import OLEH_BATAS_HARIAN


@pytest.fixture
def atur(client):
    """Setelan tertentu untuk test ini, apa pun isi .env mesin ini."""

    def terapkan(**nilai):
        dasar = {
            "rate_limit_per_session": "0",
            "rate_limit_per_ip": "0",
            "rate_limit_per_embed_site": "0",
            "client_ip_header": "X-Forwarded-For",
            "chat_daily_limit": 0,
        }
        settings = get_settings().model_copy(update={**dasar, **nilai})
        client.app.dependency_overrides[get_settings] = lambda: settings
        return client

    return terapkan


def tanya(client, payload, *, sesi="sesi-uji-12345", ip="203.0.113.1", **headers):
    return client.post(
        "/api/chat",
        json={**payload, "session_id": sesi},
        headers={"X-Session-Id": sesi, "X-Forwarded-For": ip, **headers},
    )


class TestBatasLaju:
    def test_per_sesi(self, atur, payload):
        client = atur(rate_limit_per_session="2/minute")
        assert tanya(client, payload).status_code == 200
        assert tanya(client, payload).status_code == 200
        r = tanya(client, payload)
        assert r.status_code == 429
        assert r.json()["detail"] == TERLALU_BANYAK_PERTANYAAN
        assert 1 <= int(r.headers["Retry-After"]) <= 60
        assert tanya(client, payload, sesi="sesi-lain-67890").status_code == 200

    def test_per_ip_menahan_yang_berganti_sesi(self, atur, payload):
        """Postman cukup mengganti X-Session-Id; IP-nya tetap sama."""
        client = atur(rate_limit_per_ip="2/minute")
        for i in range(2):
            assert tanya(client, payload, sesi=f"sesi-acak-{i:05}").status_code == 200
        assert tanya(client, payload, sesi="sesi-acak-99999").status_code == 429
        assert tanya(client, payload, ip="198.51.100.7").status_code == 200

    def test_tanpa_header_sesi_tetap_kena_batas_ip(self, atur, payload):
        client = atur(rate_limit_per_ip="1/minute")
        kirim = {"json": payload, "headers": {"X-Forwarded-For": "203.0.113.1"}}
        assert client.post("/api/chat", **kirim).status_code == 200
        assert client.post("/api/chat", **kirim).status_code == 429

    def test_ip_palsu_di_depan_x_forwarded_for_tidak_menolong(self, atur, payload):
        client = atur(rate_limit_per_ip="1/minute")
        assert tanya(client, payload, ip="203.0.113.1").status_code == 200
        r = tanya(client, payload, ip="6.6.6.6, 203.0.113.1")
        assert r.status_code == 429

    def test_per_kunci_sematan(self, atur, payload, embed_keys):
        client = atur(rate_limit_per_embed_site="1/minute")
        kunci = embed_keys.add().key
        assert tanya(client, payload, **{"X-Embed-Key": kunci}).status_code == 200
        lain = {"sesi": "sesi-lain-67890", "ip": "198.51.100.7"}
        r = tanya(client, payload, **lain, **{"X-Embed-Key": kunci})
        assert r.status_code == 429
        # Portal sendiri, tanpa kunci, tidak ikut kena.
        assert tanya(client, payload, sesi="sesi-portal-1111").status_code == 200

    def test_streaming_ikut_dibatasi(self, atur, payload):
        client = atur(rate_limit_per_session="1/minute")
        kirim = {
            "json": payload,
            "headers": {
                "X-Session-Id": payload["session_id"],
                "X-Forwarded-For": "203.0.113.1",
            },
        }
        assert client.post("/api/chat/stream", **kirim).status_code == 200
        assert client.post("/api/chat/stream", **kirim).status_code == 429

    def test_yang_ditolak_tidak_memanggil_llm_dan_tidak_dicatat(
        self, atur, payload, api_llm, chat_logger
    ):
        client = atur(rate_limit_per_session="1/minute")
        tanya(client, payload)
        api_llm.calls.clear()
        jumlah_log = len(chat_logger.entries)
        assert tanya(client, payload).status_code == 429
        assert not api_llm.called
        assert len(chat_logger.entries) == jumlah_log

    def test_menilai_punya_jatah_sendiri(self, atur, payload):
        client = atur(rate_limit_per_session="1/minute")
        assert tanya(client, payload).status_code == 200
        assert tanya(client, payload).status_code == 429

        headers = {"X-Session-Id": "sesi-uji-12345", "X-Forwarded-For": "203.0.113.1"}
        # Body sengaja tidak sah: batas laju diperiksa lebih dulu, database tidak disentuh.
        body = {"message_id": "bukan-uuid", "helpful": True}
        assert client.post("/api/feedback", json=body, headers=headers).status_code == 422
        r = client.post("/api/feedback", json=body, headers=headers)
        assert r.status_code == 429
        assert r.json()["detail"] == TERLALU_BANYAK_PENILAIAN


class TestBatasHarian:
    def test_melewati_batas_menyalakan_kill_switch(self, atur, payload, kill_switch):
        client = atur(chat_daily_limit=2)
        assert tanya(client, payload).status_code == 200
        assert tanya(client, payload).status_code == 200
        r = tanya(client, payload)
        assert r.status_code == 503
        assert r.json()["detail"] == kill_switch.message
        assert kill_switch.engaged
        assert kill_switch.engaged_by == OLEH_BATAS_HARIAN
        assert "Batas harian 2 pertanyaan" in (kill_switch.reason or "")
        # Admin melihatnya di dashboard lewat status kill switch yang sama.
        assert tanya(client, payload, sesi="sesi-lain-67890").status_code == 503

    def test_menyalakan_lagi_tanpa_menaikkan_batas_mati_lagi(self, atur, payload, kill_switch):
        client = atur(chat_daily_limit=1)
        tanya(client, payload)
        tanya(client, payload)
        kill_switch.release()
        assert tanya(client, payload).status_code == 503
        assert kill_switch.engaged

    def test_menaikkan_batas_dari_konfigurasi_langsung_berlaku(
        self, atur, payload, kill_switch, runtime_config, admin_headers
    ):
        client = atur(chat_daily_limit=1)
        tanya(client, payload)
        assert tanya(client, payload).status_code == 503

        r = client.patch(
            "/api/admin/config", json={"chat_daily_limit": 100}, headers=admin_headers
        )
        assert r.status_code == 200, r.text
        assert r.json()["values"]["chat_daily_limit"] == 100
        kill_switch.release()
        assert tanya(client, payload).status_code == 200

    def test_nol_berarti_tanpa_batas(self, atur, payload, kill_switch):
        client = atur(chat_daily_limit=0)
        for _ in range(5):
            assert tanya(client, payload).status_code == 200
        assert not kill_switch.engaged

    def test_yang_ditolak_batas_laju_tidak_menghabiskan_jatah_harian(
        self, atur, payload, kill_switch
    ):
        client = atur(chat_daily_limit=2, rate_limit_per_session="1/minute")
        assert tanya(client, payload).status_code == 200
        for _ in range(3):
            assert tanya(client, payload).status_code == 429
        assert tanya(client, payload, sesi="sesi-lain-67890").status_code == 200
        assert not kill_switch.engaged


class TestBatasPanjang:
    def test_pertanyaan_lebih_dari_500_ditolak(self, client, payload):
        r = client.post("/api/chat", json={**payload, "question": "a" * 501})
        assert r.status_code == 422

    def test_pertanyaan_siap_klik_terpanjang_diterima(self, client, payload):
        """Entri tanya jawab admin boleh 500 karakter; mengkliknya tidak boleh ditolak."""
        r = client.post("/api/chat", json={**payload, "question": "kapan KRS " * 50})
        assert r.status_code == 200

    def test_riwayat_dipangkas_ke_tiga_giliran_terakhir(self, client, payload, api_rewriter):
        riwayat = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"giliran-{i} " + "x" * 5000,
            }
            for i in range(10)
        ]
        r = client.post("/api/chat", json={**payload, "history": riwayat})
        assert r.status_code == 200
        _, teks_riwayat = api_rewriter.calls[-1]
        assert "giliran-6" not in teks_riwayat
        assert all(f"giliran-{i}" in teks_riwayat for i in (7, 8, 9))
        # Setiap giliran dipotong 2000 karakter: 3 x 2000 + label peran.
        assert len(teks_riwayat) < 3 * 2100

    def test_riwayat_raksasa_ditolak(self, client, payload):
        riwayat = [{"role": "user", "content": "halo"}] * 51
        r = client.post("/api/chat", json={**payload, "history": riwayat})
        assert r.status_code == 422
