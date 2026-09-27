"""Kunci sematan: dikelola superadmin, diperiksa portal, dibawa setiap pertanyaan.

Level akses (403 untuk staf dan admin) sudah dibangkitkan dari `x-min-role`
di `test_admin_roles.py`. Test di sini menjaga yang khas kunci sematan: situs
yang dicabut berhenti saat itu juga -- termasuk panel yang terlanjur terbuka --
sementara portal sendiri, yang tidak membawa kunci, tidak terpengaruh.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.config import get_settings
from app.deps import KUNCI_TIDAK_BERLAKU

URL = "/api/admin/embed-keys"
PERIKSA = "/api/embed/keys"


def dengan_kunci(kunci: str) -> dict[str, str]:
    return {"X-Embed-Key": kunci}


@pytest.fixture
def portal_client(client):
    """`PORTAL_URL` tertentu, apa pun isi .env mesin ini."""
    settings = get_settings().model_copy(update={"portal_url": "https://sads.instiki.ac.id"})
    client.app.dependency_overrides[get_settings] = lambda: settings
    return client


class TestBuat:
    def test_kunci_dibuat_server_dan_asal_dirapikan(self, portal_client, admin_headers):
        r = portal_client.post(
            URL,
            json={
                "name": " Situs PMB ",
                "allowed_origins": ["HTTPS://PMB.instiki.ac.id/", "https://pmb.instiki.ac.id"],
            },
            headers=admin_headers,
        )
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["key"].startswith("emb_")
        assert data["name"] == "Situs PMB"
        assert data["allowed_origins"] == ["https://pmb.instiki.ac.id"]
        assert data["is_active"] is True
        assert data["questions_30d"] == 0
        assert data["embed_code"] == (
            '<script src="https://sads.instiki.ac.id/embed.js"'
            f' data-key="{data["key"]}" async></script>'
        )

    def test_tanpa_asal_berarti_semua_situs(self, client, admin_headers):
        r = client.post(URL, json={"name": "LMS"}, headers=admin_headers)
        assert r.status_code == 201
        assert r.json()["allowed_origins"] == []

    def test_asal_berpath_ditolak_dengan_pesan_jelas(self, client, admin_headers):
        r = client.post(
            URL,
            json={"name": "PMB", "allowed_origins": ["https://pmb.instiki.ac.id/daftar"]},
            headers=admin_headers,
        )
        assert r.status_code == 422
        assert "tanpa path" in r.text

    @pytest.mark.parametrize("nama", ["", "   ", "x" * 201])
    def test_nama_tidak_sah_422(self, client, admin_headers, nama):
        assert client.post(URL, json={"name": nama}, headers=admin_headers).status_code == 422


class TestDaftar:
    def test_termasuk_nonaktif_dan_pemakaian(self, client, admin_headers, embed_keys):
        embed_keys.add("PMB", questions_30d=12)
        embed_keys.add("Blog", is_active=False)
        r = client.get(URL, headers=admin_headers)
        assert r.status_code == 200
        assert [(k["name"], k["is_active"], k["questions_30d"]) for k in r.json()] == [
            ("PMB", True, 12),
            ("Blog", False, 0),
        ]


class TestUbahDanHapus:
    def test_ubah_asal(self, client, admin_headers, embed_keys):
        kunci = embed_keys.add("PMB").key
        r = client.patch(
            f"{URL}/{kunci}",
            json={"allowed_origins": ["https://pmb.instiki.ac.id"]},
            headers=admin_headers,
        )
        assert r.status_code == 200
        assert client.get(f"{PERIKSA}/{kunci}").json() == {
            "allowed_origins": ["https://pmb.instiki.ac.id"]
        }

    def test_field_tak_dikenal_ditolak(self, client, admin_headers, embed_keys):
        kunci = embed_keys.add().key
        r = client.patch(f"{URL}/{kunci}", json={"key": "emb_lain"}, headers=admin_headers)
        assert r.status_code == 422

    def test_nama_null_ditolak(self, client, admin_headers, embed_keys):
        kunci = embed_keys.add().key
        r = client.patch(f"{URL}/{kunci}", json={"name": None}, headers=admin_headers)
        assert r.status_code == 422

    def test_kunci_tak_dikenal_404(self, client, admin_headers):
        r = client.patch(f"{URL}/emb_{'a' * 24}", json={"name": "X"}, headers=admin_headers)
        assert r.status_code == 404

    def test_hapus(self, client, admin_headers, embed_keys):
        kunci = embed_keys.add().key
        assert client.delete(f"{URL}/{kunci}", headers=admin_headers).status_code == 204
        assert client.delete(f"{URL}/{kunci}", headers=admin_headers).status_code == 404
        assert client.get(f"{PERIKSA}/{kunci}").status_code == 404


class TestPemeriksaanPortal:
    def test_kunci_aktif(self, client, embed_keys):
        kunci = embed_keys.add(allowed_origins=["https://pmb.instiki.ac.id"]).key
        r = client.get(f"{PERIKSA}/{kunci}")
        assert r.status_code == 200
        assert r.json() == {"allowed_origins": ["https://pmb.instiki.ac.id"]}

    def test_semua_situs(self, client, embed_keys):
        kunci = embed_keys.add().key
        assert client.get(f"{PERIKSA}/{kunci}").json() == {"allowed_origins": []}

    @pytest.mark.parametrize("kunci", ["emb_" + "b" * 24, "bukan-kunci", "emb_%3Cscript%3E"])
    def test_tak_dikenal_404(self, client, kunci):
        assert client.get(f"{PERIKSA}/{kunci}").status_code == 404

    def test_nonaktif_404(self, client, embed_keys):
        kunci = embed_keys.add(is_active=False).key
        r = client.get(f"{PERIKSA}/{kunci}")
        assert r.status_code == 404
        assert r.json()["detail"] == KUNCI_TIDAK_BERLAKU

    def test_tetap_menjawab_saat_kill_switch(self, client, embed_keys, kill_switch):
        """Panel tetap dimuat dan menyampaikan pesan penutupan saat ditanya."""
        kunci = embed_keys.add().key
        kill_switch.engage("uji", by="uji")
        assert client.get(f"{PERIKSA}/{kunci}").status_code == 200


class TestPertanyaanDariSitusPenyemat:
    @pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
    def test_kunci_aktif_dijawab_dan_asalnya_dicatat(
        self, client, payload, embed_keys, chat_logger, path
    ):
        kunci = embed_keys.add().key
        r = client.post(path, json=payload, headers=dengan_kunci(kunci))
        assert r.status_code == 200
        assert chat_logger.entries[-1].embed_key == kunci

    def test_portal_tanpa_kunci_tidak_terpengaruh(self, client, payload, chat_logger):
        assert client.post("/api/chat", json=payload).status_code == 200
        assert chat_logger.entries[-1].embed_key is None

    @pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
    def test_kunci_dinonaktifkan_menolak_panel_yang_terbuka(
        self, client, payload, embed_keys, chat_logger, api_llm, path
    ):
        kunci = embed_keys.add().key
        assert client.post(path, json=payload, headers=dengan_kunci(kunci)).status_code == 200
        embed_keys.keys[kunci] = replace(embed_keys.keys[kunci], is_active=False)
        jumlah_log = len(chat_logger.entries)
        api_llm.calls.clear()

        r = client.post(path, json=payload, headers=dengan_kunci(kunci))
        assert r.status_code == 403
        assert r.json()["detail"] == KUNCI_TIDAK_BERLAKU
        assert len(chat_logger.entries) == jumlah_log
        assert not api_llm.called

    def test_kunci_karangan_ditolak(self, client, payload):
        r = client.post("/api/chat", json=payload, headers=dengan_kunci("emb_" + "c" * 24))
        assert r.status_code == 403

    def test_umpan_balik_dari_kunci_nonaktif_ditolak(self, client, embed_keys):
        kunci = embed_keys.add(is_active=False).key
        r = client.post(
            "/api/feedback",
            json={"message_id": "9c3e1a44-6b2d-4f51-8a70-2d9b5c1e7f03", "helpful": True},
            headers=dengan_kunci(kunci),
        )
        assert r.status_code == 403
