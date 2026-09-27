"""Kelola daftar unit dari dashboard (khusus superadmin).

Level akses (403 untuk staf dan admin) sudah dibangkitkan dari `x-min-role`
di `test_admin_roles.py`. Test di sini menjaga yang khas unit: daftar yang
dikelola adalah daftar yang sama dengan menu chatbot dan validasi isian unit,
dan dua unit yang hanya berbeda huruf besar tidak pernah boleh berdampingan.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes import UNIT_RESMI

URL = "/api/admin/units"


class TestDaftar:
    def test_termasuk_unit_nonaktif(self, client, admin_headers):
        client.patch(f"{URL}/PLK", json={"is_active": False}, headers=admin_headers)
        r = client.get(URL, headers=admin_headers)
        assert r.status_code == 200
        assert [u["name"] for u in r.json()] == list(UNIT_RESMI)
        plk = next(u for u in r.json() if u["name"] == "PLK")
        assert plk["is_active"] is False
        assert {"document_count", "account_count", "sort_order"} <= plk.keys()


class TestTambah:
    def test_langsung_muncul_di_menu_dan_diakui_isian(self, client, admin_headers):
        r = client.post(
            URL, json={"name": "  Perpustakaan ", "description": "UPT"}, headers=admin_headers
        )
        assert r.status_code == 201
        assert r.json()["name"] == "Perpustakaan"
        assert r.json()["sort_order"] == len(UNIT_RESMI) + 1
        assert "Perpustakaan" in [u["name"] for u in client.get("/api/units").json()]

    @pytest.mark.parametrize("nama", ["keuangan", " KEUANGAN ", "Keuangan"])
    def test_nama_kembar_ditolak(self, client, admin_headers, nama):
        r = client.post(URL, json={"name": nama}, headers=admin_headers)
        assert r.status_code == 409

    @pytest.mark.parametrize("nama", ["", "   ", "BAAK/Akademik", "x" * 201])
    def test_nama_tidak_sah_422(self, client, admin_headers, nama):
        assert client.post(URL, json={"name": nama}, headers=admin_headers).status_code == 422


class TestUbah:
    def test_ganti_nama(self, client, admin_headers):
        r = client.patch(f"{URL}/FO", json={"name": "Front Office"}, headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["name"] == "Front Office"
        menu = [u["name"] for u in client.get("/api/units").json()]
        assert "Front Office" in menu and "FO" not in menu

    def test_ganti_nama_ke_unit_lain_ditolak(self, client, admin_headers):
        r = client.patch(f"{URL}/FO", json={"name": "baak"}, headers=admin_headers)
        assert r.status_code == 409

    def test_ganti_huruf_besar_nama_sendiri_boleh(self, client, admin_headers):
        r = client.patch(f"{URL}/PLK", json={"name": "Plk"}, headers=admin_headers)
        assert r.status_code == 200

    def test_nonaktif_hilang_dari_menu_dan_isian(self, client, admin_headers):
        r = client.patch(f"{URL}/Prodi", json={"is_active": False}, headers=admin_headers)
        assert r.status_code == 200
        assert "Prodi" not in [u["name"] for u in client.get("/api/units").json()]
        faq = client.get("/api/faq/questions", params={"unit": "Prodi"})
        assert faq.status_code == 422

    def test_urutan_menentukan_urutan_menu(self, client, admin_headers):
        client.patch(f"{URL}/Akademik", json={"sort_order": 0}, headers=admin_headers)
        assert client.get("/api/units").json()[0]["name"] == "Akademik"

    def test_deskripsi_kosong_menjadi_null(self, client, admin_headers):
        client.patch(f"{URL}/UPS", json={"description": "Sertifikasi"}, headers=admin_headers)
        r = client.patch(f"{URL}/UPS", json={"description": "  "}, headers=admin_headers)
        assert r.json()["description"] is None

    def test_unit_tak_dikenal_404(self, client, admin_headers):
        r = client.patch(f"{URL}/Tidak Ada", json={"sort_order": 1}, headers=admin_headers)
        assert r.status_code == 404

    @pytest.mark.parametrize(
        "body", [{"name": None}, {"is_active": None}, {"sort_order": -1}, {"hapus": True}]
    )
    def test_isian_tidak_sah_422(self, client, admin_headers, body):
        assert client.patch(f"{URL}/BAAK", json=body, headers=admin_headers).status_code == 422
