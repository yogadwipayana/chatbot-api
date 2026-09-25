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
        assert [u["nama"] for u in r.json()] == list(UNIT_RESMI)
        plk = next(u for u in r.json() if u["nama"] == "PLK")
        assert plk["is_active"] is False
        assert {"jumlah_dokumen", "jumlah_akun", "urutan"} <= plk.keys()


class TestTambah:
    def test_langsung_muncul_di_menu_dan_diakui_isian(self, client, admin_headers):
        r = client.post(
            URL, json={"nama": "  Perpustakaan ", "deskripsi": "UPT"}, headers=admin_headers
        )
        assert r.status_code == 201
        assert r.json()["nama"] == "Perpustakaan"
        assert r.json()["urutan"] == len(UNIT_RESMI) + 1
        assert "Perpustakaan" in [u["nama"] for u in client.get("/api/units").json()]

    @pytest.mark.parametrize("nama", ["keuangan", " KEUANGAN ", "Keuangan"])
    def test_nama_kembar_ditolak(self, client, admin_headers, nama):
        r = client.post(URL, json={"nama": nama}, headers=admin_headers)
        assert r.status_code == 409

    @pytest.mark.parametrize("nama", ["", "   ", "BAAK/Akademik", "x" * 201])
    def test_nama_tidak_sah_422(self, client, admin_headers, nama):
        assert client.post(URL, json={"nama": nama}, headers=admin_headers).status_code == 422


class TestUbah:
    def test_ganti_nama(self, client, admin_headers):
        r = client.patch(f"{URL}/FO", json={"nama": "Front Office"}, headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["nama"] == "Front Office"
        menu = [u["nama"] for u in client.get("/api/units").json()]
        assert "Front Office" in menu and "FO" not in menu

    def test_ganti_nama_ke_unit_lain_ditolak(self, client, admin_headers):
        r = client.patch(f"{URL}/FO", json={"nama": "baak"}, headers=admin_headers)
        assert r.status_code == 409

    def test_ganti_huruf_besar_nama_sendiri_boleh(self, client, admin_headers):
        r = client.patch(f"{URL}/PLK", json={"nama": "Plk"}, headers=admin_headers)
        assert r.status_code == 200

    def test_nonaktif_hilang_dari_menu_dan_isian(self, client, admin_headers):
        r = client.patch(f"{URL}/Prodi", json={"is_active": False}, headers=admin_headers)
        assert r.status_code == 200
        assert "Prodi" not in [u["nama"] for u in client.get("/api/units").json()]
        faq = client.get("/api/faq/questions", params={"unit": "Prodi"})
        assert faq.status_code == 422

    def test_urutan_menentukan_urutan_menu(self, client, admin_headers):
        client.patch(f"{URL}/Akademik", json={"urutan": 0}, headers=admin_headers)
        assert client.get("/api/units").json()[0]["nama"] == "Akademik"

    def test_deskripsi_kosong_menjadi_null(self, client, admin_headers):
        client.patch(f"{URL}/UPS", json={"deskripsi": "Sertifikasi"}, headers=admin_headers)
        r = client.patch(f"{URL}/UPS", json={"deskripsi": "  "}, headers=admin_headers)
        assert r.json()["deskripsi"] is None

    def test_unit_tak_dikenal_404(self, client, admin_headers):
        r = client.patch(f"{URL}/Tidak Ada", json={"urutan": 1}, headers=admin_headers)
        assert r.status_code == 404

    @pytest.mark.parametrize(
        "body", [{"nama": None}, {"is_active": None}, {"urutan": -1}, {"hapus": True}]
    )
    def test_isian_tidak_sah_422(self, client, admin_headers, body):
        assert client.patch(f"{URL}/BAAK", json=body, headers=admin_headers).status_code == 422
