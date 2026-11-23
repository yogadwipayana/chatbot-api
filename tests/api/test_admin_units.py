"""Kelola daftar unit dari dashboard (khusus superadmin).

Level akses (403 untuk staf dan admin) sudah dibangkitkan dari `x-min-role`
di `test_admin_roles.py`. Test di sini menjaga yang khas unit: daftar yang
dikelola adalah daftar yang sama dengan menu chatbot dan validasi isian unit,
dan dua unit yang hanya berbeda huruf besar tidak pernah boleh berdampingan.
"""

from __future__ import annotations

import pytest

from app.rag.chain import OutcomeKind
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

    def test_ganti_nama_kontak_pic_ikut(self, client, admin_headers):
        """T69: dulu PIC dikunci dengan nama unit di kode dan diam-diam jatuh ke
        Front Office. Kini `unit_contacts.unit` ikut lewat ON UPDATE CASCADE."""
        baru = {"name": "Bagian Keuangan"}
        r = client.patch(f"{URL}/Keuangan", json=baru, headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["contact_count"] == 1
        kontak = client.get(f"{URL}/Bagian Keuangan/contacts", headers=admin_headers).json()
        assert [k["label"] for k in kontak] == ["Keuangan INSTIKI"]

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


KONTAK_BARU = {"label": "PIC Prodi IF", "whatsapp": "0812-3456-7890"}


def _kontak(client, admin_headers, unit: str) -> list[dict]:
    r = client.get(f"{URL}/{unit}/contacts", headers=admin_headers)
    assert r.status_code == 200
    return r.json()


class TestKontak:
    """Kontak PIC per unit: yang tampil di banner eskalasi dan penolakan widget."""

    def test_daftar_dalam_urutan_tampil(self, client, admin_headers):
        kontak = _kontak(client, admin_headers, "Keuangan")
        assert [(k["label"], k["display"]) for k in kontak] == [
            ("Keuangan INSTIKI", "WhatsApp +62 851-7426-1765")
        ]

    def test_jumlah_kontak_di_daftar_unit(self, client, admin_headers):
        units = {u["name"]: u for u in client.get(URL, headers=admin_headers).json()}
        assert units["Keuangan"]["contact_count"] == 1
        assert units["Prodi"]["contact_count"] == 0

    def test_unit_tak_dikenal_404(self, client, admin_headers):
        path = f"{URL}/Tidak Ada/contacts"
        assert client.get(path, headers=admin_headers).status_code == 404
        r = client.post(path, json=KONTAK_BARU, headers=admin_headers)
        assert r.status_code == 404

    def test_tambah_langsung_dipakai_penolakan(
        self, make_client, weak_documents, admin_headers, payload, caplog
    ):
        client = make_client(weak_documents)
        r = client.post(f"{URL}/Prodi/contacts", json=KONTAK_BARU, headers=admin_headers)
        assert r.status_code == 201
        assert r.json()["display"] == "WhatsApp 0812-3456-7890"
        assert r.json()["sort_order"] == 1
        assert r.json()["updated_by"]
        assert "Kontak PIC Prodi IF unit Prodi ditambah" in caplog.text

        data = client.post("/api/chat", json={**payload, "unit": "Prodi"}).json()
        assert data["kind"] == OutcomeKind.REFUSAL
        assert "- PIC Prodi IF: WhatsApp 0812-3456-7890" in data["text"]
        assert "Front Office" not in data["text"]

    def test_tambah_tanpa_urutan_diletakkan_paling_akhir(self, client, admin_headers):
        r = client.post(f"{URL}/Keuangan/contacts", json=KONTAK_BARU, headers=admin_headers)
        assert r.json()["sort_order"] == 2
        assert [k["label"] for k in _kontak(client, admin_headers, "Keuangan")] == [
            "Keuangan INSTIKI",
            "PIC Prodi IF",
        ]

    @pytest.mark.parametrize("label", ["keuangan instiki", " KEUANGAN INSTIKI "])
    def test_label_kembar_ditolak(self, client, admin_headers, label):
        """Widget memakai label sebagai kunci baris dan kunci banner terakhir."""
        body = {**KONTAK_BARU, "label": label}
        r = client.post(f"{URL}/Keuangan/contacts", json=body, headers=admin_headers)
        assert r.status_code == 409

    @pytest.mark.parametrize(
        "body",
        [
            {"label": "PIC"},
            {"label": "PIC", "whatsapp": "", "phone": "", "email": ""},
            {"label": "PIC", "whatsapp": "wa.me/62812"},
            {"label": "https://contoh.id", "whatsapp": "0812-3456-7890"},
            {"label": "", "whatsapp": "0812-3456-7890"},
            {"label": "PIC", "email": "bukan-email"},
            {**KONTAK_BARU, "contact": "bebas"},
        ],
    )
    def test_isian_tidak_sah_422(self, client, admin_headers, body):
        r = client.post(f"{URL}/Keuangan/contacts", json=body, headers=admin_headers)
        assert r.status_code == 422

    def test_ubah(self, client, admin_headers, caplog):
        kontak = _kontak(client, admin_headers, "Keuangan")[0]
        r = client.patch(
            f"{URL}/Keuangan/contacts/{kontak['id']}",
            json={"phone": "(0361) 256995", "service_hours": "Senin-Jumat"},
            headers=admin_headers,
        )
        assert r.status_code == 200
        assert r.json()["display"] == "Telepon (0361) 256995 / WhatsApp +62 851-7426-1765"
        assert r.json()["service_hours"] == "Senin-Jumat"
        assert "WhatsApp +62 851-7426-1765 -> Telepon (0361) 256995" in caplog.text

    def test_kata_kunci_membatasi_kontak_ke_urusannya(
        self, make_client, weak_documents, admin_headers, payload, caplog
    ):
        """Kontak berkata kunci hanya tampil bila pertanyaannya menyebut kata itu."""
        client = make_client(weak_documents)
        body = {
            "label": "Admin PLK INSTIKI",
            "whatsapp": "+62 851-1931-5346",
            "keywords": [" PLK ", "plk", "pembelajaran di luar kampus"],
        }
        r = client.post(f"{URL}/BAAK/contacts", json=body, headers=admin_headers)
        assert r.status_code == 201
        assert r.json()["keywords"] == ["PLK", "pembelajaran di luar kampus"]
        assert "[kata kunci: PLK, pembelajaran di luar kampus]" in caplog.text
        assert _kontak(client, admin_headers, "BAAK")[1]["keywords"] == r.json()["keywords"]

        def nomor_di_penolakan(pertanyaan: str) -> list[str]:
            isi = {**payload, "unit": "BAAK", "question": pertanyaan}
            data = client.post("/api/chat", json=isi).json()
            return [c["contact"] for c in data["contacts"]]

        assert nomor_di_penolakan("surat keterangan aktif kuliah") == [
            "WhatsApp +62 813-5346-2032"
        ]
        assert nomor_di_penolakan("syarat PLK apa?") == [
            "WhatsApp +62 813-5346-2032",
            "WhatsApp +62 851-1931-5346",
        ]

        # null = hapus semua kata kunci: kontak kembali selalu tampil.
        path = f"{URL}/BAAK/contacts/{r.json()['id']}"
        r = client.patch(path, json={"keywords": None}, headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["keywords"] == []
        assert len(nomor_di_penolakan("surat keterangan aktif kuliah")) == 2

    def test_kontak_lama_tanpa_kata_kunci(self, client, admin_headers):
        assert _kontak(client, admin_headers, "Keuangan")[0]["keywords"] == []

    def test_mengosongkan_satu_dari_dua_kanal_boleh(self, client, admin_headers):
        kontak = _kontak(client, admin_headers, "Keuangan")[0]
        path = f"{URL}/Keuangan/contacts/{kontak['id']}"
        client.patch(path, json={"phone": "(0361) 256995"}, headers=admin_headers)
        r = client.patch(path, json={"whatsapp": ""}, headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["display"] == "Telepon (0361) 256995"

    def test_mengosongkan_kanal_terakhir_ditolak(self, client, admin_headers):
        kontak = _kontak(client, admin_headers, "Keuangan")[0]
        r = client.patch(
            f"{URL}/Keuangan/contacts/{kontak['id']}",
            json={"whatsapp": None},
            headers=admin_headers,
        )
        assert r.status_code == 422
        assert "minimal satu" in r.json()["detail"]

    def test_ubah_ke_label_kembar_ditolak(self, client, admin_headers):
        baru = client.post(
            f"{URL}/Keuangan/contacts", json=KONTAK_BARU, headers=admin_headers
        ).json()
        r = client.patch(
            f"{URL}/Keuangan/contacts/{baru['id']}",
            json={"label": "Keuangan INSTIKI"},
            headers=admin_headers,
        )
        assert r.status_code == 409

    def test_ubah_huruf_besar_label_sendiri_boleh(self, client, admin_headers):
        kontak = _kontak(client, admin_headers, "Keuangan")[0]
        r = client.patch(
            f"{URL}/Keuangan/contacts/{kontak['id']}",
            json={"label": "KEUANGAN INSTIKI"},
            headers=admin_headers,
        )
        assert r.status_code == 200

    def test_kontak_unit_lain_404(self, client, admin_headers):
        """Path membawa unit DAN id; id kontak unit lain tidak dapat diubah lewat unit ini."""
        kontak = _kontak(client, admin_headers, "Keuangan")[0]
        path = f"{URL}/BAAK/contacts/{kontak['id']}"
        r = client.patch(path, json={"label": "X"}, headers=admin_headers)
        assert r.status_code == 404
        assert client.delete(path, headers=admin_headers).status_code == 404

    def test_hapus_kembali_ke_front_office(
        self, make_client, weak_documents, admin_headers, payload, caplog
    ):
        client = make_client(weak_documents)
        kontak = _kontak(client, admin_headers, "Keuangan")[0]
        path = f"{URL}/Keuangan/contacts/{kontak['id']}"
        assert client.delete(path, headers=admin_headers).status_code == 204
        assert "Kontak Keuangan INSTIKI unit Keuangan dihapus" in caplog.text
        assert _kontak(client, admin_headers, "Keuangan") == []
        assert client.delete(path, headers=admin_headers).status_code == 404

        data = client.post("/api/chat", json={**payload, "unit": "Keuangan"}).json()
        assert [c["unit"] for c in data["contacts"]] == ["Front Office INSTIKI"]
