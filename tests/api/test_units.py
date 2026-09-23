"""Daftar unit dan pilihan unit di chatbot.

Filter unit di retrieval hanya dapat dipercaya bila semua pihak memakai ejaan
yang sama persis. Test di sini menjaga dua sisinya: nama yang dikirim mahasiswa
dan admin selalu dipetakan ke ejaan resmi, dan nama yang tidak terdaftar
ditolak sebelum menyentuh retrieval maupun database.
"""

from __future__ import annotations

import pytest

from tests.api.conftest import STAF_EMAIL
from tests.fixtures.fakes import UNIT_RESMI, FakeRetriever

CHAT = ("/api/chat", "/api/chat/stream")


@pytest.fixture
def retriever(strong_documents) -> FakeRetriever:
    return FakeRetriever(strong_documents)


@pytest.fixture
def client_unit(make_client, strong_documents, retriever):
    return make_client(strong_documents, retriever=retriever)


class TestDaftarUnit:
    def test_berisi_unit_aktif_sesuai_urutan(self, client):
        r = client.get("/api/units")
        assert r.status_code == 200
        assert [u["nama"] for u in r.json()] == list(UNIT_RESMI)

    def test_tanpa_login(self, client):
        """Menu chatbot mahasiswa memuatnya tanpa akun apa pun."""
        assert client.get("/api/units").status_code == 200

    def test_tetap_tersedia_saat_kill_switch_aktif(self, client, kill_switch):
        """Dashboard admin memakai daftar yang sama, justru saat chat dimatikan."""
        kill_switch.engage("pemeliharaan")
        assert client.get("/api/units").status_code == 200


class TestPertanyaanPerTopik:
    """Isi menu topik: pertanyaan entri tanya jawab untuk unit yang dipilih."""

    @pytest.fixture(autouse=True)
    def isi(self, units):
        units.faq = {
            "Keuangan": ["Bagaimana cara membayar UKT?", "Kapan batas pembayaran UKT?"],
            "BAAK": ["Cara mengurus KTM hilang?"],
        }

    def test_hanya_pertanyaan_unit_itu(self, client):
        r = client.get("/api/faq/questions", params={"unit": "Keuangan"})
        assert r.status_code == 200
        assert [q["pertanyaan"] for q in r.json()] == [
            "Bagaimana cara membayar UKT?",
            "Kapan batas pembayaran UKT?",
        ]

    def test_nama_unit_dipetakan_ke_ejaan_resmi(self, client, units):
        client.get("/api/faq/questions", params={"unit": " keuangan "})
        assert units.diminta[-1][0] == "Keuangan"

    @pytest.mark.parametrize("params", [{}, {"unit": ""}])
    def test_tanpa_unit_dari_semua_unit(self, client, params):
        r = client.get("/api/faq/questions", params=params)
        assert len(r.json()) == 3

    def test_batas_jumlah(self, client, units):
        r = client.get("/api/faq/questions", params={"limit": 1})
        assert len(r.json()) == 1
        assert units.diminta[-1] == (None, 1)

    def test_topik_tanpa_entri_mengembalikan_daftar_kosong(self, client):
        r = client.get("/api/faq/questions", params={"unit": "Prodi"})
        assert r.status_code == 200
        assert r.json() == []

    def test_unit_tak_terdaftar_ditolak_422(self, client):
        r = client.get("/api/faq/questions", params={"unit": "Gudang"})
        assert r.status_code == 422

    def test_tetap_tersedia_saat_kill_switch_aktif(self, client, kill_switch):
        kill_switch.engage("pemeliharaan")
        assert client.get("/api/faq/questions").status_code == 200


class TestPilihanUnitDiChat:
    @pytest.mark.parametrize("path", CHAT)
    def test_tanpa_unit_mencari_di_semua_unit(self, client_unit, payload, retriever, path):
        client_unit.post(path, json=payload)
        assert retriever.units == [None]

    @pytest.mark.parametrize("path", CHAT)
    def test_unit_dipetakan_ke_ejaan_resmi(self, client_unit, payload, retriever, path):
        """Filter retrieval membandingkan persis; "  keuangan " harus sampai
        sebagai "Keuangan", bukan menghasilkan nol dokumen."""
        r = client_unit.post(path, json={**payload, "unit": "  keuangan "})
        assert r.status_code == 200
        assert retriever.units == ["Keuangan"]

    @pytest.mark.parametrize("kosong", ["", "   ", None])
    def test_unit_kosong_berarti_semua_unit(self, client_unit, payload, retriever, kosong):
        r = client_unit.post("/api/chat", json={**payload, "unit": kosong})
        assert r.status_code == 200
        assert retriever.units == [None]

    @pytest.mark.parametrize("path", CHAT)
    def test_unit_tak_terdaftar_ditolak_422(
        self, client_unit, payload, retriever, api_llm, path
    ):
        r = client_unit.post(path, json={**payload, "unit": "Biro Keuangan"})
        assert r.status_code == 422
        assert "Keuangan" in r.json()["detail"]
        assert retriever.queries == []
        assert not api_llm.called

    def test_unit_tercatat_di_log(self, client_unit, payload, chat_logger):
        """Tanpa ini, penolakan karena salah pilih unit tidak dapat dibedakan dari
        dokumen yang memang belum ada."""
        client_unit.post("/api/chat", json={**payload, "unit": "BAAK"})
        assert chat_logger.entries[0].unit == "BAAK"

    def test_penolakan_menyebut_unit_pilihan(self, make_client, weak_documents, payload):
        r = make_client(weak_documents).post("/api/chat", json={**payload, "unit": "prodi"})
        assert r.json()["kind"] == "refusal"
        assert "unit Prodi" in r.json()["text"]


class TestUjiCobaAdmin:
    PERTANYAAN = {"question": "kapan pengisian KRS dibuka?"}

    def test_unit_diteruskan_ke_retriever(self, client_unit, admin_headers, retriever):
        r = client_unit.post(
            "/api/admin/test-query",
            json={**self.PERTANYAAN, "unit": "baak"},
            headers=admin_headers,
        )
        assert r.status_code == 200
        assert retriever.units == ["BAAK"]

    def test_unit_tak_terdaftar_ditolak_422(self, client_unit, admin_headers, retriever):
        r = client_unit.post(
            "/api/admin/test-query",
            json={**self.PERTANYAAN, "unit": "Fakultas Teknik"},
            headers=admin_headers,
        )
        assert r.status_code == 422
        assert retriever.queries == []


class TestIsianUnitAdmin:
    """Dokumen, entri tanya jawab, dan akun staf hanya boleh memakai unit resmi."""

    def test_akun_baru_menyimpan_ejaan_resmi(self, client, admin_headers):
        r = client.post(
            "/api/admin/users",
            json={"email": "dosen@instiki.ac.id", "role": "staf", "unit": "fakultas"},
            headers=admin_headers,
        )
        assert r.status_code == 201, r.text
        assert r.json()["user"]["unit"] == "Fakultas"

    def test_akun_dengan_unit_tak_terdaftar_ditolak(self, client, admin_headers, accounts):
        r = client.post(
            "/api/admin/users",
            json={"email": "dosen@instiki.ac.id", "role": "staf", "unit": "Fakultas Teknik"},
            headers=admin_headers,
        )
        assert r.status_code == 422
        assert "Fakultas" in r.json()["detail"]
        assert len(accounts.accounts) == 3

    def test_ubah_unit_akun_ke_unit_tak_terdaftar_ditolak(
        self, client, admin_headers, accounts
    ):
        akun = accounts.by_email(STAF_EMAIL)
        r = client.patch(
            f"/api/admin/users/{akun.id}", json={"unit": "Gudang"}, headers=admin_headers
        )
        assert r.status_code == 422
        assert accounts.by_email(STAF_EMAIL).unit == akun.unit

    def test_unggah_dokumen_unit_tak_terdaftar_ditolak_sebelum_berkas_diproses(
        self, client, admin_headers
    ):
        r = client.post(
            "/api/admin/documents",
            headers=admin_headers,
            files={"file": ("panduan.pdf", b"%PDF-1.4 isi", "application/pdf")},
            data={"judul": "Panduan Akademik", "unit": "Biro Administrasi Akademik"},
        )
        assert r.status_code == 422
        assert "BAAK" in r.json()["detail"]

    def test_entri_tanya_jawab_unit_tak_terdaftar_ditolak(self, client, admin_headers):
        r = client.post(
            "/api/admin/faq",
            json={
                "pertanyaan": "Bagaimana cara mengurus KTM yang hilang?",
                "jawaban": "Bawa surat kehilangan dari kepolisian ke loket 3.",
                "unit": "Loket 3",
            },
            headers=admin_headers,
        )
        assert r.status_code == 422
