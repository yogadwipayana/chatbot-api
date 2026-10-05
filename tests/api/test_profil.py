"""Profil penanya (prodi dan angkatan dari NIM) di chatbot mahasiswa.

Yang dijaga di sini: profil hanya sampai ke LLM dan log analitik -- tidak
pernah menyaring retrieval -- dan profil yang mustahil ditolak sebelum
menyentuh retrieval maupun LLM.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.prodi import DAFTAR_PRODI
from tests.fixtures.fakes import FakeRetriever

CHAT = ("/api/chat", "/api/chat/stream")
PROFIL = {"program_code": "1010", "intake_year": 2024}
BARIS_PROFIL = (
    "Profil mahasiswa penanya: Informatika (S1; disebut juga Teknik Informatika, TI), "
    "Fakultas Teknik Informatika, angkatan 2024"
)


@pytest.fixture
def retriever(strong_documents) -> FakeRetriever:
    return FakeRetriever(strong_documents)


@pytest.fixture
def client_profil(make_client, strong_documents, retriever):
    return make_client(strong_documents, retriever=retriever)


class TestDaftarProdi:
    def test_berisi_semua_prodi_sesuai_urutan(self, client):
        r = client.get("/api/programs")
        assert r.status_code == 200
        assert [p["code"] for p in r.json()] == [p.code for p in DAFTAR_PRODI]
        assert r.json()[0] == {
            "code": "1010",
            "name": "Informatika",
            "level": "S1",
            "faculty": "Fakultas Teknik Informatika",
        }

    def test_tetap_tersedia_saat_kill_switch_aktif(self, client, kill_switch):
        kill_switch.engage("pemeliharaan")
        assert client.get("/api/programs").status_code == 200


class TestProfilDiChat:
    @pytest.mark.parametrize("path", CHAT)
    def test_profil_sampai_ke_llm(self, client_profil, payload, api_llm, path):
        r = client_profil.post(path, json={**payload, "profile": PROFIL})
        assert r.status_code == 200
        assert BARIS_PROFIL in api_llm.last_question.split("\n")

    @pytest.mark.parametrize("path", CHAT)
    def test_profil_bukan_filter_retrieval(self, client_profil, payload, retriever, path):
        """Ketentuan per prodi tertulis di dalam dokumen umum; menyaring
        retrieval menurut prodi justru membuang potongan yang menjawab."""
        client_profil.post(path, json={**payload, "unit": "UPS", "profile": PROFIL})
        assert retriever.queries == [payload["question"]]
        assert retriever.units == ["UPS"]

    def test_tanpa_profil_tidak_ada_baris_profil(self, client_profil, payload, api_llm):
        client_profil.post("/api/chat", json={**payload, "profile": None})
        assert "Profil mahasiswa penanya" not in api_llm.last_question

    def test_profil_tercatat_di_log(self, client_profil, payload, chat_logger):
        client_profil.post("/api/chat", json={**payload, "profile": PROFIL})
        profil = chat_logger.entries[0].profile
        assert (profil.prodi.code, profil.angkatan) == ("1010", 2024)

    @pytest.mark.parametrize("path", CHAT)
    def test_kode_prodi_tak_dikenal_ditolak_422(
        self, client_profil, payload, retriever, api_llm, path
    ):
        r = client_profil.post(
            path, json={**payload, "profile": {"program_code": "9999", "intake_year": 2024}}
        )
        assert r.status_code == 422
        assert "1010 (Informatika)" in r.json()["detail"]
        assert retriever.queries == []
        assert not api_llm.called

    def test_angkatan_yang_belum_tiba_ditolak_422(self, client_profil, payload, retriever):
        tahun_depan = date.today().year + 1
        r = client_profil.post(
            "/api/chat",
            json={**payload, "profile": {"program_code": "1010", "intake_year": tahun_depan}},
        )
        assert r.status_code == 422
        assert retriever.queries == []

    @pytest.mark.parametrize(
        "profil",
        [
            {"program_code": "101", "intake_year": 2024},
            {"program_code": "10a0", "intake_year": 2024},
            {"program_code": "1010", "intake_year": 1999},
            {"program_code": "1010"},
        ],
    )
    def test_bentuk_profil_tidak_sah_ditolak_422(self, client_profil, payload, profil):
        r = client_profil.post("/api/chat", json={**payload, "profile": profil})
        assert r.status_code == 422

    def test_nim_utuh_tidak_diterima(self, client_profil, payload, chat_logger):
        """Kontrak hanya punya kode prodi dan angkatan; NIM yang terselip di
        field lain diabaikan dan tidak pernah sampai ke log."""
        r = client_profil.post(
            "/api/chat", json={**payload, "profile": {**PROFIL, "nim": "2401010101"}}
        )
        assert r.status_code == 200
        assert "2401010101" not in repr(chat_logger.entries[0])
