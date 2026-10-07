"""NIM penanya dan profil (prodi dan angkatan) yang diurai darinya.

Yang dijaga di sini: NIM wajib dan diurai API sendiri; profilnya sampai ke LLM
dan log -- tidak pernah menyaring retrieval -- sedangkan NIM-nya hanya sampai
ke log; NIM yang mustahil ditolak sebelum menyentuh retrieval maupun LLM.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.prodi import DAFTAR_PRODI
from tests.fixtures.fakes import FakeRetriever

CHAT = ("/api/chat", "/api/chat/stream")
NIM = "2401010101"
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
    def test_profil_sampai_ke_llm_nim_tidak(self, client_profil, payload, api_llm, path):
        r = client_profil.post(path, json=payload)
        assert r.status_code == 200
        assert BARIS_PROFIL in api_llm.last_question.split("\n")
        assert NIM not in repr(api_llm.calls)

    @pytest.mark.parametrize(
        "nim,baris",
        [
            ("2502010101", "Desain Komunikasi Visual (S1; disebut juga DKV), "),
            ("2300301999", "Magister Informatika (S2; "),
        ],
    )
    def test_prodi_dan_angkatan_diurai_dari_nim(
        self, client_profil, payload, api_llm, nim, baris
    ):
        client_profil.post("/api/chat", json={**payload, "nim": nim})
        profil = next(b for b in api_llm.last_question.split("\n") if b.startswith("Profil"))
        assert baris in profil
        assert profil.endswith(f"angkatan 20{nim[:2]}")

    @pytest.mark.parametrize("path", CHAT)
    def test_profil_bukan_filter_retrieval(self, client_profil, payload, retriever, path):
        """Ketentuan per prodi tertulis di dalam dokumen umum; menyaring
        retrieval menurut prodi justru membuang potongan yang menjawab."""
        client_profil.post(path, json={**payload, "unit": "UPS"})
        assert retriever.queries == [payload["question"]]
        assert retriever.units == ["UPS"]

    @pytest.mark.parametrize("path", CHAT)
    def test_nim_dan_profil_tercatat_di_log(self, client_profil, payload, chat_logger, path):
        client_profil.post(path, json=payload)
        entri = chat_logger.entries[0]
        assert entri.nim == NIM
        assert (entri.profile.prodi.code, entri.profile.angkatan) == ("1010", 2024)

    @pytest.mark.parametrize("path", CHAT)
    def test_nim_tidak_masuk_log_aplikasi(self, client_profil, payload, log_sink, path):
        """Log SQLite dibaca untuk diagnosis dan berumur pendek; NIM cukup di
        `messages.meta`."""
        client_profil.post(path, json=payload)
        assert log_sink.rows
        assert NIM not in repr(log_sink.rows)


class TestNimWajib:
    @pytest.mark.parametrize("path", CHAT)
    def test_tanpa_nim_ditolak_422(self, client_profil, payload, retriever, api_llm, path):
        tanpa = {k: v for k, v in payload.items() if k != "nim"}
        assert client_profil.post(path, json=tanpa).status_code == 422
        assert client_profil.post(path, json={**payload, "nim": None}).status_code == 422
        assert retriever.queries == []
        assert not api_llm.called

    @pytest.mark.parametrize(
        "nim", ["", "240101010", "24010101010", "24010101a1", " 2401010101", "2401 10101"]
    )
    def test_bentuk_nim_tidak_sah_ditolak_422(self, client_profil, payload, nim):
        r = client_profil.post("/api/chat", json={**payload, "nim": nim})
        assert r.status_code == 422

    @pytest.mark.parametrize("path", CHAT)
    def test_kode_prodi_tak_dikenal_ditolak_422(
        self, client_profil, payload, retriever, api_llm, path
    ):
        r = client_profil.post(path, json={**payload, "nim": "2409990101"})
        assert r.status_code == 422
        assert "'9990'" in r.json()["detail"]
        assert "1010 (Informatika)" in r.json()["detail"]
        assert retriever.queries == []
        assert not api_llm.called

    def test_angkatan_yang_belum_tiba_ditolak_422(self, client_profil, payload, retriever):
        tahun_depan = str(date.today().year + 1)[2:]
        r = client_profil.post("/api/chat", json={**payload, "nim": f"{tahun_depan}01010101"})
        assert r.status_code == 422
        assert retriever.queries == []

    def test_field_profil_lama_diabaikan(self, client_profil, payload, chat_logger):
        """Profil yang dikirim sendiri tidak boleh menimpa hasil urai NIM:
        yang dicatat dan diteruskan ke LLM selalu sesuai NIM-nya."""
        palsu = {"program_code": "2010", "intake_year": 2020}
        r = client_profil.post("/api/chat", json={**payload, "profile": palsu})
        assert r.status_code == 200
        assert chat_logger.entries[0].profile.prodi.code == "1010"
