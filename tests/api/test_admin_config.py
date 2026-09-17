"""Halaman Konfigurasi: setelan retrieval dan chunking dari dashboard.

Yang dibuktikan di sini bukan sekadar "nilainya tersimpan", melainkan bahwa
nilai yang disimpan benar-benar dipakai saat menjawab -- setelan yang tampak
berubah di layar tetapi tidak mengubah apa pun adalah kegagalan yang paling
membingungkan untuk didiagnosis.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.admin.runtime_config import NilaiTersimpan
from app.config import get_settings
from tests.api.conftest import ADMIN_EMAIL

ENDPOINT = "/api/admin/config"


@pytest.fixture
def env():
    """Nilai `.env` yang dipakai test; pembanding untuk setiap balasan."""
    return get_settings()


class TestMembaca:
    def test_tanpa_penimpaan_mengikuti_env(self, client, admin_headers, env):
        data = client.get(ENDPOINT, headers=admin_headers).json()
        assert data["nilai"] == data["nilai_env"]
        assert data["nilai"]["vector_threshold"] == env.vector_threshold
        assert data["diubah"] == []
        assert data["diperbarui_oleh"] is None

    def test_membawa_keterangan_model_tanpa_kunci_api(self, client, admin_headers, env):
        data = client.get(ENDPOINT, headers=admin_headers).json()
        assert data["chat_model"] == env.chat_model
        assert data["api_key_terisi"] == (env.kunci_api() is not None)
        assert "api_key" not in data

    def test_penimpaan_tampil_sebagai_diubah(self, client, admin_headers, runtime_config):
        client.patch(ENDPOINT, json={"vector_threshold": 0.6}, headers=admin_headers)
        data = client.get(ENDPOINT, headers=admin_headers).json()
        assert data["nilai"]["vector_threshold"] == 0.6
        assert data["nilai_env"]["vector_threshold"] != 0.6
        assert data["diubah"] == ["vector_threshold"]
        assert data["diperbarui_oleh"] == ADMIN_EMAIL
        assert data["diperbarui_at"]

    def test_baris_rusak_dilaporkan_sebagai_peringatan(
        self, client, admin_headers, runtime_config, env
    ):
        """`CHUNK_SIZE` di server diturunkan setelah overlap besar tersimpan.

        Layanan sementara kembali ke `.env`, dan halaman Konfigurasi mengatakan
        kenapa -- bukan menampilkan nilai simpanan yang sebenarnya tidak dipakai.
        """
        runtime_config.values["chunk_overlap"] = NilaiTersimpan(
            str(env.chunk_size + 50), datetime.now(UTC), ADMIN_EMAIL
        )
        data = client.get(ENDPOINT, headers=admin_headers).json()
        assert "chunk_overlap" in data["peringatan"]
        assert data["nilai"] == data["nilai_env"]
        assert data["diubah"] == ["chunk_overlap"]


class TestMengubah:
    def test_hanya_field_yang_dikirim_yang_berubah(self, client, admin_headers, env):
        data = client.patch(
            ENDPOINT, json={"retrieval_top_n": 3}, headers=admin_headers
        ).json()
        assert data["nilai"]["retrieval_top_n"] == 3
        assert data["nilai"]["retrieval_candidates"] == env.retrieval_candidates

    def test_nilai_sama_dengan_env_menghapus_penimpaan(
        self, client, admin_headers, env, runtime_config
    ):
        client.patch(ENDPOINT, json={"retrieval_top_n": 3}, headers=admin_headers)
        data = client.patch(
            ENDPOINT, json={"retrieval_top_n": env.retrieval_top_n}, headers=admin_headers
        ).json()
        assert data["diubah"] == []
        assert runtime_config.values == {}

    def test_null_mengembalikan_satu_field_ke_env(
        self, client, admin_headers, env, runtime_config
    ):
        """Cara mengembalikan satu parameter tanpa perlu tahu nilai `.env`-nya."""
        client.patch(ENDPOINT, json={"vector_threshold": 0.6}, headers=admin_headers)
        data = client.patch(
            ENDPOINT, json={"vector_threshold": None}, headers=admin_headers
        ).json()
        assert data["nilai"]["vector_threshold"] == env.vector_threshold
        assert data["diubah"] == []
        assert runtime_config.values == {}

    def test_permintaan_kosong_tidak_mengubah_apa_pun(self, client, admin_headers):
        data = client.patch(ENDPOINT, json={}, headers=admin_headers).json()
        assert data["diubah"] == []

    @pytest.mark.parametrize(
        "body",
        [
            {"vector_threshold": 1.5},
            {"vector_threshold": -0.1},
            {"retrieval_top_n": 0},
            {"chunk_size": 100},
            {"rrf_weight_vector": 9},
        ],
    )
    def test_nilai_di_luar_batas_ditolak(self, client, admin_headers, body):
        assert client.patch(ENDPOINT, json=body, headers=admin_headers).status_code == 422

    def test_field_tak_dikenal_ditolak(self, client, admin_headers):
        r = client.patch(ENDPOINT, json={"chat_model": "gpt-4o"}, headers=admin_headers)
        assert r.status_code == 422

    def test_melanggar_aturan_antar_field_ditolak_dengan_kalimat_jelas(
        self, client, admin_headers
    ):
        r = client.patch(ENDPOINT, json={"retrieval_top_n": 40}, headers=admin_headers)
        assert r.status_code == 422
        assert "retrieval_candidates" in r.json()["detail"]

    def test_melanggar_nilai_yang_sudah_tersimpan_ditolak(self, client, admin_headers):
        """Dinilai terhadap keadaan sesudahnya, bukan hanya isi permintaan."""
        assert (
            client.patch(
                ENDPOINT, json={"chunk_size": 400}, headers=admin_headers
            ).status_code
            == 200
        )
        r = client.patch(ENDPOINT, json={"chunk_overlap": 500}, headers=admin_headers)
        assert r.status_code == 422
        assert "chunk_size" in r.json()["detail"]

    def test_perubahan_yang_ditolak_tidak_ikut_tersimpan(
        self, client, admin_headers, runtime_config
    ):
        client.patch(ENDPOINT, json={"retrieval_top_n": 40}, headers=admin_headers)
        assert "retrieval_top_n" not in runtime_config.values

    def test_dua_field_sekaligus_yang_saling_membenarkan(self, client, admin_headers):
        """Menaikkan top_n dan kandidat bersamaan sah, walau satu per satu tidak."""
        r = client.patch(
            ENDPOINT,
            json={"retrieval_top_n": 40, "retrieval_candidates": 60},
            headers=admin_headers,
        )
        assert r.status_code == 200, r.text
        assert r.json()["nilai"]["retrieval_top_n"] == 40


class TestMengembalikan:
    def test_hapus_semua_penimpaan(self, client, admin_headers, runtime_config):
        client.patch(
            ENDPOINT,
            json={"retrieval_top_n": 3, "vector_threshold": 0.6},
            headers=admin_headers,
        )
        data = client.delete(ENDPOINT, headers=admin_headers).json()
        assert data["diubah"] == []
        assert data["nilai"] == data["nilai_env"]
        assert runtime_config.values == {}

    def test_tanpa_penimpaan_tetap_aman(self, client, admin_headers):
        assert client.delete(ENDPOINT, headers=admin_headers).status_code == 200


class TestBerlakuSaatMenjawab:
    """Bukti bahwa setelan benar-benar sampai ke pipeline, bukan hanya ke layar."""

    PERTANYAAN = {"question": "kapan pengisian KRS dibuka?"}

    def test_ambang_baru_langsung_dipakai_uji_coba(self, client, admin_headers):
        """`strong_documents` berskor 0.82; ambang 0.95 harus membuatnya ditolak."""
        awal = client.post(
            "/api/admin/test-query", json=self.PERTANYAAN, headers=admin_headers
        )
        assert awal.json()["kind"] == "answer"

        client.patch(ENDPOINT, json={"vector_threshold": 0.95}, headers=admin_headers)

        sesudah = client.post(
            "/api/admin/test-query", json=self.PERTANYAAN, headers=admin_headers
        )
        assert sesudah.json()["kind"] == "refusal"
        assert sesudah.json()["ambang"]["vector"] == 0.95

    def test_ambang_baru_langsung_dipakai_chat_mahasiswa(self, client, admin_headers):
        payload = {**self.PERTANYAAN, "session_id": "sesi-uji-12345"}
        assert client.post("/api/chat", json=payload).json()["kind"] == "answer"

        client.patch(ENDPOINT, json={"vector_threshold": 0.95}, headers=admin_headers)

        assert client.post("/api/chat", json=payload).json()["kind"] == "refusal"

    def test_dikembalikan_berarti_jawaban_kembali(self, client, admin_headers):
        client.patch(ENDPOINT, json={"vector_threshold": 0.95}, headers=admin_headers)
        client.delete(ENDPOINT, headers=admin_headers)
        r = client.post("/api/admin/test-query", json=self.PERTANYAAN, headers=admin_headers)
        assert r.json()["kind"] == "answer"
