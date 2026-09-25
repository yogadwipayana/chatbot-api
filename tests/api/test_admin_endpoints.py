"""Lapisan API dashboard admin -- AD-1, AD-6, FR-9.

Yang diuji di sini adalah yang dapat dibuktikan tanpa database: autentikasi,
pembatasan login, kill switch, dan uji coba retrieval. Endpoint yang isinya
query SQL (dokumen, pertanyaan tak terjawab, statistik) butuh PostgreSQL
sungguhan.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.config import get_settings
from app.rag.chain import OutcomeKind
from app.security.auth import create_access_token
from app.security.ratelimit import LOGIN_MAX_FAILURES
from tests.api.conftest import ADMIN_EMAIL, SANDI

SPEC = yaml.safe_load(
    (Path(__file__).resolve().parents[2] / "api.yaml").read_text(encoding="utf-8")
)
UUID_CONTOH = "3f1a2b3c-4d5e-6f70-8192-a3b4c5d6e7f8"


def konkret(path: str) -> str:
    for parameter in ("{document_id}", "{turn_id}", "{unanswered_id}", "{user_id}"):
        path = path.replace(parameter, UUID_CONTOH)
    return path.replace("{nama}", "Keuangan")


def operasi_terlindungi():
    """Setiap operasi yang api.yaml tandai butuh bearer token."""
    for path, item in SPEC["paths"].items():
        for metode, definisi in item.items():
            if isinstance(definisi, dict) and definisi.get("security"):
                yield pytest.param(
                    metode.upper(), konkret(path), id=f"{metode.upper()} {path}"
                )


OPERASI_TERLINDUNGI = list(operasi_terlindungi())


class TestTanpaTokenValid:
    def test_ada_operasi_yang_diperiksa(self):
        assert len(OPERASI_TERLINDUNGI) >= 19

    @pytest.mark.parametrize("metode,path", OPERASI_TERLINDUNGI)
    def test_tanpa_token_401(self, client, metode, path):
        r = client.request(metode, path, json={})
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "Bearer"

    @pytest.mark.parametrize("metode,path", OPERASI_TERLINDUNGI)
    def test_token_ngawur_401(self, client, metode, path):
        r = client.request(
            metode, path, json={}, headers={"Authorization": "Bearer bukan.token.sah"}
        )
        assert r.status_code == 401

    def test_token_kedaluwarsa_diminta_masuk_kembali(self, client):
        token = create_access_token(
            ADMIN_EMAIL, get_settings().admin_jwt_secret.get_secret_value(), ttl_minutes=-1
        )
        r = client.get("/api/admin/kill-switch", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401
        assert "masuk kembali" in r.json()["detail"]

    def test_token_dari_secret_lain_ditolak(self, client):
        token = create_access_token(ADMIN_EMAIL, "secret-lain-yang-juga-panjangnya-32-byte")
        r = client.get("/api/admin/kill-switch", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401


class TestLogin:
    @pytest.fixture
    def login_client(self, make_client):
        return make_client([])

    def masuk(self, client, email=ADMIN_EMAIL, password=SANDI):
        return client.post("/api/admin/login", json={"email": email, "password": password})

    def test_berhasil_mengembalikan_token_yang_berlaku(self, login_client):
        r = self.masuk(login_client)
        assert r.status_code == 200
        assert r.json()["token_type"] == "bearer"
        token = r.json()["access_token"]
        cek = login_client.get(
            "/api/admin/kill-switch", headers={"Authorization": f"Bearer {token}"}
        )
        assert cek.status_code == 200

    def test_email_tidak_peka_huruf_besar(self, login_client):
        assert self.masuk(login_client, email="Admin@Instiki.AC.ID").status_code == 200

    def test_pesan_gagal_tidak_membedakan_penyebab(self, login_client):
        """Tidak boleh bisa dipakai memetakan email mana yang punya akun admin."""
        sandi_salah = self.masuk(login_client, password="bukan-sandi-yang-benar")
        tak_terdaftar = self.masuk(login_client, email="orang@lain.ac.id")
        assert sandi_salah.status_code == tak_terdaftar.status_code == 401
        assert sandi_salah.json() == tak_terdaftar.json()

    def test_terkunci_setelah_terlalu_banyak_gagal(self, login_client):
        for _ in range(LOGIN_MAX_FAILURES):
            assert self.masuk(login_client, password="tebakan-salah").status_code == 401
        r = self.masuk(login_client)
        assert r.status_code == 429, "sandi benar pun ditolak selama terkunci"
        assert int(r.headers["retry-after"]) > 0

    def test_berhasil_mengosongkan_hitungan_gagal(self, login_client):
        for _ in range(LOGIN_MAX_FAILURES - 1):
            self.masuk(login_client, password="salah-ketik")
        assert self.masuk(login_client).status_code == 200
        for _ in range(LOGIN_MAX_FAILURES - 1):
            assert self.masuk(login_client, password="salah-ketik").status_code == 401

    def test_email_tidak_sah_422(self, login_client):
        assert self.masuk(login_client, email="bukan-email").status_code == 422

    def test_waktu_masuk_terakhir_dicatat(self, login_client, accounts):
        self.masuk(login_client)
        assert accounts.by_email(ADMIN_EMAIL).last_login_at is not None

    def test_akun_nonaktif_ditolak_setelah_sandi_benar(self, login_client, accounts):
        accounts.change(ADMIN_EMAIL, is_active=False)
        r = self.masuk(login_client)
        assert r.status_code == 403
        assert "dinonaktifkan" in r.json()["detail"]

    def test_akun_nonaktif_dengan_sandi_salah_tetap_pesan_seragam(
        self, login_client, accounts
    ):
        """Status nonaktif tidak boleh bocor ke orang yang tidak tahu kata sandinya."""
        accounts.change(ADMIN_EMAIL, is_active=False)
        r = self.masuk(login_client, password="tebakan-salah")
        assert r.status_code == 401
        assert r.json()["detail"] == "Email atau kata sandi salah"


class TestKillSwitchAdmin:
    def nyalakan(self, client, headers, alasan="jawaban keliru soal UKT"):
        return client.post(
            "/api/admin/kill-switch", json={"engaged": True, "alasan": alasan}, headers=headers
        )

    def test_status_awal_mati(self, client, admin_headers):
        r = client.get("/api/admin/kill-switch", headers=admin_headers)
        assert r.json() == {
            "engaged": False,
            "reason": None,
            "engaged_at": None,
            "engaged_by": None,
        }

    def test_menyalakan_memblokir_chat_dan_mencatat_pelaku(
        self, client, admin_headers, payload
    ):
        r = self.nyalakan(client, admin_headers)
        assert r.status_code == 200
        data = r.json()
        assert data["engaged"] is True
        assert data["reason"] == "jawaban keliru soal UKT"
        assert data["engaged_by"] == ADMIN_EMAIL
        assert data["engaged_at"]
        assert client.post("/api/chat", json=payload).status_code == 503

    @pytest.mark.parametrize("body", [{"engaged": True}, {"engaged": True, "alasan": "   "}])
    def test_alasan_wajib_saat_menyalakan(self, client, admin_headers, body, kill_switch):
        r = client.post("/api/admin/kill-switch", json=body, headers=admin_headers)
        assert r.status_code == 422
        assert kill_switch.engaged is False

    def test_melepas_memulihkan_chat(self, client, admin_headers, payload):
        self.nyalakan(client, admin_headers)
        r = client.post(
            "/api/admin/kill-switch", json={"engaged": False}, headers=admin_headers
        )
        assert r.json()["engaged"] is False
        assert r.json()["reason"] is None
        assert client.post("/api/chat", json=payload).status_code == 200

    def test_dashboard_tetap_hidup_saat_aktif(self, client, admin_headers):
        """Justru saat insiden admin perlu masuk dan mendiagnosis."""
        self.nyalakan(client, admin_headers)
        assert client.get("/api/admin/kill-switch", headers=admin_headers).status_code == 200
        r = client.post(
            "/api/admin/test-query", json={"question": "kapan KRS?"}, headers=admin_headers
        )
        assert r.status_code == 200

    def test_health_melaporkan_status(self, client, admin_headers):
        self.nyalakan(client, admin_headers)
        data = client.get("/health").json()
        assert data["chat_enabled"] is False
        assert data["kill_switch_reason"] == "jawaban keliru soal UKT"


class TestUjiCoba:
    def uji(self, client, headers, **body):
        return client.post(
            "/api/admin/test-query",
            json={"question": "kapan pengisian KRS dibuka?", **body},
            headers=headers,
        )

    def test_menampilkan_chunk_beserta_skor_mentah(self, client, admin_headers):
        data = self.uji(client, admin_headers).json()
        assert data["kind"] == OutcomeKind.ANSWER
        pertama = data["retrieved"][0]
        assert pertama["raw_scores"]["vector"] == pytest.approx(0.82)
        assert pertama["ranks"]["vector"] == 1
        assert "fulltext" not in pertama["raw_scores"], (
            "sumber yang tidak menemukan chunk absen"
        )
        assert pertama["konten"]
        assert data["decision"] == {
            "decision": "proceed",
            "reason": "ok",
            "top_vector_score": pytest.approx(0.82),
            "top_lexical_score": None,
        }

    def test_ambang_yang_dipakai_dilaporkan(self, client, admin_headers):
        settings = get_settings()
        data = self.uji(client, admin_headers).json()
        assert data["ambang"] == {
            "vector": pytest.approx(settings.vector_threshold),
            "fulltext": pytest.approx(settings.lexical_threshold),
        }

    def test_ambang_sementara_memicu_penolakan_tanpa_llm(self, client, admin_headers, api_llm):
        data = self.uji(client, admin_headers, vector_threshold=0.9).json()
        assert data["kind"] == OutcomeKind.REFUSAL
        assert data["decision"]["reason"] == "below_threshold"
        assert data["ambang"]["vector"] == pytest.approx(0.9)
        assert data["retrieved"], (
            "chunk tetap ditampilkan walau ditolak -- itu gunanya diagnosa"
        )
        assert api_llm.calls == []

    def test_ambang_di_luar_rentang_ditolak(self, client, admin_headers):
        assert self.uji(client, admin_headers, vector_threshold=1.5).status_code == 422

    def test_pertanyaan_sensitif_tanpa_retrieval(self, client, admin_headers):
        r = client.post(
            "/api/admin/test-query", json={"question": "saya depresi"}, headers=admin_headers
        )
        data = r.json()
        assert data["kind"] == OutcomeKind.SUPPORT
        assert data["decision"] is None
        assert data["retrieved"] == []
        assert data["contacts"]

    def test_tidak_dicatat_ke_log_percakapan(self, client, admin_headers, chat_logger):
        """Uji coba admin tidak boleh mencemari statistik AD-5 dan daftar AD-4."""
        self.uji(client, admin_headers)
        assert chat_logger.entries == []
