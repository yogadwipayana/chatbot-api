"""Lapisan API dashboard admin -- AD-1, AD-6, FR-9.

Yang diuji di sini adalah yang dapat dibuktikan tanpa database: autentikasi,
pembatasan login, kill switch, dan uji coba retrieval. Endpoint yang isinya
query SQL (dokumen, pertanyaan tak terjawab, statistik) butuh PostgreSQL
sungguhan.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import openai
import pytest
import yaml

from app.admin.grouping import UnansweredItem
from app.config import Settings, get_settings
from app.deps import build_llm_call
from app.rag.chain import OutcomeKind
from app.rag.providers import GalatGateway
from app.routers.common import LAYANAN_AI_BERMASALAH
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
    return path.replace("{name}", "Keuangan")


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
            "/api/admin/kill-switch", json={"engaged": True, "reason": alasan}, headers=headers
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

    @pytest.mark.parametrize("body", [{"engaged": True}, {"engaged": True, "reason": "   "}])
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

    def test_menyalakan_dan_melepas_disimpan_ke_database(
        self, client, admin_headers, kill_switch_store
    ):
        """Supaya restart atau deploy tidak diam-diam menyalakan lagi layanan."""
        self.nyalakan(client, admin_headers)
        tersimpan = kill_switch_store.tersimpan
        assert tersimpan is not None
        assert tersimpan.reason == "jawaban keliru soal UKT"
        assert tersimpan.engaged_by == ADMIN_EMAIL
        client.post("/api/admin/kill-switch", json={"engaged": False}, headers=admin_headers)
        assert kill_switch_store.tersimpan is None

    def test_gagal_menyimpan_tidak_menggagalkan_mematikan_layanan(
        self, client, admin_headers, payload, kill_switch_store, caplog
    ):
        """Saat insiden, mematikan layanan harus berhasil walau database menolak;
        yang hilang hanya ketahanannya setelah restart, dan itu tercatat."""
        kill_switch_store.gagal = True
        r = self.nyalakan(client, admin_headers)
        assert r.status_code == 200
        assert r.json()["engaged"] is True
        assert client.post("/api/chat", json=payload).status_code == 503
        assert "tidak akan bertahan setelah restart" in caplog.text

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
        assert "kill_switch_reason" not in data
        alasan = client.get("/api/admin/kill-switch", headers=admin_headers).json()["reason"]
        assert alasan == "jawaban keliru soal UKT"


class TestTakTerjawab:
    """Penomoran halaman AD-4. Query SQL-nya diganti: yang diuji di sini
    potongan halaman dan angka ringkasan, yang harus mencakup semua kelompok."""

    @pytest.fixture
    def butir(self, monkeypatch):
        teks = ["cara daftar wisuda"] * 3 + ["jadwal krs"] * 2
        teks += ["syarat cuti", "beasiswa prestasi"]
        t0 = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
        data = [
            UnansweredItem(f"id-{i}", q, 0.2, t0 + timedelta(minutes=i), False)
            for i, q in enumerate(teks)
        ]

        async def palsu(session, **_):
            return data

        monkeypatch.setattr("app.routers.admin_quality.fetch_items", palsu)

    def ambil(self, client, headers, **query):
        r = client.get("/api/admin/unanswered", params=query, headers=headers)
        assert r.status_code == 200
        return r.json()

    def test_halaman_pertama_kelompok_terbesar(self, client, admin_headers, butir):
        data = self.ambil(client, admin_headers, limit=2)
        assert [g["count"] for g in data["items"]] == [3, 2]

    def test_ringkasan_mencakup_halaman_lain(self, client, admin_headers, butir):
        data = self.ambil(client, admin_headers, limit=2, offset=2)
        assert [g["count"] for g in data["items"]] == [1, 1]
        assert data["total"] == 4
        assert data["question_count"] == 7
        assert data["max_count"] == 3

    def test_kosong(self, client, admin_headers, monkeypatch):
        async def palsu(session, **_):
            return []

        monkeypatch.setattr("app.routers.admin_quality.fetch_items", palsu)
        assert self.ambil(client, admin_headers) == {
            "items": [],
            "total": 0,
            "question_count": 0,
            "max_count": 0,
        }


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
        assert pertama["content"]
        assert data["decision"] == {
            "decision": "proceed",
            "reason": "ok",
            "top_vector_score": pytest.approx(0.82),
            "top_lexical_score": None,
        }

    def test_ambang_yang_dipakai_dilaporkan(self, client, admin_headers):
        settings = get_settings()
        data = self.uji(client, admin_headers).json()
        assert data["thresholds"] == {
            "vector": pytest.approx(settings.vector_threshold),
            "fulltext": pytest.approx(settings.lexical_threshold),
        }

    def test_ambang_sementara_memicu_penolakan_tanpa_llm(self, client, admin_headers, api_llm):
        data = self.uji(client, admin_headers, vector_threshold=0.9).json()
        assert data["kind"] == OutcomeKind.REFUSAL
        assert data["decision"]["reason"] == "below_threshold"
        assert data["thresholds"]["vector"] == pytest.approx(0.9)
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

    def test_ambang_menolak_dengan_sumber_threshold(self, client, admin_headers):
        """T15: halaman bisa membedakan penolakan ambang dari penolakan LLM."""
        data = self.uji(client, admin_headers, vector_threshold=0.9).json()
        assert data["refusal_source"] == "threshold"
        assert data["llm_called"] is False

    def test_penolakan_llm_dilaporkan_sebagai_llm(self, client, admin_headers, api_llm):
        """Lolos ambang, tetapi LLM menilai isinya tidak menjawab (T1). Halaman
        dulu menulis "Model AI tidak dipanggil" untuk kasus ini."""
        from app.rag.prompts import NOT_FOUND_MARKER

        api_llm.reply = NOT_FOUND_MARKER
        data = self.uji(client, admin_headers).json()
        assert data["kind"] == OutcomeKind.REFUSAL
        assert data["refusal_source"] == "llm"
        assert data["llm_called"] is True
        assert data["decision"]["decision"] == "proceed"

    def test_jawaban_tanpa_sumber_penolakan(self, client, admin_headers):
        data = self.uji(client, admin_headers).json()
        assert data["refusal_source"] is None
        assert data["llm_called"] is True

    def test_memakai_gerbang_jev_seperti_chat_mahasiswa(
        self, make_client, strong_documents, admin_headers, api_llm
    ):
        """T14: "yang dilihat mahasiswa" harus sama dengan chat sungguhan --
        pesan yang diblokir JEV di chat juga diblokir di sini."""
        from app.rag.gate import REPLIES, GateLabel, GateVerdict

        async def gerbang(question, history=(), unit=None):
            return GateVerdict(GateLabel.OUT_OF_SCOPE, 1.0, blocked=True)

        client = make_client(strong_documents, gate=gerbang)
        data = self.uji(client, admin_headers, question="resep nasi goreng dong").json()
        assert data["kind"] == OutcomeKind.REJECTED
        assert data["text"] == REPLIES[GateLabel.OUT_OF_SCOPE]
        assert data["gate"] == {
            "label": "out_of_scope",
            "confidence": 1.0,
            "blocked": True,
            "error": None,
            "source": "jev",
        }
        assert data["rejection_source"] == "jev"
        assert data["decision"] is None
        assert data["retrieved"] == []
        assert api_llm.calls == []

    def test_vonis_jev_yang_meloloskan_ikut_dilaporkan(
        self, make_client, strong_documents, admin_headers
    ):
        from app.rag.gate import GateLabel, GateVerdict

        async def gerbang(question, history=(), unit=None):
            return GateVerdict(GateLabel.ACADEMIC, 0.97, blocked=False)

        data = self.uji(make_client(strong_documents, gate=gerbang), admin_headers).json()
        assert data["kind"] == OutcomeKind.ANSWER
        assert data["gate"]["label"] == "academic"
        assert data["gate"]["blocked"] is False

    def test_potongan_lanjutan_menyebut_sumbernya(
        self, make_client, strong_documents, admin_headers
    ):
        """T16: tanpa ini potongan lanjutan tampil tanpa skor dan tanpa keterangan."""
        from tests.fixtures.fakes import make_document

        lanjutan = make_document("c1b", halaman=13, vector_score=None, rrf_score=0.0)
        lanjutan.metadata["neighbor_of"] = "c1"
        dokumen = [strong_documents[0], lanjutan, strong_documents[1]]
        data = self.uji(make_client(dokumen), admin_headers).json()
        assert [c["neighbor_of"] for c in data["retrieved"]] == [None, "c1", None]
        assert data["retrieved"][1]["raw_scores"] == {}


class LLMGagal:
    def __init__(self, galat: Exception) -> None:
        self.galat = galat

    async def __call__(self, wrapped_question: str, documents) -> str:
        raise self.galat


class TestUjiCobaSaatLayananAiGagal:
    """T26: sebelumnya galat LLM menjadi 500 tanpa header CORS, dan dashboard
    menampilkan "Tidak dapat terhubung ke server" -- seolah jaringan admin putus."""

    def uji(self, make_client, strong_documents, headers, galat: Exception):
        klien = make_client(strong_documents)
        klien.app.dependency_overrides[build_llm_call] = lambda: LLMGagal(galat)
        return klien.post(
            "/api/admin/test-query",
            json={"question": "kapan pengisian KRS dibuka?"},
            headers=headers,
        )

    @pytest.mark.parametrize(
        "galat",
        [
            GalatGateway("[Error] Our servers are currently overloaded."),
            openai.APITimeoutError(request=httpx.Request("POST", "https://gateway.contoh/v1")),
        ],
        ids=["galat-gateway", "batas-waktu"],
    )
    def test_dibalas_502_berpesan(self, make_client, strong_documents, admin_headers, galat):
        r = self.uji(make_client, strong_documents, admin_headers, galat)
        assert r.status_code == 502
        assert r.json()["detail"] == LAYANAN_AI_BERMASALAH

    def test_balasan_terbaca_dashboard(
        self, monkeypatch, make_client, strong_documents, admin_headers
    ):
        asal = "https://admin.dwipa.my.id"
        monkeypatch.setattr(
            "app.main.get_settings", lambda: Settings(_env_file=None, cors_origins=asal)
        )
        klien = make_client(strong_documents)
        klien.app.dependency_overrides[build_llm_call] = lambda: LLMGagal(GalatGateway("x"))
        r = klien.post(
            "/api/admin/test-query",
            json={"question": "kapan pengisian KRS dibuka?"},
            headers={**admin_headers, "Origin": asal},
        )
        assert r.status_code == 502
        assert r.headers["access-control-allow-origin"] == asal

    def test_bug_tetap_500(self, make_client, strong_documents, admin_headers):
        """Galat kode kita tidak boleh menyamar sebagai gangguan layanan AI."""
        with pytest.raises(ZeroDivisionError):
            self.uji(make_client, strong_documents, admin_headers, ZeroDivisionError())
