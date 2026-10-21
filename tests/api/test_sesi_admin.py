"""Sesi dashboard dan jejak auditnya (H21, H27, H28, H30, H31; OWASP A07, A09).

- Token login berumur pendek dan diperpanjang selama admin aktif, tetapi sesi
  tidak pernah melewati `ADMIN_TOKEN_TTL_MINUTES` sejak login.
- Login, penggantian kata sandi, token palsu, dan akses ditolak tercatat di
  `app.audit`; yang dapat dipicu berulang dari luar dibatasi jumlahnya.
- Kata sandi baru yang mudah ditebak ditolak.
- Uji coba jawaban dibatasi per akun.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.admin.permissions import AdminRole
from app.config import Settings, get_settings
from app.security.auth import create_access_token, decode_access_token
from app.security.ratelimit import LOGIN_MAX_FAILURES
from tests.api.conftest import ADMIN_EMAIL, SANDI, STAF_EMAIL


def rahasia() -> str:
    return get_settings().admin_jwt_secret.get_secret_value()


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def isi_token(response) -> dict:
    return decode_access_token(response.json()["access_token"], rahasia())


def catatan_audit(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "app.audit"]


@pytest.fixture
def login_client(make_client):
    return make_client([])


def masuk(client, email=ADMIN_EMAIL, password=SANDI):
    return client.post("/api/admin/login", json={"email": email, "password": password})


class TestTokenSesi:
    def test_token_login_berumur_idle_bukan_8_jam(self, login_client):
        r = masuk(login_client)
        assert r.status_code == 200
        isi = isi_token(r)
        sisa = isi["exp"] - datetime.now(UTC).timestamp()
        menit = get_settings().admin_idle_timeout_minutes
        assert menit * 60 - 60 < sisa <= menit * 60
        assert abs(isi["auth_time"] - isi["iat"]) <= 1

    def test_perpanjangan_mempertahankan_waktu_login(self, login_client):
        masuk_lama = datetime.now(UTC) - timedelta(minutes=20)
        # Tinggal 5 menit: perpanjangan memberi umur idle penuh lagi.
        lama = create_access_token(
            ADMIN_EMAIL, rahasia(), ttl_minutes=5, auth_time=masuk_lama, max_age_minutes=480
        )
        r = login_client.post("/api/admin/token/refresh", headers=bearer(lama))
        assert r.status_code == 200
        baru = isi_token(r)
        assert baru["auth_time"] == int(masuk_lama.timestamp())
        assert baru["exp"] > decode_access_token(lama, rahasia())["exp"]
        me = login_client.get("/api/admin/me", headers=bearer(r.json()["access_token"]))
        assert me.status_code == 200

    def test_perpanjangan_tidak_melewati_umur_sesi(self, login_client):
        """Sesi yang sudah 475 menit hanya diperpanjang sampai menit ke-480."""
        masuk_lama = datetime.now(UTC) - timedelta(minutes=475)
        lama = create_access_token(
            ADMIN_EMAIL, rahasia(), ttl_minutes=30, auth_time=masuk_lama
        )
        r = login_client.post("/api/admin/token/refresh", headers=bearer(lama))
        assert r.status_code == 200
        batas = masuk_lama + timedelta(minutes=get_settings().admin_token_ttl_minutes)
        assert isi_token(r)["exp"] <= int(batas.timestamp()) + 1

    def test_sesi_melewati_umur_maksimum_ditolak(self, login_client):
        """Token berumur sah (exp belum lewat) tetap ditolak bila login-nya terlalu lama."""
        masuk_lama = datetime.now(UTC) - timedelta(
            minutes=get_settings().admin_token_ttl_minutes + 1
        )
        token = create_access_token(
            ADMIN_EMAIL, rahasia(), ttl_minutes=30, auth_time=masuk_lama
        )
        for metode, jalur in (("GET", "/api/admin/me"), ("POST", "/api/admin/token/refresh")):
            r = login_client.request(metode, jalur, headers=bearer(token))
            assert r.status_code == 401
            assert "masuk kembali" in r.json()["detail"]

    def test_akun_nonaktif_tidak_dapat_memperpanjang(self, login_client, accounts):
        token = create_access_token(STAF_EMAIL, rahasia())
        accounts.change(STAF_EMAIL, is_active=False)
        r = login_client.post("/api/admin/token/refresh", headers=bearer(token))
        assert r.status_code == 401

    def test_token_sebelum_sandi_diganti_tidak_dapat_diperpanjang(
        self, login_client, accounts
    ):
        token = jwt.encode(
            {
                **decode_access_token(create_access_token(STAF_EMAIL, rahasia()), rahasia()),
                "iat": int((datetime.now(UTC) - timedelta(minutes=5)).timestamp()),
            },
            rahasia(),
            algorithm="HS256",
        )
        accounts.change(STAF_EMAIL, password_changed_at=datetime.now(UTC))
        r = login_client.post("/api/admin/token/refresh", headers=bearer(token))
        assert r.status_code == 401
        assert "diganti" in r.json()["detail"]

    def test_tanpa_token_401(self, login_client):
        assert login_client.post("/api/admin/token/refresh").status_code == 401


class TestJejakLogin:
    def test_login_gagal_dicatat_dengan_email_dan_ip(self, login_client, caplog):
        caplog.set_level(logging.INFO, logger="app.audit")
        masuk(login_client, password="salah-sekali-panjang")
        masuk(login_client, email="tidak.ada@instiki.ac.id")
        pesan = catatan_audit(caplog)
        assert any(
            "Login gagal untuk admin@instiki.ac.id" in p and "kata sandi salah" in p
            for p in pesan
        )
        assert any("tidak.ada@instiki.ac.id" in p and "tidak terdaftar" in p for p in pesan)
        assert all("salah-sekali-panjang" not in p for p in pesan)

    def test_login_berhasil_dicatat(self, login_client, caplog):
        caplog.set_level(logging.INFO, logger="app.audit")
        assert masuk(login_client).status_code == 200
        assert any(
            p.startswith(f"Login {ADMIN_EMAIL} (superadmin)") for p in catatan_audit(caplog)
        )

    def test_terkunci_dicatat_sekali(self, login_client, caplog):
        caplog.set_level(logging.INFO, logger="app.audit")
        for _ in range(LOGIN_MAX_FAILURES + 3):
            masuk(login_client, password="salah-sekali-panjang")
        terkunci = [p for p in catatan_audit(caplog) if "dikunci sementara" in p]
        assert len(terkunci) == 1

    def test_akun_nonaktif_dicatat(self, login_client, accounts, caplog):
        caplog.set_level(logging.INFO, logger="app.audit")
        accounts.change(STAF_EMAIL, is_active=False)
        assert masuk(login_client, email=STAF_EMAIL).status_code == 403
        assert any("akun nonaktif" in p for p in catatan_audit(caplog))


class TestJejakAkses:
    def test_token_palsu_dicatat_kedaluwarsa_tidak(self, login_client, caplog):
        caplog.set_level(logging.INFO, logger="app.audit")
        login_client.get("/api/admin/me", headers=bearer("bukan.token.sah"))
        palsu = create_access_token(ADMIN_EMAIL, "rahasia-lain-yang-panjangnya-32-byte-lebih")
        login_client.get("/api/admin/me", headers=bearer(palsu))
        kedaluwarsa = create_access_token(ADMIN_EMAIL, rahasia(), ttl_minutes=-1)
        login_client.get("/api/admin/me", headers=bearer(kedaluwarsa))
        pesan = [p for p in catatan_audit(caplog) if "Token admin tidak sah" in p]
        assert len(pesan) == 2
        assert "DecodeError" in pesan[0] and "InvalidSignatureError" in pesan[1]

    def test_banjir_token_palsu_dibatasi(self, login_client, caplog):
        """Pemindai tidak boleh dapat menulis jutaan baris audit."""
        caplog.set_level(logging.INFO, logger="app.audit")
        for _ in range(30):
            login_client.get("/api/admin/me", headers=bearer("bukan.token.sah"))
        assert len([p for p in catatan_audit(caplog) if "tidak sah" in p]) == 5

    def test_level_kurang_dicatat(self, login_client, headers_for, caplog):
        caplog.set_level(logging.INFO, logger="app.audit")
        r = login_client.get("/api/admin/users", headers=headers_for(STAF_EMAIL))
        assert r.status_code == 403
        pesan = [p for p in catatan_audit(caplog) if p.startswith("Akses ditolak")]
        assert pesan == [
            f"Akses ditolak: {STAF_EMAIL} (staf) ke GET /api/admin/users, "
            "butuh level superadmin"
        ]

    def test_unit_lain_dicatat(self, make_client, headers_for, caplog):
        caplog.set_level(logging.INFO, logger="app.audit")
        client = make_client([])
        r = client.post(
            "/api/admin/faq",
            json={
                "question": "Kapan batas bayar UKT?",
                "answer": "Paling lambat tanggal 10 setiap bulan.",
                "unit": "Kemahasiswaan",
            },
            headers=headers_for(STAF_EMAIL),
        )
        assert r.status_code == 403
        assert any(
            "Akses ditolak" in p and "unit Kemahasiswaan" in p for p in catatan_audit(caplog)
        )


class TestGantiSandi:
    def ganti(self, client, headers, baru):
        return client.post(
            "/api/admin/me/password",
            json={"current_password": SANDI, "new_password": baru},
            headers=headers,
        )

    @pytest.mark.parametrize("baru", ["Instiki2026!!", "password1234", "123456789012"])
    def test_sandi_lemah_ditolak_400(self, login_client, headers_for, baru):
        r = self.ganti(login_client, headers_for(STAF_EMAIL), baru)
        assert r.status_code == 400
        assert "mudah ditebak" in r.json()["detail"]

    def test_sandi_berisi_email_ditolak(self, login_client, headers_for):
        r = self.ganti(login_client, headers_for(STAF_EMAIL), "staf.keuangan.2026")
        assert r.status_code == 400
        assert "email" in r.json()["detail"]

    def test_berhasil_dicatat_dan_memulai_sesi_baru(self, login_client, headers_for, caplog):
        caplog.set_level(logging.INFO, logger="app.audit")
        r = self.ganti(login_client, headers_for(STAF_EMAIL), "kopi-bali-pagi-hari-2026")
        assert r.status_code == 200
        assert any("diganti sendiri" in p for p in catatan_audit(caplog))
        isi = isi_token(r)
        assert abs(isi["auth_time"] - datetime.now(UTC).timestamp()) < 5

    def test_sandi_lama_salah_dicatat(self, login_client, headers_for, caplog):
        caplog.set_level(logging.INFO, logger="app.audit")
        login_client.post(
            "/api/admin/me/password",
            json={"current_password": "bukan-sandi-lama", "new_password": "kopi-bali-pagi"},
            headers=headers_for(STAF_EMAIL),
        )
        assert any("kata sandi lama salah" in p for p in catatan_audit(caplog))


class TestBatasUjiCoba:
    def test_lewat_batas_429_per_akun(self, make_client, strong_documents, headers_for):
        client = make_client(strong_documents)
        settings = Settings(admin_jwt_secret=rahasia(), rate_limit_per_admin="2/minute")
        client.app.dependency_overrides[get_settings] = lambda: settings
        body = {"question": "kapan pengisian KRS dibuka?"}
        staf = headers_for(STAF_EMAIL)
        assert client.post("/api/admin/test-query", json=body, headers=staf).status_code == 200
        assert client.post("/api/admin/test-query", json=body, headers=staf).status_code == 200
        r = client.post("/api/admin/test-query", json=body, headers=staf)
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) >= 1
        # Jatah akun lain terpisah.
        lain = headers_for(ADMIN_EMAIL)
        assert client.post("/api/admin/test-query", json=body, headers=lain).status_code == 200

    def test_nol_mematikan_batas(self, make_client, strong_documents, admin_headers):
        client = make_client(strong_documents)
        settings = Settings(admin_jwt_secret=rahasia(), rate_limit_per_admin="0")
        client.app.dependency_overrides[get_settings] = lambda: settings
        body = {"question": "kapan pengisian KRS dibuka?"}
        for _ in range(5):
            r = client.post("/api/admin/test-query", json=body, headers=admin_headers)
            assert r.status_code == 200


def test_level_akun_tetap_dari_database_bukan_token(login_client, accounts):
    """Token perpanjangan membawa `role` terbaru, tetapi yang menentukan tetap DB."""
    token = create_access_token(STAF_EMAIL, rahasia(), role="superadmin")
    accounts.change(STAF_EMAIL, role=AdminRole.STAF)
    assert login_client.get("/api/admin/users", headers=bearer(token)).status_code == 403
