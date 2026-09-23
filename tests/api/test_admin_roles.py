"""Level akses dashboard: staf/dosen, admin, superadmin.

Kasus "level di bawah minimum" dibangkitkan dari `x-min-role` di api.yaml,
sehingga operasi admin baru otomatis ikut diuji -- dan kontrak yang
menjanjikan level yang tidak ditegakkan kode langsung ketahuan.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import jwt
import pytest
import yaml

from app.admin.permissions import ROLE_LEVEL, AdminRole
from app.config import get_settings
from tests.api.conftest import (
    ADMIN_BIASA_EMAIL,
    ADMIN_EMAIL,
    EMAIL_PER_LEVEL,
    SANDI,
    STAF_EMAIL,
    STAF_UNIT,
)

SPEC = yaml.safe_load(
    (Path(__file__).resolve().parents[2] / "api.yaml").read_text(encoding="utf-8")
)
UUID_CONTOH = "3f1a2b3c-4d5e-6f70-8192-a3b4c5d6e7f8"
SANDI_BARU = "kata-sandi-baru-yang-juga-panjang"


def konkret(path: str) -> str:
    for parameter in ("{document_id}", "{entry_id}", "{unanswered_id}", "{user_id}"):
        path = path.replace(parameter, UUID_CONTOH)
    return path


def kasus_di_bawah_minimum():
    for path, item in SPEC["paths"].items():
        for metode, definisi in item.items():
            if not (isinstance(definisi, dict) and definisi.get("security")):
                continue
            minimum = AdminRole(definisi["x-min-role"])
            for level in AdminRole:
                if ROLE_LEVEL[level] < ROLE_LEVEL[minimum]:
                    yield pytest.param(
                        metode.upper(),
                        konkret(path),
                        level,
                        id=f"{level}-{metode.upper()} {path}",
                    )


KASUS = list(kasus_di_bawah_minimum())


def token_lama(email: str) -> dict[str, str]:
    """Token yang terbit satu menit lalu -- sebelum perubahan apa pun di test."""
    sekarang = int(time.time())
    token = jwt.encode(
        {"sub": email, "iat": sekarang - 60, "exp": sekarang + 600},
        get_settings().admin_jwt_secret.get_secret_value(),
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def bearer(response) -> dict[str, str]:
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


class TestLevelMinimum:
    def test_kasus_mencakup_setiap_level_terbatas(self):
        levels = {level for _, _, level in (p.values for p in KASUS)}
        assert levels == {AdminRole.STAF, AdminRole.ADMIN}
        # staf: 8 operasi admin/superadmin; admin: 6 operasi superadmin.
        assert len(KASUS) >= 14

    @pytest.mark.parametrize("metode,path,level", KASUS)
    def test_level_di_bawah_minimum_ditolak_403(
        self, client, headers_for, metode, path, level
    ):
        r = client.request(metode, path, json={}, headers=headers_for(EMAIL_PER_LEVEL[level]))
        assert r.status_code == 403, r.text
        assert r.json()["detail"]

    def test_superadmin_boleh_mengatur_kill_switch(self, client, admin_headers):
        r = client.post(
            "/api/admin/kill-switch", json={"engaged": False}, headers=admin_headers
        )
        assert r.status_code == 200

    @pytest.mark.parametrize("email", [STAF_EMAIL, ADMIN_BIASA_EMAIL, ADMIN_EMAIL])
    def test_semua_level_boleh_uji_coba_dan_membaca_kill_switch(
        self, client, headers_for, email
    ):
        headers = headers_for(email)
        assert client.get("/api/admin/kill-switch", headers=headers).status_code == 200
        r = client.post(
            "/api/admin/test-query", json={"question": "kapan KRS?"}, headers=headers
        )
        assert r.status_code == 200


class TestPerubahanBerlakuSegera:
    """Level dibaca dari database per permintaan, bukan dari token."""

    def test_akun_dinonaktifkan_langsung_ditolak(self, client, accounts, headers_for):
        headers = headers_for(STAF_EMAIL)
        assert client.get("/api/admin/me", headers=headers).status_code == 200
        accounts.change(STAF_EMAIL, is_active=False)
        r = client.get("/api/admin/me", headers=headers)
        assert r.status_code == 401
        assert "tidak aktif" in r.json()["detail"]

    def test_akun_dihapus_langsung_ditolak(self, client, accounts, headers_for):
        headers = headers_for(STAF_EMAIL)
        accounts.accounts.pop(accounts.by_email(STAF_EMAIL).id)
        assert client.get("/api/admin/me", headers=headers).status_code == 401

    def test_naik_level_langsung_berlaku(self, client, accounts, headers_for):
        headers = headers_for(STAF_EMAIL)
        body = {"engaged": False}
        assert (
            client.post("/api/admin/kill-switch", json=body, headers=headers).status_code
            == 403
        )
        accounts.change(STAF_EMAIL, role=AdminRole.SUPERADMIN)
        assert (
            client.post("/api/admin/kill-switch", json=body, headers=headers).status_code
            == 200
        )

    def test_turun_level_langsung_berlaku(self, client, accounts, admin_headers):
        body = {"engaged": False}
        assert (
            client.post("/api/admin/kill-switch", json=body, headers=admin_headers).status_code
            == 200
        )
        accounts.change(ADMIN_EMAIL, role=AdminRole.ADMIN)
        r = client.post("/api/admin/kill-switch", json=body, headers=admin_headers)
        assert r.status_code == 403

    def test_token_terbit_sebelum_sandi_diganti_ditolak(self, client, accounts):
        headers = token_lama(STAF_EMAIL)
        assert client.get("/api/admin/me", headers=headers).status_code == 200
        accounts.change(STAF_EMAIL, password_changed_at=datetime.now(UTC))
        r = client.get("/api/admin/me", headers=headers)
        assert r.status_code == 401
        assert "diganti" in r.json()["detail"]


class TestAkunSendiri:
    def test_me_membawa_level_dan_unit_tanpa_hash(self, client, headers_for):
        data = client.get("/api/admin/me", headers=headers_for(STAF_EMAIL)).json()
        assert data["role"] == "staf"
        assert data["unit"] == STAF_UNIT
        assert "password_hash" not in data

    def test_ganti_sandi_mengakhiri_sesi_lain_tetapi_bukan_sesi_ini(self, client):
        lama = token_lama(STAF_EMAIL)
        r = client.post(
            "/api/admin/me/password",
            json={"password_lama": SANDI, "password_baru": SANDI_BARU},
            headers=lama,
        )
        assert r.status_code == 200
        assert client.get("/api/admin/me", headers=bearer(r)).status_code == 200
        assert client.get("/api/admin/me", headers=lama).status_code == 401

        masuk = lambda sandi: client.post(  # noqa: E731
            "/api/admin/login", json={"email": STAF_EMAIL, "password": sandi}
        )
        assert masuk(SANDI_BARU).status_code == 200
        assert masuk(SANDI).status_code == 401

    def test_sandi_lama_salah_400(self, client, headers_for):
        r = client.post(
            "/api/admin/me/password",
            json={"password_lama": "bukan-sandi-lama", "password_baru": SANDI_BARU},
            headers=headers_for(STAF_EMAIL),
        )
        assert r.status_code == 400
        assert "lama salah" in r.json()["detail"]

    def test_sandi_baru_sama_dengan_lama_400(self, client, headers_for):
        r = client.post(
            "/api/admin/me/password",
            json={"password_lama": SANDI, "password_baru": SANDI},
            headers=headers_for(STAF_EMAIL),
        )
        assert r.status_code == 400

    @pytest.mark.parametrize("sandi_baru", ["pendek", "ä" * 40])
    def test_sandi_baru_terlalu_pendek_atau_melebihi_batas_bcrypt_422(
        self, client, headers_for, sandi_baru
    ):
        r = client.post(
            "/api/admin/me/password",
            json={"password_lama": SANDI, "password_baru": sandi_baru},
            headers=headers_for(STAF_EMAIL),
        )
        assert r.status_code == 422


class TestKelolaAkun:
    def id_of(self, accounts, email) -> str:
        return str(accounts.by_email(email).id)

    def test_daftar_tanpa_hash_superadmin_lebih_dulu(self, client, admin_headers):
        data = client.get("/api/admin/users", headers=admin_headers).json()
        assert len(data) == 3
        assert data[0]["role"] == "superadmin"
        assert all("password_hash" not in akun for akun in data)

    def test_buat_akun_lalu_masuk_dengan_sandi_sementara(self, client, admin_headers):
        r = client.post(
            "/api/admin/users",
            json={
                "email": "Dosen.Baru@Instiki.ac.id",
                "role": "staf",
                "unit": "Fakultas",
                "nama": "Dosen Baru",
            },
            headers=admin_headers,
        )
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["user"]["email"] == "dosen.baru@instiki.ac.id"
        assert len(data["password_sementara"]) >= 16

        masuk = client.post(
            "/api/admin/login",
            json={"email": "dosen.baru@instiki.ac.id", "password": data["password_sementara"]},
        )
        assert masuk.status_code == 200
        me = client.get("/api/admin/me", headers=bearer(masuk)).json()
        assert (me["role"], me["unit"]) == ("staf", "Fakultas")

    def test_staf_tanpa_unit_ditolak(self, client, admin_headers):
        r = client.post(
            "/api/admin/users",
            json={"email": "baru@instiki.ac.id", "role": "staf", "unit": "   "},
            headers=admin_headers,
        )
        assert r.status_code == 409
        assert "unit" in r.json()["detail"]

    def test_email_ganda_tidak_peka_huruf_besar_ditolak(self, client, admin_headers):
        r = client.post(
            "/api/admin/users",
            json={"email": "ADMIN@instiki.ac.id", "role": "admin"},
            headers=admin_headers,
        )
        assert r.status_code == 409

    @pytest.mark.parametrize("perubahan", [{"role": "admin"}, {"is_active": False}])
    def test_tidak_bisa_menurunkan_atau_menonaktifkan_diri_sendiri(
        self, client, accounts, admin_headers, perubahan
    ):
        r = client.patch(
            f"/api/admin/users/{self.id_of(accounts, ADMIN_EMAIL)}",
            json=perubahan,
            headers=admin_headers,
        )
        assert r.status_code == 409

    def test_boleh_mengubah_nama_sendiri(self, client, accounts, admin_headers):
        r = client.patch(
            f"/api/admin/users/{self.id_of(accounts, ADMIN_EMAIL)}",
            json={"nama": "Nama Baru"},
            headers=admin_headers,
        )
        assert r.status_code == 200
        assert r.json()["nama"] == "Nama Baru"

    def test_menjadikan_staf_wajib_unit(self, client, accounts, admin_headers):
        url = f"/api/admin/users/{self.id_of(accounts, ADMIN_BIASA_EMAIL)}"
        assert (
            client.patch(url, json={"role": "staf"}, headers=admin_headers).status_code == 409
        )
        r = client.patch(
            url, json={"role": "staf", "unit": "Kemahasiswaan"}, headers=admin_headers
        )
        assert r.status_code == 200
        assert (r.json()["role"], r.json()["unit"]) == ("staf", "Kemahasiswaan")

    def test_menonaktifkan_akun_lain_mengakhiri_sesinya(
        self, client, accounts, admin_headers, headers_for
    ):
        headers_staf = headers_for(STAF_EMAIL)
        r = client.patch(
            f"/api/admin/users/{self.id_of(accounts, STAF_EMAIL)}",
            json={"is_active": False},
            headers=admin_headers,
        )
        assert r.json()["is_active"] is False
        assert client.get("/api/admin/me", headers=headers_staf).status_code == 401

    def test_atur_ulang_sandi_mengakhiri_sesi_target(self, client, accounts, admin_headers):
        lama = token_lama(STAF_EMAIL)
        r = client.post(
            f"/api/admin/users/{self.id_of(accounts, STAF_EMAIL)}/reset-password",
            headers=admin_headers,
        )
        assert r.status_code == 200
        assert client.get("/api/admin/me", headers=lama).status_code == 401
        masuk = client.post(
            "/api/admin/login",
            json={"email": STAF_EMAIL, "password": r.json()["password_sementara"]},
        )
        assert masuk.status_code == 200

    def test_atur_ulang_sandi_sendiri_ditolak(self, client, accounts, admin_headers):
        r = client.post(
            f"/api/admin/users/{self.id_of(accounts, ADMIN_EMAIL)}/reset-password",
            headers=admin_headers,
        )
        assert r.status_code == 409

    def test_hapus_akun_lain(self, client, accounts, admin_headers, headers_for):
        headers_staf = headers_for(STAF_EMAIL)
        r = client.delete(
            f"/api/admin/users/{self.id_of(accounts, STAF_EMAIL)}", headers=admin_headers
        )
        assert r.status_code == 204
        assert len(client.get("/api/admin/users", headers=admin_headers).json()) == 2
        assert client.get("/api/admin/me", headers=headers_staf).status_code == 401

    def test_hapus_diri_sendiri_ditolak(self, client, accounts, admin_headers):
        r = client.delete(
            f"/api/admin/users/{self.id_of(accounts, ADMIN_EMAIL)}", headers=admin_headers
        )
        assert r.status_code == 409

    def test_akun_tak_dikenal_404(self, client, admin_headers):
        r = client.patch(
            f"/api/admin/users/{uuid.uuid4()}", json={"nama": "x"}, headers=admin_headers
        )
        assert r.status_code == 404

    def test_field_tak_dikenal_422(self, client, accounts, admin_headers):
        r = client.patch(
            f"/api/admin/users/{self.id_of(accounts, STAF_EMAIL)}",
            json={"level": "admin"},
            headers=admin_headers,
        )
        assert r.status_code == 422


class TestDokumenStaf:
    def test_unggah_ke_unit_lain_ditolak_sebelum_berkas_diproses(self, client, headers_for):
        r = client.post(
            "/api/admin/documents",
            headers=headers_for(STAF_EMAIL),
            files={"file": ("panduan.pdf", b"%PDF-1.4 isi", "application/pdf")},
            data={"judul": "Panduan Akademik", "unit": "BAAK"},
        )
        assert r.status_code == 403
        assert STAF_UNIT in r.json()["detail"]


class TestTanyaJawabStaf:
    """Entri tanya jawab mengikuti batas unit yang sama dengan dokumen."""

    ENTRI = {
        "pertanyaan": "Bagaimana cara mengurus KTM yang hilang?",
        "jawaban": "Bawa surat kehilangan dari kepolisian ke loket 3.",
    }

    def test_menambah_untuk_unit_lain_ditolak(self, client, headers_for):
        r = client.post(
            "/api/admin/faq",
            json={**self.ENTRI, "unit": "BAAK"},
            headers=headers_for(STAF_EMAIL),
        )
        assert r.status_code == 403
        assert STAF_UNIT in r.json()["detail"]

    @pytest.mark.parametrize(
        "payload",
        [
            {"pertanyaan": "?", "jawaban": "Cukup panjang.", "unit": STAF_UNIT},
            {"pertanyaan": "Kapan wisuda?", "jawaban": "Okt", "unit": STAF_UNIT},
            {"pertanyaan": "Kapan wisuda?", "jawaban": "Bulan Oktober."},
        ],
    )
    def test_isian_tidak_lengkap_ditolak_422(self, client, headers_for, payload):
        r = client.post("/api/admin/faq", json=payload, headers=headers_for(STAF_EMAIL))
        assert r.status_code == 422
