"""FR-9 / AD-1 -- autentikasi admin dan kill switch."""

from __future__ import annotations

import time

import jwt
import pytest

from app.security.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.security.killswitch import KillSwitch
from app.security.ratelimit import FailureLimiter

SECRET = "rahasia-uji-yang-panjangnya-lebih-dari-32-byte"
SANDI = "kata-sandi-admin-panjang"


class TestKataSandi:
    def test_hash_lalu_verifikasi(self):
        assert verify_password(SANDI, hash_password(SANDI))

    def test_sandi_salah_ditolak(self):
        assert not verify_password("salah-sekali-panjang", hash_password(SANDI))

    def test_hash_tidak_menyimpan_sandi_apa_adanya(self):
        assert SANDI not in hash_password(SANDI)

    def test_dua_hash_dari_sandi_sama_berbeda(self):
        """Salt acak; tanpa ini, hash yang sama membocorkan sandi yang sama."""
        assert hash_password(SANDI) != hash_password(SANDI)

    def test_sandi_pendek_ditolak(self):
        with pytest.raises(ValueError, match="minimal 12 karakter"):
            hash_password("pendek")

    def test_hash_rusak_tidak_melempar(self):
        """Baris DB yang korup harus jadi gagal login, bukan 500."""
        assert verify_password(SANDI, "bukan-hash-bcrypt") is False


class TestToken:
    def test_bolak_balik(self):
        token = create_access_token("admin@kampus.ac.id", SECRET)
        assert decode_access_token(token, SECRET)["sub"] == "admin@kampus.ac.id"

    def test_role_ikut_terbawa(self):
        token = create_access_token("a@b.c", SECRET, role="superadmin")
        assert decode_access_token(token, SECRET)["role"] == "superadmin"

    def test_secret_salah_ditolak(self):
        token = create_access_token("a@b.c", SECRET)
        with pytest.raises(jwt.InvalidSignatureError):
            decode_access_token(token, "secret-lain-yang-juga-panjang-32-byte-lebih")

    def test_secret_terlalu_pendek_ditolak(self):
        """RFC 7518 §3.2. PyJWT hanya memperingatkan; di sini harus gagal,
        supaya konfigurasi lemah tidak diam-diam sampai ke produksi."""
        with pytest.raises(ValueError, match="terlalu pendek"):
            create_access_token("a@b.c", "pendek")

    def test_secret_tepat_32_byte_diterima(self):
        assert create_access_token("a@b.c", "a" * 32)

    def test_token_kedaluwarsa_ditolak(self):
        token = create_access_token("a@b.c", SECRET, ttl_minutes=-1)
        with pytest.raises(jwt.ExpiredSignatureError):
            decode_access_token(token, SECRET)

    def test_token_ngawur_ditolak(self):
        with pytest.raises(jwt.DecodeError):
            decode_access_token("bukan.token.sama.sekali", SECRET)

    def test_masa_berlaku_tercatat(self):
        token = create_access_token("a@b.c", SECRET, ttl_minutes=10)
        payload = decode_access_token(token, SECRET)
        sisa = payload["exp"] - time.time()
        assert 500 < sisa <= 600


class TestKillSwitch:
    def test_default_mati(self):
        assert KillSwitch().engaged is False

    def test_dinyalakan_dengan_alasan(self):
        switch = KillSwitch()
        switch.engage("jawaban salah pada topik pembayaran")
        assert switch.engaged
        assert switch.reason == "jawaban salah pada topik pembayaran"
        assert switch.engaged_at is not None

    def test_alasan_wajib_diisi(self):
        """Insiden tanpa catatan alasan tidak bisa diaudit setelahnya."""
        with pytest.raises(ValueError, match="alasan"):
            KillSwitch().engage("   ")

    def test_dimatikan_kembali_membersihkan_status(self):
        switch = KillSwitch()
        switch.engage("uji coba")
        switch.release()
        assert not switch.engaged
        assert switch.reason is None
        assert switch.engaged_at is None

    def test_pesan_untuk_mahasiswa_menyebut_kontak_manusia(self):
        """Layanan mati bukan alasan meninggalkan mahasiswa tanpa jalan keluar."""
        pesan = KillSwitch().message
        assert "Akademik" in pesan

    def test_pelaku_tercatat_dan_dibersihkan(self):
        switch = KillSwitch()
        switch.engage("insiden", by="admin@kampus.ac.id")
        assert switch.engaged_by == "admin@kampus.ac.id"
        switch.release()
        assert switch.engaged_by is None

    def test_menyalakan_ulang_mempertahankan_awal_insiden(self):
        """Melengkapi catatan alasan tidak boleh menghapus durasi gangguan."""
        switch = KillSwitch()
        switch.engage("insiden")
        awal = switch.engaged_at
        switch.engage("insiden: jawaban UKT keliru", by="admin@kampus.ac.id")
        assert switch.engaged_at == awal
        assert switch.reason == "insiden: jawaban UKT keliru"


class TestKillSwitchDariKonfigurasi:
    def test_kill_switch_enabled_menyalakan_saat_start(self):
        from app.config import Settings
        from app.main import apply_initial_kill_switch

        switch = KillSwitch()
        apply_initial_kill_switch(Settings(_env_file=None, kill_switch_enabled=True), switch)
        assert switch.engaged
        assert switch.reason

    def test_default_tidak_menyalakan(self):
        from app.config import Settings
        from app.main import apply_initial_kill_switch

        switch = KillSwitch()
        apply_initial_kill_switch(Settings(_env_file=None), switch)
        assert not switch.engaged


class JamPalsu:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class TestPembatasLogin:
    @pytest.fixture
    def jam(self) -> JamPalsu:
        return JamPalsu()

    @pytest.fixture
    def pembatas(self, jam) -> FailureLimiter:
        return FailureLimiter(3, 60, clock=jam)

    def test_di_bawah_batas_tidak_diblokir(self, pembatas):
        pembatas.record_failure("ip:1")
        pembatas.record_failure("ip:1")
        assert pembatas.retry_after("ip:1") is None

    def test_mencapai_batas_diblokir(self, pembatas):
        for _ in range(3):
            pembatas.record_failure("ip:1")
        assert pembatas.retry_after("ip:1") == pytest.approx(60)

    def test_terbuka_lagi_setelah_jendela_lewat(self, pembatas, jam):
        for _ in range(3):
            pembatas.record_failure("ip:1")
        jam.t += 60
        assert pembatas.retry_after("ip:1") is None

    def test_sisa_waktu_dihitung_dari_kegagalan_tertua_yang_menentukan(self, pembatas, jam):
        for _ in range(3):
            pembatas.record_failure("ip:1")
            jam.t += 10
        # Kegagalan pada t=1000, 1010, 1020; sekarang t=1030. Yang tertua
        # kedaluwarsa pada 1060, saat itulah jumlahnya turun di bawah batas.
        assert pembatas.retry_after("ip:1") == pytest.approx(30)

    def test_reset_mengosongkan(self, pembatas):
        for _ in range(3):
            pembatas.record_failure("ip:1")
        pembatas.reset("ip:1")
        assert pembatas.retry_after("ip:1") is None

    def test_kunci_terpisah(self, pembatas):
        for _ in range(3):
            pembatas.record_failure("ip:1")
        assert pembatas.retry_after("ip:2") is None

    @pytest.mark.parametrize("maks,jendela", [(0, 60), (3, 0)])
    def test_parameter_tidak_sah(self, maks, jendela):
        with pytest.raises(ValueError):
            FailureLimiter(maks, jendela)
