"""Kunci sematan: bentuk kunci, alamat situs yang diizinkan, dan kode sematan."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.embed_keys import (
    MAKS_ASAL,
    bentuk_sah,
    buat_kunci,
    kode_sematan,
    normalisasi_asal,
    rapikan_daftar_asal,
)


class TestKunci:
    def test_bentuk_dan_panjang(self):
        kunci = buat_kunci()
        assert kunci.startswith("emb_")
        assert len(kunci) == 28
        assert bentuk_sah(kunci)

    def test_tidak_berulang(self):
        assert len({buat_kunci() for _ in range(200)}) == 200

    @pytest.mark.parametrize(
        "kunci",
        [
            None,
            "",
            "emb_",
            "emb_pendek",
            "emb_" + "a" * 25,
            "key_" + "a" * 24,
            "emb_" + "a" * 23 + "-",
        ],
    )
    def test_bentuk_salah_tidak_sah(self, kunci):
        """Tidak perlu ditanyakan ke database."""
        assert not bentuk_sah(kunci)


class TestNormalisasiAsal:
    @pytest.mark.parametrize(
        "masukan,hasil",
        [
            ("https://pmb.instiki.ac.id", "https://pmb.instiki.ac.id"),
            ("  HTTPS://PMB.Instiki.ac.id/ ", "https://pmb.instiki.ac.id"),
            ("https://pmb.instiki.ac.id:443", "https://pmb.instiki.ac.id"),
            ("http://localhost:5500", "http://localhost:5500"),
            ("http://127.0.0.1:5500", "http://127.0.0.1:5500"),
            ("https://lms.instiki.ac.id:8443", "https://lms.instiki.ac.id:8443"),
            ("https://*.instiki.ac.id", "https://*.instiki.ac.id"),
        ],
    )
    def test_dirapikan_ke_bentuk_csp(self, masukan, hasil):
        assert normalisasi_asal(masukan) == hasil

    @pytest.mark.parametrize(
        "masukan",
        [
            "",
            "pmb.instiki.ac.id",
            "ftp://pmb.instiki.ac.id",
            "https://pmb.instiki.ac.id/daftar",
            "https://pmb.instiki.ac.id?utm=1",
            "https://pmb.instiki.ac.id#asisten",
            "https://admin@pmb.instiki.ac.id",
            "https://",
            "https://pmb instiki.ac.id",
            "https://-pmb.instiki.ac.id",
            "https://pmb.instiki.ac.id:99999",
            "https://*.id",
            "https://pmb.*.ac.id",
        ],
    )
    def test_bukan_asal_ditolak(self, masukan):
        """Path atau domain yang salah tidak pernah cocok dengan `frame-ancestors`,
        dan panelnya ditolak tanpa pesan yang jelas -- lebih baik ditolak di sini."""
        with pytest.raises(ValueError):
            normalisasi_asal(masukan)

    def test_daftar_tanpa_duplikat_setelah_dirapikan(self):
        assert rapikan_daftar_asal(
            ["https://a.instiki.ac.id/", "HTTPS://A.instiki.ac.id", "https://b.instiki.ac.id"]
        ) == ["https://a.instiki.ac.id", "https://b.instiki.ac.id"]

    def test_daftar_dibatasi(self):
        with pytest.raises(ValueError):
            rapikan_daftar_asal([f"https://s{i}.instiki.ac.id" for i in range(MAKS_ASAL + 1)])


class TestKodeSematan:
    def test_siap_tempel(self):
        assert kode_sematan("https://sads.instiki.ac.id", "emb_x") == (
            '<script src="https://sads.instiki.ac.id/embed.js"'
            ' data-key="emb_x" async></script>'
        )

    def test_portal_lokal_bawaan(self):
        assert Settings(_env_file=None).url_portal() == "http://localhost:3001"

    def test_portal_tanpa_garis_miring_akhir(self):
        settings = Settings(_env_file=None, portal_url="https://sads.instiki.ac.id/")
        assert settings.url_portal() == "https://sads.instiki.ac.id"

    def test_portal_kosong_di_produksi_tidak_ditebak(self):
        settings = Settings(
            _env_file=None,
            environment="production",
            api_key="sk-uji",
            admin_jwt_secret="x" * 48,
        )
        assert settings.url_portal() is None
