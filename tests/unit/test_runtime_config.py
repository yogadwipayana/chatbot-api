"""Lapisan penimpaan setelan (`app.admin.runtime_config`).

Dua hal yang dijaga di sini:

- daftar parameter yang boleh diubah, batasnya, dan bentuk balasannya tidak
  boleh saling menyimpang -- ketiganya ditulis di berkas yang berbeda;
- baris simpanan yang tidak dapat dipakai tidak boleh menjatuhkan layanan.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.admin.runtime_config import (
    DAPAT_DIUBAH,
    NilaiTersimpan,
    bangun,
    keluhan,
    terapkan,
)
from app.config import Settings
from app.schemas.admin import RuntimeConfigUpdate, RuntimeConfigValues


def simpan(**nilai: str) -> dict[str, NilaiTersimpan]:
    waktu = datetime.now(UTC)
    return {k: NilaiTersimpan(v, waktu, "admin@kampus.ac.id") for k, v in nilai.items()}


@pytest.fixture
def base() -> Settings:
    return Settings(
        chunk_size=900,
        chunk_overlap=105,
        retrieval_candidates=20,
        retrieval_top_n=5,
        vector_threshold=0.35,
    )


class TestDaftarParameter:
    def test_semuanya_field_settings_sungguhan(self):
        """Salah ketik satu nama membuat penimpaannya diam-diam tidak berlaku."""
        assert set(DAPAT_DIUBAH) <= set(Settings.model_fields)

    def test_skema_perubahan_sama_dengan_daftar(self):
        assert set(RuntimeConfigUpdate.model_fields) == set(DAPAT_DIUBAH)

    def test_skema_nilai_sama_dengan_daftar(self):
        assert set(RuntimeConfigValues.model_fields) == set(DAPAT_DIUBAH)

    def test_rahasia_tidak_ikut_dapat_diubah(self):
        """Kredensial dan nama model bukan urusan dashboard."""
        terlarang = {
            "api_key",
            "admin_jwt_secret",
            "database_url",
            "chat_model",
            "embed_model",
        }
        assert terlarang.isdisjoint(DAPAT_DIUBAH)


class TestBangun:
    def test_tanpa_penimpaan_mengembalikan_objek_yang_sama(self, base):
        """Keadaan normal tidak boleh menyusun ulang Settings di tiap permintaan."""
        assert bangun(base, {}) is base

    def test_nilai_teks_diubah_ke_tipe_field(self, base):
        efektif = bangun(base, {"retrieval_top_n": "7", "vector_threshold": "0.5"})
        assert (efektif.retrieval_top_n, efektif.vector_threshold) == (7, 0.5)

    def test_field_lain_ikut_utuh(self, base):
        efektif = bangun(base, {"retrieval_top_n": "7"})
        assert efektif.chat_model == base.chat_model
        assert efektif.timezone == base.timezone

    def test_base_tidak_ikut_berubah(self, base):
        bangun(base, {"retrieval_top_n": "7"})
        assert base.retrieval_top_n == 5

    def test_kunci_di_luar_daftar_diabaikan(self, base):
        """Baris nyasar di database tidak boleh menjadi pintu belakang."""
        efektif = bangun(base, {"admin_jwt_secret": "x" * 40})
        assert efektif.admin_jwt_secret == base.admin_jwt_secret

    def test_aturan_antar_field_tetap_berlaku(self, base):
        with pytest.raises(ValidationError):
            bangun(base, {"retrieval_top_n": "50"})

    def test_melanggar_nilai_env_yang_tidak_ikut_diubah(self, base):
        """`chunk_overlap` sah sendirian, tetapi tidak terhadap chunk_size 900."""
        with pytest.raises(ValidationError):
            bangun(base, {"chunk_overlap": "950"})


class TestTerapkan:
    def test_nilai_sah_dipakai(self, base):
        assert terapkan(base, simpan(vector_threshold="0.6")).vector_threshold == 0.6

    def test_baris_rusak_membuat_layanan_kembali_ke_env(self, base, caplog):
        """Mahasiswa tetap dijawab; kegagalannya masuk log, bukan ke respons."""
        efektif = terapkan(base, simpan(chunk_overlap="950"))
        assert efektif is base
        assert "diabaikan" in caplog.text.lower()

    def test_keluhan_menjelaskan_sebabnya(self, base):
        pesan = keluhan(base, simpan(chunk_overlap="950"))
        assert pesan and "chunk_overlap" in pesan

    def test_keluhan_kosong_saat_sehat(self, base):
        assert keluhan(base, simpan(chunk_overlap="100")) is None
