"""FR-8 / PRD §12 -- estimasi biaya token.

PRD menandai "biaya API membengkak" sebagai risiko dengan dampak layanan mati
mendadak. Estimasi per pesan inilah yang membuat AD-5 bisa menampilkan biaya
berjalan sebelum tagihan datang.
"""

from __future__ import annotations

import pytest

from app.observability.costs import (
    PRICES_PER_MTOK,
    estimate_cost,
    price_for,
    try_estimate_cost,
)


class TestEstimasi:
    def test_perhitungan_dasar(self):
        hasil = estimate_cost("claude-sonnet-5", 1_000_000, 1_000_000)
        assert hasil.usd == pytest.approx(18.0)

    def test_skala_kecil_tidak_dibulatkan_menjadi_nol(self):
        """Ribuan pertanyaan kecil menumpuk; pembulatan ke nol menyembunyikannya."""
        hasil = estimate_cost("claude-sonnet-5", 1_500, 300)
        assert hasil.usd > 0

    def test_nol_token_berbiaya_nol(self):
        assert estimate_cost("claude-sonnet-5", 0, 0).usd == 0.0

    def test_token_ikut_dikembalikan(self):
        hasil = estimate_cost("gpt-4o-mini", 120, 45)
        assert (hasil.input_tokens, hasil.output_tokens) == (120, 45)

    def test_output_lebih_mahal_dari_input(self):
        masuk = estimate_cost("claude-sonnet-5", 10_000, 0).usd
        keluar = estimate_cost("claude-sonnet-5", 0, 10_000).usd
        assert keluar > masuk

    def test_embedding_tidak_menagih_output(self):
        assert estimate_cost("text-embedding-3-large", 0, 10_000).usd == 0.0


class TestValidasi:
    def test_model_tak_dikenal_melempar(self):
        """Melaporkan biaya nol untuk model tak dikenal lebih menyesatkan
        daripada gagal terang-terangan."""
        with pytest.raises(KeyError):
            estimate_cost("model-yang-belum-didaftarkan", 100, 100)

    @pytest.mark.parametrize("masuk,keluar", [(-1, 0), (0, -1)])
    def test_token_negatif_ditolak(self, masuk, keluar):
        with pytest.raises(ValueError, match="negatif"):
            estimate_cost("claude-sonnet-5", masuk, keluar)


class TestDaftarHarga:
    def test_model_default_terdaftar(self):
        """Model di config harus punya tarif, kalau tidak AD-5 kosong."""
        from app.config import Settings

        s = Settings(_env_file=None)
        assert s.chat_model in PRICES_PER_MTOK
        assert s.embed_model in PRICES_PER_MTOK

    def test_semua_tarif_tidak_negatif(self):
        for model, (masuk, keluar) in PRICES_PER_MTOK.items():
            assert masuk >= 0 and keluar >= 0, model


class TestTarifLewatGateway:
    def test_awalan_penyedia_dilepas(self):
        assert price_for("openrouter/openai/gpt-4o-mini") == PRICES_PER_MTOK["gpt-4o-mini"]

    def test_id_persis_tetap_dikenali(self):
        assert price_for("gpt-4o-mini") == PRICES_PER_MTOK["gpt-4o-mini"]

    def test_model_tak_terdaftar_none(self):
        assert price_for("penyedia/model-yang-belum-didaftarkan") is None

    def test_try_estimate_none_untuk_model_tak_dikenal(self):
        """Tidak melempar: dipakai saat mencatat jawaban yang sudah jadi."""
        assert try_estimate_cost("penyedia/model-yang-belum-didaftarkan", 10, 10) is None

    def test_try_estimate_sama_dengan_estimate(self):
        assert try_estimate_cost("openrouter/gpt-4o-mini", 1500, 300) == estimate_cost(
            "gpt-4o-mini", 1500, 300
        )
