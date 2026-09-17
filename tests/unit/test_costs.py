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

CHAT_MODEL = "cx/gpt-5.5"
EMBED_MODEL = "openrouter/openai/text-embedding-3-small"


class TestEstimasi:
    def test_perhitungan_dasar(self):
        hasil = estimate_cost(CHAT_MODEL, 1_000_000, 1_000_000)
        assert hasil.usd == pytest.approx(35.0)

    def test_skala_kecil_tidak_dibulatkan_menjadi_nol(self):
        """Ribuan pertanyaan kecil menumpuk; pembulatan ke nol menyembunyikannya."""
        hasil = estimate_cost(CHAT_MODEL, 1_500, 300)
        assert hasil.usd > 0

    def test_nol_token_berbiaya_nol(self):
        assert estimate_cost(CHAT_MODEL, 0, 0).usd == 0.0

    def test_token_ikut_dikembalikan(self):
        hasil = estimate_cost(CHAT_MODEL, 120, 45)
        assert (hasil.input_tokens, hasil.output_tokens) == (120, 45)

    def test_output_lebih_mahal_dari_input(self):
        masuk = estimate_cost(CHAT_MODEL, 10_000, 0).usd
        keluar = estimate_cost(CHAT_MODEL, 0, 10_000).usd
        assert keluar > masuk

    def test_embedding_tidak_menagih_output(self):
        assert estimate_cost(EMBED_MODEL, 0, 10_000).usd == 0.0


class TestValidasi:
    def test_model_tak_dikenal_melempar(self):
        """Melaporkan biaya nol untuk model tak dikenal lebih menyesatkan
        daripada gagal terang-terangan."""
        with pytest.raises(KeyError):
            estimate_cost("model-yang-belum-didaftarkan", 100, 100)

    @pytest.mark.parametrize("masuk,keluar", [(-1, 0), (0, -1)])
    def test_token_negatif_ditolak(self, masuk, keluar):
        with pytest.raises(ValueError, match="negatif"):
            estimate_cost(CHAT_MODEL, masuk, keluar)


class TestDaftarHarga:
    def test_model_operasional_terdaftar(self):
        """Model yang dipakai deployment harus punya tarif agar AD-5 tidak kosong."""
        assert CHAT_MODEL in PRICES_PER_MTOK
        assert EMBED_MODEL in PRICES_PER_MTOK

    def test_semua_tarif_tidak_negatif(self):
        for model, (masuk, keluar) in PRICES_PER_MTOK.items():
            assert masuk >= 0 and keluar >= 0, model


class TestTarifLewatGateway:
    def test_id_gateway_lengkap_dikenali(self):
        assert price_for(EMBED_MODEL) == PRICES_PER_MTOK[EMBED_MODEL]

    def test_id_persis_tetap_dikenali(self):
        assert price_for(CHAT_MODEL) == PRICES_PER_MTOK[CHAT_MODEL]

    def test_model_tak_terdaftar_none(self):
        assert price_for("penyedia/model-yang-belum-didaftarkan") is None

    def test_try_estimate_none_untuk_model_tak_dikenal(self):
        """Tidak melempar: dipakai saat mencatat jawaban yang sudah jadi."""
        assert try_estimate_cost("penyedia/model-yang-belum-didaftarkan", 10, 10) is None

    def test_try_estimate_sama_dengan_estimate(self):
        assert try_estimate_cost(CHAT_MODEL, 1500, 300) == estimate_cost(
            CHAT_MODEL, 1500, 300
        )
