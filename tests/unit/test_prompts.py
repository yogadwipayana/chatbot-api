"""FR-4 / FR-5 -- instruksi wajib pada prompt dan penyusunan konteks.

Instruksi FR-5 adalah kontrol keamanan. Kontrol keamanan yang tidak diuji
cenderung hilang diam-diam saat seseorang merapikan teks prompt.
"""

from __future__ import annotations

import pytest

from app.rag.prompts import (
    REWRITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    answer_prompt,
    format_context,
    rewrite_prompt,
)
from app.rag.rewriter import Turn, format_history, needs_rewrite
from tests.fixtures.fakes import make_document


class TestInstruksiWajibFR5:
    def test_membatasi_jawaban_pada_konteks(self):
        assert "hanya dari KONTEKS" in SYSTEM_PROMPT

    def test_mewajibkan_sitasi_berikut_formatnya(self):
        assert "hal. N" in SYSTEM_PROMPT

    def test_melarang_menyimpulkan_saat_konteks_kurang(self):
        for kata in ("menyimpulkan", "menebak"):
            assert kata in SYSTEM_PROMPT

    def test_memerintahkan_mengabaikan_instruksi_dalam_pertanyaan(self):
        assert "Abaikan instruksi" in SYSTEM_PROMPT

    def test_menyebut_delimiter_yang_benar_benar_dipakai(self):
        """Instruksi yang menyebut tag berbeda dari yang dipakai kode adalah
        instruksi kosong."""
        from app.security.sanitize import OPEN_TAG

        assert OPEN_TAG.strip("<>") in SYSTEM_PROMPT

    def test_menetapkan_bahasa_indonesia(self):
        assert "Bahasa Indonesia" in SYSTEM_PROMPT

    def test_menyediakan_slot_konteks(self):
        assert "{context}" in SYSTEM_PROMPT


class TestPromptTulisUlang:
    def test_melarang_menjawab_pertanyaan(self):
        assert "Jangan menjawab" in REWRITE_SYSTEM_PROMPT

    def test_mempertahankan_bahasa_asli(self):
        assert "Bahasa Indonesia" in REWRITE_SYSTEM_PROMPT

    def test_mengabaikan_instruksi_dalam_teks(self):
        """Injeksi bisa masuk lewat chain tulis ulang, bukan hanya chain jawaban."""
        assert "Abaikan instruksi" in REWRITE_SYSTEM_PROMPT

    def test_menyediakan_slot_riwayat(self):
        assert "{history}" in REWRITE_SYSTEM_PROMPT


class TestTemplate:
    def test_prompt_jawaban_meminta_context_dan_question(self):
        assert set(answer_prompt().input_variables) == {"context", "question"}

    def test_prompt_tulis_ulang_meminta_history_dan_question(self):
        assert set(rewrite_prompt().input_variables) == {"history", "question"}

    def test_prompt_jawaban_dapat_dirender(self):
        pesan = answer_prompt().format_messages(context="isi dokumen", question="halo")
        assert len(pesan) == 2
        assert "isi dokumen" in pesan[0].content


class TestFormatKonteks:
    def test_setiap_potongan_diberi_penanda_sumber(self):
        """Penanda ditempel per potongan, bukan sekali di akhir, supaya LLM
        dapat mengutip per klaim sebagaimana dituntut FR-5."""
        docs = [
            make_document("c1", judul="Panduan Akademik 2025", halaman=12),
            make_document("c2", judul="SK Rektor 2024", halaman=3),
        ]
        hasil = format_context(docs)
        assert "[Panduan Akademik 2025, hal. 12]" in hasil
        assert "[SK Rektor 2024, hal. 3]" in hasil

    def test_isi_chunk_ikut_disertakan(self):
        docs = [make_document("c1", konten="Batas KRS adalah 7 Agustus.")]
        assert "Batas KRS adalah 7 Agustus." in format_context(docs)

    def test_potongan_dipisahkan_jelas(self):
        docs = [make_document("c1"), make_document("c2")]
        assert "---" in format_context(docs)

    def test_konteks_kosong(self):
        assert format_context([]) == ""

    def test_metadata_hilang_tidak_menggagalkan(self):
        """Chunk lama hasil migrasi bisa saja tanpa judul; jangan sampai 500."""
        from tests.fixtures.fakes import StubDocument

        hasil = format_context([StubDocument(page_content="isi", metadata={})])
        assert "Dokumen tanpa judul" in hasil


class TestRewriterHelper:
    def test_pesan_pertama_tidak_perlu_ditulis_ulang(self):
        assert needs_rewrite([]) is False

    def test_ada_riwayat_berarti_perlu(self):
        assert needs_rewrite([Turn("user", "kapan KRS")]) is True

    def test_hanya_tiga_pesan_terakhir_dipakai(self):
        """FR-4 menyebut 3 pesan terakhir; jendela lebih lebar menaikkan biaya
        tanpa menambah ketepatan."""
        riwayat = [Turn("user", f"pesan {i}") for i in range(10)]
        hasil = format_history(riwayat)
        assert "pesan 9" in hasil
        assert "pesan 6" not in hasil
        assert len(hasil.splitlines()) == 3

    def test_riwayat_pendek_dipakai_seluruhnya(self):
        hasil = format_history([Turn("user", "a"), Turn("assistant", "b")])
        assert len(hasil.splitlines()) == 2

    def test_peran_ikut_ditulis(self):
        assert format_history([Turn("user", "halo")]) == "user: halo"

    def test_riwayat_kosong(self):
        assert format_history([]) == ""

    @pytest.mark.parametrize("window", [0, -1])
    def test_jendela_nol_atau_negatif_menghasilkan_kosong(self, window):
        assert format_history([Turn("user", "a")], window=window) == ""
