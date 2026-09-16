"""FR-5 / FE-2 -- sitasi wajib dan dapat diverifikasi."""

from __future__ import annotations

import pytest

from app.rag.citations import (
    Citation,
    extract_citations,
    format_citation,
    validate_answer,
)

KONTEKS = {("Panduan Akademik 2025", 12), ("SK Rektor 2024", 3)}


class TestFormat:
    def test_bentuk_baku(self):
        hasil = format_citation("Panduan Akademik 2025", 12)
        assert hasil == "[Panduan Akademik 2025, hal. 12]"

    def test_judul_dirapikan(self):
        assert format_citation("  Panduan  ", 1) == "[Panduan, hal. 1]"

    def test_judul_kosong_ditolak(self):
        with pytest.raises(ValueError, match="judul"):
            format_citation("   ", 5)

    @pytest.mark.parametrize("halaman", [0, -3])
    def test_halaman_tidak_valid_ditolak(self, halaman):
        with pytest.raises(ValueError, match="halaman"):
            format_citation("Panduan", halaman)

    def test_hasil_format_dapat_diurai_kembali(self):
        """FE-2 butuh perjalanan bolak-balik: teks -> (dokumen, halaman) -> PDF."""
        teks = format_citation("Panduan Akademik 2025", 12)
        assert extract_citations(teks) == (Citation("Panduan Akademik 2025", 12),)


class TestEkstraksi:
    def test_beberapa_sitasi_terurut_kemunculan(self):
        jawaban = (
            "Syaratnya A [Panduan Akademik 2025, hal. 12] "
            "dan B [SK Rektor 2024, hal. 3]."
        )
        assert extract_citations(jawaban) == (
            Citation("Panduan Akademik 2025", 12),
            Citation("SK Rektor 2024", 3),
        )

    def test_duplikat_dibuang(self):
        jawaban = "A [Panduan, hal. 2]. B [Panduan, hal. 2]."
        assert len(extract_citations(jawaban)) == 1

    @pytest.mark.parametrize(
        "teks",
        [
            "[Panduan, hal. 4]",
            "[Panduan, hal 4]",
            "[Panduan,hal.4]",
            "[ Panduan , hal.  4 ]",
            "[Panduan, HAL. 4]",
        ],
    )
    def test_variasi_penulisan_tetap_terbaca(self, teks):
        """LLM tidak selalu konsisten spasinya; sitasi sah jangan dianggap hilang."""
        assert extract_citations(teks) == (Citation("Panduan", 4),)

    def test_jawaban_tanpa_sitasi(self):
        assert extract_citations("Silakan hubungi bagian akademik.") == ()

    def test_kurung_biasa_bukan_sitasi(self):
        assert extract_citations("Lihat bagian (hal. 12) dokumen.") == ()


class TestValidasi:
    def test_jawaban_dengan_sumber_sah(self):
        hasil = validate_answer("Jawaban [Panduan Akademik 2025, hal. 12].", KONTEKS)
        assert hasil.is_valid
        assert hasil.has_citation
        assert hasil.unknown == ()

    def test_jawaban_tanpa_sitasi_tidak_valid(self):
        """FR-5 mewajibkan sumber pada setiap klaim."""
        hasil = validate_answer("Syaratnya tiga hal.", KONTEKS)
        assert not hasil.has_citation
        assert not hasil.is_valid

    def test_sitasi_ke_dokumen_di_luar_konteks_ditandai(self):
        """Sumber karangan lebih berbahaya daripada tidak menjawab: ia tampak sah."""
        hasil = validate_answer("Menurut [Peraturan Fiktif 2030, hal. 9].", KONTEKS)
        assert hasil.unknown == (Citation("Peraturan Fiktif 2030", 9),)
        assert not hasil.is_valid

    def test_halaman_salah_pada_dokumen_benar_ditandai(self):
        """Judul benar tetapi halaman meleset tetap membuat verifikasi FE-2 gagal."""
        hasil = validate_answer("Lihat [Panduan Akademik 2025, hal. 99].", KONTEKS)
        assert hasil.unknown == (Citation("Panduan Akademik 2025", 99),)

    def test_perbedaan_kapitalisasi_judul_bukan_karangan(self):
        hasil = validate_answer("Lihat [panduan akademik 2025, hal. 12].", KONTEKS)
        assert hasil.unknown == ()
        assert hasil.is_valid

    def test_campuran_sitasi_sah_dan_karangan_tetap_tidak_valid(self):
        jawaban = "A [Panduan Akademik 2025, hal. 12] dan B [Entah Apa, hal. 1]."
        hasil = validate_answer(jawaban, KONTEKS)
        assert len(hasil.citations) == 2
        assert len(hasil.unknown) == 1
        assert not hasil.is_valid

    def test_konteks_kosong_membuat_semua_sitasi_karangan(self):
        hasil = validate_answer("Lihat [Panduan Akademik 2025, hal. 12].", set())
        assert len(hasil.unknown) == 1
