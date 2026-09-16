"""FR-7 -- penanganan pertanyaan sensitif."""

from __future__ import annotations

import pytest

from app.rag.sensitive import SensitivityLevel, detect


class TestTingkatDistress:
    @pytest.mark.parametrize(
        "pertanyaan",
        [
            "saya stres banget mikirin skripsi",
            "aku depresi karena nilai jelek",
            "sudah putus asa mau ngurus KRS",
            "saya gak kuat lagi kuliah",
            "rasanya mau menyerah aja",
            "saya dibully teman sekelas",
            "ada konflik dengan dosen pembimbing saya",
            "saya takut di-DO semester ini",
        ],
    )
    def test_terdeteksi_distress(self, pertanyaan):
        assert detect(pertanyaan).level is SensitivityLevel.DISTRESS

    def test_kontak_konseling_disertakan(self):
        hasil = detect("saya stres berat")
        assert hasil.contacts
        assert any("Konseling" in c.unit for c in hasil.contacts)


class TestTingkatCrisis:
    @pytest.mark.parametrize(
        "pertanyaan",
        [
            "saya kepikiran bunuh diri",
            "rasanya ingin mengakhiri hidup",
            "saya sudah tidak mau hidup lagi",
            "kadang saya menyakiti diri sendiri",
        ],
    )
    def test_terdeteksi_crisis(self, pertanyaan):
        assert detect(pertanyaan).level is SensitivityLevel.CRISIS

    def test_crisis_menang_atas_distress(self):
        """Kalimat yang mengandung keduanya harus naik ke tingkat tertinggi."""
        hasil = detect("saya stres berat dan ingin mengakhiri hidup")
        assert hasil.level is SensitivityLevel.CRISIS

    def test_kontak_krisis_tersedia_24_jam(self):
        hasil = detect("saya ingin mengakhiri hidup")
        assert any("24 jam" in c.jam_layanan for c in hasil.contacts)


class TestPertanyaanAdministrasiBiasa:
    @pytest.mark.parametrize(
        "pertanyaan",
        [
            "kapan batas akhir pengisian KRS?",
            "bagaimana cara mengajukan cuti kuliah",
            "syarat wisuda apa saja",
            "berapa biaya UKT semester depan",
        ],
    )
    def test_tidak_terdeteksi_sensitif(self, pertanyaan):
        hasil = detect(pertanyaan)
        assert hasil.level is SensitivityLevel.NONE
        assert not hasil.bypasses_rag
        assert hasil.contacts == ()


class TestPrioritasAtasFR6:
    def test_takut_di_DO_ditangani_sebagai_sensitif_bukan_administrasi(self):
        """Mahasiswa yang menulis ketakutan tidak boleh dibalas kutipan pasal.

        Kalimat ini juga memicu FR-6 (kata kunci DO). Yang menang harus FR-7:
        arahkan ke konseling, jangan jelaskan tata cara DO.
        """
        hasil = detect("saya stres, takut di-DO semester ini")
        assert hasil.level is SensitivityLevel.DISTRESS
        assert hasil.bypasses_rag

    def test_pertanyaan_prosedural_tentang_DO_tetap_administratif(self):
        """Sebaliknya, pertanyaan netral soal aturan DO bukan urusan konseling."""
        hasil = detect("berapa batas maksimal semester sebelum kena DO?")
        assert hasil.level is SensitivityLevel.NONE


class TestBypass:
    def test_distress_melewati_rag(self):
        assert detect("saya stres").bypasses_rag is True

    def test_crisis_melewati_rag(self):
        assert detect("ingin mengakhiri hidup").bypasses_rag is True

    def test_none_tidak_melewati_rag(self):
        assert detect("kapan wisuda").bypasses_rag is False
