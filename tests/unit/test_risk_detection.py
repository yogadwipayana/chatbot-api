"""FR-6 -- deteksi topik berisiko tinggi dan eskalasi kontak."""

from __future__ import annotations

import pytest

from app.rag.risk import RiskTopic, UnitContact, detect


class TestDeteksiPerTopik:
    @pytest.mark.parametrize(
        "pertanyaan,topik",
        [
            ("kapan deadline pengisian KRS?", RiskTopic.DEADLINE),
            ("batas akhir pembatalan mata kuliah kapan", RiskTopic.DEADLINE),
            ("tenggat pengumpulan skripsi", RiskTopic.DEADLINE),
            ("apa syarat kelulusan S1", RiskTopic.SYARAT_KELULUSAN),
            ("IPK minimal untuk yudisium berapa", RiskTopic.SYARAT_KELULUSAN),
            ("bagaimana cara pembayaran UKT", RiskTopic.PEMBAYARAN),
            ("ada denda kalau telat bayar?", RiskTopic.PEMBAYARAN),
            ("sanksi kalau tidak isi KRS apa", RiskTopic.SANKSI),
            ("apakah saya bisa kena skorsing", RiskTopic.SANKSI),
            ("berapa lama sampai kena DO", RiskTopic.DROP_OUT),
            ("aturan drop out di kampus", RiskTopic.DROP_OUT),
        ],
    )
    def test_topik_terdeteksi(self, pertanyaan, topik):
        assert topik in detect(pertanyaan).topics

    def test_pertanyaan_biasa_tidak_memicu_eskalasi(self):
        hasil = detect("di mana ruang tata usaha fakultas?")
        assert hasil.topics == ()
        assert not hasil.needs_escalation
        assert hasil.contacts == ()


class TestBatasKataDO:
    """DO adalah akronim dua huruf. Dicocokkan sembarangan, ia kena di mana-mana."""

    @pytest.mark.parametrize(
        "pertanyaan",
        [
            "siapa dosen pembimbing akademik saya",
            "saya dapat kado dari kampus",
            "peraturan di Indonesia soal ijazah",
            "cara membuat todo list kuliah",
            "kegiatan donor darah kampus",
        ],
    )
    def test_kata_lain_tidak_ikut_terdeteksi(self, pertanyaan):
        assert RiskTopic.DROP_OUT not in detect(pertanyaan).topics

    @pytest.mark.parametrize(
        "pertanyaan",
        ["saya takut kena DO", "syarat mahasiswa di-DO", "terancam do semester ini"],
    )
    def test_penggunaan_sah_tetap_terdeteksi(self, pertanyaan):
        assert RiskTopic.DROP_OUT in detect(pertanyaan).topics

    def test_do_lepas_tanpa_kata_pendahulu_tidak_memicu(self):
        """Bentuk lepas huruf kecil terlalu sering muncul di teks campuran;
        hanya bentuk kapital atau frasa jelas yang dihitung."""
        assert RiskTopic.DROP_OUT not in detect("what should i do next semester").topics


class TestKontakEskalasi:
    def test_kontak_disertakan_untuk_topik_berisiko(self):
        hasil = detect("kapan deadline pembayaran UKT?")
        assert hasil.needs_escalation
        assert len(hasil.contacts) >= 1

    def test_kontak_dideduplikasi(self):
        """Deadline dan syarat kelulusan ditangani unit sama -> satu banner saja."""
        hasil = detect("apa syarat kelulusan dan kapan deadline-nya")
        assert {RiskTopic.DEADLINE, RiskTopic.SYARAT_KELULUSAN} <= set(hasil.topics)
        assert len(hasil.contacts) == 1

    def test_unit_berbeda_memunculkan_kontak_berbeda(self):
        hasil = detect("kapan deadline bayar UKT dan apa sanksinya")
        units = {c.unit for c in hasil.contacts}
        assert len(units) == len(hasil.contacts)
        assert len(units) >= 2

    def test_kontak_dapat_diganti(self):
        """Fase 0 menetapkan daftar kontak; modul ini tidak boleh menguncinya."""
        kustom = {
            RiskTopic.PEMBAYARAN: UnitContact("Bagian Keuangan FT", "08-12", "ft@x.ac.id")
        }
        hasil = detect("cara pembayaran UKT", contacts=kustom)
        assert hasil.contacts[0].unit == "Bagian Keuangan FT"

    def test_topik_tanpa_kontak_tidak_menggagalkan(self):
        hasil = detect("cara pembayaran UKT", contacts={})
        assert RiskTopic.PEMBAYARAN in hasil.topics
        assert hasil.contacts == ()


class TestDeterminisme:
    def test_urutan_topik_mengikuti_deklarasi_enum(self):
        hasil = detect("sanksi dan deadline dan pembayaran")
        urutan_enum = list(RiskTopic)
        assert list(hasil.topics) == sorted(hasil.topics, key=urutan_enum.index)

    def test_case_insensitive(self):
        assert detect("KAPAN DEADLINE KRS").topics == detect("kapan deadline krs").topics
