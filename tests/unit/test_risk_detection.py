"""FR-6 -- deteksi topik berisiko tinggi dan eskalasi kontak."""

from __future__ import annotations

import pytest

from app.rag.risk import (
    DEFAULT_CONTACTS,
    FRONT_OFFICE,
    UNIT_CONTACTS,
    RiskTopic,
    UnitContact,
    contacts_for_unit,
    detect,
)


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
        kustom = {
            RiskTopic.DEADLINE: UnitContact("BAAK", "08-15", "a"),
            RiskTopic.PEMBAYARAN: UnitContact("Keuangan", "08-14", "k"),
            RiskTopic.SANKSI: UnitContact("Kemahasiswaan", "08-15", "m"),
        }
        hasil = detect("kapan deadline bayar UKT dan apa sanksi telat bayar", contacts=kustom)
        units = {c.unit for c in hasil.contacts}
        assert len(units) == len(hasil.contacts)
        assert units == {"BAAK", "Keuangan", "Kemahasiswaan"}

    @pytest.mark.parametrize("topik", list(RiskTopic))
    def test_setiap_topik_punya_kontak_bawaan(self, topik):
        """Topik tanpa kontak berarti banner FR-6 diam-diam hilang."""
        assert DEFAULT_CONTACTS[topik].kontak.strip()

    def test_kontak_bawaan_front_office_satu_banner(self):
        """Tanpa unit yang dipilih, semua topik diarahkan ke Front Office --
        dan cukup satu banner walau beberapa topik terdeteksi."""
        hasil = detect("kapan deadline bayar UKT dan apa sanksinya")
        assert hasil.contacts == (FRONT_OFFICE,)
        assert "(0361) 256995" in FRONT_OFFICE.kontak

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


class TestKontakPerUnit:
    """Kontak mengikuti unit (topik) yang dipilih mahasiswa di widget."""

    @pytest.mark.parametrize(
        "unit,nomor",
        [
            ("Kemahasiswaan", "+62 813-3936-5103"),
            ("BAAK", "+62 813-5346-2032"),
            ("UPS", "+62 813-3734-8654"),
            ("PLK", "+62 851-1931-5346"),
            ("Keuangan", "+62 851-7426-1765"),
            ("FO", "+62 813-3982-2383"),
        ],
    )
    def test_pic_utama_per_unit(self, unit, nomor):
        assert nomor in contacts_for_unit(unit)[0].kontak

    def test_kemahasiswaan_membawa_pic_per_bidang(self):
        kontak = contacts_for_unit("Kemahasiswaan")
        assert [c.kontak for c in kontak[1:]] == [
            "WhatsApp +62 812-3673-3230",
            "WhatsApp +62 819-9843-9707",
            "WhatsApp +62 812-3608-8993",
        ]

    def test_label_unik_dalam_satu_unit(self):
        """Widget memakai label sebagai kunci baris dan kunci banner terakhir."""
        for kontak in UNIT_CONTACTS.values():
            assert len({c.unit for c in kontak}) == len(kontak)

    def test_nomor_ponsel_berlabel_whatsapp(self):
        """Tanpa label ini widget membuka nomornya lewat `tel:` (client `kontak.ts`)."""
        for unit, kontak in UNIT_CONTACTS.items():
            for c in kontak:
                assert "WhatsApp +62 8" in c.kontak, unit

    def test_nama_unit_tidak_peka_huruf_besar(self):
        assert contacts_for_unit(" baak ") == contacts_for_unit("BAAK")

    @pytest.mark.parametrize("unit", [None, "", "Prodi", "Fakultas", "Akademik"])
    def test_unit_tanpa_pic_kosong(self, unit):
        """Pemanggil yang memutuskan cadangannya (Front Office)."""
        assert contacts_for_unit(unit) == ()


class TestDeterminisme:
    def test_urutan_topik_mengikuti_deklarasi_enum(self):
        hasil = detect("sanksi dan deadline dan pembayaran")
        urutan_enum = list(RiskTopic)
        assert list(hasil.topics) == sorted(hasil.topics, key=urutan_enum.index)

    def test_case_insensitive(self):
        assert detect("KAPAN DEADLINE KRS").topics == detect("kapan deadline krs").topics
