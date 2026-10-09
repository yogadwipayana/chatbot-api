"""Query fulltext dari pertanyaan mahasiswa (T17).

Dulu setiap kata digabung DAN dan kata tanya ikut dicari, sehingga pertanyaan
utuh hampir tidak pernah cocok dengan potongan mana pun (enam pertanyaan uji:
semuanya 0 potongan). Dua hal yang dijaga di sini: kata umum benar-benar
terbuang, dan kata isi TIDAK ikut terbuang.
"""

from __future__ import annotations

import pytest

from app.rag.fts_query import STOPWORDS, fulltext_queries, fulltext_query


class TestKataUmumDibuang:
    @pytest.mark.parametrize(
        ("pertanyaan", "query"),
        [
            ("berapa harga sertifikasi TOEIC?", "harga or sertifikasi or toeic"),
            (
                "Siapa yang menerbitkan sertifikat pelatihan sertifikasi?",
                "menerbitkan or sertifikat or pelatihan or sertifikasi",
            ),
            (
                "Bagaimana cara bayar VA BNI lewat SMS banking?",
                "bayar or va or bni or sms or banking",
            ),
            ("min mau tanya dong, syarat cuti apa aja ya kak?", "syarat or cuti"),
            # T40: stemmer menyamakan "urus" dengan "pengurus" (tabel poin organisasi).
            ("cara urus skp gimana?", "skp"),
            ("gimana mengurusnya?", ""),
            ("poin pengurus inti BEM", "poin or pengurus or inti or bem"),
        ],
    )
    def test_pertanyaan_menjadi_query_or(self, pertanyaan, query):
        assert fulltext_query(pertanyaan) == query

    @pytest.mark.parametrize("teks", ["apa itu?", "min, mau tanya dong", "???", ""])
    def test_semua_kata_umum_menghasilkan_kosong(self, teks):
        assert fulltext_query(teks) == ""
        assert fulltext_queries(teks) == []

    def test_akhiran_klitik_ikut_dikenali_sebagai_kata_umum(self):
        """apakah, caranya, infonya -- stemmer Postgres melepas akhiran itu."""
        assert fulltext_query("apakah caranya ada infonya?") == ""

    def test_kata_duplikat_sekali_saja(self):
        assert fulltext_query("KRS krs Krs") == "krs"

    def test_huruf_dan_digit_tunggal_dibuang_angka_panjang_tidak(self):
        """"1", "2", "3" ada di 40-47% potongan; tahun tetap bermakna."""
        query = fulltext_query("wisuda 2026 tahap 2 kelas B")
        assert query == "wisuda or 2026 or tahap or kelas"


class TestKataIsiTidakIkutTerbuang:
    """Penyaringan memakai kata mentah, bukan stem: stemmer Postgres menyatukan
    kata isi dengan stopword (peserta -> serta, permohonan -> mohon)."""

    @pytest.mark.parametrize(
        "kata", ["peserta", "permohonan", "pendapatan", "media", "sekolah", "diadakan"]
    )
    def test_kata_isi_yang_stemnya_mirip_stopword(self, kata):
        assert fulltext_query(f"syarat {kata}") == f"syarat or {kata}"

    @pytest.mark.parametrize("kata", ["syarat", "biaya", "daftar", "jadwal", "sore", "pagi"])
    def test_kata_yang_menentukan_maksud_dipertahankan(self, kata):
        """"kelas sore" dan "jadwal pagi" adalah pertanyaan sungguhan."""
        assert kata not in STOPWORDS
        assert kata in fulltext_query(f"apa {kata} ya?")


class TestIstilahKampus:
    def test_istilah_multikata_dikirim_sebagai_frasa(self):
        """Terpecah jadi kata umum ("unit", "sistem"), ia cocok ke mana-mana."""
        assert fulltext_query("biaya di Unit Pelaksana Sertifikasi?") == (
            'biaya or "unit pelaksana sertifikasi"'
        )

    def test_stopword_di_dalam_frasa_tidak_dibuang(self):
        assert '"pembelajaran di luar kampus"' in fulltext_query(
            "syarat Pembelajaran di Luar Kampus"
        )

    def test_istilah_bertanda_hubung_utuh(self):
        assert fulltext_query("pendaftaran KIP-K") == "pendaftaran or kip-k"

    def test_setiap_varian_kamus_dibersihkan(self):
        varian = fulltext_queries("apa syarat PLK?")
        assert varian[0] == "syarat or plk"
        assert 'syarat or "pembelajaran di luar kampus"' in varian


class TestSintaksPenggunaDinetralkan:
    """Teks ini masuk ke `websearch_to_tsquery`: tanda kutip (frasa), minus di
    depan kata (NOT), dan "or" punya arti di sana."""

    def test_kutip_minus_dan_or_tidak_menjadi_operator(self):
        assert fulltext_query('syarat "wisuda" -online or') == "syarat or wisuda or online"

    def test_hanya_kata_yang_lolos(self):
        query = fulltext_query("biaya (UKT) & denda | 'cicil' !")
        assert query == "biaya or ukt or denda or cicil"
