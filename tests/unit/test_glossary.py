"""Kamus sinonim kampus untuk jalur fulltext.

Dua sisi yang sama pentingnya: istilah kampus harus menghasilkan padanannya,
dan pertanyaan tanpa istilah itu TIDAK BOLEH berubah -- jalur fulltext untuk
pertanyaan biasa harus tetap sama persis seperti sebelum kamus ada.
"""

from __future__ import annotations

import pytest

from app.rag.glossary import KELOMPOK, MAKS_VARIAN, fulltext_variants


def dinormalkan(varian: list[str]) -> set[str]:
    return {" ".join(v.split()).casefold() for v in varian}


class TestTanpaIstilah:
    @pytest.mark.parametrize(
        "query", ["pengisian KRS", "syarat wisuda", "cara cuti akademik", "", "   "]
    )
    def test_hanya_query_asli(self, query):
        assert fulltext_variants(query) == [query]

    @pytest.mark.parametrize(
        "query",
        [
            "kampus instikiku",  # bukan kata utuh
            "sadsad",
            "rsk2",
        ],
    )
    def test_bagian_kata_tidak_dicocokkan(self, query):
        assert fulltext_variants(query) == [query]


class TestNamaKampus:
    def test_nama_lama_mencari_nama_sekarang(self):
        varian = dinormalkan(fulltext_variants("akreditasi STIKI"))
        assert "akreditasi instiki" in varian
        assert "akreditasi stmik stikom indonesia" in varian

    def test_nama_sekarang_mencari_nama_lama(self):
        """Dokumen lama dan nama UKM masih menulis STIKI."""
        assert "akreditasi stiki" in dinormalkan(fulltext_variants("akreditasi INSTIKI"))

    def test_instiki_tidak_dibaca_sebagai_stiki(self):
        """Kata "instiki" mengandung "stiki"; tanpa batas kata yang benar, kata
        itu ikut diganti dan menghasilkan "inINSTIKI"."""
        for v in fulltext_variants("akreditasi instiki"):
            assert "ininstiki" not in v.casefold()
            assert "instmik" not in v.casefold()

    def test_frasa_terpanjang_menang(self):
        """Frasa "STIKI Indonesia" diganti utuh, bukan "STIKI"-nya saja."""
        varian = dinormalkan(fulltext_variants("alamat STIKI Indonesia"))
        assert "alamat instiki" in varian
        assert "alamat instiki indonesia" not in varian


class TestSingkatan:
    @pytest.mark.parametrize(
        ("query", "harus_ada"),
        [
            ("sertifikasi di UPS", "sertifikasi di unit pelaksana sertifikasi"),
            ("sertifikasi di unit pelaksana sertifikasi", "sertifikasi di ups"),
            ("jadwal PLK", "jadwal pembelajaran di luar kampus"),
            ("syarat KP", "syarat kerja praktik"),
            ("laporan kerja praktek", "laporan kp"),
            ("daftar RPL", "daftar rekognisi pembelajaran lampau"),
            ("berkas KIP-K", "berkas kip kuliah"),
            ("login SADS", "login sistem akademik"),
            ("akreditasi teknik informatika", "akreditasi informatika"),
            ("akreditasi RSK", "akreditasi rekayasa sistem komputer"),
            ("portofolio DKV", "portofolio desain komunikasi visual"),
        ],
    )
    def test_padanan_dihasilkan(self, query, harus_ada):
        assert harus_ada in dinormalkan(fulltext_variants(query))

    def test_tidak_peka_huruf_besar_dan_spasi(self):
        varian = dinormalkan(fulltext_variants("daftar  kerja   PRAKTIK"))
        assert "daftar kp" in varian

    @pytest.mark.parametrize("query", ["surat SP 1", "nilai TI", "TA 2026/2027"])
    def test_singkatan_bermakna_ganda_tidak_diperluas(self, query):
        """SP = Semester Pendek atau Surat Peringatan; TI dan TA juga ganda."""
        assert fulltext_variants(query) == [query]


class TestBentukHasil:
    def test_query_asli_selalu_pertama(self):
        assert fulltext_variants("sertifikasi UPS")[0] == "sertifikasi UPS"

    def test_tanpa_duplikat(self):
        varian = fulltext_variants("akreditasi instiki")
        assert len(dinormalkan(varian)) == len(varian)

    def test_dibatasi(self):
        query = "akreditasi RSK STIKI dan DKV serta sertifikasi UPS"
        assert len(fulltext_variants(query)) == MAKS_VARIAN

    def test_varian_nama_sekarang_tetap_ikut_walau_terbatas(self):
        """Dokumen terbaru memakai nama sekarang; varian itu tidak boleh
        terpotong oleh batas jumlah varian."""
        query = "akreditasi RSK STIKI dan DKV serta sertifikasi UPS"
        assert (
            "akreditasi rekayasa sistem komputer instiki dan desain komunikasi visual "
            "serta sertifikasi ups"
        ) in dinormalkan(fulltext_variants(query))

    def test_setiap_istilah_hanya_di_satu_kelompok(self):
        """Istilah di dua kelompok membuat padanannya bergantung urutan."""
        semua = [" ".join(i.split()).casefold() for k in KELOMPOK for i in k]
        assert len(semua) == len(set(semua))
