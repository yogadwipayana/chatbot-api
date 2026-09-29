"""Deteksi sapaan dan basa-basi.

Dua sisi yang sama pentingnya: basa-basi harus dikenali supaya tidak berakhir
sebagai penolakan FR-3, dan pertanyaan sungguhan TIDAK BOLEH ikut tertelan --
sapaan palsu atas pertanyaan administrasi berarti mahasiswa tidak dijawab sama
sekali.
"""

from __future__ import annotations

import pytest

from app.rag.smalltalk import REPLIES, SmallTalkKind, detect, reply_for


class TestDikenali:
    @pytest.mark.parametrize(
        "teks",
        ["hai", "Halo!", "hi", "  halo  ", "halo min", "selamat pagi", "pagi kak", "assalamualaikum", "permisi"],
    )
    def test_sapaan(self, teks):
        assert detect(teks).kind is SmallTalkKind.GREETING

    @pytest.mark.parametrize(
        "teks", ["Om Swastyastu", "om swastiastu kak", "Om Swastyastu 🙏", "swastyastu"]
    )
    def test_salam_bali(self, teks):
        assert detect(teks).kind is SmallTalkKind.GREETING

    def test_salam_bali_bersama_pertanyaan_bukan_basa_basi(self):
        assert not detect("om swastyastu, kapan KRS dibuka?").handled

    def test_sapaan_menyebut_nama_kampus(self):
        assert "INSTIKI" in detect("halo").reply

    @pytest.mark.parametrize(
        "teks", ["terima kasih", "makasih ya", "makasih banyak", "thanks", "suksma"]
    )
    def test_terima_kasih(self, teks):
        assert detect(teks).kind is SmallTalkKind.THANKS

    @pytest.mark.parametrize("teks", ["oke", "sip", "baik kak", "sampai jumpa"])
    def test_penutup(self, teks):
        assert detect(teks).kind is SmallTalkKind.CLOSING

    def test_membawa_balasan_siap_tampil(self):
        hasil = detect("hai")
        assert hasil.handled
        assert hasil.reply.strip()

    @pytest.mark.parametrize(
        "teks",
        [
            # Tiga pesan dari uji browser yang dulu berakhir "tidak ditemukan" (T12).
            "oke siap min, nanti saya coba dulu ya",
            "baik min, sudah jelas penjelasannya",
            "sip, itu saja dulu pertanyaan saya",
            "oke kalau begitu",
            "sekian dulu min",
            "sudah cukup",
            "noted kak",
        ],
    )
    def test_penutup_dengan_kata_pengiring(self, teks):
        assert detect(teks).kind is SmallTalkKind.CLOSING

    @pytest.mark.parametrize(
        "teks",
        [
            "terima kasih atas bantuannya",
            "makasih min, semoga sehat selalu",
            "oke makasih ya",
            "oke terima kasih",
            "siap, terima kasih banyak kak",
            "sangat membantu, thanks",
        ],
    )
    def test_campuran_dengan_terima_kasih_dijawab_sama_sama(self, teks):
        """Terima kasih mengalahkan penutup dan sapaan: yang pamit tidak disapa ulang."""
        assert detect(teks).kind is SmallTalkKind.THANKS


class TestTidakDikenali:
    @pytest.mark.parametrize(
        "teks",
        [
            "halo, kapan pengisian KRS dibuka?",
            "pagi, bagaimana cara mengajukan cuti",
            "terima kasih, tapi bagaimana cara cuti?",
            "kapan KRS dibuka",
            "syarat wisuda",
            "halo saya mau tanya soal skripsi dan wisuda",
            "kelas malam",
            # Kata netral baru tidak boleh membuka celah untuk pertanyaan.
            "saya sudah bayar ukt tapi belum masuk",
            "oke, lalu syarat cuti?",
            "sudah jelas, tapi bagaimana cara daftar ulang",
            "oke saya coba ke BAAK besok",
            "nanti KRS dibuka kapan",
            "makasih, saya mau tanya lagi soal beasiswa",
            "saya belum paham",
        ],
    )
    def test_pertanyaan_tidak_pernah_tertelan(self, teks):
        assert detect(teks).kind is SmallTalkKind.NONE

    @pytest.mark.parametrize("teks", ["itu saja", "sudah jelas", "saya coba dulu"])
    def test_hanya_kata_pengiring_bukan_basa_basi(self, teks):
        """Tanpa satu pun kata kunci, pesan diteruskan -- bisa jadi kalimat terpotong."""
        assert not detect(teks).handled


class TestBalasanUntukVonisJev:
    """JEV hanya memvonis "smalltalk"; nada balasannya dibaca dari pesannya."""

    def test_pamit_dibalas_penutup(self):
        assert reply_for("oke deh kalau begitu, nanti saya ke kampus aja") == REPLIES[
            SmallTalkKind.CLOSING
        ]

    def test_terima_kasih_dibalas_sama_sama(self):
        assert reply_for("wah makasih banyak, sangat membantu skripsi saya") == REPLIES[
            SmallTalkKind.THANKS
        ]

    def test_tanpa_kata_dikenal_diserahkan_ke_jev(self):
        assert reply_for("apa kabar bot") is None

    def test_sapaan_bersama_kata_lain_bukan_basa_basi(self):
        """"halo skripsi" adalah awal pertanyaan yang terpotong, bukan sapaan."""
        assert not detect("halo skripsi").handled

    @pytest.mark.parametrize("teks", ["   ", "", "???"])
    def test_pesan_tanpa_kata(self, teks):
        assert not detect(teks).handled
