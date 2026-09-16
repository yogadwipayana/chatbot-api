"""Deteksi sapaan dan basa-basi.

Dua sisi yang sama pentingnya: basa-basi harus dikenali supaya tidak berakhir
sebagai penolakan FR-3, dan pertanyaan sungguhan TIDAK BOLEH ikut tertelan --
sapaan palsu atas pertanyaan administrasi berarti mahasiswa tidak dijawab sama
sekali.
"""

from __future__ import annotations

import pytest

from app.rag.smalltalk import SmallTalkKind, detect


class TestDikenali:
    @pytest.mark.parametrize(
        "teks",
        ["hai", "Halo!", "hi", "  halo  ", "halo min", "selamat pagi", "pagi kak", "assalamualaikum", "permisi"],
    )
    def test_sapaan(self, teks):
        assert detect(teks).kind is SmallTalkKind.GREETING

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
        ],
    )
    def test_pertanyaan_tidak_pernah_tertelan(self, teks):
        assert detect(teks).kind is SmallTalkKind.NONE

    def test_sapaan_bersama_kata_lain_bukan_basa_basi(self):
        """"halo skripsi" adalah awal pertanyaan yang terpotong, bukan sapaan."""
        assert not detect("halo skripsi").handled

    @pytest.mark.parametrize("teks", ["   ", "", "???"])
    def test_pesan_tanpa_kata(self, teks):
        assert not detect(teks).handled
