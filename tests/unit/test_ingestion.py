"""FR-1 -- ingestion dokumen: deteksi PDF scan dan pemecahan chunk."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import split_pages
from app.ingestion.loader import (
    LoadedPage,
    empty_page_ratio,
    is_probably_scanned,
)

TEKS_PENUH = "Panduan akademik. " * 20


class TestDeteksiPdfScan:
    """PDF scan menghasilkan chunk kosong yang merusak retrieval tanpa terlihat:
    dokumen tampak masuk indeks, tetapi tidak pernah terambil."""

    def test_dokumen_teks_penuh_bukan_scan(self):
        assert not is_probably_scanned([TEKS_PENUH] * 10)

    def test_dokumen_kosong_seluruhnya_terdeteksi_scan(self):
        assert is_probably_scanned(["", "", ""])

    def test_halaman_hanya_spasi_dihitung_kosong(self):
        assert is_probably_scanned(["   \n\n  ", "  ", " "])

    def test_sedikit_halaman_kosong_masih_diterima(self):
        """Halaman sampul dan halaman pemisah wajar kosong; jangan tolak dokumen sah."""
        halaman = [TEKS_PENUH] * 9 + [""]
        assert not is_probably_scanned(halaman)

    def test_mayoritas_halaman_kosong_ditolak(self):
        halaman = [TEKS_PENUH] * 3 + [""] * 7
        assert is_probably_scanned(halaman)

    def test_dokumen_tanpa_halaman_dianggap_scan(self):
        assert is_probably_scanned([])

    def test_rasio_dihitung_benar(self):
        assert empty_page_ratio([TEKS_PENUH, "", TEKS_PENUH, ""]) == 0.5

    def test_rasio_dokumen_kosong(self):
        assert empty_page_ratio([]) == 1.0

    def test_ambang_dapat_diatur(self):
        halaman = [TEKS_PENUH] * 5 + [""] * 5
        assert is_probably_scanned(halaman, max_empty_ratio=0.4)
        assert not is_probably_scanned(halaman, max_empty_ratio=0.6)

    def test_ambang_karakter_dapat_diatur(self):
        pendek = ["Bab I", "Bab II"]
        assert is_probably_scanned(pendek, min_chars=100)
        assert not is_probably_scanned(pendek, min_chars=3)


@pytest.mark.langchain
class TestPemecahanChunk:
    def test_nomor_halaman_terbawa_ke_setiap_chunk(self):
        """FE-2 menunjuk ke halaman tertentu; kalau nomor halaman hilang di
        sini, sitasi tidak akan pernah bisa diverifikasi."""
        halaman = [LoadedPage(1, TEKS_PENUH), LoadedPage(2, TEKS_PENUH)]
        chunks = split_pages(halaman, chunk_size=100, chunk_overlap=10)
        assert {c.halaman for c in chunks} == {1, 2}

    def test_satu_chunk_tidak_pernah_mencakup_dua_halaman(self):
        halaman = [LoadedPage(1, "Halaman satu."), LoadedPage(2, "Halaman dua.")]
        chunks = split_pages(halaman, chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            asal = "satu" if chunk.halaman == 1 else "dua"
            assert asal in chunk.konten

    def test_urutan_berjalan_menaik_tanpa_bolong(self):
        halaman = [LoadedPage(1, TEKS_PENUH), LoadedPage(2, TEKS_PENUH)]
        chunks = split_pages(halaman, chunk_size=100, chunk_overlap=10)
        assert [c.urutan for c in chunks] == list(range(len(chunks)))

    def test_chunk_kosong_dibuang(self):
        halaman = [LoadedPage(1, "Isi.\n\n\n\n"), LoadedPage(2, "   ")]
        chunks = split_pages(halaman, chunk_size=50, chunk_overlap=5)
        assert all(c.konten.strip() for c in chunks)

    def test_metadata_tambahan_ikut_disalin(self):
        halaman = [LoadedPage(3, TEKS_PENUH)]
        chunks = split_pages(
            halaman,
            chunk_size=100,
            chunk_overlap=10,
            metadata={"unit": "Biro Akademik", "tahun_berlaku": 2025},
        )
        assert chunks[0].metadata["unit"] == "Biro Akademik"
        assert chunks[0].metadata["tahun_berlaku"] == 2025
        assert chunks[0].metadata["halaman"] == 3

    def test_overlap_lebih_besar_dari_chunk_ditolak(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            split_pages([LoadedPage(1, TEKS_PENUH)], chunk_size=100, chunk_overlap=200)

    def test_halaman_kosong_tidak_menghasilkan_chunk(self):
        assert split_pages([LoadedPage(1, "")], chunk_size=100, chunk_overlap=10) == []

    def test_dokumen_tanpa_halaman(self):
        assert split_pages([], chunk_size=100, chunk_overlap=10) == []
