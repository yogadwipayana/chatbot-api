"""FR-1 -- ingestion dokumen: deteksi PDF scan dan pemecahan chunk."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import LANJUTAN, split_pages, split_qa
from app.ingestion.loader import (
    LoadedLine,
    LoadedPage,
    empty_page_ratio,
    is_probably_scanned,
    peringatan_kepadatan,
    tabel_ke_baris,
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


class TestPemecahanTanyaJawab:
    PERTANYAAN = "Bagaimana cara mengurus KTM yang hilang?"

    def test_pertanyaan_dan_jawaban_menjadi_satu_chunk(self):
        chunks = split_qa(self.PERTANYAAN, "Bawa surat kehilangan ke loket 3.")
        assert len(chunks) == 1
        assert self.PERTANYAAN in chunks[0].konten
        assert "Bawa surat kehilangan ke loket 3." in chunks[0].konten
        assert (chunks[0].halaman, chunks[0].urutan) == (1, 0)

    def test_jawaban_panjang_dipecah_dan_setiap_potongan_membawa_pertanyaan(self):
        """Tanpa pertanyaan di setiap potongan, potongan kedua tidak akan pernah
        cocok dengan pertanyaan mahasiswa mana pun."""
        jawaban = " ".join(f"Langkah {i} dijelaskan panjang lebar." for i in range(80))
        chunks = split_qa(self.PERTANYAAN, jawaban, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        assert all(self.PERTANYAAN in c.konten for c in chunks)
        assert [c.urutan for c in chunks] == list(range(len(chunks)))
        assert all(c.halaman == 1 for c in chunks)

    def test_spasi_berlebih_pada_pertanyaan_dirapikan(self):
        chunks = split_qa("  Kapan   wisuda?  ", "Bulan Oktober.")
        assert chunks[0].konten.startswith("Pertanyaan: Kapan wisuda?")

    def test_metadata_ikut_disalin(self):
        chunks = split_qa(self.PERTANYAAN, "Bawa surat kehilangan.", metadata={"unit": "BAAK"})
        assert chunks[0].metadata == {"unit": "BAAK", "halaman": 1}

    @pytest.mark.parametrize(
        "pertanyaan,jawaban", [("   ", "Jawaban ada."), ("Ada pertanyaan?", "  ")]
    )
    def test_isi_kosong_ditolak(self, pertanyaan, jawaban):
        with pytest.raises(ValueError):
            split_qa(pertanyaan, jawaban)


@pytest.mark.langchain
class TestPemecahanSadarStruktur:
    """FR-1 -- batas chunk mengikuti judul bagian, bukan hitungan karakter.

    Memecah per ukuran tetap menghasilkan potongan yatim: langkah "1. Ketik
    alamat https://ibank..." tanpa kata "iBank Personal" di mana pun. Isinya
    benar, tetapi tidak mengandung satu pun kata yang akan diketik mahasiswa.
    """

    @staticmethod
    def _halaman(nomor: int, *baris: tuple[str, str]) -> LoadedPage:
        isi = tuple(LoadedLine(teks, jenis) for teks, jenis in baris)
        return LoadedPage(nomor, "\n".join(b.teks for b in isi), isi)

    def test_judul_bagian_ikut_pada_chunk(self):
        halaman = self._halaman(
            1,
            ("ATM BNI", "judul"),
            ("1. Masukkan kartu.", "teks"),
            ("2. Pilih Transfer.", "teks"),
        )
        chunks = split_pages([halaman], chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0].konten.startswith("ATM BNI\n")

    def test_tiap_bagian_menjadi_chunk_sendiri(self):
        halaman = self._halaman(
            1,
            ("ATM BNI", "judul"),
            ("1. Masukkan kartu.", "teks"),
            ("Mobile Banking", "judul"),
            ("1. Buka aplikasi.", "teks"),
        )
        chunks = split_pages([halaman], chunk_size=500, chunk_overlap=50)
        assert [c.konten.splitlines()[0] for c in chunks] == ["ATM BNI", "Mobile Banking"]

    def test_bagian_kepanjangan_dipecah_dengan_judul_diulang(self):
        panjang = [(f"{i}. Langkah yang dijelaskan panjang lebar.", "teks") for i in range(20)]
        chunks = split_pages(
            [self._halaman(1, ("iBank Personal", "judul"), *panjang)],
            chunk_size=300,
            chunk_overlap=30,
        )
        assert len(chunks) > 1
        assert all(c.konten.startswith("iBank Personal") for c in chunks)
        assert all(LANJUTAN in c.konten.splitlines()[0] for c in chunks[1:])
        assert LANJUTAN not in chunks[0].konten.splitlines()[0]

    def test_judul_diwariskan_ke_halaman_berikutnya(self):
        """Bagian yang menyeberang pergantian halaman tetap punya identitas.

        Chunk-nya tetap terpisah demi sitasi FE-2, tetapi tanpa pewarisan ini
        potongan pertama tiap halaman akan kehilangan judulnya."""
        halaman = [
            self._halaman(1, ("Transfer dari Bank Lain", "judul"), ("1. Pilih menu.", "teks")),
            self._halaman(2, ("2. Masukkan kode bank.", "teks")),
        ]
        chunks = split_pages(halaman, chunk_size=500, chunk_overlap=50)
        assert chunks[1].halaman == 2
        assert chunks[1].konten.startswith("Transfer dari Bank Lain")

    def test_baris_tabel_tidak_pernah_terpotong(self):
        """Baris tabel yang terbelah memisahkan nilai dari nama kolomnya,
        menyisakan deret angka tanpa arti."""
        baris = [
            (f"No.: {i} | Pertanyaan: Pertanyaan nomor {i} apa?", "tabel") for i in range(12)
        ]
        chunks = split_pages(
            [self._halaman(1, ("BAAK", "judul"), *baris)], chunk_size=200, chunk_overlap=20
        )
        utuh = [b[0] for b in baris]
        tergabung = "\n".join(c.konten for c in chunks)
        assert all(u in tergabung for u in utuh)

    def test_halaman_tanpa_struktur_tetap_terpecah(self):
        """`LoadedPage` tanpa `baris` -- mundur ke pemecahan berbasis teks."""
        chunks = split_pages([LoadedPage(1, TEKS_PENUH)], chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1
        assert all(c.halaman == 1 for c in chunks)


class TestPerataanTabel:
    class _Tabel:
        def __init__(self, baris):
            self._baris = baris

        def extract(self):
            return self._baris

    def test_sel_merge_yang_terduplikasi_diruntuhkan(self):
        """PyMuPDF mengulang nilai sel yang di-merge ke tiap kolom yang dilewatinya."""
        tabel = self._Tabel(
            [
                ["", "No.", "", "Pertanyaan", "", "Jawaban", ""],
                ["1", "1", "1", "Kapan KRS dibuka?", "Kapan KRS dibuka?", "", ""],
            ]
        )
        assert tabel_ke_baris(tabel) == ["No.: 1 | Pertanyaan: Kapan KRS dibuka?"]

    def test_tanpa_header_baris_tetap_terbaca(self):
        tabel = self._Tabel([["Senin", "08.00"], ["Selasa", "09.00"]])
        assert tabel_ke_baris(tabel) == ["Senin | 08.00", "Selasa | 09.00"]

    def test_tabel_kosong_tidak_menghasilkan_baris(self):
        assert tabel_ke_baris(self._Tabel([["", ""], ["", ""]])) == []


class TestPeringatanKepadatan:
    """Panduan berbasis tangkapan layar lolos deteksi scan -- tiap halaman punya
    satu-dua kalimat keterangan -- padahal langkahnya ada di dalam gambar."""

    PADAT = "Panduan akademik yang isinya benar-benar padat teks. " * 12

    def test_halaman_padat_tidak_diperingatkan(self):
        assert peringatan_kepadatan([self.PADAT] * 5) is None

    def test_halaman_tipis_diperingatkan(self):
        tipis = ["1. Login pada sads.instiki.ac.id, klik menu KRS."] * 6
        pesan = peringatan_kepadatan(tipis)
        assert pesan is not None
        assert "gambar" in pesan

    def test_sedikit_halaman_tipis_masih_diterima(self):
        assert peringatan_kepadatan([self.PADAT] * 5 + ["Sampul"]) is None

    def test_dokumen_tanpa_halaman(self):
        assert peringatan_kepadatan([]) is None
