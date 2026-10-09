"""FR-1 -- ingestion dokumen: deteksi PDF scan dan pemecahan chunk."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ingestion.chunker import LANJUTAN, split_pages, split_qa
from app.ingestion.loader import (
    LoadedLine,
    LoadedPage,
    empty_page_ratio,
    is_probably_scanned,
    load_pdf,
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

    @staticmethod
    def _judul_chunk(chunks) -> list[str]:
        return [c.konten.splitlines()[0] for c in chunks]

    def test_nama_bab_ikut_pada_setiap_sub_bagian(self):
        """T9: enam bab pedoman beasiswa memakai sub-judul yang sama (Gambaran
        Umum, Kuota, ...). Tanpa nama BAB, potongannya tidak menyebut beasiswa
        mana yang dibahas."""
        halaman = self._halaman(
            1,
            ("BAB II", "judul"),
            ("BEASISWA KIP KULIAH", "judul"),
            ("2.1. Gambaran Umum", "judul"),
            ("KIP Kuliah adalah bantuan pemerintah.", "teks"),
            ("2.5. Kuota", "judul"),
            ("Kuota nasional.", "teks"),
            ("BAB III", "judul"),
            ("BEASISWA ADIK DIFABEL", "judul"),
            ("3.5 Kuota", "judul"),
            ("Kuota 10 orang.", "teks"),
        )
        chunks = split_pages([halaman], chunk_size=500, chunk_overlap=50)
        assert self._judul_chunk(chunks) == [
            "BAB II BEASISWA KIP KULIAH › 2.1. Gambaran Umum",
            "BAB II BEASISWA KIP KULIAH › 2.5. Kuota",
            "BAB III BEASISWA ADIK DIFABEL › 3.5 Kuota",
        ]

    def test_nomor_dan_judul_di_baris_terpisah_disatukan(self):
        halaman = self._halaman(
            1,
            ("2.10", "judul"),
            ("Mekanisme Pendaftaran", "judul"),
            ("Daftar lewat portal.", "teks"),
            ("Pasal 24", "judul"),
            ("HAK DAN KEWAJIBAN ORGANISASI", "judul"),
            ("KEMAHASISWAAN", "judul"),
            ("Organisasi berhak ...", "teks"),
        )
        chunks = split_pages([halaman], chunk_size=500, chunk_overlap=50)
        assert self._judul_chunk(chunks) == [
            "2.10 Mekanisme Pendaftaran",
            "Pasal 24 HAK DAN KEWAJIBAN ORGANISASI KEMAHASISWAAN",
        ]

    def test_jejak_bab_menyeberang_halaman(self):
        """Judul BAB di dasar halaman, judul bab dan sub-bagian di halaman berikut."""
        halaman = [
            self._halaman(1, ("Penutup bab sebelumnya.", "teks"), ("BAB II", "judul")),
            self._halaman(
                2,
                ("BEASISWA KIP KULIAH", "judul"),
                ("2.1 Gambaran Umum", "judul"),
                ("Isi.", "teks"),
            ),
            self._halaman(3, ("Lanjutan isi.", "teks")),
        ]
        chunks = split_pages(halaman, chunk_size=500, chunk_overlap=50)
        assert self._judul_chunk(chunks)[1:] == [
            "BAB II BEASISWA KIP KULIAH › 2.1 Gambaran Umum",
            "BAB II BEASISWA KIP KULIAH › 2.1 Gambaran Umum",
        ]
        assert [c.halaman for c in chunks[1:]] == [2, 3]

    def test_bagian_dan_pasal_bertingkat(self):
        halaman = self._halaman(
            1,
            ("BAB III", "judul"),
            ("Bagian Pertama", "judul"),
            ("RUANG LINGKUP", "judul"),
            ("Pasal 3", "judul"),
            ("Isi pasal 3.", "teks"),
            ("Bagian Kedua", "judul"),
            ("KODE ETIK DENGAN DOSEN", "judul"),
            ("Pasal 4", "judul"),
            ("Isi pasal 4.", "teks"),
            ("Pasal 5", "judul"),
            ("Isi pasal 5.", "teks"),
            ("BAB IV", "judul"),
            ("LARANGAN", "judul"),
            ("Pasal 10", "judul"),
            ("Mahasiswa dilarang ...", "teks"),
        )
        chunks = split_pages([halaman], chunk_size=500, chunk_overlap=50)
        assert self._judul_chunk(chunks) == [
            "BAB III › Bagian Pertama RUANG LINGKUP › Pasal 3",
            "BAB III › Bagian Kedua KODE ETIK DENGAN DOSEN › Pasal 4",
            "BAB III › Bagian Kedua KODE ETIK DENGAN DOSEN › Pasal 5",
            "BAB IV LARANGAN › Pasal 10",
        ]

    def test_nomor_saudara_menggantikan_dan_anak_mengikuti_awalannya(self):
        halaman = self._halaman(
            1,
            ("2. Uraian Pedoman", "judul"),
            ("2.1 Peserta", "judul"),
            ("Isi 2.1.", "teks"),
            ("2.2 Asesor", "judul"),
            ("2.2.1 Asesor Lisensi", "judul"),
            ("Isi 2.2.1.", "teks"),
            ("2.3 Instruktur", "judul"),
            ("Isi 2.3.", "teks"),
            ("3.7 Cakupan", "judul"),
            ("1. Biaya Pendidikan:", "judul"),
            ("Isi 1.", "teks"),
            ("2. Biaya Hidup:", "judul"),
            ("Isi 2.", "teks"),
            ("3.8 Persyaratan", "judul"),
            ("Isi 3.8.", "teks"),
        )
        chunks = split_pages([halaman], chunk_size=500, chunk_overlap=50)
        assert self._judul_chunk(chunks) == [
            "2. Uraian Pedoman › 2.1 Peserta",
            "2. Uraian Pedoman › 2.2 Asesor › 2.2.1 Asesor Lisensi",
            "2. Uraian Pedoman › 2.3 Instruktur",
            "3.7 Cakupan › 1. Biaya Pendidikan:",
            "3.7 Cakupan › 2. Biaya Hidup:",
            "3.8 Persyaratan",
        ]

    def test_judul_tanpa_nomor_tetap_di_bawah_bab(self):
        halaman = self._halaman(
            1,
            ("BAB IV", "judul"),
            ("BEASISWA SKSS", "judul"),
            ("4.1", "judul"),
            ("Profil Program", "judul"),
            ("Isi.", "teks"),
            ("Catatan Khusus", "judul"),
            ("Isi catatan.", "teks"),
            ("Catatan Lain", "judul"),
            ("Isi catatan lain.", "teks"),
        )
        chunks = split_pages([halaman], chunk_size=500, chunk_overlap=50)
        assert self._judul_chunk(chunks) == [
            "BAB IV BEASISWA SKSS › 4.1 Profil Program",
            "BAB IV BEASISWA SKSS › 4.1 Profil Program › Catatan Khusus",
            "BAB IV BEASISWA SKSS › 4.1 Profil Program › Catatan Lain",
        ]

    def test_judul_kapital_tanpa_nomor_mengakhiri_bab(self):
        """Blok tanda tangan dan buku panduan yang disatukan sesudah SK bukan
        bagian dari BAB terakhir SK itu."""
        halaman = self._halaman(
            1,
            ("BAB VIII", "judul"),
            ("KETENTUAN PENUTUP", "judul"),
            ("Pasal 14", "judul"),
            ("Berlaku sejak ditetapkan.", "teks"),
            ("SATUAN KREDIT PARTISIPASI", "judul"),
            ("A. PENGERTIAN", "judul"),
            ("SKP adalah ...", "teks"),
            ("B. TUJUAN", "judul"),
            ("Tujuannya ...", "teks"),
        )
        chunks = split_pages([halaman], chunk_size=500, chunk_overlap=50)
        assert self._judul_chunk(chunks) == [
            "BAB VIII KETENTUAN PENUTUP › Pasal 14",
            "SATUAN KREDIT PARTISIPASI › A. PENGERTIAN",
            "SATUAN KREDIT PARTISIPASI › B. TUJUAN",
        ]

    def test_bab_di_tengah_runtun_tidak_menjadi_anak_judul_sebelumnya(self):
        halaman = self._halaman(
            1,
            ("PROGRAM STUDI SISTEM KOMPUTER", "judul"),
            ("B. MISI", "judul"),
            ("Misi prodi.", "teks"),
            ("Lampiran Surat Keputusan Rektor", "judul"),
            ("BAB I", "judul"),
            ("KETENTUAN UMUM", "judul"),
            ("Pasal 1", "judul"),
            ("Pengertian.", "teks"),
        )
        chunks = split_pages([halaman], chunk_size=500, chunk_overlap=50)
        assert self._judul_chunk(chunks)[-1] == "BAB I KETENTUAN UMUM › Pasal 1"

    def test_tahun_di_awal_judul_bukan_nomor_bagian(self):
        halaman = self._halaman(
            1,
            ("1. Ketentuan", "judul"),
            ("Isi.", "teks"),
            ("2026 Jadwal Baru", "judul"),
            ("Isi jadwal.", "teks"),
        )
        chunks = split_pages([halaman], chunk_size=500, chunk_overlap=50)
        assert self._judul_chunk(chunks)[-1] == "1. Ketentuan › 2026 Jadwal Baru"

    def test_halaman_tanpa_struktur_tetap_terpecah(self):
        """`LoadedPage` tanpa `baris` -- mundur ke pemecahan berbasis teks."""
        chunks = split_pages([LoadedPage(1, TEKS_PENUH)], chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1
        assert all(c.halaman == 1 for c in chunks)


class _Tabel:
    """Tiruan `pymupdf.table.Table`: sel (i, j) berkotak 10x10 di kolom j, baris i.

    `rentang` memberi sel (i, j) tinggi beberapa baris (di-merge ke bawah);
    posisi yang ditutupinya ditulis `None`, seperti keluaran PyMuPDF. `geser`
    menggeser seluruh kolom, seperti tabel yang bersambung di halaman lain.
    """

    def __init__(self, baris, rentang=None, geser=0.0):
        self._baris = baris
        rentang = rentang or {}
        self.rows = [
            SimpleNamespace(
                cells=[
                    None
                    if nilai is None
                    else (
                        10 * j + geser,
                        10 * i,
                        10 * (j + 1) + geser,
                        10 * (i + rentang.get((i, j), 1)),
                    )
                    for j, nilai in enumerate(row)
                ]
            )
            for i, row in enumerate(baris)
        ]

    def extract(self):
        return self._baris


KEPALA_POIN = ["No", "Kegiatan", "Tingkat", "Jabatan", "Poin"]


class TestPerataanTabel:
    _Tabel = _Tabel

    def test_sel_merge_yang_terduplikasi_diruntuhkan(self):
        """PyMuPDF mengulang nilai sel yang di-merge ke tiap kolom yang dilewatinya."""
        tabel = self._Tabel(
            [
                ["", "No.", "", "Pertanyaan", "", "Jawaban", ""],
                ["1", "1", "1", "Kapan KRS dibuka?", "Kapan KRS dibuka?", "", ""],
            ]
        )
        assert tabel_ke_baris(tabel).baris == ["No.: 1 | Pertanyaan: Kapan KRS dibuka?"]

    def test_tanpa_header_baris_tetap_terbaca(self):
        tabel = self._Tabel([["Senin", "08.00"], ["Selasa", "09.00"]])
        assert tabel_ke_baris(tabel) == (["Senin | 08.00", "Selasa | 09.00"], None)

    def test_tabel_kosong_tidak_menghasilkan_baris(self):
        assert tabel_ke_baris(self._Tabel([["", ""], ["", ""]])) == ([], None)

    def test_sel_kosong_di_tengah_tidak_menggeser_kolom(self):
        """T52: dulu "25" jatuh ke kolom Jabatan karena sel kosong dibuang lalu
        sisanya diratakan dari kiri."""
        tabel = self._Tabel([KEPALA_POIN, ["3", "Pendukung", "Internasional", "", "25"]])
        assert tabel_ke_baris(tabel).baris == [
            "No: 3 | Kegiatan: Pendukung | Tingkat: Internasional | Poin: 25"
        ]

    def test_sel_merge_ke_bawah_diwariskan(self):
        """T52: tabel poin Buku SKP -- kegiatan dan tingkat di-merge ke bawah."""
        tabel = self._Tabel(
            [
                KEPALA_POIN,
                ["1", "Pengurus Organisasi", "Nasional", "Pengurus Inti", "40"],
                [None, None, None, "Anggota", "20"],
                [None, None, "Regional", "Pengurus Inti", "30"],
                [None, None, None, "Anggota", "10"],
            ],
            rentang={(1, 0): 4, (1, 1): 4, (1, 2): 2, (3, 2): 2},
        )
        assert tabel_ke_baris(tabel).baris == [
            "No: 1 | Kegiatan: Pengurus Organisasi | Tingkat: Nasional"
            " | Jabatan: Pengurus Inti | Poin: 40",
            "No: 1 | Kegiatan: Pengurus Organisasi | Tingkat: Nasional"
            " | Jabatan: Anggota | Poin: 20",
            "No: 1 | Kegiatan: Pengurus Organisasi | Tingkat: Regional"
            " | Jabatan: Pengurus Inti | Poin: 30",
            "No: 1 | Kegiatan: Pengurus Organisasi | Tingkat: Regional"
            " | Jabatan: Anggota | Poin: 10",
        ]

    def test_sel_kosong_sesudah_sel_merge_tidak_mewarisi(self):
        """Sel merge hanya diwariskan ke baris yang benar-benar dijangkaunya."""
        tabel = self._Tabel(
            [
                KEPALA_POIN,
                ["1", "Organisasi", "Nasional", "Ketua", "40"],
                [None, None, None, "Anggota", "20"],
                ["2", "Seminar", "", "", "10"],
            ],
            rentang={(1, 0): 2, (1, 1): 2, (1, 2): 2},
        )
        assert tabel_ke_baris(tabel).baris[-1] == "No: 2 | Kegiatan: Seminar | Poin: 10"

    def test_header_subtabel_menggantikan_header_awal(self):
        """T37: HARGA SERTIFIKASI menumpuk tiga subtabel dalam satu tabel. Grid
        bawaan PyMuPDF untuk tabel ini tidak sejajar dengan header-nya, jadi
        perataan dari kiri yang dipakai."""
        tabel = self._Tabel(
            [
                ["", "SERTIFIKASI DASAR", "", "HARGA", "", "PRODI"],
                ["IC3 GS6", "", "RP 1.050.000", "", "TI, RSK, BD", ""],
                ["", "SERTIFIKASI BIDANG", "", "HARGA", "", "PRODI"],
                ["META DIGITAL MARKETING", "", "RP 1.300.000", "", "BD", ""],
                ["", "SERTIFIKASI TOEIC", "", "Harga", "", "PRODI"],
                ["TOEIC ENGLISH", "", "RP 675.000", "", "TI, RSK, BD, DKV", ""],
            ]
        )
        assert tabel_ke_baris(tabel).baris == [
            "SERTIFIKASI DASAR: IC3 GS6 | HARGA: RP 1.050.000 | PRODI: TI, RSK, BD",
            "SERTIFIKASI BIDANG: META DIGITAL MARKETING | HARGA: RP 1.300.000 | PRODI: BD",
            "SERTIFIKASI TOEIC: TOEIC ENGLISH | Harga: RP 675.000 | PRODI: TI, RSK, BD, DKV",
        ]

    def test_header_terulang_persis_dilewati(self):
        tabel = self._Tabel(
            [["Hari", "Jam"], ["Senin", "08.00"], ["Hari", "Jam"], ["Selasa", "09.00"]]
        )
        assert tabel_ke_baris(tabel).baris == [
            "Hari: Senin | Jam: 08.00",
            "Hari: Selasa | Jam: 09.00",
        ]

    def test_data_tanpa_angka_bukan_header_subtabel(self):
        """Tampang header saja tidak cukup: harus mengulang nama kolom."""
        tabel = self._Tabel([["Nama", "Jabatan"], ["Budi", "Ketua"], ["Sari", "Sekretaris"]])
        assert tabel_ke_baris(tabel).baris == [
            "Nama: Budi | Jabatan: Ketua",
            "Nama: Sari | Jabatan: Sekretaris",
        ]


class TestTabelBersambung:
    """Tabel poin Buku SKP bersambung beberapa halaman. Sel yang di-merge ke
    bawah terpotong pergantian halaman dan tampil kosong di halaman baru."""

    def _ekor_halaman_1(self):
        tabel = _Tabel(
            [KEPALA_POIN, ["1", "Pengurus Organisasi", "Internasional", "Pengurus Inti", "50"]]
        )
        return tabel_ke_baris(tabel).ekor

    def test_header_diulang_kolom_awal_diisi_dari_halaman_sebelumnya(self):
        tabel = _Tabel(
            [
                KEPALA_POIN,
                ["", "", "", "Anggota", "30"],
                [None, None, "Nasional", "Pengurus Inti", "40"],
            ],
            rentang={(1, 0): 2, (1, 1): 2},
        )
        assert tabel_ke_baris(tabel, self._ekor_halaman_1()).baris == [
            "No: 1 | Kegiatan: Pengurus Organisasi | Tingkat: Internasional"
            " | Jabatan: Anggota | Poin: 30",
            "No: 1 | Kegiatan: Pengurus Organisasi | Tingkat: Nasional"
            " | Jabatan: Pengurus Inti | Poin: 40",
        ]

    def test_tanpa_header_memakai_header_halaman_sebelumnya(self):
        """Batas kolom Word bergeser 1-2 pt antarhalaman."""
        tabel = _Tabel(
            [
                ["", "", "Regional", "Anggota", "10"],
                ["2", "Panitia", "Nasional", "Anggota", "20"],
            ],
            geser=2,
        )
        rakitan = tabel_ke_baris(tabel, self._ekor_halaman_1())
        assert rakitan.baris == [
            "No: 1 | Kegiatan: Pengurus Organisasi | Tingkat: Regional"
            " | Jabatan: Anggota | Poin: 10",
            "No: 2 | Kegiatan: Panitia | Tingkat: Nasional | Jabatan: Anggota | Poin: 20",
        ]
        assert rakitan.ekor is not None
        assert rakitan.ekor.nilai == ("2", "Panitia", "Nasional", "Anggota", "20")

    def test_kolom_berbeda_bukan_sambungan(self):
        tabel = _Tabel([["", "Catatan", "lain"]], geser=40)
        assert tabel_ke_baris(tabel, self._ekor_halaman_1()) == (["Catatan | lain"], None)

    def test_header_lain_tidak_mewarisi(self):
        tabel = _Tabel([["Hari", "Jam"], ["", "08.00"]])
        assert tabel_ke_baris(tabel, self._ekor_halaman_1()).baris == ["Jam: 08.00"]


class TestTabelPdf:
    """Ujung ke ujung lewat `load_pdf` dengan PDF yang meniru tabel Word: tiap
    sel diberi kotak latar putih yang ikut terbaca sebagai garis oleh strategi
    bawaan PyMuPDF, sehingga sel yang di-merge ke bawah terpecah (T52)."""

    KOLOM = (50, 80, 200, 280, 320)
    PARAGRAF = "Tabel berikut memuat poin kegiatan kemahasiswaan yang diakui institut."

    def _tabel(self, page, baris, y=40):
        """`baris`: (isi sel, tinggi); `None` = tertutup sel di atasnya."""
        pymupdf = pytest.importorskip("pymupdf")
        y_awal = y
        for sel, tinggi in baris:
            for j, isi in enumerate(sel):
                if isi is None:
                    continue
                x0, x1 = self.KOLOM[j], self.KOLOM[j + 1]
                page.draw_line((x0, y), (x1, y))
                page.draw_rect(
                    pymupdf.Rect(x0 + 1, y + 1, x1 - 1, y + tinggi - 1),
                    color=None,
                    fill=(1, 1, 1),
                )
                if isi:
                    page.insert_text((x0 + 3, y + 13), isi, fontsize=8)
            y += tinggi
        page.draw_line((self.KOLOM[0], y), (self.KOLOM[-1], y))
        for x in self.KOLOM:
            page.draw_line((x, y_awal), (x, y))

    def test_sel_merge_dan_tabel_bersambung_tanpa_header(self, tmp_path):
        pymupdf = pytest.importorskip("pymupdf")
        dokumen = pymupdf.open()
        satu = dokumen.new_page()
        satu.insert_text((50, 30), self.PARAGRAF, fontsize=8)
        self._tabel(
            satu,
            [
                (["No", "Kegiatan", "Tingkat", "Poin"], 20),
                (["1", "Pengurus Organisasi", "Nasional", "40"], 20),
                ([None, None, "Regional", "30"], 20),
            ],
        )
        dua = dokumen.new_page()
        dua.insert_text((50, 30), self.PARAGRAF, fontsize=8)
        self._tabel(
            dua,
            [
                (["", "", "Himaprodi", "20"], 20),
                ([None, None, "UKM", "10"], 20),
                (["2", "Panitia", "Nasional", "25"], 20),
            ],
        )
        path = tmp_path / "skp.pdf"
        dokumen.save(path)

        tabel = [b.teks for p in load_pdf(path) for b in p.baris if b.jenis == "tabel"]
        assert tabel == [
            "No: 1 | Kegiatan: Pengurus Organisasi | Tingkat: Nasional | Poin: 40",
            "No: 1 | Kegiatan: Pengurus Organisasi | Tingkat: Regional | Poin: 30",
            "No: 1 | Kegiatan: Pengurus Organisasi | Tingkat: Himaprodi | Poin: 20",
            "No: 1 | Kegiatan: Pengurus Organisasi | Tingkat: UKM | Poin: 10",
            "No: 2 | Kegiatan: Panitia | Tingkat: Nasional | Poin: 25",
        ]


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
