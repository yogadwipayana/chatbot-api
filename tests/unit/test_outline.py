"""Daftar bab dari jejak judul potongan (T9, `app.rag.outline`)."""

from __future__ import annotations

from app.rag.outline import MAKS_BAB, Bab, susun_daftar_bab


def baris(chunk_id: str, halaman: int, kepala: str) -> dict:
    return {"chunk_id": chunk_id, "page": halaman, "kepala": kepala}


PEDOMAN = [
    baris("p0", 2, "BUKU PEDOMAN PROGRAM BEASISWA"),
    baris("p1", 11, "BAB II BEASISWA KIP KULIAH › 2.1. Gambaran Umum"),
    baris("p2", 12, "BAB II BEASISWA KIP KULIAH › 2.8 Persyaratan"),
    baris("p3", 16, "BAB III BEASISWA ADIK DIFABEL › 3.1 Gambaran Umum"),
    baris("p4", 27, "BAB VI BEASISWA MAHASISWA BERPRESTASI › 6.1 Gambaran Umum"),
    baris("p5", 27, "BAB VI BEASISWA MAHASISWA BERPRESTASI › 6.1 Gambaran Umum (lanjutan)"),
    baris("p6", 32, "BAB VII BEASISWA TALENTA INSTIKI › 7.1 Ketentuan Umum"),
]

TRANSKRIP = [
    baris("t0", 1, "KNOWLEDGE BASE CHATBOT › 1. Tanggung Jawab"),
    baris("t1", 2, "KNOWLEDGE BASE CHATBOT › 6. Beasiswa"),
    baris("t2", 2, "KNOWLEDGE BASE CHATBOT › 6. Beasiswa (lanjutan)"),
    baris("t3", 3, "KNOWLEDGE BASE CHATBOT › 9. Proses Pengajuan SKP"),
    baris("t4", 5, "KNOWLEDGE BASE CHATBOT › 16. Kalender › Sumber Knowledge"),
]


class TestSusunan:
    def test_bab_tingkat_teratas_menurut_urutan_kemunculan(self):
        daftar = susun_daftar_bab(PEDOMAN)
        assert daftar is not None
        assert [b.judul for b in daftar.bab] == [
            "BUKU PEDOMAN PROGRAM BEASISWA",
            "BAB II BEASISWA KIP KULIAH",
            "BAB III BEASISWA ADIK DIFABEL",
            "BAB VI BEASISWA MAHASISWA BERPRESTASI",
            "BAB VII BEASISWA TALENTA INSTIKI",
        ]

    def test_halaman_bab_adalah_halaman_potongan_pertamanya(self):
        daftar = susun_daftar_bab(PEDOMAN)
        assert daftar.bab[1] == Bab("BAB II BEASISWA KIP KULIAH", 11)

    def test_penanda_lanjutan_tidak_menjadi_bab_sendiri(self):
        rows = [
            baris("a", 1, "BAB I UMUM"),
            baris("b", 2, "BAB I UMUM (lanjutan)"),
            baris("c", 3, "BAB II KHUSUS"),
            baris("d", 4, "BAB III PENUTUP"),
        ]
        assert [b.judul for b in susun_daftar_bab(rows).bab] == [
            "BAB I UMUM",
            "BAB II KHUSUS",
            "BAB III PENUTUP",
        ]

    def test_judul_yang_memayungi_separuh_dokumen_diganti_anaknya(self):
        """TRANSKRIP memayungi semua bagian dengan satu judul; daftar berisi
        judul itu saja tidak menggambarkan apa pun."""
        daftar = susun_daftar_bab(TRANSKRIP)
        assert [b.judul for b in daftar.bab] == [
            "1. Tanggung Jawab",
            "6. Beasiswa",
            "9. Proses Pengajuan SKP",
            "16. Kalender",
        ]
        assert daftar.bab[1].halaman == 2

    def test_bab_kecil_di_samping_payung_tetap_ada(self):
        """Pedoman sertifikasi: halaman depan sendiri-sendiri, bab 1-7 di bawah
        satu judul."""
        rows = [
            baris("k", 7, "KATA PENGANTAR"),
            baris("a", 8, "PEDOMAN › 1. Ruang Lingkup"),
            baris("b", 8, "PEDOMAN › 2. Uraian"),
            baris("c", 10, "PEDOMAN › 3. Publikasi"),
            baris("d", 10, "PEDOMAN › 3. Publikasi (lanjutan)"),
        ]
        assert [b.judul for b in susun_daftar_bab(rows).bab] == [
            "KATA PENGANTAR",
            "1. Ruang Lingkup",
            "2. Uraian",
            "3. Publikasi",
        ]


class TestBabTersentuh:
    def test_potongan_dipetakan_ke_babnya(self):
        daftar = susun_daftar_bab(PEDOMAN)
        assert daftar.tersentuh(["p2", "p5", "p4"]) == {1, 3}

    def test_potongan_anak_dipetakan_ke_anaknya(self):
        daftar = susun_daftar_bab(TRANSKRIP)
        assert daftar.tersentuh(["t1", "t2"]) == {1}
        assert daftar.tersentuh(["t4"]) == {3}

    def test_potongan_tak_dikenal_diabaikan(self):
        assert susun_daftar_bab(PEDOMAN).tersentuh(["x"]) == set()


class TestBukanDaftarBab:
    def test_kosong(self):
        assert susun_daftar_bab([]) is None

    def test_kurang_dari_tiga_bab(self):
        rows = [
            baris("a", 1, "BAB I"),
            baris("b", 2, "BAB I (lanjutan)"),
            baris("c", 3, "BAB II"),
        ]
        assert susun_daftar_bab(rows) is None

    def test_setiap_potongan_babnya_sendiri(self):
        """Dokumen tanpa judul bagian: baris pertama potongannya kalimat isi
        (Panduan KRS MBKM)."""
        rows = [baris(str(i), i, f"{i}. Klik menu ke-{i}") for i in range(1, 7)]
        assert susun_daftar_bab(rows) is None

    def test_terlalu_banyak_bab(self):
        rows = [
            baris(f"{i}{s}", i, f"BAGIAN {i}{' (lanjutan)' if s else ''}")
            for i in range(MAKS_BAB + 1)
            for s in (0, 1)
        ]
        assert susun_daftar_bab(rows) is None


class TestTeks:
    def test_memuat_judul_dokumen_dan_halaman_setiap_bab(self):
        teks = susun_daftar_bab(PEDOMAN).teks("Pedoman Beasiswa")
        baris_teks = teks.splitlines()
        assert baris_teks[0].startswith('Daftar bab dokumen "Pedoman Beasiswa"')
        assert "- BAB VI BEASISWA MAHASISWA BERPRESTASI (hal. 27)" in baris_teks
        assert len(baris_teks) == 1 + 5

    def test_bukan_penanda_sitasi(self):
        """Satu penanda `[Judul, hal. N]` akan disalin LLM untuk semua bab."""
        teks = susun_daftar_bab(PEDOMAN).teks("Pedoman Beasiswa")
        assert "[" not in teks


class TestKonteksDanSitasi:
    """Daftar bab di prompt (aturan 9) dan kartu sumbernya (FE-2)."""

    @staticmethod
    def dokumen_daftar_bab():
        from tests.fixtures.fakes import make_document

        daftar = susun_daftar_bab(PEDOMAN)
        dok = make_document(
            "x", judul="Pedoman Beasiswa", halaman=2, konten=daftar.teks("Pedoman Beasiswa")
        )
        del dok.metadata["chunk_id"]
        dok.metadata["daftar_bab"] = [
            {"judul": b.judul, "halaman": b.halaman} for b in daftar.bab
        ]
        return dok

    def test_konteks_tanpa_penanda_halaman(self):
        from app.rag.prompts import format_context

        dok = self.dokumen_daftar_bab()
        assert format_context([dok]) == dok.page_content

    def test_aturan_9_menyebut_kepala_yang_benar_benar_dipakai(self):
        from app.rag.prompts import SYSTEM_PROMPT, TOOL_SYSTEM_PROMPT

        kepala = self.dokumen_daftar_bab().page_content.split(" ", 3)[:3]
        assert '"' + " ".join(kepala) + '"' in SYSTEM_PROMPT
        assert "9. Blok" in TOOL_SYSTEM_PROMPT

    def test_bab_yang_dikutip_menjadi_kartu_di_halamannya(self):
        from app.rag.chain import OutcomeKind, PipelineOutcome
        from app.routers.chat import citations_for

        outcome = PipelineOutcome(
            kind=OutcomeKind.ANSWER,
            text="Beasiswa Berprestasi [Pedoman Beasiswa, hal. 27] dan Talenta "
            "[Pedoman Beasiswa, hal. 32].",
            documents=(self.dokumen_daftar_bab(),),
        )
        kartu = citations_for(outcome)
        assert [(k.title, k.page) for k in kartu] == [
            ("Pedoman Beasiswa", 27),
            ("Pedoman Beasiswa", 32),
        ]
        assert {k.document_id for k in kartu} == {"d1"}

    def test_halaman_di_luar_daftar_tetap_tanpa_kartu(self):
        from app.rag.chain import OutcomeKind, PipelineOutcome
        from app.routers.chat import citations_for

        outcome = PipelineOutcome(
            kind=OutcomeKind.ANSWER,
            text="Rinciannya [Pedoman Beasiswa, hal. 29].",
            documents=(self.dokumen_daftar_bab(),),
        )
        assert citations_for(outcome) == []
