"""Validasi skema dan pembantu dashboard admin yang tidak butuh database."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from app.admin.documents import stale_clause
from app.admin.stats import ratio, rincian_profil
from app.prodi import DAFTAR_PRODI
from app.rag.filters import active_document_clause
from app.routers.admin_documents import nama_berkas_aman
from app.schemas.admin import (
    AdminUserCreate,
    AdminUserUpdate,
    DocumentUpdate,
    KillSwitchRequest,
)


class TestUnitAkun:
    def test_spasi_berlebih_dirapikan_huruf_besar_dibiarkan(self):
        akun = AdminUserCreate(email="a@instiki.ac.id", role="staf", unit="  Biro   Keuangan ")
        assert akun.unit == "Biro Keuangan"

    @pytest.mark.parametrize("unit", ["", "   "])
    def test_unit_kosong_menjadi_none(self, unit):
        assert AdminUserUpdate(unit=unit).unit is None
        assert "unit" in AdminUserUpdate(unit=unit).model_fields_set

    def test_level_tidak_boleh_null(self):
        with pytest.raises(ValidationError, match="tidak boleh kosong"):
            AdminUserUpdate(role=None)


class TestKillSwitchRequest:
    @pytest.mark.parametrize("alasan", [None, "", "   "])
    def test_menyalakan_tanpa_alasan_ditolak(self, alasan):
        with pytest.raises(ValidationError, match="alasan"):
            KillSwitchRequest(engaged=True, reason=alasan)

    def test_mematikan_tanpa_alasan_boleh(self):
        assert KillSwitchRequest(engaged=False).engaged is False


class TestDocumentUpdate:
    @pytest.mark.parametrize("nama", ["title", "unit", "is_active"])
    def test_field_wajib_tidak_boleh_null(self, nama):
        with pytest.raises(ValidationError, match="tidak boleh kosong"):
            DocumentUpdate(**{nama: None})

    def test_masa_berlaku_boleh_dikosongkan(self):
        """null = berlaku tanpa batas; harus dapat dibedakan dari 'tidak diubah'."""
        perubahan = DocumentUpdate(valid_until=None).model_dump(exclude_unset=True)
        assert perubahan == {"valid_until": None}

    def test_field_tak_dikenal_ditolak(self):
        """Salah ketik `is_aktif` tidak boleh diam-diam diabaikan."""
        with pytest.raises(ValidationError):
            DocumentUpdate(is_aktif=False)

    def test_spasi_dirapikan_sebelum_panjang_diperiksa(self):
        assert DocumentUpdate(title="  Panduan 2026  ").title == "Panduan 2026"
        with pytest.raises(ValidationError):
            DocumentUpdate(title="  ab  ")

    def test_tanggal_diurai(self):
        assert DocumentUpdate(valid_until="2027-01-31").valid_until == date(2027, 1, 31)


class TestPredikatUsang:
    def test_kedaluwarsa_berlawanan_persis_dengan_filter_retrieval(self):
        """Badge 'kedaluwarsa' AD-2 harus tepat dokumen yang berhenti terambil FR-2."""
        assert "d.valid_until > now()" in active_document_clause("d")
        assert "d.valid_until <= now()" in stale_clause("d")

    def test_alias_disuntik_ditolak(self):
        with pytest.raises(ValueError):
            stale_clause("d; DROP TABLE documents")


class TestRasio:
    def test_tanpa_penyebut_none(self):
        assert ratio(0, 0) is None

    def test_nilai_biasa(self):
        assert ratio(3, 4) == pytest.approx(0.75)

    def test_dibatasi_satu(self):
        assert ratio(5, 4) == 1.0


def _baris(code, tahun, jumlah, ditolak=0):
    return {
        "code": code,
        "intake_year": tahun,
        "question_count": jumlah,
        "refusal_count": ditolak,
    }


class TestRincianProfil:
    def test_dirangkum_per_prodi_dan_per_angkatan(self):
        hasil = rincian_profil(
            [
                _baris("1010", 2024, 5, 1),
                _baris("1010", 2025, 3),
                _baris("2010", 2024, 2, 2),
            ]
        )
        assert hasil["questions_with_profile"] == 10
        informatika = hasil["program_breakdown"][0]
        assert informatika == {
            "code": "1010",
            "name": "Informatika",
            "level": "S1",
            "question_count": 8,
            "refusal_count": 1,
        }
        assert hasil["intake_year_breakdown"] == [
            {"intake_year": 2025, "question_count": 3, "refusal_count": 0},
            {"intake_year": 2024, "question_count": 7, "refusal_count": 3},
        ]

    def test_prodi_yang_belum_bertanya_tetap_muncul_sebagai_nol(self):
        """Prodi yang belum pernah bertanya adalah temuan, bukan baris yang hilang."""
        hasil = rincian_profil([_baris("2020", 2024, 4)])
        kode = [p["code"] for p in hasil["program_breakdown"]]
        assert kode[0] == "2020"
        assert sorted(kode) == sorted(p.code for p in DAFTAR_PRODI)
        nol = [p for p in hasil["program_breakdown"] if p["code"] != "2020"]
        assert all(p["question_count"] == 0 for p in nol)

    def test_seri_nol_mengikuti_urutan_daftar_prodi(self):
        hasil = rincian_profil([])
        assert [p["code"] for p in hasil["program_breakdown"]] == [
            p.code for p in DAFTAR_PRODI
        ]
        assert hasil["questions_with_profile"] == 0
        assert hasil["intake_year_breakdown"] == []

    def test_kode_yang_sudah_tidak_terdaftar_tetap_terhitung(self):
        """Tanpa ini jumlah per prodi tidak lagi sama dengan
        `questions_with_profile` setelah satu prodi dihapus dari daftar."""
        hasil = rincian_profil([_baris("9901", 2020, 2)])
        lama = next(p for p in hasil["program_breakdown"] if p["code"] == "9901")
        assert (lama["name"], lama["level"]) == ("9901", None)
        assert sum(p["question_count"] for p in hasil["program_breakdown"]) == 2

    def test_angkatan_tak_terbaca_tetap_masuk_rincian_prodi(self):
        hasil = rincian_profil([_baris("1010", None, 3)])
        assert hasil["program_breakdown"][0]["question_count"] == 3
        assert hasil["intake_year_breakdown"] == []


class TestNamaBerkas:
    @pytest.mark.parametrize(
        "masuk,keluar",
        [
            ("Panduan Akademik 2025.pdf", "Panduan Akademik 2025.pdf"),
            ("../../etc/passwd", "passwd"),
            ("C:\\Users\\staf\\Panduan.pdf", "Panduan.pdf"),
            ('a<b>:"c|?*.pdf', "a_b___c___.pdf"),
            (None, "dokumen.pdf"),
            ("", "dokumen.pdf"),
            ("...", "dokumen.pdf"),
        ],
    )
    def test_tanpa_unsur_lintasan(self, masuk, keluar):
        assert nama_berkas_aman(masuk) == keluar
