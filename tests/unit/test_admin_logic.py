"""Validasi skema dan pembantu dashboard admin yang tidak butuh database."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from app.admin.documents import stale_clause
from app.admin.stats import ratio
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
        akun = AdminUserCreate(email="a@kampus.ac.id", role="staf", unit="  Biro   Keuangan ")
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
            KillSwitchRequest(engaged=True, alasan=alasan)

    def test_mematikan_tanpa_alasan_boleh(self):
        assert KillSwitchRequest(engaged=False).engaged is False


class TestDocumentUpdate:
    @pytest.mark.parametrize("nama", ["judul", "unit", "is_active"])
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
        assert DocumentUpdate(judul="  Panduan 2026  ").judul == "Panduan 2026"
        with pytest.raises(ValidationError):
            DocumentUpdate(judul="  ab  ")

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
