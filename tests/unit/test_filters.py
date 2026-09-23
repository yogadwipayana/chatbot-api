"""FR-2 -- filter dokumen aktif di level SQL.

Predikat ini satu-satunya yang mencegah dokumen kedaluwarsa terambil. Metrik
PRD §3 menargetkan 0 dokumen kedaluwarsa aktif di indeks, dan angka itu hanya
tercapai kalau predikat ini benar-benar ikut di setiap query.
"""

from __future__ import annotations

import pytest

from app.rag import retriever as retriever_module
from app.rag.filters import ACTIVE_DOCUMENT_PREDICATE, active_document_clause


class TestPredikat:
    def test_memeriksa_is_active(self):
        assert "is_active = true" in active_document_clause()

    def test_memeriksa_valid_until_null_atau_belum_lewat(self):
        klausa = active_document_clause()
        assert "valid_until IS NULL" in klausa
        assert "valid_until > now()" in klausa

    def test_valid_until_null_dan_belum_lewat_digabung_dengan_OR(self):
        """Dokumen tanpa masa berlaku harus tetap lolos; kalau dua syarat ini
        digabung dengan AND, seluruh dokumen permanen ikut tersaring habis."""
        assert " OR " in active_document_clause()

    def test_dua_syarat_digabung_dengan_AND(self):
        assert " AND " in active_document_clause()

    def test_alias_diterapkan_ke_semua_kolom(self):
        klausa = active_document_clause("dok")
        assert klausa.count("dok.") == 3
        assert "d.is_active" not in klausa

    def test_konstanta_konsisten_dengan_fungsi(self):
        """Konstanta dipakai sebagai dokumentasi; jangan sampai menyimpang."""
        assert active_document_clause("d") == ACTIVE_DOCUMENT_PREDICATE

    @pytest.mark.parametrize("alias", ["", "1abc", "a b", "d;DROP TABLE documents", "d-1"])
    def test_alias_tidak_valid_ditolak(self, alias):
        """Alias masuk ke SQL sebagai teks, jadi ia tidak boleh sembarang string."""
        with pytest.raises(ValueError, match="alias"):
            active_document_clause(alias)


class TestDipakaiDiQueryRetrieval:
    def test_query_vektor_menyertakan_filter(self):
        sql = str(retriever_module.VECTOR_SQL)
        assert "is_active" in sql
        assert "valid_until" in sql

    def test_query_fulltext_menyertakan_filter(self):
        sql = str(retriever_module.FULLTEXT_SQL)
        assert "is_active" in sql
        assert "valid_until" in sql

    def test_kedua_query_memakai_predikat_yang_sama(self):
        """Dua salinan predikat yang berbeda adalah cara paling mudah untuk
        diam-diam membocorkan dokumen kedaluwarsa lewat salah satu jalur."""
        klausa = active_document_clause("d")
        assert klausa in str(retriever_module.VECTOR_SQL)
        assert klausa in str(retriever_module.FULLTEXT_SQL)


class TestFilterUnit:
    def test_kedua_query_memakai_klausa_unit_yang_sama(self):
        """Sama alasannya dengan predikat dokumen aktif: dua salinan yang berbeda
        membiarkan dokumen unit lain bocor lewat salah satu jalur."""
        klausa = retriever_module._UNIT
        assert klausa in str(retriever_module.VECTOR_SQL)
        assert klausa in str(retriever_module.FULLTEXT_SQL)

    def test_null_berarti_semua_unit(self):
        assert "IS NULL OR d.unit =" in retriever_module._UNIT

    def test_parameter_diberi_tipe_eksplisit(self):
        """asyncpg tidak dapat menebak tipe parameter yang hanya muncul di
        `IS NULL`; tanpa CAST query gagal saat unit dikirim sebagai None."""
        assert retriever_module._UNIT.count("CAST(:unit AS text)") == 2
