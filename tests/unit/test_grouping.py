"""AD-4 -- pengelompokan pertanyaan tak terjawab.

"12 mahasiswa menanyakan ini" hanya berguna bila kelompoknya benar: variasi
kalimat untuk hal yang sama harus bergabung, hal yang berbeda harus terpisah.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.admin.grouping import UnansweredItem, group_questions, jaccard, key_terms

T0 = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)


def item(
    id_: str,
    teks: str,
    *,
    menit: int = 0,
    skor: float | None = 0.2,
    resolved: bool = False,
) -> UnansweredItem:
    return UnansweredItem(id_, teks, skor, T0 + timedelta(minutes=menit), resolved)


class TestKataKunci:
    def test_kata_tanya_dan_sapaan_dibuang(self):
        assert key_terms("Min, bagaimana cara daftar beasiswa prestasi?") == {
            "daftar",
            "beasiswa",
            "prestasi",
        }

    def test_huruf_besar_dan_tanda_baca_diabaikan(self):
        assert key_terms("KRS!!!") == key_terms("krs")

    def test_akhiran_nya_dilepas(self):
        assert "wisuda" in key_terms("syarat wisudanya apa")

    def test_kata_pendek_berakhiran_nya_tidak_dipotong(self):
        assert "punya" in key_terms("punya surat")

    def test_jaccard(self):
        assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == pytest.approx(1 / 3)


class TestPengelompokan:
    def test_variasi_kalimat_digabung(self):
        grup = group_questions(
            [
                item("1", "Bagaimana cara daftar beasiswa prestasi?"),
                item("2", "cara daftar beasiswa prestasi gimana", menit=5),
                item("3", "Pendaftaran beasiswa prestasi kapan?", menit=10),
            ]
        )
        assert len(grup) == 1
        assert grup[0].jumlah == 3
        assert set(grup[0].ids) == {"1", "2", "3"}

    def test_topik_berbeda_dipisah(self):
        grup = group_questions(
            [item("1", "Syarat pindah program studi apa?"), item("2", "Jadwal bus kampus")]
        )
        assert len(grup) == 2

    def test_contoh_adalah_pertanyaan_terbaru_apa_adanya(self):
        grup = group_questions(
            [
                item("lama", "cara pindah program studi", menit=0),
                item("baru", "Bisa PINDAH program studi tidak?", menit=30),
            ]
        )
        assert grup[0].representative.pertanyaan == "Bisa PINDAH program studi tidak?"
        assert grup[0].terakhir_ditanyakan == T0 + timedelta(minutes=30)

    def test_kelompok_terbesar_lebih_dulu(self):
        grup = group_questions(
            [
                item("a", "jadwal bus kampus", menit=50),
                item("b1", "syarat pindah program studi", menit=1),
                item("b2", "cara pindah program studi", menit=2),
            ]
        )
        assert [g.jumlah for g in grup] == [2, 1]

    def test_sudah_dan_belum_ditindaklanjuti_tidak_dicampur(self):
        grup = group_questions(
            [
                item("1", "lupa password portal akademik", resolved=True),
                item("2", "lupa password portal akademik", menit=5),
            ]
        )
        assert len(grup) == 2
        assert {g.resolved for g in grup} == {True, False}

    def test_tanpa_kata_bermakna_hanya_digabung_bila_persis_sama(self):
        grup = group_questions(
            [item("1", "??"), item("2", "??", menit=1), item("3", "apa?", menit=2)]
        )
        assert sorted(g.jumlah for g in grup) == [1, 2]

    def test_tidak_melebar_berantai(self):
        """A mirip B dan B mirip C tidak berarti A dan C membahas hal yang sama."""
        grup = group_questions(
            [
                item("A", "beasiswa prestasi akademik", menit=20),
                item("B", "prestasi akademik olahraga", menit=10),
                item("C", "akademik olahraga nasional", menit=0),
            ]
        )
        assert sorted(g.jumlah for g in grup) == [1, 2]

    def test_rata_rata_skor_mengabaikan_null(self):
        grup = group_questions(
            [
                item("1", "jadwal KKN", skor=0.2),
                item("2", "jadwal KKN", skor=0.4, menit=1),
                item("3", "jadwal KKN", skor=None, menit=2),
            ]
        )
        assert grup[0].top_score_rata2 == pytest.approx(0.3)

    def test_skor_semua_null(self):
        assert group_questions([item("1", "jadwal KKN", skor=None)])[0].top_score_rata2 is None

    def test_kosong(self):
        assert group_questions([]) == []

    @pytest.mark.parametrize("ambang", [0, -0.1, 1.5])
    def test_ambang_tidak_sah_ditolak(self, ambang):
        with pytest.raises(ValueError):
            group_questions([item("1", "x")], threshold=ambang)
