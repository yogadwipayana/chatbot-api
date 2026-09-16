"""PRD §3 / §14 -- metrik evaluasi retrieval.

Angka Recall@5 dan MRR yang dilaporkan di skripsi berasal dari fungsi-fungsi
ini. Kalau metriknya sendiri salah, seluruh perbandingan vector-only melawan
hybrid tidak berarti apa-apa.
"""

from __future__ import annotations

import pytest

from eval.metrics import EvalCase, evaluate, recall_at_k, reciprocal_rank


class TestRecallAtK:
    def test_semua_relevan_terambil(self):
        assert recall_at_k(["a", "b", "c"], frozenset({"a", "b"}), 5) == 1.0

    def test_sebagian_terambil(self):
        assert recall_at_k(["a", "x", "y"], frozenset({"a", "b"}), 5) == 0.5

    def test_tidak_ada_yang_terambil(self):
        assert recall_at_k(["x", "y"], frozenset({"a"}), 5) == 0.0

    def test_hanya_menghitung_k_teratas(self):
        """Chunk relevan di peringkat 6 tidak boleh dihitung untuk Recall@5."""
        hasil = ["x", "x", "x", "x", "x", "a"]
        assert recall_at_k(hasil, frozenset({"a"}), 5) == 0.0
        assert recall_at_k(hasil, frozenset({"a"}), 6) == 1.0

    def test_k_lebih_besar_dari_jumlah_hasil(self):
        assert recall_at_k(["a"], frozenset({"a"}), 100) == 1.0

    def test_hasil_kosong(self):
        assert recall_at_k([], frozenset({"a"}), 5) == 0.0

    @pytest.mark.parametrize("k", [0, -1])
    def test_k_tidak_valid_ditolak(self, k):
        with pytest.raises(ValueError, match="k harus"):
            recall_at_k(["a"], frozenset({"a"}), k)

    def test_himpunan_relevan_kosong_ditolak(self):
        """Pembagian dengan nol akan menghasilkan 0% palsu, bukan error."""
        with pytest.raises(ValueError, match="kosong"):
            recall_at_k(["a"], frozenset(), 5)


class TestReciprocalRank:
    @pytest.mark.parametrize(
        "hasil,harapan",
        [
            (["a", "x", "y"], 1.0),
            (["x", "a", "y"], 0.5),
            (["x", "y", "a"], pytest.approx(1 / 3)),
        ],
    )
    def test_sesuai_posisi_relevan_pertama(self, hasil, harapan):
        assert reciprocal_rank(hasil, frozenset({"a"})) == harapan

    def test_nol_bila_tidak_ada_yang_relevan(self):
        assert reciprocal_rank(["x", "y"], frozenset({"a"})) == 0.0

    def test_memakai_kemunculan_pertama_saja(self):
        assert reciprocal_rank(["a", "b"], frozenset({"a", "b"})) == 1.0

    def test_hasil_kosong(self):
        assert reciprocal_rank([], frozenset({"a"})) == 0.0


class TestEvalCase:
    def test_kasus_tanpa_chunk_relevan_ditolak(self):
        """Baris set evaluasi tanpa jawaban benar hanya menurunkan skor secara
        palsu; lebih baik gagal saat memuat dataset."""
        with pytest.raises(ValueError, match="tanpa chunk relevan"):
            EvalCase("kapan wisuda?", frozenset())


class TestAgregasi:
    def test_rata_rata_antar_kasus(self):
        kasus = [
            (EvalCase("q1", frozenset({"a"})), ["a", "x"]),
            (EvalCase("q2", frozenset({"b"})), ["x", "y"]),
        ]
        laporan = evaluate(kasus, k=5)
        assert laporan.n_cases == 2
        assert laporan.recall_at_k == 0.5
        assert laporan.mrr == 0.5

    def test_k_tercatat_di_laporan(self):
        laporan = evaluate([(EvalCase("q", frozenset({"a"})), ["a"])], k=3)
        assert laporan.k == 3

    def test_set_evaluasi_kosong_ditolak(self):
        with pytest.raises(ValueError, match="kosong"):
            evaluate([])

    def test_target_rilis_terpenuhi(self):
        """§14: Recall@5 >= 85%."""
        kasus = [(EvalCase(f"q{i}", frozenset({"a"})), ["a"]) for i in range(10)]
        assert evaluate(kasus).meets_target()

    def test_target_rilis_tidak_terpenuhi(self):
        kasus = [
            (EvalCase(f"q{i}", frozenset({"a"})), ["a"] if i < 8 else ["x"])
            for i in range(10)
        ]
        laporan = evaluate(kasus)
        assert laporan.recall_at_k == pytest.approx(0.8)
        assert not laporan.meets_target()

    def test_ambang_target_dapat_diubah(self):
        kasus = [(EvalCase("q", frozenset({"a"})), ["x"])]
        assert not evaluate(kasus).meets_target(target_recall=0.5)
        assert evaluate(kasus).meets_target(target_recall=0.0)
