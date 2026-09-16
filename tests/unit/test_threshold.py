"""FR-3 -- ambang penolakan."""

from __future__ import annotations

import pytest

from app.rag.fusion import FusedHit
from app.rag.threshold import Decision, Reason, ThresholdPolicy, evaluate


def hit(chunk_id="c1", *, vector=None, lexical=None, rrf=0.016) -> FusedHit:
    raw = {}
    if vector is not None:
        raw["vector"] = vector
    if lexical is not None:
        raw["fulltext"] = lexical
    return FusedHit(chunk_id=chunk_id, rrf_score=rrf, ranks={}, raw_scores=raw)


POLICY = ThresholdPolicy(vector_threshold=0.35, lexical_threshold=0.05)


class TestKeputusan:
    def test_vektor_kuat_lolos(self):
        keputusan = evaluate([hit(vector=0.80)], POLICY)
        assert keputusan.decision is Decision.PROCEED
        assert keputusan.reason is Reason.OK
        assert keputusan.should_call_llm

    def test_semua_lemah_ditolak(self):
        keputusan = evaluate([hit(vector=0.10, lexical=0.001)], POLICY)
        assert keputusan.decision is Decision.REFUSE
        assert keputusan.reason is Reason.BELOW_THRESHOLD
        assert not keputusan.should_call_llm

    def test_hasil_kosong_ditolak(self):
        keputusan = evaluate([], POLICY)
        assert keputusan.decision is Decision.REFUSE
        assert keputusan.reason is Reason.NO_RESULTS
        assert keputusan.top_vector_score is None
        assert keputusan.top_lexical_score is None

    def test_tepat_di_ambang_lolos(self):
        """Perbandingan memakai >=. Diuji agar perubahan ke > tidak lolos diam-diam."""
        assert evaluate([hit(vector=0.35)], POLICY).decision is Decision.PROCEED

    def test_sedikit_di_bawah_ambang_ditolak(self):
        assert evaluate([hit(vector=0.3499)], POLICY).decision is Decision.REFUSE

    def test_leksikal_kuat_menyelamatkan_vektor_lemah(self):
        """Kecocokan frasa persis sering punya kemiripan vektor rendah tetapi
        ts_rank tinggi. Menolaknya akan membuang dokumen yang paling tepat."""
        keputusan = evaluate([hit(vector=0.10, lexical=0.42)], POLICY)
        assert keputusan.decision is Decision.PROCEED

    def test_ambang_dinilai_atas_hit_terbaik_bukan_hit_pertama(self):
        hits = [hit("c1", vector=0.20), hit("c2", vector=0.90)]
        assert evaluate(hits, POLICY).top_vector_score == pytest.approx(0.90)


class TestInvarianLLM:
    def test_ditolak_berarti_llm_tidak_boleh_dipanggil(self):
        assert evaluate([hit(vector=0.01)], POLICY).should_call_llm is False

    def test_ditolak_berarti_dicatat_ke_unanswered(self):
        """FR-3: pertanyaan yang ditolak masuk tabel unanswered (sumber AD-4)."""
        assert evaluate([hit(vector=0.01)], POLICY).should_log_unanswered is True

    def test_lolos_tidak_dicatat_sebagai_unanswered(self):
        assert evaluate([hit(vector=0.9)], POLICY).should_log_unanswered is False


class TestSkorRRFTidakDipakai:
    def test_skor_rrf_tinggi_tidak_menyelamatkan_hasil_tidak_relevan(self):
        """Alasan raw_scores dibawa dari tahap fusi.

        Chunk peringkat 1 SELALU memperoleh skor RRF maksimum, termasuk saat
        seluruh kandidat tidak relevan. Menilai ambang atas skor RRF berarti
        sistem tidak pernah menolak apa pun -- FR-3 mati diam-diam.
        """
        rrf_maksimum = 1 / 61
        keputusan = evaluate([hit(vector=0.02, lexical=0.0001, rrf=rrf_maksimum)], POLICY)
        assert keputusan.decision is Decision.REFUSE

    def test_top_score_yang_dicatat_bukan_skor_rrf(self):
        keputusan = evaluate([hit(vector=0.62, rrf=0.016)], POLICY)
        assert keputusan.top_score == pytest.approx(0.62)


class TestTopScore:
    def test_mengambil_nilai_terbesar_antar_sumber(self):
        assert evaluate([hit(vector=0.4, lexical=0.9)], POLICY).top_score == pytest.approx(0.9)

    def test_nol_bila_tidak_ada_hasil(self):
        assert evaluate([], POLICY).top_score == 0.0


class TestValidasiPolicy:
    @pytest.mark.parametrize("nilai", [-0.1, 1.5])
    def test_vector_threshold_di_luar_rentang_ditolak(self, nilai):
        with pytest.raises(ValueError, match="vector_threshold"):
            ThresholdPolicy(vector_threshold=nilai)

    def test_lexical_threshold_negatif_ditolak(self):
        with pytest.raises(ValueError, match="lexical_threshold"):
            ThresholdPolicy(lexical_threshold=-1.0)

    def test_policy_default_dipakai_bila_tidak_diberikan(self):
        assert evaluate([hit(vector=0.99)]).decision is Decision.PROCEED
