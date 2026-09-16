"""FR-2 -- Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from app.rag.fusion import DEFAULT_K, FusedHit, RankedHit, reciprocal_rank_fusion


def hits(*pairs: tuple[str, float]) -> list[RankedHit]:
    return [RankedHit(chunk_id, score) for chunk_id, score in pairs]


class TestPeringkat:
    def test_chunk_yang_muncul_di_kedua_sumber_menang(self):
        """Inti RRF: kesepakatan dua sumber mengalahkan peringkat 1 di satu sumber."""
        fused = reciprocal_rank_fusion(
            {
                "vector": hits(("a", 0.9), ("b", 0.8)),
                "fulltext": hits(("b", 0.5), ("c", 0.4)),
            }
        )
        assert fused[0].chunk_id == "b"
        assert fused[0].sources == {"vector", "fulltext"}

    def test_skor_mengikuti_rumus(self):
        fused = reciprocal_rank_fusion({"vector": hits(("a", 0.9))}, k=60)
        assert fused[0].rrf_score == pytest.approx(1 / 61)

    def test_bobot_mengubah_pemenang(self):
        """Bobot harus benar-benar berpengaruh -- itu alasan RRF ditulis sendiri."""
        lists = {
            "vector": hits(("a", 0.9)),
            "fulltext": hits(("b", 0.9)),
        }
        seimbang = reciprocal_rank_fusion(lists)
        assert seimbang[0].chunk_id == "a"  # tie -> chunk_id menaik

        berat_leksikal = reciprocal_rank_fusion(
            lists, weights={"vector": 0.2, "fulltext": 5.0}
        )
        assert berat_leksikal[0].chunk_id == "b"

    def test_k_besar_mempersempit_jarak_antar_peringkat(self):
        lists = {"vector": hits(("a", 0.9), ("b", 0.8))}
        sempit = reciprocal_rank_fusion(lists, k=1)
        lebar = reciprocal_rank_fusion(lists, k=10_000)
        selisih_sempit = sempit[0].rrf_score - sempit[1].rrf_score
        selisih_lebar = lebar[0].rrf_score - lebar[1].rrf_score
        assert selisih_sempit > selisih_lebar

    def test_urutan_deterministik_saat_skor_sama(self):
        """Evaluasi §14 harus dapat direproduksi, jadi tie-break tidak boleh acak."""
        lists = {"vector": hits(("z", 0.5)), "fulltext": hits(("a", 0.5))}
        pertama = [h.chunk_id for h in reciprocal_rank_fusion(lists)]
        for _ in range(20):
            assert [h.chunk_id for h in reciprocal_rank_fusion(lists)] == pertama
        assert pertama == ["a", "z"]


class TestSkorMentah:
    def test_raw_scores_dipertahankan_per_sumber(self):
        """FR-3 menilai skor mentah, bukan skor RRF -- jadi ini tidak boleh hilang."""
        fused = reciprocal_rank_fusion(
            {"vector": hits(("a", 0.83)), "fulltext": hits(("a", 0.04))}
        )
        assert fused[0].raw_scores == {"vector": 0.83, "fulltext": 0.04}

    def test_ranks_dicatat_1_based(self):
        fused = reciprocal_rank_fusion({"vector": hits(("a", 0.9), ("b", 0.8))})
        by_id = {h.chunk_id: h for h in fused}
        assert by_id["a"].ranks == {"vector": 1}
        assert by_id["b"].ranks == {"vector": 2}

    def test_chunk_hanya_di_satu_sumber_tidak_punya_skor_sumber_lain(self):
        fused = reciprocal_rank_fusion(
            {"vector": hits(("a", 0.9)), "fulltext": hits(("b", 0.3))}
        )
        by_id = {h.chunk_id: h for h in fused}
        assert "fulltext" not in by_id["a"].raw_scores
        assert "vector" not in by_id["b"].raw_scores


class TestKasusTepi:
    def test_semua_sumber_kosong(self):
        assert reciprocal_rank_fusion({"vector": [], "fulltext": []}) == []

    def test_satu_sumber_kosong_tidak_menggagalkan(self):
        fused = reciprocal_rank_fusion({"vector": hits(("a", 0.9)), "fulltext": []})
        assert [h.chunk_id for h in fused] == ["a"]

    def test_top_n_memotong_hasil(self):
        lists = {"vector": hits(("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6))}
        assert len(reciprocal_rank_fusion(lists, top_n=2)) == 2

    def test_top_n_lebih_besar_dari_hasil(self):
        fused = reciprocal_rank_fusion({"vector": hits(("a", 0.9))}, top_n=99)
        assert len(fused) == 1

    def test_top_n_nol_mengembalikan_kosong(self):
        assert reciprocal_rank_fusion({"vector": hits(("a", 0.9))}, top_n=0) == []

    def test_duplikat_dalam_satu_sumber_dihitung_sekali(self):
        """Peringkat pertama yang dipakai; tanpa ini satu chunk bisa menggandakan skor."""
        fused = reciprocal_rank_fusion({"vector": hits(("a", 0.9), ("a", 0.5))})
        assert len(fused) == 1
        assert fused[0].rrf_score == pytest.approx(1 / (DEFAULT_K + 1))

    def test_hasil_bertipe_FusedHit(self):
        fused = reciprocal_rank_fusion({"vector": hits(("a", 0.9))})
        assert isinstance(fused[0], FusedHit)


class TestValidasiArgumen:
    def test_k_nol_ditolak(self):
        with pytest.raises(ValueError, match="k harus > 0"):
            reciprocal_rank_fusion({"vector": hits(("a", 0.9))}, k=0)

    def test_k_negatif_ditolak(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion({"vector": hits(("a", 0.9))}, k=-5)

    def test_top_n_negatif_ditolak(self):
        with pytest.raises(ValueError, match="top_n"):
            reciprocal_rank_fusion({"vector": hits(("a", 0.9))}, top_n=-1)

    def test_bobot_untuk_sumber_tak_dikenal_ditolak(self):
        """Salah ketik nama sumber akan diam-diam mengabaikan bobot; jangan biarkan."""
        with pytest.raises(ValueError, match="sumber tak dikenal"):
            reciprocal_rank_fusion(
                {"vector": hits(("a", 0.9))}, weights={"vetcor": 2.0}
            )
