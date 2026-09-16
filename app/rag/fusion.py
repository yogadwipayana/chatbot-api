"""Reciprocal Rank Fusion (FR-2).

Ditulis sendiri, bukan `EnsembleRetriever`, karena PRD §6 menuntut kendali penuh
atas pembobotan dan atas skor mentah tiap sumber.

Catatan penting untuk tahap threshold (FR-3): skor RRF bersifat *rank-based*.
Dokumen peringkat 1 selalu mendapat skor RRF maksimum, bahkan ketika kemiripan
aslinya rendah. Karena itu `FusedHit` tetap membawa `raw_scores` per sumber, dan
FR-3 harus menilai kemiripan mentah -- bukan skor RRF. Lihat `threshold.py`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

DEFAULT_K = 60
"""Konstanta peredam RRF. 60 mengikuti Cormack et al. (2009)."""


@dataclass(frozen=True)
class RankedHit:
    """Satu hasil dari satu sumber pencarian, dengan skor asli sumber tersebut."""

    chunk_id: str
    score: float


@dataclass(frozen=True)
class FusedHit:
    """Hasil gabungan lintas sumber."""

    chunk_id: str
    rrf_score: float
    ranks: dict[str, int] = field(default_factory=dict)
    """Nama sumber -> peringkat 1-based di sumber itu. Absen bila tidak muncul."""
    raw_scores: dict[str, float] = field(default_factory=dict)
    """Nama sumber -> skor mentah. Dipakai FR-3, jangan dibuang."""

    @property
    def sources(self) -> set[str]:
        return set(self.ranks)


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[RankedHit]],
    *,
    weights: Mapping[str, float] | None = None,
    k: int = DEFAULT_K,
    top_n: int | None = None,
) -> list[FusedHit]:
    """Gabungkan beberapa daftar terurut menjadi satu peringkat.

    score(d) = sum_over_sources( weight_source / (k + rank_source(d)) )

    Args:
        ranked_lists: nama sumber -> hasil, sudah terurut dari paling relevan.
        weights: bobot per sumber, default 1.0. Sumber tak dikenal ditolak.
        k: konstanta peredam; makin besar makin rata pengaruh antar peringkat.
        top_n: potong hasil akhir. None berarti kembalikan semua.

    Urutan hasil deterministik: skor menurun, lalu peringkat terbaik menaik,
    lalu `chunk_id` menaik -- supaya evaluasi bisa direproduksi.
    """
    if k <= 0:
        raise ValueError(f"k harus > 0, diberi {k}")
    if top_n is not None and top_n < 0:
        raise ValueError(f"top_n tidak boleh negatif, diberi {top_n}")

    weights = dict(weights or {})
    unknown = set(weights) - set(ranked_lists)
    if unknown:
        raise ValueError(f"bobot untuk sumber tak dikenal: {sorted(unknown)}")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    raw: dict[str, dict[str, float]] = {}

    for source, hits in ranked_lists.items():
        weight = weights.get(source, 1.0)
        seen: set[str] = set()
        for position, hit in enumerate(hits, start=1):
            if hit.chunk_id in seen:
                # Sumber yang sama tidak boleh menyumbang dua kali untuk chunk
                # yang sama; peringkat pertama yang dipakai.
                continue
            seen.add(hit.chunk_id)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + position)
            ranks.setdefault(hit.chunk_id, {})[source] = position
            raw.setdefault(hit.chunk_id, {})[source] = hit.score

    fused = [
        FusedHit(
            chunk_id=chunk_id,
            rrf_score=score,
            ranks=dict(ranks[chunk_id]),
            raw_scores=dict(raw[chunk_id]),
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda h: (-h.rrf_score, min(h.ranks.values()), h.chunk_id))
    return fused if top_n is None else fused[:top_n]
