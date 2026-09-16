"""Ambang penolakan (FR-3).

Dijalankan *setelah* retrieval dan *sebelum* LLM. Bila keputusannya REFUSE,
LLM tidak boleh dipanggil sama sekali -- ini invarian yang diuji di
`tests/unit/test_threshold.py`.

Kenapa tidak menilai skor RRF: skor RRF hanya mencerminkan peringkat, bukan
kemiripan. Chunk terbaik dari sekumpulan chunk yang sama-sama tidak relevan
tetap memperoleh skor RRF tertinggi. Yang dinilai di sini adalah skor mentah
(`FusedHit.raw_scores`) dari retrieval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.rag.fusion import FusedHit

VECTOR_SOURCE = "vector"
LEXICAL_SOURCE = "fulltext"


class Decision(StrEnum):
    PROCEED = "proceed"
    REFUSE = "refuse"


class Reason(StrEnum):
    OK = "ok"
    NO_RESULTS = "no_results"
    BELOW_THRESHOLD = "below_threshold"


@dataclass(frozen=True)
class ThresholdPolicy:
    """Ambang ditentukan empiris dari distribusi skor (FR-3), bukan ditebak.

    Jalankan `eval/calibrate_threshold.py` terhadap set evaluasi untuk
    memperolehnya; nilai default di bawah hanyalah placeholder agar sistem
    tidak berjalan tanpa ambang sama sekali.
    """

    vector_threshold: float = 0.35
    """Cosine similarity minimum (0..1, makin tinggi makin mirip)."""

    lexical_threshold: float = 0.05
    """`ts_rank` minimum. Kecocokan frasa persis boleh lolos walau vektor lemah."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.vector_threshold <= 1.0:
            raise ValueError(f"vector_threshold di luar 0..1: {self.vector_threshold}")
        if self.lexical_threshold < 0.0:
            raise ValueError(f"lexical_threshold negatif: {self.lexical_threshold}")


@dataclass(frozen=True)
class ThresholdDecision:
    decision: Decision
    reason: Reason
    top_vector_score: float | None
    top_lexical_score: float | None

    @property
    def should_call_llm(self) -> bool:
        return self.decision is Decision.PROCEED

    @property
    def should_log_unanswered(self) -> bool:
        """FR-3: pertanyaan yang ditolak masuk tabel `unanswered`."""
        return self.decision is Decision.REFUSE

    @property
    def top_score(self) -> float:
        """Nilai yang disimpan ke kolom `messages.top_score` / `unanswered.top_score`."""
        return max(
            (s for s in (self.top_vector_score, self.top_lexical_score) if s is not None),
            default=0.0,
        )


def evaluate(
    hits: Sequence[FusedHit],
    policy: ThresholdPolicy | None = None,
) -> ThresholdDecision:
    """Putuskan apakah konteks cukup kuat untuk dijawab LLM.

    Lolos bila skor vektor ATAU skor leksikal terbaik mencapai ambangnya.
    Perbandingan memakai `>=`, sehingga nilai tepat di ambang dianggap lolos.
    """
    policy = policy or ThresholdPolicy()

    if not hits:
        return ThresholdDecision(
            decision=Decision.REFUSE,
            reason=Reason.NO_RESULTS,
            top_vector_score=None,
            top_lexical_score=None,
        )

    top_vector = _best(hits, VECTOR_SOURCE)
    top_lexical = _best(hits, LEXICAL_SOURCE)

    passes = (top_vector is not None and top_vector >= policy.vector_threshold) or (
        top_lexical is not None and top_lexical >= policy.lexical_threshold
    )

    return ThresholdDecision(
        decision=Decision.PROCEED if passes else Decision.REFUSE,
        reason=Reason.OK if passes else Reason.BELOW_THRESHOLD,
        top_vector_score=top_vector,
        top_lexical_score=top_lexical,
    )


def _best(hits: Sequence[FusedHit], source: str) -> float | None:
    scores = [h.raw_scores[source] for h in hits if source in h.raw_scores]
    return max(scores) if scores else None
