"""Metrik evaluasi retrieval (PRD §3, §14, §16).

Dipakai oleh `eval/run_eval.py` untuk membandingkan baseline vector-only
melawan hybrid + RRF. Target rilis: Recall@5 >= 85%.

Murni fungsi -- tidak menyentuh DB, LLM, maupun LangChain -- supaya angka yang
dilaporkan di skripsi dapat direproduksi siapa pun tanpa akses sistem.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    """Satu baris set evaluasi (50-100 pertanyaan, divalidasi staf akademik)."""

    question: str
    relevant_chunk_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.relevant_chunk_ids:
            raise ValueError(f"kasus tanpa chunk relevan: {self.question!r}")


@dataclass(frozen=True)
class EvalReport:
    n_cases: int
    recall_at_k: float
    mrr: float
    k: int

    def meets_target(self, target_recall: float = 0.85) -> bool:
        """Kriteria kelayakan rilis §14."""
        return self.recall_at_k >= target_recall


def recall_at_k(retrieved: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Proporsi chunk relevan yang muncul di k teratas.

    `k` lebih besar dari jumlah hasil bukan error -- hasil yang ada saja
    yang dinilai.
    """
    if k < 1:
        raise ValueError(f"k harus >= 1, diberi {k}")
    if not relevant:
        raise ValueError("himpunan chunk relevan kosong")
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: frozenset[str]) -> float:
    """1/peringkat chunk relevan pertama; 0.0 bila tidak ada yang relevan."""
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def evaluate(
    results: Iterable[tuple[EvalCase, Sequence[str]]],
    *,
    k: int = 5,
) -> EvalReport:
    """Rata-ratakan metrik atas seluruh set evaluasi (macro-average).

    Args:
        results: pasangan (kasus, chunk_id hasil retrieval terurut).
        k: nilai k untuk Recall@k. Default 5, sesuai target PRD.
    """
    recalls: list[float] = []
    rrs: list[float] = []
    for case, retrieved in results:
        recalls.append(recall_at_k(retrieved, case.relevant_chunk_ids, k))
        rrs.append(reciprocal_rank(retrieved, case.relevant_chunk_ids))

    n = len(recalls)
    if n == 0:
        raise ValueError("set evaluasi kosong")
    return EvalReport(
        n_cases=n,
        recall_at_k=sum(recalls) / n,
        mrr=sum(rrs) / n,
        k=k,
    )
