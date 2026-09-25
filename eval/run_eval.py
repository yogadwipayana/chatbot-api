"""Script evaluasi retrieval (PRD §16 'Evaluasi').

Menjalankan set evaluasi terhadap dua konfigurasi dan membandingkannya:
baseline vector-only melawan hybrid + RRF -- ditambah hybrid + RRF + rerank
bila RERANK_PROVIDER dinyalakan. Perbandingan inilah yang menjadi
bukti bahwa retriever kustom memang lebih baik -- tanpa angka pembanding,
klaim di skripsi tidak dapat dipertahankan.

    python -m eval.run_eval --dataset eval/data/eval_set.jsonl --k 5
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.db.session import SessionLocal
from app.rag.providers import build_embeddings
from app.rag.reranker import build_reranker
from app.rag.retriever import PostgresHybridRetriever
from eval.dataset import load_jsonl
from eval.metrics import EvalReport, evaluate


async def jalankan(
    dataset: Path, k: int, top_n: int
) -> tuple[EvalReport, EvalReport, EvalReport | None]:
    settings = get_settings()
    kasus = load_jsonl(dataset)
    embeddings = build_embeddings(settings)

    hybrid = PostgresHybridRetriever(
        session_factory=SessionLocal,
        embed_query=embeddings.aembed_query,
        candidates=settings.retrieval_candidates,
        top_n=top_n,
    )
    # Baseline: bobot fulltext nol -> murni vector search, jalur SQL sama
    # persis, sehingga yang dibandingkan benar-benar hanya efek fusi.
    baseline = PostgresHybridRetriever(
        session_factory=SessionLocal,
        embed_query=embeddings.aembed_query,
        candidates=settings.retrieval_candidates,
        top_n=top_n,
        weight_fulltext=0.0,
    )
    # Hybrid yang sama persis ditambah reranker: selisihnya dengan `hybrid`
    # adalah efek reranker saja.
    reranker = build_reranker(settings)
    reranked = (
        PostgresHybridRetriever(
            session_factory=SessionLocal,
            embed_query=embeddings.aembed_query,
            candidates=settings.retrieval_candidates,
            top_n=top_n,
            reranker=reranker,
            rerank_candidates=settings.rerank_candidates,
        )
        if reranker is not None
        else None
    )

    hasil_hybrid = []
    hasil_baseline = []
    hasil_rerank = []
    for kas in kasus:
        docs_h = await hybrid.ainvoke(kas.question)
        docs_b = await baseline.ainvoke(kas.question)
        hasil_hybrid.append((kas, [d.metadata["chunk_id"] for d in docs_h]))
        hasil_baseline.append((kas, [d.metadata["chunk_id"] for d in docs_b]))
        if reranked is not None:
            docs_r = await reranked.ainvoke(kas.question)
            hasil_rerank.append((kas, [d.metadata["chunk_id"] for d in docs_r]))

    return (
        evaluate(hasil_baseline, k=k),
        evaluate(hasil_hybrid, k=k),
        evaluate(hasil_rerank, k=k) if reranked is not None else None,
    )


def cetak(nama: str, laporan: EvalReport) -> None:
    tanda = "LULUS" if laporan.meets_target() else "BELUM"
    print(
        f"{nama:<24} n={laporan.n_cases:<4} "
        f"Recall@{laporan.k}={laporan.recall_at_k:.3f}  "
        f"MRR={laporan.mrr:.3f}  [{tanda} target 0.85]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluasi retrieval")
    parser.add_argument("--dataset", type=Path, default=Path("eval/data/eval_set.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    baseline, hybrid, reranked = asyncio.run(jalankan(args.dataset, args.k, args.top_n))
    cetak("vector-only", baseline)
    cetak("hybrid + RRF", hybrid)
    if reranked is not None:
        cetak("hybrid + RRF + rerank", reranked)

    selisih = hybrid.recall_at_k - baseline.recall_at_k
    print(f"\nSelisih Recall@{args.k}: {selisih:+.3f}")
    if selisih <= 0:
        print(
            "Hybrid tidak lebih baik. Sebelum menyimpulkan, periksa dulu apakah "
            "konfigurasi text search Indonesia benar-benar aktif dan kolom tsv terisi."
        )
    if reranked is not None:
        print(
            f"Efek reranker Recall@{args.k}: {reranked.recall_at_k - hybrid.recall_at_k:+.3f}"
        )
    terbaik = reranked if reranked is not None else hybrid
    return 0 if terbaik.meets_target() else 1


if __name__ == "__main__":
    raise SystemExit(main())
