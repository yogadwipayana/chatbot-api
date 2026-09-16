"""Kalibrasi ambang penolakan (FR-3).

PRD menuntut ambang ditentukan empiris dari distribusi `top_score`, bukan
ditebak. Script ini menjalankan dua kumpulan pertanyaan:

- `--dataset`      : pertanyaan yang MEMANG terjawab dokumen (set evaluasi)
- `--negatives`    : pertanyaan yang SEHARUSNYA ditolak (satu per baris, teks polos)

Lalu melaporkan distribusi skor keduanya, sehingga ambang dapat dipilih pada
titik yang memisahkan keduanya. Tanpa kumpulan negatif, ambang hanya bisa
dipilih dengan menebak berapa banyak jawaban benar yang rela dibuang.

    python -m eval.calibrate_threshold --dataset eval/data/eval_set.jsonl \
        --negatives eval/data/negatives.txt
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from pathlib import Path

from app.config import get_settings
from app.db.session import SessionLocal
from app.rag.providers import build_embeddings
from app.rag.retriever import PostgresHybridRetriever
from app.rag.threshold import VECTOR_SOURCE
from eval.dataset import load_jsonl


async def kumpulkan_skor(pertanyaan: list[str]) -> list[float]:
    from app.db.session import engine

    settings = get_settings()
    embeddings = build_embeddings(settings)
    retriever = PostgresHybridRetriever(
        session_factory=SessionLocal,
        embed_query=embeddings.aembed_query,
        candidates=settings.retrieval_candidates,
        top_n=settings.retrieval_top_n,
    )
    skor: list[float] = []
    try:
        for teks in pertanyaan:
            docs = await retriever.ainvoke(teks)
            nilai = [
                d.metadata["raw_scores"][VECTOR_SOURCE]
                for d in docs
                if VECTOR_SOURCE in d.metadata["raw_scores"]
            ]
            skor.append(max(nilai) if nilai else 0.0)
    finally:
        # main() memanggil asyncio.run() dua kali (positif, lalu negatif).
        # Koneksi asyncpg di pool terikat pada event loop yang membuatnya dan
        # tidak bisa dipakai loop berikutnya; buang pool agar dibuka ulang.
        await engine.dispose()
    return skor


def ringkas(nama: str, skor: list[float]) -> None:
    if not skor:
        print(f"{nama}: tidak ada data")
        return
    urut = sorted(skor)
    print(f"\n{nama} (n={len(urut)})")
    print(f"  min    {urut[0]:.4f}")
    for persentil in (5, 25, 50, 75, 95):
        indeks = min(len(urut) - 1, persentil * len(urut) // 100)
        print(f"  p{persentil:<5} {urut[indeks]:.4f}")
    print(f"  max    {urut[-1]:.4f}")
    print(f"  rata2  {statistics.fmean(urut):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Kalibrasi ambang penolakan FR-3")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--negatives", type=Path)
    args = parser.parse_args()

    positif = [k.question for k in load_jsonl(args.dataset)]
    skor_positif = asyncio.run(kumpulkan_skor(positif))
    ringkas("Pertanyaan terjawab", skor_positif)

    if args.negatives:
        negatif = [
            b.strip()
            for b in args.negatives.read_text(encoding="utf-8").splitlines()
            if b.strip()
        ]
        skor_negatif = asyncio.run(kumpulkan_skor(negatif))
        ringkas("Pertanyaan seharusnya ditolak", skor_negatif)

        p5_positif = sorted(skor_positif)[max(0, 5 * len(skor_positif) // 100)]
        indeks_p95 = min(len(skor_negatif) - 1, 95 * len(skor_negatif) // 100)
        p95_negatif = sorted(skor_negatif)[indeks_p95]
        print(
            f"\nUsulan vector_threshold: antara {p95_negatif:.4f} dan {p5_positif:.4f}"
        )
        if p95_negatif >= p5_positif:
            print(
                "Kedua distribusi bertumpang tindih -- tidak ada ambang yang memisahkan "
                "keduanya dengan bersih. Perbaiki kualitas retrieval dulu; menaikkan "
                "ambang di sini hanya menukar halusinasi dengan penolakan palsu."
            )
    else:
        print("\nTanpa --negatives, ambang hanya bisa ditebak. Siapkan kumpulan negatif.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
