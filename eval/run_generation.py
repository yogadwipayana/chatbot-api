"""Jalankan PANDU atas set evaluasi dan simpan hasilnya untuk RAGAS (flow.md §10).

Tahap pertama dari dua. Skrip ini memakai alur produksi apa adanya -- graf yang
sama, retriever + reranker yang sama, prompt dan LLM yang sama -- lalu menulis
satu baris JSONL per pertanyaan: pertanyaan, jawaban rujukan, konteks yang
terambil, dan jawaban yang dihasilkan. Tahap kedua (`eval/ragas/run_ragas.py`)
menilai berkas itu di environment tersendiri, karena ragas belum cocok dengan
versi LangChain yang dipin api/.

Format set evaluasi (JSONL), satu kasus per baris:

    {"question": "Apa syarat cuti kuliah?",
     "reference": "Mahasiswa aktif minimal dua semester ... (jawaban benar)",
     "unit": "Akademik"}                      # opsional

`reference` WAJIB ditulis atau divalidasi staf akademik (PRD §14): RAGAS
membandingkan jawaban dengan rujukan ini, jadi rujukan yang keliru menghasilkan
angka yang meyakinkan tetapi salah.

    python -m eval.run_generation --dataset eval/data/generation_set.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.rag.chain import OutcomeKind, PipelineOutcome

HASIL_DIR = Path(__file__).resolve().parent / "hasil"


@dataclass(frozen=True)
class GenerationCase:
    question: str
    reference: str
    unit: str | None = None


def load_cases(path: str | Path) -> list[GenerationCase]:
    """Muat set evaluasi generasi. Gagal keras pada baris cacat, seperti `eval.dataset`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"set evaluasi tidak ditemukan: {path}")
    kasus: list[GenerationCase] = []
    for nomor, baris in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        baris = baris.strip()
        if not baris or baris.startswith("//"):
            continue
        try:
            data = json.loads(baris)
            question, reference = data["question"].strip(), data["reference"].strip()
            if not question or not reference:
                raise ValueError("question dan reference tidak boleh kosong")
            kasus.append(GenerationCase(question, reference, data.get("unit") or None))
        except (json.JSONDecodeError, KeyError, ValueError, AttributeError) as exc:
            raise ValueError(f"{path}:{nomor} tidak valid -- {exc}") from exc
    if not kasus:
        raise ValueError(f"{path} tidak berisi satu kasus pun")
    return kasus


def build_record(case: GenerationCase, outcome: PipelineOutcome, latency_ms: int) -> dict:
    """Satu baris hasil. Nama kolom mengikuti `SingleTurnSample` ragas."""
    decision = outcome.decision
    return {
        "user_input": case.question,
        "reference": case.reference,
        "response": outcome.text,
        "retrieved_contexts": [d.page_content for d in outcome.documents],
        "kind": outcome.kind.value,
        "unit": case.unit,
        "sources": [
            {"judul": d.metadata.get("judul"), "halaman": d.metadata.get("halaman")}
            for d in outcome.documents
        ],
        "top_score": decision.top_score if decision else None,
        "top_rerank_score": decision.top_rerank_score if decision else None,
        "gate_label": outcome.gate.label.value if outcome.gate else None,
        "latency_ms": latency_ms,
    }


async def jalankan(dataset: Path, keluaran: Path) -> list[dict]:
    from app.config import get_settings
    from app.db.session import engine
    from app.deps import LLMCall, build_gate_call, build_retriever
    from app.rag.chain import run_pipeline
    from app.routers.chat import policy_from

    settings = get_settings()
    kasus = load_cases(dataset)
    retriever = build_retriever(settings)
    gate = build_gate_call(settings)

    hasil: list[dict] = []
    try:
        for i, kas in enumerate(kasus, start=1):
            mulai = time.perf_counter()
            outcome = await run_pipeline(
                kas.question,
                retriever=retriever,
                # Instans baru per pertanyaan: LLMCall menyimpan `usage` giliran terakhir.
                llm_call=LLMCall(settings),
                gate_call=gate,
                policy=policy_from(settings),
                unit=kas.unit,
            )
            rekaman = build_record(kas, outcome, round((time.perf_counter() - mulai) * 1000))
            hasil.append(rekaman)
            print(f"  [{i}/{len(kasus)}] {rekaman['kind']:<9} {kas.question[:60]}")
    finally:
        await engine.dispose()

    keluaran.parent.mkdir(parents=True, exist_ok=True)
    keluaran.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in hasil), encoding="utf-8"
    )
    return hasil


def ringkas(hasil: list[dict[str, Any]]) -> str:
    dijawab = sum(1 for r in hasil if r["kind"] == OutcomeKind.ANSWER)
    return (
        f"{len(hasil)} pertanyaan: {dijawab} dijawab, {len(hasil) - dijawab} tidak "
        f"(hanya yang dijawab dinilai RAGAS)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Jalankan PANDU untuk evaluasi RAGAS")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=HASIL_DIR / f"generation-{datetime.now():%Y%m%d-%H%M%S}.jsonl",
    )
    args = parser.parse_args()
    hasil = asyncio.run(jalankan(args.dataset, args.out))
    print(ringkas(hasil))
    print(f"Tersimpan: {args.out}")
    print(f"Lanjutkan: cd eval/ragas && uv run python run_ragas.py {args.out.resolve()}")


if __name__ == "__main__":
    main()
