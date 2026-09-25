"""Nilai hasil `eval/run_generation.py` dengan RAGAS (flow.md §10).

Tahap kedua dari dua, di environment uv tersendiri (lihat pyproject.toml di
folder ini). Empat metrik:

- faithfulness          -- apakah setiap klaim jawaban didukung konteks terambil
- answer_relevancy      -- apakah jawaban menjawab pertanyaannya
- context_precision     -- apakah chunk yang relevan berada di peringkat atas
- context_recall        -- apakah konteks memuat semua yang dibutuhkan rujukan

Hanya baris `kind == "answer"` yang dinilai. Penolakan FR-3 tidak punya jawaban
untuk dinilai kesetiaannya; jumlahnya dilaporkan terpisah, karena sistem yang
menolak semua pertanyaan sulit akan tampak "sangat setia" bila penolakan ikut
dibuang diam-diam.

Model penilai dan embedding dibaca dari api/.env (BASE_URL, API_KEY, CHAT_MODEL,
EMBED_MODEL) dan dapat ditimpa lewat argumen. Pakai penilai yang setara atau
lebih kuat dari model yang dinilai.

    cd api/eval/ragas
    uv run python run_ragas.py ../hasil/generation-20260925-101500.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

API_DIR = Path(__file__).resolve().parents[2]
USER_AGENT = "chatbot-administrasi/0.1.0"
"""Sama dengan app/rag/providers.py: gateway proyek ini memblokir UA bawaan SDK OpenAI."""


def muat_rekaman(path: Path) -> tuple[list[dict], int]:
    """Baris yang dijawab, dan jumlah baris yang tidak dijawab."""
    semua = [json.loads(b) for b in path.read_text(encoding="utf-8").splitlines() if b.strip()]
    dijawab = [r for r in semua if r.get("kind") == "answer" and r.get("retrieved_contexts")]
    return dijawab, len(semua) - len(dijawab)


def bangun_model(judge_model: str | None, embed_model: str | None):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    load_dotenv(API_DIR / ".env")
    # api/.env menyalakan tracing LangSmith untuk server; ratusan panggilan penilai
    # RAGAS tidak perlu ikut mengisi proyek LangSmith chatbot.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    base_url = os.environ.get("BASE_URL") or None
    api_key = os.environ.get("API_KEY")
    if not api_key:
        sys.exit("API_KEY kosong: isi api/.env atau set variabel lingkungan API_KEY")
    headers = {"User-Agent": USER_AGENT}

    llm = ChatOpenAI(
        model=judge_model or os.environ.get("CHAT_MODEL", "gpt-4o-mini"),
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        default_headers=headers,
    )
    emb_kwargs = {}
    if base_url:
        # Sama dengan app/rag/providers.py: kirim teks, bukan token ID tiktoken.
        emb_kwargs["check_embedding_ctx_length"] = False
    embeddings = OpenAIEmbeddings(
        model=embed_model or os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
        base_url=base_url,
        api_key=api_key,
        default_headers=headers,
        **emb_kwargs,
    )
    return llm, embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Nilai hasil PANDU dengan RAGAS")
    parser.add_argument("hasil", type=Path, help="berkas JSONL dari eval/run_generation.py")
    parser.add_argument("--judge-model", help="model penilai; default CHAT_MODEL")
    parser.add_argument(
        "--embed-model",
        help="embedding API untuk answer_relevancy; default EMBED_MODEL. Wajib diisi "
        "bila EMBED_PROVIDER=local, karena model lokal tidak tersedia lewat BASE_URL.",
    )
    args = parser.parse_args()

    # Impor metrik lewat `ragas.metrics` masih didukung evaluate() di 0.4 tetapi
    # sudah ditandai usang; peringatannya hanya mengotori keluaran.
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    rekaman, tidak_dijawab = muat_rekaman(args.hasil)
    print(f"{len(rekaman)} jawaban dinilai, {tidak_dijawab} tidak dijawab (penolakan/lainnya)")
    if not rekaman:
        sys.exit("Tidak ada jawaban untuk dinilai.")

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=r["user_input"],
                retrieved_contexts=r["retrieved_contexts"],
                response=r["response"],
                reference=r["reference"],
            )
            for r in rekaman
        ]
    )
    llm, embeddings = bangun_model(args.judge_model, args.embed_model)
    metrik = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    hasil = evaluate(
        dataset=dataset,
        metrics=metrik,
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        allow_nest_asyncio=False,
    )

    df = hasil.to_pandas()
    print("\nRata-rata (0..1, makin tinggi makin baik)")
    for m in metrik:
        if m.name in df:
            nilai = df[m.name]
            print(f"  {m.name:<38} {nilai.mean():.3f}   (n={nilai.notna().sum()})")
    print(f"  {'tingkat dijawab':<38} {len(rekaman) / (len(rekaman) + tidak_dijawab):.3f}")

    keluaran = args.hasil.with_name(args.hasil.stem + "-ragas.csv")
    df.to_csv(keluaran, index=False, encoding="utf-8")
    print(f"\nPer pertanyaan: {keluaran}")


if __name__ == "__main__":
    main()
