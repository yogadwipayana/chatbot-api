"""Pemuatan set evaluasi (PRD §14, §16).

Format JSONL, satu kasus per baris:
    {"question": "kapan pengisian KRS?", "relevant_chunk_ids": ["uuid", "uuid"]}

Set evaluasi WAJIB divalidasi staf akademik, bukan hanya oleh pengembang
(kriteria kelayakan rilis §14). Pemuat ini gagal keras pada baris yang cacat --
kasus tanpa jawaban benar hanya menurunkan skor secara palsu.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.metrics import EvalCase


def load_jsonl(path: str | Path) -> list[EvalCase]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"set evaluasi tidak ditemukan: {path}")

    kasus: list[EvalCase] = []
    for nomor, baris in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        baris = baris.strip()
        if not baris or baris.startswith("//"):
            continue
        try:
            data = json.loads(baris)
            kasus.append(
                EvalCase(
                    question=data["question"],
                    relevant_chunk_ids=frozenset(data["relevant_chunk_ids"]),
                )
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise ValueError(f"{path}:{nomor} tidak valid -- {exc}") from exc

    if not kasus:
        raise ValueError(f"{path} tidak berisi satu kasus pun")
    return kasus
