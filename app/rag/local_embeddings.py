"""Embedding lokal lewat sentence-transformers, mis. multilingual-e5-small (flow.md §2).

Dua hal yang membuat model e5 berbeda dari embedding API:

1. **Awalan wajib.** e5 dilatih dengan `query: ` di depan pertanyaan dan
   `passage: ` di depan dokumen. Tanpa awalan itu kualitas retrieval turun
   nyata -- dan tidak ada galat yang memberi tahu. Awalan dipasang otomatis
   untuk setiap model yang namanya mengandung `e5`.
2. **Dimensi.** e5-small menghasilkan 384 dimensi, kolom `chunks.embedding`
   berukuran 1024. Vektor dinormalisasi lalu diisi nol di ekornya: hasil kali
   titik dan norma tidak berubah oleh nol tambahan, jadi cosine similarity --
   satu-satunya yang dibaca retriever -- identik dengan vektor 384 aslinya.
   Skema database dan index HNSW tidak perlu disentuh; yang wajib hanya
   re-index (`python -m scripts.reindex_embeddings`), karena vektor dari dua
   model berbeda tidak dapat dibandingkan.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def pakai_awalan_e5(model: str) -> bool:
    return "e5" in model.lower()


def pad(vector: Any, dim: int) -> list[float]:
    """Isi nol sampai `dim`. Vektor yang lebih panjang dari kolomnya ditolak."""
    nilai = [float(x) for x in vector]
    if len(nilai) > dim:
        raise ValueError(
            f"model menghasilkan {len(nilai)} dimensi, melebihi kolom chunks.embedding ({dim})"
        )
    return nilai + [0.0] * (dim - len(nilai))


class LocalEmbeddings(Embeddings):
    """`Embeddings` LangChain di atas SentenceTransformer, dengan awalan dan padding."""

    lokal = True
    """Dibaca `EmbedQuery`: biaya embedding lokal nol, bukan "tidak diketahui"."""

    def __init__(self, model: str, dim: int, *, encoder: Any = None) -> None:
        self.model = model
        self.dim = dim
        self._encoder = encoder
        """Disuntikkan di test; default dimuat sekali per proses."""
        self._awalan = pakai_awalan_e5(model)

    @property
    def encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        return _sentence_transformer(self.model)

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        if self._awalan:
            texts = [prefix + t for t in texts]
        vektor = self.encoder.encode(texts, normalize_embeddings=True)
        return [pad(v, self.dim) for v in vektor]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(list(texts), PASSAGE_PREFIX)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], QUERY_PREFIX)[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        # encode() memakan CPU; di thread supaya event loop tetap melayani
        # permintaan lain selama dokumen di-embed.
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


@lru_cache(maxsize=2)
def _sentence_transformer(model: str) -> Any:
    """Satu salinan model per proses; `build_embeddings` dipanggil per permintaan."""
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:  # pragma: no cover - tergantung instalasi
        raise RuntimeError(
            "EMBED_PROVIDER=local butuh sentence-transformers: uv sync --extra local"
        ) from exc
    return SentenceTransformer(model)
