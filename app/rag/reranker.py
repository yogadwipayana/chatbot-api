"""Reranking setelah RRF (flow.md §7).

RRF hanya menggabungkan PERINGKAT dua pencarian; ia tidak pernah membaca
pertanyaan dan chunk bersamaan. Cross-encoder melakukannya: setiap pasangan
(pertanyaan, chunk) dinilai utuh, sehingga chunk yang kebetulan berbagi kata
kunci tetapi tidak menjawab turun ke bawah.

Skor reranker juga bersifat absolut (0..1), tidak seperti skor RRF. Karena itu
ia boleh menjadi dasar FR-3 bila `RERANK_THRESHOLD` diisi -- lihat `threshold.py`.

Kegagalan reranker TIDAK menggagalkan jawaban: urutan RRF dipakai apa adanya
dan galatnya dicatat. Reranker adalah perbaikan mutu, bukan syarat menjawab.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

logger = logging.getLogger(__name__)

RERANK_SCORE_KEY = "rerank_score"
"""Kunci di `Document.metadata`. Absen berarti reranker mati atau gagal."""


class Reranker(Protocol):
    model: str

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Skor relevansi 0..1 per teks, urutan sama dengan `texts`."""
        ...


class ApiReranker:
    """Endpoint `POST {base_url}/rerank` gaya Cohere/Jina.

    Permintaan `{model, query, documents, top_n}`, jawaban
    `{results: [{index, relevance_score}]}` -- bentuk yang dipakai Cohere, Jina,
    Voyage, dan gateway yang menirunya.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 10.0,
        client: Any = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/rerank"
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout
        self._client = client
        """Disuntikkan di test (`httpx.AsyncClient` dengan MockTransport)."""

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        import httpx

        from app.rag.providers import USER_AGENT

        body = {
            "model": self.model,
            "query": query,
            "documents": list(texts),
            "top_n": len(texts),
        }
        headers = {**self._headers, "User-Agent": USER_AGENT}
        if self._client is not None:
            resp = await self._client.post(self.url, json=body, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self.url, json=body, headers=headers)
        resp.raise_for_status()
        return skor_dari_respons(resp.json(), len(texts))


def skor_dari_respons(data: Any, n: int) -> list[float]:
    """Susun ulang `results` penyedia menurut indeks masukan.

    Penyedia mengembalikan hasil terurut menurut skor, bukan menurut masukan;
    indeks yang tidak dikembalikan mendapat 0.0 -- penyedia yang memotong
    `top_n` sendiri berarti menilainya tidak relevan.
    """
    hasil = data.get("results") if isinstance(data, dict) else None
    if not isinstance(hasil, list):
        raise ValueError("respons rerank tanpa daftar `results`")
    skor = [0.0] * n
    for item in hasil:
        indeks = int(item["index"])
        if not 0 <= indeks < n:
            raise ValueError(f"indeks rerank di luar jangkauan: {indeks}")
        skor[indeks] = float(item.get("relevance_score", item.get("score", 0.0)))
    return skor


class LocalReranker:
    """Cross-encoder sentence-transformers di server ini (`uv sync --extra local`)."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        encoder = _cross_encoder(self.model)
        pasangan = [(query, t) for t in texts]
        # predict() memakan CPU; di thread supaya event loop tetap melayani
        # permintaan lain selama menunggu.
        nilai = await asyncio.to_thread(encoder.predict, pasangan)
        return [float(x) for x in nilai]


@lru_cache(maxsize=2)
def _cross_encoder(model: str) -> Any:
    """Satu salinan model per proses; memuat model memakan detik, bukan milidetik."""
    try:
        from sentence_transformers import CrossEncoder
    except ModuleNotFoundError as exc:  # pragma: no cover - tergantung instalasi
        raise RuntimeError(
            "RERANK_PROVIDER=local butuh sentence-transformers: uv sync --extra local"
        ) from exc
    # CrossEncoder dengan satu label memakai aktivasi sigmoid secara bawaan,
    # jadi skornya sudah 0..1 dan setara dengan skor reranker API.
    return CrossEncoder(model)


def build_reranker(settings: Any) -> Reranker | None:
    """Reranker sesuai RERANK_PROVIDER, atau None bila dimatikan."""
    if settings.rerank_provider == "api":
        kunci = settings.kunci_rerank()
        return ApiReranker(
            base_url=settings.url_rerank(),
            api_key=kunci.get_secret_value() if kunci else None,
            model=settings.rerank_model,
            timeout=settings.rerank_timeout_seconds,
        )
    if settings.rerank_provider == "local":
        return LocalReranker(settings.rerank_model)
    return None


async def rerank_documents(
    query: str,
    documents: Sequence[Any],
    reranker: Reranker | None,
    *,
    top_n: int,
) -> list[Any]:
    """Urutkan ulang `documents` menurut reranker lalu potong ke `top_n`.

    Tanpa reranker, atau bila reranker gagal, urutan RRF dipertahankan. Urutan
    untuk skor yang sama mengikuti urutan RRF (sort stabil), supaya hasil
    evaluasi dapat direproduksi.
    """
    if reranker is None or not documents:
        return list(documents)[:top_n]
    try:
        skor = await reranker.score(query, [d.page_content for d in documents])
    except Exception:
        logger.warning("Reranker %s gagal; memakai urutan RRF", reranker.model, exc_info=True)
        return list(documents)[:top_n]
    if len(skor) != len(documents):
        logger.warning(
            "Reranker mengembalikan %d skor untuk %d dokumen", len(skor), len(documents)
        )
        return list(documents)[:top_n]

    for doc, nilai in zip(documents, skor, strict=True):
        doc.metadata[RERANK_SCORE_KEY] = nilai
    terurut = sorted(documents, key=lambda d: -d.metadata[RERANK_SCORE_KEY])
    return terurut[:top_n]
