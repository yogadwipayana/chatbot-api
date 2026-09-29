"""Reranking setelah RRF (flow.md §7).

RRF hanya menggabungkan PERINGKAT dua pencarian; ia tidak pernah membaca
pertanyaan dan chunk bersamaan. Cross-encoder melakukannya: setiap pasangan
(pertanyaan, chunk) dinilai utuh, sehingga chunk yang kebetulan berbagi kata
kunci tetapi tidak menjawab turun ke bawah.

Skor reranker juga bersifat absolut (0..1), tidak seperti skor RRF. Karena itu
ia boleh menjadi dasar FR-3 bila `RERANK_THRESHOLD` diisi -- lihat `threshold.py`.

Kegagalan reranker TIDAK menggagalkan jawaban: urutan RRF dipakai apa adanya
dan galatnya dicatat. Reranker adalah perbaikan mutu, bukan syarat menjawab.

Sakelarnya `RERANK_ENABLED`. `RERANK_PROVIDER` hanya memilih bentuk endpoint
(TEI, Cohere/Jina, atau lokal); model diganti lewat URL, kunci, dan nama model.
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


TEI_MAX_BATCH = 32
"""Bawaan `MAX_CLIENT_BATCH_SIZE` TEI; lebih dari itu per permintaan dijawab 422."""


class _HttpReranker:
    """Dasar reranker lewat `POST {base_url}/rerank`."""

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

    async def _kirim(self, bodies: Sequence[dict[str, Any]]) -> list[Any]:
        """POST setiap badan berurutan lewat satu klien; JSON jawabannya berurutan sama."""
        import httpx

        from app.rag.providers import USER_AGENT

        headers = {**self._headers, "User-Agent": USER_AGENT}

        async def kirim_semua(client: Any) -> list[Any]:
            hasil = []
            for body in bodies:
                resp = await client.post(self.url, json=body, headers=headers)
                resp.raise_for_status()
                hasil.append(resp.json())
            return hasil

        if self._client is not None:
            return await kirim_semua(self._client)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await kirim_semua(client)


class ApiReranker(_HttpReranker):
    """Endpoint `/rerank` gaya Cohere/Jina (`RERANK_PROVIDER=api`).

    Permintaan `{model, query, documents, top_n}`, jawaban
    `{results: [{index, relevance_score}]}` -- bentuk yang dipakai Cohere, Jina,
    Voyage, Infinity, dan gateway yang menirunya.
    """

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        body = {
            "model": self.model,
            "query": query,
            "documents": list(texts),
            "top_n": len(texts),
        }
        (data,) = await self._kirim([body])
        return skor_dari_respons(data, len(texts))


class TeiReranker(_HttpReranker):
    """Text Embeddings Inference (`RERANK_PROVIDER=tei`), mis. container di `/rerank` server.

    Permintaan `{query, texts, raw_scores, truncate}`, jawaban daftar
    `[{index, score}]`. Satu server TEI melayani satu model, jadi nama model tidak
    dikirim. `raw_scores=false` membuat TEI memasang sigmoid sehingga skornya
    0..1 seperti bentuk Cohere; `truncate=true` memotong potongan yang melebihi
    panjang maksimum model alih-alih menolaknya. Kandidat dikirim per
    `TEI_MAX_BATCH`.
    """

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        kelompok = [
            list(texts[i : i + TEI_MAX_BATCH]) for i in range(0, len(texts), TEI_MAX_BATCH)
        ]
        bodies = [
            {"query": query, "texts": k, "raw_scores": False, "truncate": True}
            for k in kelompok
        ]
        skor: list[float] = []
        for k, data in zip(kelompok, await self._kirim(bodies), strict=True):
            skor.extend(skor_dari_respons(data, len(k)))
        return skor


def skor_dari_respons(data: Any, n: int) -> list[float]:
    """Susun ulang hasil penyedia menurut indeks masukan.

    Dua bentuk diterima: `{results: [{index, relevance_score}]}` (Cohere dan
    tiruannya) dan daftar `[{index, score}]` (TEI). Penyedia mengembalikan hasil
    terurut menurut skor, bukan menurut masukan; indeks yang tidak dikembalikan
    mendapat 0.0 -- penyedia yang memotong `top_n` sendiri berarti menilainya
    tidak relevan.
    """
    if isinstance(data, list):
        hasil = data
    else:
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
    """Reranker sesuai RERANK_PROVIDER, atau None bila RERANK_ENABLED=false."""
    if not settings.rerank_enabled:
        return None
    if settings.rerank_provider == "local":
        return LocalReranker(settings.rerank_model)
    kelas = TeiReranker if settings.rerank_provider == "tei" else ApiReranker
    kunci = settings.kunci_rerank()
    return kelas(
        base_url=settings.rerank_base_url,
        api_key=kunci.get_secret_value() if kunci else None,
        model=settings.rerank_model,
        timeout=settings.rerank_timeout_seconds,
    )


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
