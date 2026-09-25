"""Pabrik model chat dan embedding (PRD §6).

Satu endpoint OpenAI-compatible untuk keduanya, dikonfigurasi lewat empat
variabel: BASE_URL, API_KEY, CHAT_MODEL, EMBED_MODEL. Endpoint apa pun yang
meniru API OpenAI dapat dipakai -- OpenAI sendiri, gateway, atau penyedia lain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar

from app.config import Settings
from app.db.models import EMBEDDING_DIM

USER_AGENT = "chatbot-administrasi/0.1.0"
"""User-Agent untuk semua permintaan ke BASE_URL, menggantikan bawaan SDK
`OpenAI/Python x.y.z`.

Firewall di depan sebagian gateway memblokir UA bawaan SDK itu secara spesifik.
Terbukti pada gateway proyek ini: UA `OpenAI/Python` mendapat 403 "Your request
was blocked." dari Cloudflare, sementara UA lain -- termasuk yang ini -- lolos.
UA yang jujur juga membuat permintaan dari sistem ini mudah dikenali di log
gateway, tidak seperti menyamar sebagai peramban.
"""

_HEADERS = {"User-Agent": USER_AGENT}


def _kunci(settings: Settings) -> str:
    kunci = settings.kunci_api()
    if kunci is None:
        raise ValueError("API_KEY belum diisi")
    return kunci.get_secret_value()


def build_llm(settings: Settings, *, streaming: bool = True) -> Any:
    """Bangun chat model dari CHAT_MODEL."""
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": settings.chat_model,
        "api_key": _kunci(settings),
        "streaming": streaming,
        "temperature": 0,
        "default_headers": dict(_HEADERS),
    }
    if streaming:
        # langchain-openai menyalakan `stream_usage` sendiri HANYA untuk endpoint
        # resmi OpenAI: begitu `base_url` terisi, defaultnya mati karena banyak
        # endpoint tiruan tidak mendukung `stream_options`. Tanpa nilai eksplisit
        # di sini, setiap jawaban yang dialirkan kehilangan `usage_metadata` --
        # dan biaya AD-5 diam-diam menjadi null untuk seluruh chat mahasiswa.
        kwargs["stream_usage"] = settings.llm_stream_usage
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return ChatOpenAI(**kwargs)


def build_embeddings(settings: Settings) -> Any:
    """Bangun model embedding dari EMBED_MODEL.

    Peringatan (PRD §10): mengganti model embedding setelah ingestion berarti
    re-index seluruh dokumen. Uji kualitas bahasa Indonesia dulu.

    `EMBED_PROVIDER=local` memuat EMBED_MODEL lewat sentence-transformers di
    server ini (`app.rag.local_embeddings`), tanpa BASE_URL maupun API_KEY.
    """
    if settings.embed_provider == "local":
        from app.rag.local_embeddings import LocalEmbeddings

        return LocalEmbeddings(settings.embed_model, EMBEDDING_DIM)

    from langchain_openai import OpenAIEmbeddings

    from app.rag.local_embeddings import pakai_awalan_e5

    e5 = pakai_awalan_e5(settings.embed_model)
    kwargs: dict[str, Any] = {
        "model": settings.embed_model,
        # e5 berdimensi tetap dan tidak mengenal `dimensions` (fitur Matryoshka
        # text-embedding-3); vektornya diisi nol sampai kolom oleh `E5ApiEmbeddings`.
        "dimensions": None if e5 else EMBEDDING_DIM,
        "api_key": _kunci(settings),
        "default_headers": dict(_HEADERS),
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
        # Default `check_embedding_ctx_length=True` men-tokenisasi teks dengan
        # tiktoken lalu mengirim *token ID*, bukan teks. Endpoint resmi OpenAI
        # menerimanya; banyak endpoint OpenAI-compatible tidak. Dokumentasi
        # langchain-openai sendiri menyarankan False untuk penyedia lain.
        # Yang hilang hanya pemecahan otomatis teks di atas batas konteks --
        # chunk kita ~700 token (FR-1), jauh di bawah batas itu.
        kwargs["check_embedding_ctx_length"] = False
    if e5:
        return _e5_api_class()(**kwargs)
    return OpenAIEmbeddings(**kwargs)


@lru_cache(maxsize=1)
def _e5_api_class() -> type:
    """Kelas dibuat saat dibutuhkan, supaya langchain-openai tetap diimpor malas."""
    from langchain_openai import OpenAIEmbeddings

    from app.rag.local_embeddings import PASSAGE_PREFIX, QUERY_PREFIX, pad

    class E5ApiEmbeddings(OpenAIEmbeddings):
        """multilingual-e5 lewat endpoint OpenAI-compatible (`EMBED_PROVIDER=api`).

        Dua hal yang dilakukan `LocalEmbeddings` untuk e5 lokal, diulang di sini
        karena endpoint-nya hanya menerima teks dan mengembalikan vektor apa adanya:

        - awalan `query: ` / `passage: `. Tanpa itu mutu retrieval e5 turun tanpa
          pesan galat, dan endpoint tidak memasangnya sendiri;
        - isi nol sampai `EMBEDDING_DIM` (384 untuk e5-small). Cosine similarity
          tidak berubah, jadi kolom `chunks.embedding` tidak perlu dimigrasi.

        `aembed_query` bawaan meneruskan ke `aembed_documents`, yang di sini
        memasang awalan passage -- karena itu keduanya ditimpa terpisah.
        """

        awalan_e5: ClassVar[bool] = True

        def embed_documents(self, texts, chunk_size=None, **kwargs):  # type: ignore[override]
            vektor = super().embed_documents(
                [PASSAGE_PREFIX + t for t in texts], chunk_size, **kwargs
            )
            return [pad(v, EMBEDDING_DIM) for v in vektor]

        async def aembed_documents(self, texts, chunk_size=None, **kwargs):  # type: ignore[override]
            vektor = await super().aembed_documents(
                [PASSAGE_PREFIX + t for t in texts], chunk_size, **kwargs
            )
            return [pad(v, EMBEDDING_DIM) for v in vektor]

        def embed_query(self, text, **kwargs):  # type: ignore[override]
            vektor = super().embed_documents([QUERY_PREFIX + text], **kwargs)[0]
            return pad(vektor, EMBEDDING_DIM)

        async def aembed_query(self, text, **kwargs):  # type: ignore[override]
            vektor = (await super().aembed_documents([QUERY_PREFIX + text], **kwargs))[0]
            return pad(vektor, EMBEDDING_DIM)

    return E5ApiEmbeddings


@dataclass(frozen=True)
class EmbedResult:
    """Vektor beserta laporan pemakaian dari satu panggilan embedding."""

    vectors: list[list[float]]
    tokens: int | None
    """`usage.prompt_tokens`. None bila endpoint tidak melaporkannya."""
    biaya_usd: float | None
    """`usage.cost` -- biaya sebenarnya menurut penyedia, bukan taksiran kita.

    Bukan bagian spesifikasi OpenAI: gateway seperti OpenRouter menambahkannya,
    endpoint lain tidak. None berarti "tidak dilaporkan", bukan "gratis" --
    pemanggil jatuh ke `costs.biaya_embedding` untuk menaksirnya sendiri.
    """
    model: str | None
    """Nama model menurut respons. Boleh berbeda dari yang diminta: gateway
    proyek ini menjawab `text-embedding-3-small` untuk permintaan
    `openrouter/openai/text-embedding-3-small`."""
    is_byok: bool | None
    """True bila kunci API penyedia hulu milik sendiri. Saat itu `cost` berarti
    ongkos platform, bukan belanja ke penyedia hulu -- dua angka yang tidak
    boleh dijumlahkan begitu saja."""


async def embed_with_usage(
    embeddings: Any, texts: Sequence[str], *, sebagai_query: bool = False
) -> EmbedResult:
    """Embed `texts` dalam SATU panggilan, sekalian membaca laporan pemakaiannya.

    `sebagai_query=True` untuk pertanyaan mahasiswa. Embedding API tidak
    membedakan keduanya, tetapi model e5 lokal memberi awalan berbeda untuk
    query dan dokumen -- memperlakukan pertanyaan sebagai dokumen diam-diam
    menurunkan mutu retrieval.

    `aembed_documents`/`aembed_query` hanya mengembalikan vektor: antarmuka
    `Embeddings` LangChain tidak punya kanal usage seperti `usage_metadata` milik
    chat model, sehingga `response["usage"]` -- termasuk jumlah token dan biaya
    yang dilaporkan penyedia -- dibuang begitu saja. Fungsi ini memanggil klien
    yang sama persis seperti langchain-openai, lalu menyimpan bagian yang dibuang.

    `texts` tidak dipecah menjadi beberapa batch: satu panggilan menghasilkan
    satu laporan pemakaian, dan memecahnya di sini akan membuat angka yang
    dikembalikan hanya mewakili batch terakhir. Pemanggil yang mengatur ukuran
    batch -- lihat `app.ingestion.embedder.BATCH_SIZE`.
    """
    if getattr(embeddings, "lokal", False):
        # Model di server sendiri: tidak ada tagihan, jadi biayanya nol -- bukan
        # None, yang di AD-5 berarti "berbiaya tetapi tidak terhitung".
        if sebagai_query:
            vektor = [await embeddings.aembed_query(t) for t in texts]
        else:
            vektor = await embeddings.aembed_documents(list(texts))
        return EmbedResult(vektor, None, 0.0, embeddings.model, None)

    if getattr(embeddings, "check_embedding_ctx_length", True):
        # Jalur "len-safe" langchain-openai men-tokenisasi dengan tiktoken lalu
        # mengirim token ID, dan memecah sendiri teks yang melebihi batas konteks.
        # Menirunya di sini berarti menyalin logika itu; kehilangan angka biaya
        # lebih murah daripada dua implementasi yang bisa berbeda diam-diam.
        # `build_embeddings` mematikannya untuk setiap BASE_URL, jadi jalur ini
        # hanya terpakai pada endpoint resmi OpenAI.
        vektor = await embeddings.aembed_documents(list(texts))
        return EmbedResult(vektor, None, None, None, None)

    params: dict[str, Any] = {"model": embeddings.model}
    if embeddings.dimensions is not None:
        params["dimensions"] = embeddings.dimensions
    e5 = getattr(embeddings, "awalan_e5", False)
    if e5:
        # Klien dipanggil langsung, melewati `E5ApiEmbeddings.aembed_*`: awalan
        # dan padding-nya harus dipasang di sini juga.
        from app.rag.local_embeddings import PASSAGE_PREFIX, QUERY_PREFIX

        awalan = QUERY_PREFIX if sebagai_query else PASSAGE_PREFIX
        texts = [awalan + t for t in texts]
    response = await embeddings.async_client.create(input=list(texts), **params)
    if not isinstance(response, dict):
        response = response.model_dump()

    usage = response.get("usage") or {}
    vektor = [r["embedding"] for r in response["data"]]
    if e5:
        from app.rag.local_embeddings import pad

        vektor = [pad(v, EMBEDDING_DIM) for v in vektor]
    return EmbedResult(
        vectors=vektor,
        tokens=_bilangan(usage.get("prompt_tokens", usage.get("total_tokens")), int),
        biaya_usd=_bilangan(usage.get("cost"), float),
        model=response.get("model") or None,
        is_byok=usage.get("is_byok") if isinstance(usage.get("is_byok"), bool) else None,
    )


def _bilangan(nilai: Any, tipe: type) -> Any:
    """Angka dari respons penyedia, atau None bila bentuknya tidak terduga.

    Laporan pemakaian adalah tambahan di luar spesifikasi OpenAI, jadi bentuknya
    tidak dijamin. Nilai yang tidak dapat dibaca harus menjadi None -- "tidak
    tahu" -- bukan menggagalkan jawaban yang sudah terlanjur benar.
    """
    if isinstance(nilai, bool) or nilai is None:
        return None
    try:
        return tipe(nilai)
    except (TypeError, ValueError):
        return None
