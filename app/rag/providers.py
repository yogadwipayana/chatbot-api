"""Pabrik model chat dan embedding (PRD §6).

Satu endpoint OpenAI-compatible untuk keduanya, dikonfigurasi lewat empat
variabel: BASE_URL, API_KEY, CHAT_MODEL, EMBED_MODEL. Endpoint apa pun yang
meniru API OpenAI dapat dipakai -- OpenAI sendiri, gateway, atau penyedia lain.
"""

from __future__ import annotations

from typing import Any

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
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return ChatOpenAI(**kwargs)


def build_embeddings(settings: Settings) -> Any:
    """Bangun model embedding dari EMBED_MODEL.

    Peringatan (PRD §10): mengganti model embedding setelah ingestion berarti
    re-index seluruh dokumen. Uji kualitas bahasa Indonesia dulu.
    """
    from langchain_openai import OpenAIEmbeddings

    kwargs: dict[str, Any] = {
        "model": settings.embed_model,
        "dimensions": EMBEDDING_DIM,
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
    return OpenAIEmbeddings(**kwargs)
