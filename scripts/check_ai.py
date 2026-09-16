"""Verifikasi BASE_URL, API_KEY, CHAT_MODEL, dan EMBED_MODEL terhadap endpoint sungguhan.

Mengirim satu prompt pendek ke CHAT_MODEL dan satu kalimat ke EMBED_MODEL --
biayanya hanya beberapa token. Unit test tidak bisa menangkap masalah di sini:
kunci salah, model tidak dikenal endpoint, atau model yang menolak parameter
yang kita kirim.

    python -m scripts.check_ai
"""

from __future__ import annotations

import asyncio
import sys
import time
from urllib.parse import urlsplit

from app.config import Settings, get_settings
from app.db.models import EMBEDDING_DIM
from app.rag.providers import build_embeddings, build_llm


def baris(label: str, nilai: object) -> None:
    print(f"  {label:<12} {nilai}")


def _ringkas(exc: Exception, settings: Settings) -> str:
    """Pesan galat singkat, dengan API_KEY disamarkan bila ikut tergema."""
    pesan = str(exc)
    kunci = settings.kunci_api()
    if kunci is not None:
        pesan = pesan.replace(kunci.get_secret_value(), "***")
    return pesan[:300]


def _petunjuk(pesan: str, variabel_model: str) -> str:
    rendah = pesan.lower()
    if "blocked" in rendah:
        return (
            "  Petunjuk: permintaan ditolak firewall di depan BASE_URL (mis. Cloudflare "
            "WAF), bukan karena API_KEY. Izinkan lalu lintas API di aturan firewall."
        )
    if "401" in pesan or "unauthorized" in rendah or "api key" in rendah:
        return "  Petunjuk: API_KEY salah, kedaluwarsa, atau bukan untuk BASE_URL ini."
    if "403" in pesan or "forbidden" in rendah or "permission" in rendah:
        return "  Petunjuk: API_KEY tidak punya akses ke model ini di endpoint tersebut."
    if "temperature" in rendah:
        return "  Petunjuk: model menolak temperature=0 (umum pada model reasoning)."
    if "dimensions" in rendah:
        return (
            f"  Petunjuk: endpoint menolak parameter dimensions; pilih {variabel_model} "
            f"yang berdimensi bawaan {EMBEDDING_DIM}."
        )
    if "404" in pesan or "not found" in rendah or "does not exist" in rendah:
        return (
            f"  Petunjuk: {variabel_model} tidak dikenal endpoint ini, "
            "atau path BASE_URL keliru (umumnya diakhiri /v1)."
        )
    if "429" in pesan or "quota" in rendah or "rate limit" in rendah:
        return "  Petunjuk: kuota habis atau kena rate limit di penyedia."
    if "connect" in rendah or "timeout" in rendah or "resolve" in rendah:
        return "  Petunjuk: BASE_URL tidak terjangkau dari mesin ini."
    return ""


async def cek_chat(settings: Settings) -> bool:
    print(f"\nChat ({settings.chat_model})")
    try:
        llm = build_llm(settings, streaming=False)
        mulai = time.perf_counter()
        hasil = await llm.ainvoke("Balas hanya dengan satu kata: OK")
        durasi = (time.perf_counter() - mulai) * 1000
    except Exception as exc:
        pesan = _ringkas(exc, settings)
        print(f"  GAGAL  {type(exc).__name__}: {pesan}")
        print(_petunjuk(pesan, "CHAT_MODEL"))
        return False

    teks = hasil.content if isinstance(hasil.content, str) else str(hasil.content)
    baris("balasan", repr(teks.strip()[:60]))
    baris("latency", f"{durasi:.0f} ms")
    return True


async def cek_embedding(settings: Settings) -> bool:
    print(f"\nEmbedding ({settings.embed_model})")
    try:
        emb = build_embeddings(settings)
        vektor = await emb.aembed_query("Kapan batas akhir pengisian KRS semester ganjil?")
    except Exception as exc:
        pesan = _ringkas(exc, settings)
        print(f"  GAGAL  {type(exc).__name__}: {pesan}")
        print(_petunjuk(pesan, "EMBED_MODEL"))
        return False

    baris("dimensi", len(vektor))
    if len(vektor) != EMBEDDING_DIM:
        print(
            f"  GAGAL  kolom database vector({EMBEDDING_DIM}); vektor {len(vektor)} "
            "dimensi akan ditolak saat ingestion."
        )
        return False
    return True


async def jalankan() -> int:
    settings = get_settings()
    if settings.base_url:
        p = urlsplit(settings.base_url)
        endpoint = f"{p.scheme}://{p.hostname}{p.path}"
    else:
        endpoint = "(OpenAI resmi)"
    print("Konfigurasi")
    baris("endpoint", endpoint)
    baris("API_KEY", "terisi" if settings.kunci_api() else "KOSONG")

    chat_ok = await cek_chat(settings)
    embed_ok = await cek_embedding(settings)

    print()
    if chat_ok and embed_ok:
        print("Model AI siap dipakai.")
        return 0
    print("Ada yang gagal; lihat petunjuk di atas.")
    return 1


def main() -> int:
    return asyncio.run(jalankan())


if __name__ == "__main__":
    sys.exit(main())
