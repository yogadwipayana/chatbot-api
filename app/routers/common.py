"""Penerjemah galat layanan AI menjadi balasan berbahasa admin.

Dipakai setiap endpoint yang memanggil layanan AI saat admin menunggu: unggah
dokumen (AD-3), entri tanya jawab, dan uji coba jawaban (AD-6). Kalimatnya
sengaja tunggal di satu tempat -- dua salinan akan berbeda bunyi setelah
suntingan pertama, dan admin yang
membaca "coba lagi" di satu layar lalu jargon teknis di layar lain akan
menyimpulkan yang satu lebih serius daripada yang lain padahal sama saja.

PRD §9: pesan galat untuk admin harus non-teknis. Rincian teknisnya hanya masuk
log server.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import HTTPException, status

from app.ingestion.embedder import EmbeddingDimensionError, galat_layanan_ai

logger = logging.getLogger(__name__)

MODEL_TIDAK_SESUAI = (
    "Dokumen tidak dapat diproses karena pengaturan model AI tidak sesuai. "
    "Hubungi pengelola teknis."
)
LAYANAN_AI_MATI = (
    "Layanan AI untuk memproses dokumen sedang tidak dapat dihubungi. "
    "Coba lagi beberapa saat lagi."
)
LAYANAN_AI_BERMASALAH = (
    "Layanan AI sedang bermasalah, jadi jawaban uji coba tidak dapat disusun. "
    "Coba lagi beberapa saat lagi."
)


@contextmanager
def terjemahkan_galat_ai(apa: str, *, pesan: str = LAYANAN_AI_MATI) -> Iterator[None]:
    """Ubah kegagalan layanan AI menjadi 502 berisi `pesan`; galat lain diteruskan.

    `apa` hanya untuk log, mis. "Panduan Akademik.pdf" atau "entri tanya jawab".
    Galat yang bukan berasal dari layanan AI sengaja tidak disentuh: ia harus
    tetap menjadi 500 dan terlihat sebagai bug, bukan menyamar sebagai gangguan
    pihak lain yang "coba lagi nanti" tidak akan pernah memperbaikinya.

    Tanpa terjemahan ini galat layanan AI juga menjadi 500, dan balasan 500
    tidak membawa header CORS: dashboard admin melihatnya sebagai koneksi
    putus ("Tidak dapat terhubung ke server"), bukan layanan AI yang bermasalah.
    """
    try:
        yield
    except EmbeddingDimensionError as exc:
        logger.error("Pemrosesan %s gagal: %s", apa, exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, MODEL_TIDAK_SESUAI) from exc
    except Exception as exc:
        if not galat_layanan_ai(exc):
            raise
        logger.exception("Layanan AI gagal saat memproses %s", apa)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, pesan) from exc
