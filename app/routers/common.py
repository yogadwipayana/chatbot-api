"""Penerjemah galat layanan AI menjadi balasan berbahasa admin.

Dipakai setiap endpoint yang menghitung embedding saat admin menunggu: unggah
dokumen (AD-3) dan entri tanya jawab. Kalimatnya sengaja tunggal di satu tempat
-- dua salinan akan berbeda bunyi setelah suntingan pertama, dan admin yang
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


@contextmanager
def terjemahkan_galat_ai(apa: str) -> Iterator[None]:
    """Ubah kegagalan layanan embedding menjadi 502; galat lain diteruskan.

    `apa` hanya untuk log, mis. "Panduan Akademik.pdf" atau "entri tanya jawab".
    Galat yang bukan berasal dari layanan AI sengaja tidak disentuh: ia harus
    tetap menjadi 500 dan terlihat sebagai bug, bukan menyamar sebagai gangguan
    pihak lain yang "coba lagi nanti" tidak akan pernah memperbaikinya.
    """
    try:
        yield
    except EmbeddingDimensionError as exc:
        logger.error("Pemrosesan %s gagal: %s", apa, exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, MODEL_TIDAK_SESUAI) from exc
    except Exception as exc:
        if not galat_layanan_ai(exc):
            raise
        logger.exception("Layanan embedding gagal saat memproses %s", apa)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, LAYANAN_AI_MATI) from exc
