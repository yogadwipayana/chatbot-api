"""Riwayat percakapan dari klien: giliran asisten dicocokkan ke database.

Portal mengirim riwayat yang terlihat di layar (FR-4), termasuk jawaban bot.
Tanpa pemeriksaan, siapa pun dapat menulis "jawaban bot" palsu di riwayat dan
mengarahkan penulisan ulang query serta jawaban dengan instruksi yang tampak
datang dari asisten sendiri.

Riwayat sengaja TIDAK diambil seluruhnya dari database berdasarkan
`session_id`. `session_id` bertahan berbulan-bulan di localStorage, juga di
komputer lab yang dipakai bergantian: mahasiswa berikutnya akan mewarisi
percakapan mahasiswa sebelumnya sebagai konteks yang tidak terlihat, dan
setelah reload server mengingat percakapan yang sudah hilang dari layar.

Jadi yang dipakai tetap riwayat di layar, tetapi setiap giliran asisten harus
cocok dengan jawaban yang benar-benar pernah dikirim ke sesi itu (tabel
`messages`). Giliran pengguna diterima apa adanya: isinya memang ketikan
pengguna sendiri, sama seperti pertanyaannya.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.rewriter import Turn
from app.schemas.chat import MAKS_KONTEN_RIWAYAT, TurnIn

logger = logging.getLogger(__name__)

JAWABAN_DIPERIKSA = 20
"""Jawaban terakhir sesi yang dicocokkan. Portal mengirim paling banyak tiga
giliran; sisanya ruang untuk beberapa tab yang berbagi `session_id`."""

_JAWABAN_TERAKHIR_SQL = text(
    """
    SELECT m.content
    FROM messages m
    JOIN conversations c ON c.id = m.conversation_id
    WHERE c.session_id = :session_id AND m.role = 'assistant'
    ORDER BY m.created_at DESC
    LIMIT :batas
    """
)


class JawabanTerkirim(Protocol):
    async def terakhir(self, session_id: str, batas: int) -> list[str]: ...


class SqlJawabanTerkirim:
    """Jawaban yang tercatat untuk satu `session_id`, terbaru lebih dulu."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def terakhir(self, session_id: str, batas: int) -> list[str]:
        try:
            hasil = await self.session.execute(
                _JAWABAN_TERAKHIR_SQL, {"session_id": session_id, "batas": batas}
            )
        except Exception:
            await self.session.rollback()
            raise
        return list(hasil.scalars())


async def riwayat_tepercaya(
    history: Sequence[TurnIn], session_id: str, store: JawabanTerkirim
) -> list[Turn]:
    """Riwayat klien tanpa giliran asisten yang tidak pernah dikirim ke sesi ini.

    Database hanya ditanya bila riwayat memuat giliran asisten, jadi pesan
    pertama tidak membayar query tambahan. Giliran yang tidak cocok dibuang,
    bukan ditolak: penyebab sahnya ada (pencatatan jawaban itu gagal), dan
    pertanyaan tetap dijawab, hanya tanpa konteks itu. Database yang tidak
    dapat dibaca diperlakukan sama -- semua giliran asisten dibuang.
    """
    if not any(t.role == "assistant" for t in history):
        return [Turn(t.role, t.content) for t in history]

    try:
        tercatat = await store.terakhir(session_id, JAWABAN_DIPERIKSA)
    except Exception:
        logger.exception(
            "Jawaban tercatat sesi ini tidak dapat dibaca; riwayat asisten diabaikan"
        )
        tercatat = []
    # Riwayat klien sudah dipotong sepanjang ini (`TurnIn`); potong juga
    # pembandingnya supaya jawaban panjang tetap cocok.
    terkirim = {isi[:MAKS_KONTEN_RIWAYAT] for isi in tercatat}

    hasil: list[Turn] = []
    dibuang = 0
    for t in history:
        if t.role == "assistant" and t.content not in terkirim:
            dibuang += 1
            continue
        hasil.append(Turn(t.role, t.content))
    if dibuang:
        logger.warning(
            "%d giliran asisten di riwayat klien tidak pernah dikirim ke sesi ini; diabaikan",
            dibuang,
        )
    return hasil
