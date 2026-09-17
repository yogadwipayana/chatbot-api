"""Umpan balik mahasiswa atas jawaban chatbot (FE-5), dibaca dashboard.

Tabel `feedback` sebelumnya hanya masuk statistik sebagai satu angka rasio
(AD-5). Angka itu memberi tahu ada yang salah, tidak memberi tahu apanya:
jawaban mana yang ditandai tidak membantu, dan pertanyaan apa yang
mendahuluinya. Modul ini menyajikan barisnya apa adanya beserta pertanyaan
yang bersangkutan, supaya jawaban buruk dapat ditelusuri ke dokumen sumbernya.

Yang tampil adalah isi `messages`, jadi pertanyaan sensitif (FR-7) muncul
sebagai penanda tetap yang disimpan chatlog -- bukan curahan hati mahasiswa.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_PERTANYAAN_SUBQUERY = """
    LEFT JOIN LATERAL (
        SELECT u.konten
        FROM messages u
        WHERE u.conversation_id = m.conversation_id
          AND u.role = 'user'
          AND u.created_at <= m.created_at
        ORDER BY u.created_at DESC
        LIMIT 1
    ) q ON true
"""
"""Pertanyaan yang memicu jawaban ini.

`messages` tidak menyimpan penunjuk dari jawaban ke pertanyaannya, jadi
pasangannya dicari dari urutan waktu dalam percakapan yang sama. Inilah
alasan `chatlog` memakai `clock_timestamp()`: dengan `now()` seluruh baris satu
transaksi bercap waktu identik dan pasangan ini tidak dapat ditentukan.
"""


def _kondisi(helpful: bool | None, sejak: date | None, timezone: str) -> tuple[str, dict]:
    kondisi: list[str] = []
    params: dict[str, Any] = {}
    if helpful is not None:
        kondisi.append("f.helpful = :helpful")
        params["helpful"] = helpful
    if sejak is not None:
        kondisi.append("(f.created_at AT TIME ZONE :tz)::date >= :sejak")
        params.update(tz=timezone, sejak=sejak)
    return ("WHERE " + " AND ".join(kondisi)) if kondisi else "", params


async def fetch_feedback(
    session: AsyncSession,
    *,
    helpful: bool | None,
    sejak: date | None,
    timezone: str,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    where, params = _kondisi(helpful, sejak, timezone)
    rows = await session.execute(
        text(
            "SELECT f.id::text AS id, f.message_id::text AS message_id, f.helpful,"
            " f.catatan, f.created_at, m.konten AS jawaban, m.meta->>'kind' AS kind,"
            " m.top_score, q.konten AS pertanyaan"
            " FROM feedback f"
            " JOIN messages m ON m.id = f.message_id"
            f" {_PERTANYAAN_SUBQUERY} {where}"
            " ORDER BY f.created_at DESC LIMIT :limit OFFSET :offset"
        ),
        {**params, "limit": limit, "offset": offset},
    )

    # Rekap dihitung tanpa filter `helpful` supaya jumlah pada kedua tab tetap
    # terlihat saat salah satunya sedang dipilih.
    rekap_where, rekap_params = _kondisi(None, sejak, timezone)
    rekap = (
        (
            await session.execute(
                text(
                    "SELECT count(*) FILTER (WHERE f.helpful) AS positif,"
                    " count(*) FILTER (WHERE NOT f.helpful) AS negatif"
                    f" FROM feedback f {rekap_where}"
                ),
                rekap_params,
            )
        )
        .mappings()
        .one()
    )
    positif, negatif = rekap["positif"], rekap["negatif"]
    if helpful is None:
        total = positif + negatif
    else:
        total = positif if helpful else negatif

    return {
        "items": [dict(row) for row in rows.mappings()],
        "total": total,
        "jumlah_positif": positif,
        "jumlah_negatif": negatif,
    }
