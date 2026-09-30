"""Query pertanyaan tak terjawab (AD-4)."""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.grouping import UnansweredItem

MAX_ROWS = 5000
"""Pengelompokan berjalan di Python atas baris terbaru. Pada beban PRD §11
(500 pertanyaan/hari, <15% ditolak) ini mencakup berbulan-bulan data."""


async def fetch_items(
    session: AsyncSession,
    *,
    resolved: bool | None,
    sejak: date | None,
    timezone: str,
) -> list[UnansweredItem]:
    kondisi: list[str] = []
    params: dict[str, object] = {"batas": MAX_ROWS}
    if resolved is not None:
        kondisi.append("u.resolved = :resolved")
        params["resolved"] = resolved
    if sejak is not None:
        kondisi.append("(u.created_at AT TIME ZONE :tz)::date >= :sejak")
        params.update(tz=timezone, sejak=sejak)
    where = ("WHERE " + " AND ".join(kondisi)) if kondisi else ""

    # Unit tidak disimpan di baris tak terjawab, tetapi di meta jawaban
    # penolakannya. LEFT JOIN: baris yang pesannya sudah terhapus tetap tampil.
    rows = await session.execute(
        text(
            "SELECT u.id::text AS id, u.question, u.top_score, u.created_at,"
            " u.resolved, m.meta->>'unit' AS unit"
            " FROM unanswered_questions u LEFT JOIN messages m ON m.id = u.message_id"
            f" {where} ORDER BY u.created_at DESC LIMIT :batas"
        ),
        params,
    )
    return [
        UnansweredItem(
            id=row["id"],
            pertanyaan=row["question"],
            top_score=row["top_score"],
            created_at=row["created_at"],
            resolved=row["resolved"],
            unit=row["unit"],
        )
        for row in rows.mappings()
    ]


async def set_resolved(
    session: AsyncSession, unanswered_id: uuid.UUID, resolved: bool
) -> bool:
    """Return: False bila id tidak ada."""
    hasil = await session.execute(
        text(
            "UPDATE unanswered_questions SET resolved = :resolved WHERE id = :id RETURNING id"
        ),
        {"id": unanswered_id, "resolved": resolved},
    )
    if hasil.first() is None:
        await session.rollback()
        return False
    await session.commit()
    return True
