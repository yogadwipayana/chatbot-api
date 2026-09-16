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
        kondisi.append("resolved = :resolved")
        params["resolved"] = resolved
    if sejak is not None:
        kondisi.append("(created_at AT TIME ZONE :tz)::date >= :sejak")
        params.update(tz=timezone, sejak=sejak)
    where = ("WHERE " + " AND ".join(kondisi)) if kondisi else ""

    rows = await session.execute(
        text(
            "SELECT id::text AS id, pertanyaan, top_score, created_at, resolved"
            f" FROM unanswered {where} ORDER BY created_at DESC LIMIT :batas"
        ),
        params,
    )
    return [UnansweredItem(**row) for row in rows.mappings()]


async def set_resolved(
    session: AsyncSession, unanswered_id: uuid.UUID, resolved: bool
) -> bool:
    """Return: False bila id tidak ada."""
    hasil = await session.execute(
        text("UPDATE unanswered SET resolved = :resolved WHERE id = :id RETURNING id"),
        {"id": unanswered_id, "resolved": resolved},
    )
    if hasil.first() is None:
        await session.rollback()
        return False
    await session.commit()
    return True
