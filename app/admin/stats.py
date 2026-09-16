"""Statistik penggunaan untuk AD-5.

Seluruh angka dihitung langsung dari tabel log (FR-8) dan dikelompokkan per
hari menurut `TIMEZONE` -- bukan UTC, supaya pertanyaan pukul 06.00 WIB tidak
tercatat sebagai pertanyaan hari sebelumnya.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

TOPIK_TERATAS = 10


def _rentang(kolom: str) -> str:
    return f"({kolom} AT TIME ZONE :tz)::date BETWEEN :sejak AND :sampai"


_PERCAKAPAN_SQL = text(
    f"SELECT count(*) FROM conversations c WHERE {_rentang('c.created_at')}"
)

_PESAN_SQL = text(
    f"""
    SELECT
        count(*) FILTER (WHERE m.role = 'user') AS pertanyaan,
        count(*) FILTER (
            WHERE m.role = 'assistant' AND m.meta->>'kind' = 'answer'
        ) AS answer,
        count(*) FILTER (
            WHERE m.role = 'assistant' AND m.meta->>'kind' = 'refusal'
        ) AS refusal,
        count(*) FILTER (
            WHERE m.role = 'assistant' AND m.meta->>'kind' = 'support'
        ) AS support,
        count(*) FILTER (
            WHERE m.role = 'assistant' AND m.meta->>'kind' = 'smalltalk'
        ) AS smalltalk,
        coalesce(
            sum((m.meta->>'biaya_usd')::float) FILTER (WHERE m.role = 'assistant'), 0
        ) AS biaya,
        count(*) FILTER (
            WHERE m.role = 'assistant'
              AND m.meta->>'llm_dipanggil' = 'true'
              AND m.meta->>'biaya_usd' IS NULL
        ) AS tanpa_biaya,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY m.latency_ms)
            FILTER (WHERE m.role = 'assistant' AND m.latency_ms IS NOT NULL) AS p95
    FROM messages m
    WHERE {_rentang("m.created_at")}
    """
)

_FEEDBACK_SQL = text(
    f"""
    SELECT count(*) AS jumlah, count(*) FILTER (WHERE f.helpful) AS positif
    FROM feedback f
    WHERE {_rentang("f.created_at")}
    """
)

_UNANSWERED_SQL = text(f"SELECT count(*) FROM unanswered u WHERE {_rentang('u.created_at')}")

# Deret hari dibangkitkan dari offset bilangan bulat, bukan generate_series atas
# tanggal: hari tanpa pertanyaan tetap muncul sebagai nol, dan tidak ada
# ambiguitas konversi date -> timestamptz yang bergantung TimeZone sesi.
_VOLUME_SQL = text(
    f"""
    SELECT (CAST(:sejak AS date) + s.i) AS tanggal, coalesce(v.jumlah, 0) AS jumlah
    FROM generate_series(0, :hari) AS s(i)
    LEFT JOIN (
        SELECT (m.created_at AT TIME ZONE :tz)::date AS tanggal, count(*) AS jumlah
        FROM messages m
        WHERE m.role = 'user' AND {_rentang("m.created_at")}
        GROUP BY 1
    ) v ON v.tanggal = CAST(:sejak AS date) + s.i
    ORDER BY s.i
    """
)

_TOPIK_SQL = text(
    f"""
    SELECT t.topik, count(*) AS jumlah
    FROM messages m
    CROSS JOIN LATERAL jsonb_array_elements_text(
        CASE WHEN jsonb_typeof(m.meta->'topik') = 'array' THEN m.meta->'topik'
             ELSE '[]'::jsonb END
    ) AS t(topik)
    WHERE m.role = 'assistant' AND {_rentang("m.created_at")}
    GROUP BY t.topik
    ORDER BY jumlah DESC, t.topik
    LIMIT :batas
    """
)


async def today(session: AsyncSession, timezone: str) -> date:
    return (
        await session.execute(text("SELECT (now() AT TIME ZONE :tz)::date"), {"tz": timezone})
    ).scalar_one()


def ratio(bagian: int, total: int) -> float | None:
    """None bila penyebutnya nol: rasio 0% dan 'belum ada data' berbeda arti."""
    if not total:
        return None
    return min(bagian / total, 1.0)


async def compute_stats(
    session: AsyncSession, *, sejak: date, sampai: date, timezone: str
) -> dict[str, Any]:
    p = {"tz": timezone, "sejak": sejak, "sampai": sampai}

    percakapan = (await session.execute(_PERCAKAPAN_SQL, p)).scalar_one()
    pesan = (await session.execute(_PESAN_SQL, p)).mappings().one()
    feedback = (await session.execute(_FEEDBACK_SQL, p)).mappings().one()
    tak_terjawab = (await session.execute(_UNANSWERED_SQL, p)).scalar_one()
    volume = (
        (await session.execute(_VOLUME_SQL, {**p, "hari": (sampai - sejak).days}))
        .mappings()
        .all()
    )
    topik = (await session.execute(_TOPIK_SQL, {**p, "batas": TOPIK_TERATAS})).mappings().all()

    return {
        "sejak": sejak,
        "sampai": sampai,
        "total_percakapan": percakapan,
        "total_pertanyaan": pesan["pertanyaan"],
        "rincian_jenis": {
            "answer": pesan["answer"],
            "refusal": pesan["refusal"],
            "support": pesan["support"],
            "smalltalk": pesan["smalltalk"],
        },
        "jumlah_feedback": feedback["jumlah"],
        "rasio_feedback_positif": ratio(feedback["positif"], feedback["jumlah"]),
        "rasio_tak_terjawab": ratio(tak_terjawab, pesan["pertanyaan"]),
        "volume_harian": [dict(r) for r in volume],
        "topik_populer": [dict(r) for r in topik],
        "biaya_usd_berjalan": round(float(pesan["biaya"]), 6),
        "pesan_tanpa_estimasi_biaya": pesan["tanpa_biaya"],
        "latency_p95_ms": round(pesan["p95"]) if pesan["p95"] is not None else None,
    }
