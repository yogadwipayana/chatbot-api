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
        coalesce(
            sum((m.meta->>'embed_biaya_usd')::float) FILTER (WHERE m.role = 'assistant'), 0
        ) AS biaya_embed,
        count(*) FILTER (
            WHERE m.role = 'assistant'
              AND m.meta->>'llm_dipanggil' = 'true'
              AND m.meta->>'biaya_usd' IS NULL
        ) AS tanpa_biaya,
        count(*) FILTER (
            WHERE m.role = 'assistant'
              AND m.meta->>'embed_dipanggil' = 'true'
              AND m.meta->>'embed_biaya_usd' IS NULL
        ) AS embed_tanpa_biaya,
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

_BIAYA_SQL = text(
    f"""
    SELECT
        count(*) FILTER (WHERE m.meta->>'llm_dipanggil' = 'true') AS jumlah_panggilan,
        coalesce(sum((m.meta->>'input_tokens')::bigint)
            FILTER (WHERE m.meta->>'llm_dipanggil' = 'true'), 0) AS input_tokens,
        coalesce(sum((m.meta->>'output_tokens')::bigint)
            FILTER (WHERE m.meta->>'llm_dipanggil' = 'true'), 0) AS output_tokens,
        coalesce(sum((m.meta->>'biaya_usd')::double precision)
            FILTER (WHERE m.meta->>'llm_dipanggil' = 'true'), 0) AS biaya_llm,
        count(*) FILTER (
            WHERE m.meta->>'llm_dipanggil' = 'true' AND m.meta->>'biaya_usd' IS NULL
        ) AS llm_tanpa_biaya,
        count(*) FILTER (WHERE m.meta->>'embed_dipanggil' = 'true') AS jumlah_embed,
        coalesce(sum((m.meta->>'embed_tokens')::bigint)
            FILTER (WHERE m.meta->>'embed_dipanggil' = 'true'), 0) AS embed_tokens,
        coalesce(sum((m.meta->>'embed_biaya_usd')::double precision)
            FILTER (WHERE m.meta->>'embed_dipanggil' = 'true'), 0) AS biaya_embed,
        count(*) FILTER (
            WHERE m.meta->>'embed_dipanggil' = 'true'
              AND m.meta->>'embed_biaya_usd' IS NULL
        ) AS embed_tanpa_biaya
    FROM messages m
    WHERE m.role = 'assistant' AND {_rentang("m.created_at")}
    """
)

_BIAYA_MODEL_SQL = text(
    f"""
    SELECT * FROM (
        SELECT 'llm_chat' AS jenis,
            coalesce(nullif(m.meta->>'model', ''), 'Tidak diketahui') AS model,
            count(*) AS jumlah_panggilan,
            coalesce(sum((m.meta->>'input_tokens')::bigint), 0) AS input_tokens,
            coalesce(sum((m.meta->>'output_tokens')::bigint), 0) AS output_tokens,
            coalesce(sum((m.meta->>'input_tokens')::bigint), 0)
              + coalesce(sum((m.meta->>'output_tokens')::bigint), 0) AS tokens,
            coalesce(sum((m.meta->>'biaya_usd')::double precision), 0) AS biaya_usd
        FROM messages m
        WHERE m.role = 'assistant' AND m.meta->>'llm_dipanggil' = 'true'
          AND {_rentang("m.created_at")}
        GROUP BY 2
        UNION ALL
        SELECT 'embedding_chat' AS jenis,
            coalesce(nullif(m.meta->>'embed_model', ''), 'Tidak diketahui') AS model,
            count(*) AS jumlah_panggilan, 0 AS input_tokens, 0 AS output_tokens,
            coalesce(sum((m.meta->>'embed_tokens')::bigint), 0) AS tokens,
            coalesce(sum((m.meta->>'embed_biaya_usd')::double precision), 0) AS biaya_usd
        FROM messages m
        WHERE m.role = 'assistant' AND m.meta->>'embed_dipanggil' = 'true'
          AND {_rentang("m.created_at")}
        GROUP BY 2
        UNION ALL
        SELECT 'embedding_ingestion' AS jenis, u.model, count(*) AS jumlah_panggilan,
            0 AS input_tokens, 0 AS output_tokens, coalesce(sum(u.tokens), 0) AS tokens,
            coalesce(sum(u.biaya_usd), 0) AS biaya_usd
        FROM usage_log u
        WHERE {_rentang("u.created_at")}
        GROUP BY u.model
    ) rincian
    ORDER BY biaya_usd DESC, model
    """
)

_BIAYA_HARIAN_SQL = text(
    f"""
    WITH semua AS (
        SELECT (m.created_at AT TIME ZONE :tz)::date AS tanggal,
            (CASE WHEN m.meta->>'llm_dipanggil' = 'true' THEN 1 ELSE 0 END)
              + (CASE WHEN m.meta->>'embed_dipanggil' = 'true' THEN 1 ELSE 0 END) AS jumlah_panggilan,
            coalesce((m.meta->>'input_tokens')::bigint, 0)
              + coalesce((m.meta->>'output_tokens')::bigint, 0) AS llm_tokens,
            coalesce((m.meta->>'embed_tokens')::bigint, 0) AS embed_tokens,
            coalesce((m.meta->>'biaya_usd')::double precision, 0) AS biaya_llm_usd,
            coalesce((m.meta->>'embed_biaya_usd')::double precision, 0) AS biaya_embedding_usd,
            0::double precision AS biaya_ingestion_usd
        FROM messages m
        WHERE m.role = 'assistant' AND {_rentang("m.created_at")}
        UNION ALL
        SELECT (u.created_at AT TIME ZONE :tz)::date, 1, 0,
            coalesce(u.tokens, 0), 0, 0, coalesce(u.biaya_usd, 0)
        FROM usage_log u WHERE {_rentang("u.created_at")}
    )
    SELECT tanggal, sum(jumlah_panggilan) AS jumlah_panggilan,
        sum(llm_tokens) AS llm_tokens, sum(embed_tokens) AS embed_tokens,
        sum(biaya_llm_usd) AS biaya_llm_usd,
        sum(biaya_embedding_usd) AS biaya_embedding_usd,
        sum(biaya_ingestion_usd) AS biaya_ingestion_usd,
        sum(biaya_llm_usd + biaya_embedding_usd + biaya_ingestion_usd) AS biaya_usd
    FROM semua
    GROUP BY tanggal
    ORDER BY 1
    """
)

_USAGE_LOG_SQL = text(
    f"""
    SELECT count(*) AS jumlah, coalesce(sum(tokens), 0) AS tokens,
        coalesce(sum(biaya_usd), 0) AS biaya,
        count(*) FILTER (WHERE biaya_usd IS NULL) AS tanpa_biaya
    FROM usage_log u WHERE {_rentang("u.created_at")}
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
    usage = (await session.execute(_USAGE_LOG_SQL, p)).mappings().one()

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
        "biaya_usd_berjalan": round(
            float(pesan["biaya"]) + float(pesan["biaya_embed"]) + float(usage["biaya"]), 6
        ),
        "pesan_tanpa_estimasi_biaya": (
            pesan["tanpa_biaya"] + pesan["embed_tanpa_biaya"] + usage["tanpa_biaya"]
        ),
        "latency_p95_ms": round(pesan["p95"]) if pesan["p95"] is not None else None,
    }


async def compute_costs(
    session: AsyncSession, *, sejak: date, sampai: date, timezone: str
) -> dict[str, Any]:
    p = {"tz": timezone, "sejak": sejak, "sampai": sampai}
    ringkasan = (await session.execute(_BIAYA_SQL, p)).mappings().one()
    usage = (await session.execute(_USAGE_LOG_SQL, p)).mappings().one()
    model = (await session.execute(_BIAYA_MODEL_SQL, p)).mappings().all()
    harian = (await session.execute(_BIAYA_HARIAN_SQL, p)).mappings().all()
    input_tokens = int(ringkasan["input_tokens"])
    output_tokens = int(ringkasan["output_tokens"])
    embed_chat_tokens = int(ringkasan["embed_tokens"])
    usage_tokens = int(usage["tokens"])
    biaya_llm = float(ringkasan["biaya_llm"])
    biaya_embed = float(ringkasan["biaya_embed"])
    biaya_usage = float(usage["biaya"])
    return {
        "sejak": sejak,
        "sampai": sampai,
        "jumlah_panggilan_llm": int(ringkasan["jumlah_panggilan"]),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens + embed_chat_tokens + usage_tokens,
        "biaya_usd": round(biaya_llm + biaya_embed + biaya_usage, 8),
        "biaya_llm_usd": round(biaya_llm, 8),
        "embed_chat_tokens": embed_chat_tokens,
        "biaya_embed_chat_usd": round(biaya_embed, 8),
        "jumlah_embed_chat": int(ringkasan["jumlah_embed"]),
        "usage_log_tokens": usage_tokens,
        "biaya_usage_log_usd": round(biaya_usage, 8),
        "jumlah_usage_log": int(usage["jumlah"]),
        "llm_tanpa_biaya": int(ringkasan["llm_tanpa_biaya"]),
        "embed_chat_tanpa_biaya": int(ringkasan["embed_tanpa_biaya"]),
        "usage_log_tanpa_biaya": int(usage["tanpa_biaya"]),
        "rincian_model": [
            {**dict(row), "biaya_usd": round(float(row["biaya_usd"]), 8)} for row in model
        ],
        "biaya_harian": [
            {key: round(float(value), 8) if key.startswith("biaya_") else value
             for key, value in dict(row).items()}
            for row in harian
        ],
    }
