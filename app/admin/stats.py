"""Statistik penggunaan untuk AD-5.

Seluruh angka dihitung langsung dari tabel log (FR-8) dan dikelompokkan per
hari menurut `TIMEZONE` -- bukan UTC, supaya pertanyaan pukul 06.00 WIB tidak
tercatat sebagai pertanyaan hari sebelumnya.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.prodi import DAFTAR_PRODI, cari_prodi

TOPIK_TERATAS = 10


def _rentang(kolom: str) -> str:
    return f"({kolom} AT TIME ZONE :tz)::date BETWEEN :since AND :until"


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
        count(*) FILTER (
            WHERE m.role = 'assistant' AND m.meta->>'kind' = 'rejected'
        ) AS rejected,
        coalesce(
            sum((m.meta->>'llm_cost_usd')::float) FILTER (WHERE m.role = 'assistant'), 0
        ) AS biaya,
        coalesce(
            sum((m.meta->>'embed_cost_usd')::float) FILTER (WHERE m.role = 'assistant'), 0
        ) AS biaya_embed,
        coalesce(
            sum((m.meta->>'gate_cost_usd')::float) FILTER (WHERE m.role = 'assistant'), 0
        ) AS biaya_gate,
        count(*) FILTER (
            WHERE m.role = 'assistant'
              AND m.meta->>'llm_called' = 'true'
              AND m.meta->>'llm_cost_usd' IS NULL
        ) AS tanpa_biaya,
        count(*) FILTER (
            WHERE m.role = 'assistant'
              AND m.meta->>'embed_called' = 'true'
              AND m.meta->>'embed_cost_usd' IS NULL
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

_UNANSWERED_SQL = text(
    f"SELECT count(*) FROM unanswered_questions u WHERE {_rentang('u.created_at')}"
)

# Deret hari dibangkitkan dari offset bilangan bulat, bukan generate_series atas
# tanggal: hari tanpa pertanyaan tetap muncul sebagai nol, dan tidak ada
# ambiguitas konversi date -> timestamptz yang bergantung TimeZone sesi.
_VOLUME_SQL = text(
    f"""
    SELECT (CAST(:since AS date) + s.i) AS "date", coalesce(v.jumlah, 0) AS count
    FROM generate_series(0, :hari) AS s(i)
    LEFT JOIN (
        SELECT (m.created_at AT TIME ZONE :tz)::date AS tanggal, count(*) AS jumlah
        FROM messages m
        WHERE m.role = 'user' AND {_rentang("m.created_at")}
        GROUP BY 1
    ) v ON v.tanggal = CAST(:since AS date) + s.i
    ORDER BY s.i
    """
)

_TOPIK_SQL = text(
    f"""
    SELECT t.topic, count(*) AS count
    FROM messages m
    CROSS JOIN LATERAL jsonb_array_elements_text(
        CASE WHEN jsonb_typeof(m.meta->'topics') = 'array' THEN m.meta->'topics'
             ELSE '[]'::jsonb END
    ) AS t(topic)
    WHERE m.role = 'assistant' AND {_rentang("m.created_at")}
    GROUP BY t.topic
    ORDER BY count DESC, t.topic
    LIMIT :batas
    """
)

# Satu baris per (prodi, angkatan); `rincian_profil` merangkumnya menjadi dua
# rincian sekaligus. Profil hanya ada di baris jawaban (`app.observability.
# chatlog`), dan tidak pernah untuk balasan `support`.
_PROFIL_SQL = text(
    f"""
    SELECT m.meta->>'program_code' AS code,
        CASE WHEN jsonb_typeof(m.meta->'intake_year') = 'number'
             THEN (m.meta->>'intake_year')::int END AS intake_year,
        count(*) AS question_count,
        count(*) FILTER (WHERE m.meta->>'kind' = 'refusal') AS refusal_count
    FROM messages m
    WHERE m.role = 'assistant' AND m.meta->>'program_code' IS NOT NULL
      AND {_rentang("m.created_at")}
    GROUP BY 1, 2
    """
)

_BIAYA_SQL = text(
    f"""
    SELECT
        count(*) FILTER (WHERE m.meta->>'llm_called' = 'true') AS jumlah_panggilan,
        coalesce(sum((m.meta->>'input_tokens')::bigint)
            FILTER (WHERE m.meta->>'llm_called' = 'true'), 0) AS input_tokens,
        coalesce(sum((m.meta->>'output_tokens')::bigint)
            FILTER (WHERE m.meta->>'llm_called' = 'true'), 0) AS output_tokens,
        coalesce(sum((m.meta->>'llm_cost_usd')::double precision)
            FILTER (WHERE m.meta->>'llm_called' = 'true'), 0) AS biaya_llm,
        count(*) FILTER (
            WHERE m.meta->>'llm_called' = 'true' AND m.meta->>'llm_cost_usd' IS NULL
        ) AS llm_tanpa_biaya,
        count(*) FILTER (WHERE m.meta->>'embed_called' = 'true') AS jumlah_embed,
        coalesce(sum((m.meta->>'embed_tokens')::bigint)
            FILTER (WHERE m.meta->>'embed_called' = 'true'), 0) AS embed_tokens,
        coalesce(sum((m.meta->>'embed_cost_usd')::double precision)
            FILTER (WHERE m.meta->>'embed_called' = 'true'), 0) AS biaya_embed,
        count(*) FILTER (
            WHERE m.meta->>'embed_called' = 'true'
              AND m.meta->>'embed_cost_usd' IS NULL
        ) AS embed_tanpa_biaya
    FROM messages m
    WHERE m.role = 'assistant' AND {_rentang("m.created_at")}
    """
)

_BIAYA_MODEL_SQL = text(
    f"""
    SELECT * FROM (
        SELECT 'llm_chat' AS type,
            coalesce(nullif(m.meta->>'model', ''), 'Tidak diketahui') AS model,
            count(*) AS call_count,
            coalesce(sum((m.meta->>'input_tokens')::bigint), 0) AS input_tokens,
            coalesce(sum((m.meta->>'output_tokens')::bigint), 0) AS output_tokens,
            coalesce(sum((m.meta->>'input_tokens')::bigint), 0)
              + coalesce(sum((m.meta->>'output_tokens')::bigint), 0) AS tokens,
            coalesce(sum((m.meta->>'llm_cost_usd')::double precision), 0) AS cost_usd
        FROM messages m
        WHERE m.role = 'assistant' AND m.meta->>'llm_called' = 'true'
          AND {_rentang("m.created_at")}
        GROUP BY 2
        UNION ALL
        SELECT 'embedding_chat' AS type,
            coalesce(nullif(m.meta->>'embed_model', ''), 'Tidak diketahui') AS model,
            count(*) AS call_count, 0 AS input_tokens, 0 AS output_tokens,
            coalesce(sum((m.meta->>'embed_tokens')::bigint), 0) AS tokens,
            coalesce(sum((m.meta->>'embed_cost_usd')::double precision), 0) AS cost_usd
        FROM messages m
        WHERE m.role = 'assistant' AND m.meta->>'embed_called' = 'true'
          AND {_rentang("m.created_at")}
        GROUP BY 2
    ) rincian
    ORDER BY cost_usd DESC, model
    """
)

_BIAYA_HARIAN_SQL = text(
    f"""
    WITH semua AS (
        SELECT (m.created_at AT TIME ZONE :tz)::date AS tanggal,
            (CASE WHEN m.meta->>'llm_called' = 'true' THEN 1 ELSE 0 END)
              + (CASE WHEN m.meta->>'embed_called' = 'true' THEN 1 ELSE 0 END) AS call_count,
            coalesce((m.meta->>'input_tokens')::bigint, 0)
              + coalesce((m.meta->>'output_tokens')::bigint, 0) AS llm_tokens,
            coalesce((m.meta->>'embed_tokens')::bigint, 0) AS embed_tokens,
            coalesce((m.meta->>'llm_cost_usd')::double precision, 0) AS llm_cost_usd,
            coalesce((m.meta->>'embed_cost_usd')::double precision, 0) AS embedding_cost_usd
        FROM messages m
        WHERE m.role = 'assistant' AND {_rentang("m.created_at")}
    )
    SELECT tanggal AS "date", sum(call_count) AS call_count,
        sum(llm_tokens) AS llm_tokens, sum(embed_tokens) AS embed_tokens,
        sum(llm_cost_usd) AS llm_cost_usd,
        sum(embedding_cost_usd) AS embedding_cost_usd,
        sum(llm_cost_usd + embedding_cost_usd) AS cost_usd
    FROM semua
    GROUP BY tanggal
    ORDER BY 1
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


def rincian_profil(baris: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Rincian per prodi dan per angkatan dari baris `_PROFIL_SQL`.

    Setiap prodi di `DAFTAR_PRODI` selalu muncul, juga yang nol: "DKV belum
    pernah bertanya" adalah temuan, bukan baris yang boleh hilang. Kode yang
    sudah tidak terdaftar tetap ditampilkan dengan kodenya sebagai nama, supaya
    jumlahnya tetap sama dengan `questions_with_profile`.
    """
    per_prodi: dict[str, list[int]] = {p.code: [0, 0] for p in DAFTAR_PRODI}
    per_angkatan: dict[int, list[int]] = {}
    total = 0
    for b in baris:
        jumlah, ditolak = int(b["question_count"]), int(b["refusal_count"])
        total += jumlah
        tujuan = [per_prodi.setdefault(b["code"], [0, 0])]
        if b["intake_year"] is not None:
            tujuan.append(per_angkatan.setdefault(int(b["intake_year"]), [0, 0]))
        for hitungan in tujuan:
            hitungan[0] += jumlah
            hitungan[1] += ditolak

    urutan = {p.code: i for i, p in enumerate(DAFTAR_PRODI)}
    prodi = sorted(
        per_prodi.items(),
        key=lambda item: (-item[1][0], urutan.get(item[0], len(urutan)), item[0]),
    )
    return {
        "questions_with_profile": total,
        "program_breakdown": [
            {
                "code": kode,
                "name": p.name if (p := cari_prodi(kode)) else kode,
                "level": p.level if p else None,
                "question_count": jumlah,
                "refusal_count": ditolak,
            }
            for kode, (jumlah, ditolak) in prodi
        ],
        "intake_year_breakdown": [
            {"intake_year": tahun, "question_count": jumlah, "refusal_count": ditolak}
            for tahun, (jumlah, ditolak) in sorted(per_angkatan.items(), reverse=True)
        ],
    }


async def compute_stats(
    session: AsyncSession, *, since: date, until: date, timezone: str
) -> dict[str, Any]:
    p = {"tz": timezone, "since": since, "until": until}

    percakapan = (await session.execute(_PERCAKAPAN_SQL, p)).scalar_one()
    pesan = (await session.execute(_PESAN_SQL, p)).mappings().one()
    feedback = (await session.execute(_FEEDBACK_SQL, p)).mappings().one()
    tak_terjawab = (await session.execute(_UNANSWERED_SQL, p)).scalar_one()
    volume = (
        (await session.execute(_VOLUME_SQL, {**p, "hari": (until - since).days}))
        .mappings()
        .all()
    )
    topik = (await session.execute(_TOPIK_SQL, {**p, "batas": TOPIK_TERATAS})).mappings().all()
    profil = (await session.execute(_PROFIL_SQL, p)).mappings().all()

    return {
        "since": since,
        "until": until,
        "total_conversations": percakapan,
        "total_questions": pesan["pertanyaan"],
        "kind_breakdown": {
            "answer": pesan["answer"],
            "refusal": pesan["refusal"],
            "support": pesan["support"],
            "smalltalk": pesan["smalltalk"],
            "rejected": pesan["rejected"],
        },
        "feedback_count": feedback["jumlah"],
        "positive_feedback_ratio": ratio(feedback["positif"], feedback["jumlah"]),
        "unanswered_ratio": ratio(tak_terjawab, pesan["pertanyaan"]),
        "daily_volume": [dict(r) for r in volume],
        "top_topics": [dict(r) for r in topik],
        "running_cost_usd": round(
            float(pesan["biaya"]) + float(pesan["biaya_embed"]) + float(pesan["biaya_gate"]),
            6,
        ),
        "messages_without_cost_estimate": pesan["tanpa_biaya"] + pesan["embed_tanpa_biaya"],
        "latency_p95_ms": round(pesan["p95"]) if pesan["p95"] is not None else None,
        **rincian_profil(profil),
    }


async def compute_costs(
    session: AsyncSession, *, since: date, until: date, timezone: str
) -> dict[str, Any]:
    p = {"tz": timezone, "since": since, "until": until}
    ringkasan = (await session.execute(_BIAYA_SQL, p)).mappings().one()
    model = (await session.execute(_BIAYA_MODEL_SQL, p)).mappings().all()
    harian = (await session.execute(_BIAYA_HARIAN_SQL, p)).mappings().all()
    input_tokens = int(ringkasan["input_tokens"])
    output_tokens = int(ringkasan["output_tokens"])
    embed_chat_tokens = int(ringkasan["embed_tokens"])
    biaya_llm = float(ringkasan["biaya_llm"])
    biaya_embed = float(ringkasan["biaya_embed"])
    return {
        "since": since,
        "until": until,
        "llm_call_count": int(ringkasan["jumlah_panggilan"]),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens + embed_chat_tokens,
        "cost_usd": round(biaya_llm + biaya_embed, 8),
        "llm_cost_usd": round(biaya_llm, 8),
        "chat_embed_tokens": embed_chat_tokens,
        "chat_embed_cost_usd": round(biaya_embed, 8),
        "chat_embed_count": int(ringkasan["jumlah_embed"]),
        "llm_calls_without_cost": int(ringkasan["llm_tanpa_biaya"]),
        "chat_embeds_without_cost": int(ringkasan["embed_tanpa_biaya"]),
        "model_breakdown": [
            {**dict(row), "cost_usd": round(float(row["cost_usd"]), 8)} for row in model
        ],
        "daily_costs": [
            {key: round(float(value), 8) if key.endswith("_usd") else value
             for key, value in dict(row).items()}
            for row in harian
        ],
    }
