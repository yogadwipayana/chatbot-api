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
        "latency_p95_ms": round(pesan["p95"]) if pesan["p95"] is not None else None,
        **rincian_profil(profil),
    }
