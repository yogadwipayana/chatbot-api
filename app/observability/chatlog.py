"""Pencatatan percakapan ke Postgres (FR-8).

Sumber data AD-4 (pertanyaan tak terjawab) dan AD-5 (statistik): tanpa catatan
ini dashboard admin tetap kosong walau chatbot ramai dipakai.

Kegagalan menulis log TIDAK boleh menggagalkan jawaban ke mahasiswa. Database
yang tersendat sesaat lebih baik kehilangan satu baris log daripada membuat
mahasiswa melihat galat setelah jawabannya sudah jadi.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from app.observability.costs import try_estimate_cost
from app.rag.chain import OutcomeKind, PipelineOutcome

logger = logging.getLogger(__name__)

CONVERSATION_IDLE_MINUTES = 30
"""Pesan dari sesi yang sama dianggap satu percakapan selama jedanya di bawah
ini. `session_id` tinggal di localStorage berbulan-bulan; tanpa batas jeda,
seluruh pemakaian satu peramban terhitung satu percakapan."""

SENSITIVE_PLACEHOLDER = "[disembunyikan: pertanyaan sensitif, dialihkan ke layanan konseling]"
"""Isi pertanyaan FR-7 tidak disimpan. Jumlahnya tetap tercatat untuk statistik,
tetapi curahan hati mahasiswa tidak ikut terbaca siapa pun yang membuka database."""


@dataclass(frozen=True)
class ChatLogEntry:
    session_id: str
    question: str
    """Pertanyaan yang sudah disanitasi."""
    outcome: PipelineOutcome
    latency_ms: int
    model: str | None = None
    usage: dict[str, Any] | None = None
    """`usage_metadata` LangChain: `input_tokens`, `output_tokens`."""
    langsmith_run_id: str | None = None
    """Akar trace giliran ini: penulisan ulang query, retrieval, dan penyusunan
    jawaban berada di bawahnya. None saat tracing mati -- tidak ada trace yang
    dikirim, jadi tidak ada yang bisa dirujuk."""
    unit: str | None = None
    """Unit yang dipilih mahasiswa di menu chatbot; None = semua unit. Tanpa
    ini, penolakan akibat salah pilih unit tidak dapat dibedakan dari dokumen
    yang memang belum ada."""
    embed_dipanggil: bool = False
    """False untuk FR-7 dan smalltalk: keduanya berhenti sebelum retrieval,
    sehingga pertanyaannya tidak pernah di-embed sama sekali."""
    embed_model: str | None = None
    embed_tokens: int | None = None
    embed_biaya_usd: float | None = None
    embed_biaya_sumber: str | None = None


def build_meta(entry: ChatLogEntry) -> dict[str, Any]:
    """Isi kolom `messages.meta` untuk baris jawaban (FR-6, FR-8)."""
    outcome = entry.outcome
    usage = entry.usage or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")

    biaya: float | None = None
    if (
        outcome.llm_called
        and entry.model
        and input_tokens is not None
        and output_tokens is not None
    ):
        estimasi = try_estimate_cost(entry.model, int(input_tokens), int(output_tokens))
        biaya = estimasi.usd if estimasi else None

    return {
        "kind": outcome.kind.value,
        "escalated": bool(outcome.contacts),
        "topik": [t.value for t in outcome.risk.topics] if outcome.risk else [],
        "sensitivitas": outcome.sensitivity.level.value if outcome.sensitivity else None,
        "llm_dipanggil": outcome.llm_called,
        "model": entry.model if outcome.llm_called else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "biaya_usd": biaya,
        "rewritten_query": outcome.rewritten_query,
        "unit": entry.unit,
        # Biaya meng-embed pertanyaan mahasiswa. Dipisah dari `biaya_usd`, bukan
        # dijumlahkan ke dalamnya: `biaya_usd` sudah berarti "biaya LLM" di
        # seluruh baris lama dan di `app/admin/stats.py`, dan mengubah artinya
        # diam-diam membuat baris sebelum dan sesudah hari ini tidak sebanding.
        "embed_dipanggil": entry.embed_dipanggil,
        "embed_model": entry.embed_model if entry.embed_dipanggil else None,
        "embed_tokens": entry.embed_tokens,
        "embed_biaya_usd": entry.embed_biaya_usd,
        "embed_biaya_sumber": entry.embed_biaya_sumber,
    }


def retrieved_chunk_ids(outcome: PipelineOutcome) -> list[uuid.UUID] | None:
    ids: list[uuid.UUID] = []
    for doc in outcome.documents:
        try:
            ids.append(uuid.UUID(str(doc.metadata.get("chunk_id", ""))))
        except ValueError:
            continue
    return ids or None


_PERCAKAPAN_AKTIF_SQL = text(
    """
    SELECT c.id
    FROM conversations c
    WHERE c.session_id = :session_id
      AND coalesce(
            (SELECT max(m.created_at) FROM messages m WHERE m.conversation_id = c.id),
            c.created_at
          ) > now() - make_interval(mins => :idle)
    ORDER BY c.created_at DESC
    LIMIT 1
    """
)

# clock_timestamp(), bukan now(): now() bernilai sama sepanjang transaksi, sehingga
# pertanyaan dan jawabannya akan bercap waktu identik dan urutannya tak tentu.
_PESAN_SQL = text(
    """
    INSERT INTO messages
        (id, conversation_id, role, konten, retrieved_chunk_ids, top_score,
         latency_ms, langsmith_run_id, meta, created_at)
    VALUES
        (:id, :conversation_id, :role, :konten, CAST(:chunk_ids AS uuid[]), :top_score,
         :latency_ms, :langsmith_run_id, CAST(:meta AS jsonb), clock_timestamp())
    """
)


class ChatLogger:
    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self.session_factory = session_factory

    async def log(self, entry: ChatLogEntry) -> str | None:
        """Catat satu putaran tanya-jawab. Return: id pesan jawaban, atau None bila gagal."""
        try:
            return await self._tulis(entry)
        except Exception:
            logger.exception("Gagal mencatat percakapan ke database")
            return None

    async def _tulis(self, entry: ChatLogEntry) -> str:
        outcome = entry.outcome
        async with self.session_factory() as session:
            conversation_id = (
                await session.execute(
                    _PERCAKAPAN_AKTIF_SQL,
                    {"session_id": entry.session_id, "idle": CONVERSATION_IDLE_MINUTES},
                )
            ).scalar()
            if conversation_id is None:
                conversation_id = uuid.uuid4()
                await session.execute(
                    text(
                        "INSERT INTO conversations (id, session_id) VALUES (:id, :session_id)"
                    ),
                    {"id": conversation_id, "session_id": entry.session_id},
                )

            sensitif = outcome.kind is OutcomeKind.SUPPORT
            await session.execute(
                _PESAN_SQL,
                {
                    "id": uuid.uuid4(),
                    "conversation_id": conversation_id,
                    "role": "user",
                    "konten": SENSITIVE_PLACEHOLDER if sensitif else entry.question,
                    "chunk_ids": None,
                    "top_score": None,
                    "latency_ms": None,
                    "langsmith_run_id": None,
                    "meta": None,
                },
            )

            top_score = outcome.decision.top_score if outcome.decision else None
            assistant_id = uuid.uuid4()
            await session.execute(
                _PESAN_SQL,
                {
                    "id": assistant_id,
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "konten": outcome.text,
                    "chunk_ids": retrieved_chunk_ids(outcome),
                    "top_score": top_score,
                    "latency_ms": entry.latency_ms,
                    "langsmith_run_id": entry.langsmith_run_id,
                    "meta": json.dumps(build_meta(entry), ensure_ascii=False),
                },
            )

            # FR-3: pertanyaan yang ditolak masuk `unanswered`, sumber utama AD-4.
            if outcome.kind is OutcomeKind.REFUSAL:
                await session.execute(
                    text(
                        "INSERT INTO unanswered (id, pertanyaan, top_score, message_id)"
                        " VALUES (:id, :pertanyaan, :top_score, :message_id)"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "pertanyaan": entry.question,
                        "top_score": top_score,
                        "message_id": assistant_id,
                    },
                )

            await session.commit()
        return str(assistant_id)
