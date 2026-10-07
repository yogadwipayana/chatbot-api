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
from app.prodi import ProfilMahasiswa
from app.rag.chain import OutcomeKind, PipelineOutcome, refusal_source, rejection_source

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
    embed_key: str | None = None
    """Kunci situs penyemat asal pertanyaan (`app/embed_keys.py`); None = portal."""
    profile: ProfilMahasiswa | None = None
    """Prodi dan angkatan penanya, diurai dari `nim`, untuk analitik per kohort."""
    nim: str | None = None
    """NIM penanya apa adanya. Hanya untuk `messages.meta`: tidak ikut ke LLM,
    trace, maupun log aplikasi."""
    embed_dipanggil: bool = False
    """False untuk FR-7 dan sapaan berbasis aturan: keduanya berhenti sebelum
    retrieval, sehingga pertanyaannya tidak pernah di-embed sama sekali. Pesan
    yang diblokir JEV bisa True -- pencarian berjalan paralel dengan gerbang dan
    baru dihentikan saat vonis blokir tiba."""
    embed_model: str | None = None
    embed_tokens: int | None = None
    embed_biaya_usd: float | None = None
    embed_biaya_sumber: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    """Jejak panggilan tool giliran ini (docs/tool-call.md §13): tiap entri
    berisi nama tool, argumen (mentah dari model), ok, dan latency_ms. None bila
    tool tidak dipakai (TOOLS_ENABLED=false atau pertanyaan tidak eligible)."""


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

    # Prodi dan angkatan tidak dicatat untuk pesan FR-7, sama seperti isinya:
    # rincian per kohort di statistik AD-5 tidak memuat pesan konseling. NIM
    # tetap dicatat (2026-10-07), jadi pesan FR-7 tetap dapat ditelusuri ke
    # penanyanya walau isinya disembunyikan.
    profil = None if outcome.kind is OutcomeKind.SUPPORT else entry.profile

    return {
        "kind": outcome.kind.value,
        # Penolakan yang baru diputuskan LLM berarti konteksnya lolos threshold
        # padahal tidak menjawab -- sinyal untuk kalibrasi ambang FR-3.
        "refusal_source": refusal_source(outcome),
        "escalated": bool(outcome.contacts),
        "topics": [t.value for t in outcome.risk.topics] if outcome.risk else [],
        "sensitivity": outcome.sensitivity.level.value if outcome.sensitivity else None,
        "llm_called": outcome.llm_called,
        "model": entry.model if outcome.llm_called else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "llm_cost_usd": biaya,
        "rewritten_query": outcome.rewritten_query,
        "unit": entry.unit,
        "program_code": profil.prodi.code if profil else None,
        "intake_year": profil.angkatan if profil else None,
        "nim": entry.nim,
        # Biaya meng-embed pertanyaan mahasiswa. Dipisah dari `llm_cost_usd`, bukan
        # dijumlahkan ke dalamnya: `llm_cost_usd` sudah berarti "biaya LLM" di
        # seluruh baris lama dan di `app/admin/stats.py`, dan mengubah artinya
        # diam-diam membuat baris sebelum dan sesudah hari ini tidak sebanding.
        "embed_called": entry.embed_dipanggil,
        "embed_model": entry.embed_model if entry.embed_dipanggil else None,
        "embed_tokens": entry.embed_tokens,
        "embed_cost_usd": entry.embed_biaya_usd,
        "embed_cost_source": entry.embed_biaya_sumber,
        # Gerbang JEV atau saringan aturan (`gate_source`). `gate_label` terisi
        # juga untuk pesan yang diteruskan JEV, supaya ambangnya dapat
        # dikalibrasi dari log, bukan ditebak.
        "gate_label": outcome.gate.label.value if outcome.gate else None,
        "gate_confidence": outcome.gate.confidence if outcome.gate else None,
        "gate_error": outcome.gate.error if outcome.gate else None,
        "gate_cost_usd": outcome.gate.cost_usd if outcome.gate else None,
        "gate_source": outcome.gate.source.value if outcome.gate else None,
        # `llm` = LLM penjawab membalas OFF_TOPIC_MARKER walau gerbang meloloskan.
        "rejection_source": rejection_source(outcome),
        "top_rerank_score": (
            outcome.decision.top_rerank_score if outcome.decision else None
        ),
        # Jejak tool-calling: nama, argumen, ok, latency_ms per panggilan.
        # None bila tool tidak dipakai, sehingga baris lama tetap sebanding.
        "tool_calls": entry.tool_calls or None,
    }


def retrieved_chunk_ids(outcome: PipelineOutcome) -> list[uuid.UUID] | None:
    ids: list[uuid.UUID] = []
    for doc in outcome.documents:
        try:
            ids.append(uuid.UUID(str(doc.metadata.get("chunk_id", ""))))
        except ValueError:
            continue
    return ids or None


# `embed_key` ikut dicocokkan: percakapan dari portal dan dari situs penyemat
# tidak boleh tergabung, supaya jumlah pertanyaan per situs tidak tercampur.
_PERCAKAPAN_AKTIF_SQL = text(
    """
    SELECT c.id
    FROM conversations c
    WHERE c.session_id = :session_id
      AND c.embed_key IS NOT DISTINCT FROM CAST(:embed_key AS text)
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
        (id, conversation_id, role, content, retrieved_chunk_ids, top_score,
         latency_ms, langsmith_run_id, meta, created_at)
    VALUES
        (:id, :conversation_id, :role, :content, CAST(:chunk_ids AS uuid[]), :top_score,
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
                    {
                        "session_id": entry.session_id,
                        "embed_key": entry.embed_key,
                        "idle": CONVERSATION_IDLE_MINUTES,
                    },
                )
            ).scalar()
            if conversation_id is None:
                conversation_id = uuid.uuid4()
                await session.execute(
                    text(
                        "INSERT INTO conversations (id, session_id, embed_key)"
                        " VALUES (:id, :session_id, :embed_key)"
                    ),
                    {
                        "id": conversation_id,
                        "session_id": entry.session_id,
                        "embed_key": entry.embed_key,
                    },
                )

            sensitif = outcome.kind is OutcomeKind.SUPPORT
            await session.execute(
                _PESAN_SQL,
                {
                    "id": uuid.uuid4(),
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": SENSITIVE_PLACEHOLDER if sensitif else entry.question,
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
                    "content": outcome.text,
                    "chunk_ids": retrieved_chunk_ids(outcome),
                    "top_score": top_score,
                    "latency_ms": entry.latency_ms,
                    "langsmith_run_id": entry.langsmith_run_id,
                    "meta": json.dumps(build_meta(entry), ensure_ascii=False),
                },
            )

            # FR-3: pertanyaan yang ditolak masuk `unanswered_questions`, sumber
            # utama AD-4.
            if outcome.kind is OutcomeKind.REFUSAL:
                await session.execute(
                    text(
                        "INSERT INTO unanswered_questions"
                        " (id, question, top_score, message_id)"
                        " VALUES (:id, :question, :top_score, :message_id)"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "question": entry.question,
                        "top_score": top_score,
                        "message_id": assistant_id,
                    },
                )

            await session.commit()
        return str(assistant_id)
