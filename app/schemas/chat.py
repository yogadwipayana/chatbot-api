"""Skema I/O endpoint chat."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from app.db.models import JenisDokumen
from app.rag.chain import OutcomeKind


class TurnIn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    konten: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=8, max_length=128)
    history: list[TurnIn] = Field(default_factory=list)


class CitationOut(BaseModel):
    """Isi kartu sitasi FE-2 -- cukup untuk membuka PDF di halaman yang tepat."""

    judul: str
    halaman: int
    document_id: str
    file_path: str
    """Kosong untuk sumber tanpa berkas; jangan dijadikan tautan."""
    jenis: JenisDokumen = JenisDokumen.PDF
    """`tanya_jawab` berarti sumbernya diketik admin di dashboard, bukan PDF:
    tidak ada berkas yang bisa dibuka dan nomor halaman tidak berarti apa-apa,
    jadi kartunya harus tampil tanpa tautan dan tanpa "hal. N"."""


class ContactOut(BaseModel):
    unit: str
    jam_layanan: str
    kontak: str


class ChatResponse(BaseModel):
    """Balasan endpoint chat.

    `citations`, `contacts`, dan `escalated` sengaja TANPA nilai default.
    Server selalu mengisi ketiganya, dan tanpa default Pydantic menandainya
    `required` di OpenAPI -- sehingga klien boleh menulis `data.citations.map(...)`
    tanpa penjagaan. Memberi default akan membuat kontrak menjanjikan bahwa
    ketiganya boleh absen, padahal tidak pernah absen.
    """

    kind: OutcomeKind
    text: str
    citations: list[CitationOut]
    contacts: list[ContactOut]
    escalated: bool
    message_id: str | None = None
    """Id baris jawaban di tabel `messages`, dipakai `POST /api/feedback`.
    None bila pencatatan ke database gagal -- jawaban tetap terkirim, dan
    frontend menyembunyikan tombol feedback (FE-5) saat nilainya None."""
    top_score: float | None = None


class FeedbackRequest(BaseModel):
    message_id: UUID
    helpful: bool
    catatan: str | None = Field(default=None, max_length=1000)
