"""Skema I/O endpoint chat."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.db.models import DocumentType
from app.prodi import ANGKATAN_PERTAMA
from app.rag.chain import OutcomeKind
from app.rag.rewriter import HISTORY_WINDOW

MAKS_PERTANYAAN = 500
"""Batas panjang pertanyaan mahasiswa.

Kotak pertanyaan portal membatasi 200 karakter, tetapi pertanyaan siap klik
di menu topik adalah entri tanya jawab admin, yang boleh sampai 500
(`FaqEntryCreate.question`). Di atas ini hanya pemanggil langsung --
Postman, skrip -- yang dapat mengirim, dan setiap karakternya dibayar sebagai
token di penulisan ulang, gerbang JEV, dan jawaban."""

MAKS_KONTEN_RIWAYAT = 2000
"""Satu giliran riwayat dipotong sepanjang ini, bukan ditolak. Isinya hanya
konteks penulisan ulang pertanyaan lanjutan (FR-4) -- awal jawaban sudah cukup
-- dan menolaknya akan mematahkan pertanyaan lanjutan setelah jawaban panjang."""


class TurnIn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str

    @field_validator("content")
    @classmethod
    def _potong(cls, v: str) -> str:
        return v[:MAKS_KONTEN_RIWAYAT]


class StudentProfileIn(BaseModel):
    """Angkatan dan prodi penanya, diurai widget dari NIM. Bukan NIM itu sendiri:
    nomor urutnya tidak pernah meninggalkan peramban (PRD §11)."""

    program_code: str = Field(pattern=r"^\d{4}$")
    """`code` dari `GET /api/programs` -- digit 4-7 NIM. Kode yang tidak dikenal
    ditolak 422."""
    intake_year: int = Field(ge=ANGKATAN_PERTAMA, le=2099)
    """Tahun angkatan, mis. 2024 untuk NIM berawalan "240". Tahun yang belum
    tiba ditolak 422."""


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAKS_PERTANYAAN)
    session_id: str = Field(min_length=8, max_length=128)
    history: list[TurnIn] = Field(default_factory=list, max_length=50)
    """Hanya `HISTORY_WINDOW` giliran terakhir yang disimpan; sisanya dibuang di
    sini, karena pipeline memang hanya membaca sebanyak itu. Lebih dari 50
    ditolak: portal mengirim tiga, dan daftar raksasa tetap harus diurai
    seluruhnya sebelum dipotong."""
    unit: str | None = Field(default=None, max_length=200)
    """`name` dari `GET /api/units`: retrieval hanya mencari di dokumen unit itu.
    Kosong berarti semua unit. Nama yang tidak terdaftar ditolak 422."""
    profile: StudentProfileIn | None = None
    """Bukan filter retrieval: diteruskan ke LLM supaya ketentuan yang berbeda
    per prodi atau angkatan dijawab untuk penanya (`app.prodi`). Kosong = tanpa
    penyesuaian."""

    @field_validator("history")
    @classmethod
    def _giliran_terakhir(cls, v: list[TurnIn]) -> list[TurnIn]:
        return v[-HISTORY_WINDOW:]

    @field_validator("unit")
    @classmethod
    def _kosong_berarti_semua(cls, v: str | None) -> str | None:
        return v if v and v.strip() else None


class UnitOut(BaseModel):
    """Satu pilihan di menu unit chatbot."""

    name: str
    """Dikirim kembali apa adanya sebagai `unit` pada `POST /api/chat`."""
    description: str | None = None


class ProgramOut(BaseModel):
    """Satu program studi, untuk mengurai dan menampilkan NIM di widget."""

    code: str
    """Digit 4-7 NIM; dikirim sebagai `profile.program_code`."""
    name: str
    level: str
    """`S1` atau `S2`."""
    faculty: str


class FaqQuestion(BaseModel):
    """Satu pertanyaan siap klik di menu topik chatbot."""

    question: str
    """Dikirim apa adanya sebagai `question` pada `POST /api/chat`, bersama
    unit topiknya."""


class EmbedKeyInfo(BaseModel):
    """Yang perlu diketahui portal tentang satu kunci sematan yang masih berlaku."""

    allowed_origins: list[str]
    """Menjadi `frame-ancestors` halaman `/embed`. Kosong = situs mana pun."""


class CitationOut(BaseModel):
    """Isi kartu sitasi FE-2 -- cukup untuk membuka PDF di halaman yang tepat."""

    title: str
    page: int
    document_id: str
    file_path: str
    """Kosong untuk sumber tanpa berkas; jangan dijadikan tautan."""
    type: DocumentType = DocumentType.PDF
    """`tanya_jawab` berarti sumbernya diketik admin di dashboard, bukan PDF:
    tidak ada berkas yang bisa dibuka dan nomor halaman tidak berarti apa-apa,
    jadi kartunya harus tampil tanpa tautan dan tanpa "hal. N"."""


class ContactOut(BaseModel):
    unit: str
    service_hours: str
    contact: str


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
    comment: str | None = Field(default=None, max_length=1000)
