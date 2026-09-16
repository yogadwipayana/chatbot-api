"""Skema I/O dashboard admin (AD-1..AD-6).

Nama kelas sengaja sama dengan nama skema di `api.yaml`: FastAPI menamai skema
OpenAPI sesuai nama kelas, dan `tests/api/test_openapi_contract.py`
membandingkan field wajib keduanya berdasarkan nama.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from app.admin.permissions import AdminRole
from app.rag.chain import OutcomeKind
from app.rag.threshold import Decision, Reason
from app.schemas.chat import ContactOut

# --- AD-1 -----------------------------------------------------------------


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"]


# --- AD-2, AD-3 -------------------------------------------------------------


class Document(BaseModel):
    id: str
    judul: str
    unit: str
    tahun_berlaku: int | None = None
    valid_until: date | None = None
    updated_at: datetime
    is_active: bool
    jumlah_chunk: int
    stale: bool
    """AD-2: >6 bulan tidak diperbarui, atau sudah lewat masa berlaku."""
    uploaded_by: str | None = None
    """Email admin pengunggah. PRD §12: dokumen usang dimitigasi lewat admin bernama."""


class DocumentPage(BaseModel):
    items: list[Document]
    total: int
    jumlah_stale: int
    """Dokumen AKTIF yang perlu ditinjau, untuk lencana angka di navigasi.
    Dokumen nonaktif tidak dihitung karena sudah tidak terambil retrieval."""


class DocumentUpdate(BaseModel):
    """Hanya field yang dikirim yang diubah."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    judul: str | None = Field(default=None, min_length=3, max_length=500)
    unit: str | None = Field(default=None, min_length=2, max_length=200)
    tahun_berlaku: int | None = Field(default=None, ge=2000, le=2100)
    valid_until: date | None = None
    is_active: bool | None = None

    @model_validator(mode="after")
    def _field_wajib_tidak_boleh_null(self) -> DocumentUpdate:
        """`tahun_berlaku` dan `valid_until` boleh dikosongkan; tiga ini tidak."""
        for nama in ("judul", "unit", "is_active"):
            if nama in self.model_fields_set and getattr(self, nama) is None:
                raise ValueError(f"{nama} tidak boleh kosong")
        return self


class IngestionResult(BaseModel):
    document_id: str
    jumlah_halaman: int
    jumlah_chunk: int


class Chunk(BaseModel):
    id: str
    konten: str
    halaman: int
    urutan: int


# --- AD-4 -------------------------------------------------------------------


class UnansweredGroup(BaseModel):
    """AD-4: pertanyaan mirip dikelompokkan beserta frekuensinya."""

    ids: list[str]
    """Seluruh baris `unanswered` dalam kelompok. Menandai kelompok selesai
    berarti mengirim PATCH untuk setiap id."""
    contoh_pertanyaan: str
    jumlah: int
    top_score_rata2: float | None = None
    terakhir_ditanyakan: datetime
    resolved: bool


class UnansweredUpdate(BaseModel):
    resolved: bool


# --- AD-6 -------------------------------------------------------------------


class TestQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    vector_threshold: float | None = Field(default=None, ge=0, le=1)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str | None = None
    judul: str
    halaman: int
    konten: str
    rrf_score: float
    raw_scores: dict[str, float]
    """Hanya sumber yang benar-benar menemukan chunk ini yang punya kunci."""
    ranks: dict[str, int]


class ThresholdDecisionOut(BaseModel):
    decision: Decision
    reason: Reason
    top_vector_score: float | None = None
    top_lexical_score: float | None = None


class ThresholdValues(BaseModel):
    """Ambang yang benar-benar dipakai pada uji coba ini (FR-3)."""

    vector: float
    fulltext: float


class TestQueryResponse(BaseModel):
    kind: OutcomeKind
    text: str
    rewritten_query: str | None = None
    retrieved: list[RetrievedChunk]
    decision: ThresholdDecisionOut | None
    """None bila pertanyaan dialihkan ke konseling (FR-7) sebelum retrieval."""
    ambang: ThresholdValues
    contacts: list[ContactOut]
    escalated: bool
    latency_ms: int
    langsmith_run_id: str | None = None


# --- AD-5 -------------------------------------------------------------------


class DailyVolume(BaseModel):
    tanggal: date
    jumlah: int


class TopicCount(BaseModel):
    topik: str
    jumlah: int


class KindBreakdown(BaseModel):
    answer: int
    refusal: int
    support: int
    smalltalk: int
    """Sapaan dan basa-basi. Bukan pertanyaan administrasi, jadi jangan
    dibandingkan dengan `answer` seolah keduanya setara."""


class Stats(BaseModel):
    sejak: date
    sampai: date
    total_percakapan: int
    total_pertanyaan: int
    rincian_jenis: KindBreakdown
    jumlah_feedback: int
    rasio_feedback_positif: float | None
    rasio_tak_terjawab: float | None
    volume_harian: list[DailyVolume]
    topik_populer: list[TopicCount]
    biaya_usd_berjalan: float
    pesan_tanpa_estimasi_biaya: int
    """Jawaban yang memanggil LLM tetapi modelnya belum punya tarif di
    `costs.PRICES_PER_MTOK`. Lebih dari nol berarti `biaya_usd_berjalan` kurang."""
    latency_p95_ms: int | None


# --- FR-9 -------------------------------------------------------------------


class KillSwitchRequest(BaseModel):
    engaged: bool
    alasan: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _alasan_wajib_saat_menyalakan(self) -> KillSwitchRequest:
        if self.engaged and not (self.alasan or "").strip():
            raise ValueError("alasan wajib diisi saat mematikan layanan chat")
        return self


class KillSwitchState(BaseModel):
    engaged: bool
    reason: str | None = None
    engaged_at: datetime | None = None
    engaged_by: str | None = None


# --- Akun dashboard dan level akses ------------------------------------------


def _rapikan_unit(v: str | None) -> str | None:
    """Spasi berlebih dirapikan saat disimpan, huruf besar dibiarkan.

    Pencocokan unit memang sudah mengabaikan spasi, tetapi nama unit juga
    tampil di dashboard dan di pesan galat -- "Biro  Keuangan" terlihat salah
    ketik di sana.
    """
    if v is None:
        return None
    return " ".join(v.split()) or None


class AdminUser(BaseModel):
    """Akun dashboard. Hash kata sandi tidak pernah ikut dikirim."""

    id: str
    email: str
    role: AdminRole
    is_active: bool
    nama: str | None = None
    unit: str | None = None
    """Wajib untuk staf/dosen: membatasi dokumen yang dapat dikelola."""
    created_at: datetime | None = None
    last_login_at: datetime | None = None


class AdminUserCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    email: EmailStr
    role: AdminRole
    nama: str | None = Field(default=None, max_length=200)
    unit: str | None = Field(default=None, max_length=200)

    @field_validator("unit")
    @classmethod
    def _rapikan(cls, v: str | None) -> str | None:
        return _rapikan_unit(v)


class AdminUserUpdate(BaseModel):
    """Hanya field yang dikirim yang diubah."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    nama: str | None = Field(default=None, max_length=200)
    role: AdminRole | None = None
    unit: str | None = Field(default=None, max_length=200)
    is_active: bool | None = None

    @field_validator("unit")
    @classmethod
    def _rapikan(cls, v: str | None) -> str | None:
        return _rapikan_unit(v)

    @model_validator(mode="after")
    def _field_wajib_tidak_boleh_null(self) -> AdminUserUpdate:
        for nama in ("role", "is_active"):
            if nama in self.model_fields_set and getattr(self, nama) is None:
                raise ValueError(f"{nama} tidak boleh kosong")
        return self


class AdminUserCreated(BaseModel):
    user: AdminUser
    password_sementara: str
    """Ditampilkan sekali. Serahkan lewat jalur aman; pengguna menggantinya sendiri."""


class TemporaryPassword(BaseModel):
    password_sementara: str


class PasswordChange(BaseModel):
    password_lama: str = Field(min_length=1, max_length=256)
    password_baru: str = Field(min_length=12, max_length=72)

    @field_validator("password_baru")
    @classmethod
    def _batas_bcrypt(cls, v: str) -> str:
        # bcrypt hanya membaca 72 byte pertama; sisanya diam-diam diabaikan.
        if len(v.encode("utf-8")) > 72:
            raise ValueError("kata sandi baru maksimal 72 byte")
        return v
