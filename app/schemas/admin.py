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
from app.db.models import JenisDokumen
from app.rag.chain import OutcomeKind
from app.rag.threshold import Decision, Reason
from app.schemas.chat import ContactOut


def _rapikan_unit(v: str | None) -> str | None:
    """Spasi berlebih dirapikan saat disimpan, huruf besar dibiarkan.

    Pencocokan unit memang sudah mengabaikan spasi, tetapi nama unit juga
    tampil di dashboard dan di pesan galat -- "Biro  Keuangan" terlihat salah
    ketik di sana.
    """
    if v is None:
        return None
    return " ".join(v.split()) or None


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
    peringatan: list[str] = []
    """Catatan mutu dokumen yang baru diunggah, mis. teks terbaca sangat sedikit.

    Bukan galat: unggahan tetap berhasil. Ditampilkan admin agar dokumen yang
    isinya didominasi gambar tidak diam-diam menghasilkan jawaban yang tipis."""


class Chunk(BaseModel):
    id: str
    konten: str
    halaman: int
    urutan: int


# --- Entri tanya jawab ------------------------------------------------------


class FaqEntry(BaseModel):
    """Satu pasang pertanyaan-jawaban yang dipakai chatbot seperti dokumen.

    Tidak ada `jumlah_halaman` maupun berkas: yang tersimpan hanya teks yang
    diketik admin. `jumlah_chunk` tetap ditampilkan karena jawaban panjang
    dipecah, dan jumlah potongan itulah yang benar-benar masuk indeks.
    """

    id: str
    pertanyaan: str
    jawaban: str
    unit: str
    valid_until: date | None = None
    updated_at: datetime
    is_active: bool
    jumlah_chunk: int
    stale: bool
    """Sama dengan dokumen: >6 bulan tidak diperbarui, atau sudah lewat masa berlaku."""
    uploaded_by: str | None = None


class FaqPage(BaseModel):
    items: list[FaqEntry]
    total: int


class FaqEntryCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    pertanyaan: str = Field(min_length=5, max_length=500)
    """Tampil apa adanya sebagai judul sumber pada kartu sitasi mahasiswa."""
    jawaban: str = Field(min_length=10, max_length=5000)
    unit: str = Field(min_length=2, max_length=200)
    valid_until: date | None = None

    @field_validator("unit")
    @classmethod
    def _rapikan(cls, v: str) -> str:
        return _rapikan_unit(v) or v


class FaqEntryUpdate(BaseModel):
    """Hanya field yang dikirim yang diubah."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    pertanyaan: str | None = Field(default=None, min_length=5, max_length=500)
    jawaban: str | None = Field(default=None, min_length=10, max_length=5000)
    unit: str | None = Field(default=None, min_length=2, max_length=200)
    valid_until: date | None = None
    is_active: bool | None = None

    @field_validator("unit")
    @classmethod
    def _rapikan(cls, v: str | None) -> str | None:
        return _rapikan_unit(v)

    @model_validator(mode="after")
    def _field_wajib_tidak_boleh_null(self) -> FaqEntryUpdate:
        """`valid_until` boleh dikosongkan; sisanya tidak."""
        for nama in ("pertanyaan", "jawaban", "unit", "is_active"):
            if nama in self.model_fields_set and getattr(self, nama) is None:
                raise ValueError(f"{nama} tidak boleh kosong")
        return self


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


# --- Umpan balik mahasiswa (FE-5) -------------------------------------------


class FeedbackItem(BaseModel):
    """Satu penilaian 👍/👎 beserta pasangan pertanyaan-jawaban yang dinilai.

    Rasio kepuasan di AD-5 hanya memberi tahu ada yang salah; baris inilah yang
    memberi tahu apanya. `pertanyaan` diambil dari pesan mahasiswa terakhir
    sebelum jawaban ini di percakapan yang sama.
    """

    id: str
    message_id: str
    helpful: bool
    created_at: datetime
    jawaban: str
    catatan: str | None = None
    """Isian bebas mahasiswa. FE-5 tidak mewajibkannya, jadi sebagian besar
    umpan balik hanya berupa jempol tanpa penjelasan."""
    pertanyaan: str | None = None
    """None bila pesan pertanyaannya sudah terhapus dari log. Pertanyaan
    sensitif (FR-7) berisi penanda tetap, bukan kalimat aslinya."""
    kind: str | None = None
    """`messages.meta->>'kind'` apa adanya -- bukan enum tertutup: nilai baru
    di backend tidak boleh membuat halaman ini gagal memuat."""
    top_score: float | None = None


class FeedbackPage(BaseModel):
    items: list[FeedbackItem]
    total: int
    """Jumlah baris yang cocok dengan seluruh filter, untuk penomoran halaman."""
    jumlah_positif: int
    jumlah_negatif: int
    """Keduanya dihitung mengabaikan filter `helpful`, sehingga jumlah pada
    kedua tab tetap terlihat saat salah satunya sedang dipilih."""


# --- AD-6 -------------------------------------------------------------------


class TestQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    vector_threshold: float | None = Field(default=None, ge=0, le=1)
    unit: str | None = Field(default=None, max_length=200)
    """Sama dengan `ChatRequest.unit`: uji apa yang dilihat mahasiswa yang
    memilih unit ini di menu chatbot."""


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str | None = None
    judul: str
    jenis: JenisDokumen = JenisDokumen.PDF
    """Asal potongan ini: dokumen PDF atau entri tanya jawab."""
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
    """Akar trace uji coba ini. None saat tracing mati (FR-8)."""


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


class CostByModel(BaseModel):
    jenis: Literal["llm_chat", "embedding_chat", "embedding_ingestion"]
    model: str
    jumlah_panggilan: int
    input_tokens: int
    output_tokens: int
    tokens: int
    biaya_usd: float


class DailyCost(BaseModel):
    tanggal: date
    jumlah_panggilan: int
    llm_tokens: int
    embed_tokens: int
    biaya_llm_usd: float
    biaya_embedding_usd: float
    biaya_ingestion_usd: float
    biaya_usd: float


class Costs(BaseModel):
    sejak: date
    sampai: date
    jumlah_panggilan_llm: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    biaya_usd: float
    biaya_llm_usd: float
    embed_chat_tokens: int
    biaya_embed_chat_usd: float
    jumlah_embed_chat: int
    usage_log_tokens: int
    biaya_usage_log_usd: float
    jumlah_usage_log: int
    llm_tanpa_biaya: int
    embed_chat_tanpa_biaya: int
    usage_log_tanpa_biaya: int
    rincian_model: list[CostByModel]
    biaya_harian: list[DailyCost]


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


# --- Konfigurasi runtime -----------------------------------------------------


class RuntimeConfigValues(BaseModel):
    """Parameter retrieval (FR-2, FR-3) dan chunking (FR-1) yang dapat disetel.

    Field-nya sengaja dinamai persis seperti di `.env` dan
    `app.config.Settings`: admin yang membaca dokumentasi server, isi berkas
    `.env`, dan halaman Konfigurasi harus melihat nama yang sama.
    """

    retrieval_candidates: int
    retrieval_top_n: int
    rrf_k: int
    rrf_weight_vector: float
    rrf_weight_fulltext: float
    vector_threshold: float
    lexical_threshold: float
    chunk_size: int
    chunk_overlap: int


class RuntimeConfigUpdate(BaseModel):
    """Hanya field yang dikirim yang diubah.

    Nilai yang sama dengan `.env` menghapus penimpaannya, bukan menyimpan
    salinan: dengan begitu parameter itu ikut lagi bila `.env` diubah. `null`
    berarti hal yang sama tanpa perlu tahu nilai `.env`-nya: kembalikan field
    ini ke nilai server.

    Batas di sini adalah pagar kewarasan, bukan rentang yang dianjurkan;
    aturan antar-field (`chunk_overlap < chunk_size`,
    `retrieval_top_n <= retrieval_candidates`) diperiksa `app.config.Settings`
    terhadap hasil gabungannya.
    """

    model_config = ConfigDict(extra="forbid")

    retrieval_candidates: int | None = Field(default=None, ge=1, le=100)
    retrieval_top_n: int | None = Field(default=None, ge=1, le=50)
    rrf_k: int | None = Field(default=None, ge=1, le=1000)
    rrf_weight_vector: float | None = Field(default=None, ge=0, le=5)
    rrf_weight_fulltext: float | None = Field(default=None, ge=0, le=5)
    vector_threshold: float | None = Field(default=None, ge=0, le=1)
    lexical_threshold: float | None = Field(default=None, ge=0, le=1)
    chunk_size: int | None = Field(default=None, ge=200, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)

    def perubahan(self) -> dict[str, float | None]:
        """Field yang benar-benar dikirim; `None` = kembalikan ke nilai `.env`.

        `exclude_unset` memisahkan "tidak disebut" dari "disebut sebagai null";
        keduanya terlihat sama pada model yang seluruh field-nya opsional.
        """
        return self.model_dump(exclude_unset=True)


class RuntimeConfig(BaseModel):
    """Isi halaman Konfigurasi: yang berlaku sekarang, asalnya, dan jejaknya."""

    nilai: RuntimeConfigValues
    """Yang dipakai layanan saat ini."""
    nilai_env: RuntimeConfigValues
    """Yang tertulis di `.env` server. Tombol "kembalikan" menuju ke sini."""
    diubah: list[str]
    """Nama field yang sedang ditimpa dari dashboard."""
    chat_model: str
    embed_model: str
    base_url: str | None = None
    """Endpoint OpenAI-compatible; kosong berarti OpenAI resmi."""
    api_key_terisi: bool
    """Kunci API-nya sendiri tidak pernah dikirim ke peramban."""
    diperbarui_at: datetime | None = None
    diperbarui_oleh: str | None = None
    peringatan: str | None = None
    """Terisi bila nilai tersimpan tidak dapat dipakai (mis. `.env` berubah
    sehingga kombinasinya melanggar aturan) dan layanan sementara kembali ke
    `.env`."""
