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
from app.db.models import DocumentType
from app.embed_keys import rapikan_daftar_asal
from app.rag.chain import OutcomeKind
from app.rag.threshold import Decision, Reason
from app.schemas.chat import AttachmentOut, ContactOut


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
    title: str
    unit: str
    effective_year: int | None = None
    valid_until: date | None = None
    updated_at: datetime
    is_active: bool
    chunk_count: int
    stale: bool
    """AD-2: >6 bulan tidak diperbarui, atau sudah lewat masa berlaku."""
    uploaded_by: str | None = None
    """Email admin pengunggah. PRD §12: dokumen usang dimitigasi lewat admin bernama."""


class DocumentPage(BaseModel):
    items: list[Document]
    total: int
    stale_count: int
    """Dokumen AKTIF yang perlu ditinjau, untuk lencana angka di navigasi.
    Dokumen nonaktif tidak dihitung karena sudah tidak terambil retrieval."""


class DocumentUpdate(BaseModel):
    """Hanya field yang dikirim yang diubah."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str | None = Field(default=None, min_length=3, max_length=500)
    unit: str | None = Field(default=None, min_length=2, max_length=200)
    effective_year: int | None = Field(default=None, ge=2000, le=2100)
    valid_until: date | None = None
    is_active: bool | None = None

    @model_validator(mode="after")
    def _field_wajib_tidak_boleh_null(self) -> DocumentUpdate:
        """`effective_year` dan `valid_until` boleh dikosongkan; tiga ini tidak."""
        for nama in ("title", "unit", "is_active"):
            if nama in self.model_fields_set and getattr(self, nama) is None:
                raise ValueError(f"{nama} tidak boleh kosong")
        return self


class IngestionResult(BaseModel):
    document_id: str
    page_count: int
    chunk_count: int
    warnings: list[str] = []
    """Catatan mutu dokumen yang baru diunggah, mis. teks terbaca sangat sedikit.

    Bukan galat: unggahan tetap berhasil. Ditampilkan admin agar dokumen yang
    isinya didominasi gambar tidak diam-diam menghasilkan jawaban yang tipis."""


class Chunk(BaseModel):
    id: str
    content: str
    page: int
    position: int


# --- Entri tanya jawab ------------------------------------------------------


class FaqEntry(BaseModel):
    """Satu pasang pertanyaan-jawaban yang dipakai chatbot seperti dokumen.

    Tidak ada `page_count` maupun berkas: yang tersimpan hanya teks yang
    diketik admin. `chunk_count` tetap ditampilkan karena jawaban panjang
    dipecah, dan jumlah potongan itulah yang benar-benar masuk indeks.
    """

    id: str
    question: str
    answer: str
    unit: str
    valid_until: date | None = None
    updated_at: datetime
    is_active: bool
    chunk_count: int
    stale: bool
    """Sama dengan dokumen: >6 bulan tidak diperbarui, atau sudah lewat masa berlaku."""
    uploaded_by: str | None = None


class FaqPage(BaseModel):
    items: list[FaqEntry]
    total: int


class FaqEntryCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=5, max_length=500)
    """Tampil apa adanya sebagai judul sumber pada kartu sitasi mahasiswa."""
    answer: str = Field(min_length=10, max_length=5000)
    unit: str = Field(min_length=2, max_length=200)
    valid_until: date | None = None

    @field_validator("unit")
    @classmethod
    def _rapikan(cls, v: str) -> str:
        return _rapikan_unit(v) or v


class FaqEntryUpdate(BaseModel):
    """Hanya field yang dikirim yang diubah."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str | None = Field(default=None, min_length=5, max_length=500)
    answer: str | None = Field(default=None, min_length=10, max_length=5000)
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
        for nama in ("question", "answer", "unit", "is_active"):
            if nama in self.model_fields_set and getattr(self, nama) is None:
                raise ValueError(f"{nama} tidak boleh kosong")
        return self


# --- AD-4 -------------------------------------------------------------------


class UnansweredGroup(BaseModel):
    """AD-4: pertanyaan mirip dikelompokkan beserta frekuensinya."""

    ids: list[str]
    """Seluruh baris `unanswered_questions` dalam kelompok. Menandai kelompok
    selesai berarti mengirim PATCH untuk setiap id."""
    sample_question: str
    count: int
    avg_top_score: float | None = None
    last_asked_at: datetime
    resolved: bool
    unit: str | None = None
    """Unit yang dipilih mahasiswa saat menanyakan `sample_question`. Chatbot
    hanya mencari di dokumen unit itu, jadi uji ulangnya juga harus begitu.
    None bila pesan asalnya sudah terhapus dari log."""


class UnansweredPage(BaseModel):
    items: list[UnansweredGroup]
    total: int
    """Jumlah kelompok yang cocok dengan seluruh filter, untuk penomoran halaman."""
    question_count: int
    """Jumlah pertanyaan di seluruh kelompok itu, bukan hanya di halaman ini."""
    max_count: int
    """`count` kelompok terbesar di seluruh halaman. Batang frekuensi diskalakan
    terhadap angka ini, supaya kelompok kecil di halaman berikutnya tidak tampak
    sama besar dengan kelompok terbesar."""


class UnansweredUpdate(BaseModel):
    resolved: bool


# --- Umpan balik mahasiswa (FE-5) -------------------------------------------


class FeedbackItem(BaseModel):
    """Satu penilaian 👍/👎 beserta pasangan pertanyaan-jawaban yang dinilai.

    Rasio kepuasan di AD-5 hanya memberi tahu ada yang salah; baris inilah yang
    memberi tahu apanya. `question` diambil dari pesan mahasiswa terakhir
    sebelum jawaban ini di percakapan yang sama.
    """

    id: str
    message_id: str
    helpful: bool
    created_at: datetime
    answer: str
    comment: str | None = None
    """Isian bebas mahasiswa. FE-5 tidak mewajibkannya, jadi sebagian besar
    umpan balik hanya berupa jempol tanpa penjelasan."""
    question: str | None = None
    """None bila pesan pertanyaannya sudah terhapus dari log. Pertanyaan
    sensitif (FR-7) berisi penanda tetap, bukan kalimat aslinya."""
    kind: str | None = None
    """`messages.meta->>'kind'` apa adanya -- bukan enum tertutup: nilai baru
    di backend tidak boleh membuat halaman ini gagal memuat."""
    top_score: float | None = None
    unit: str | None = None
    """`messages.meta->>'unit'`: unit yang dipilih mahasiswa saat bertanya."""


class FeedbackPage(BaseModel):
    items: list[FeedbackItem]
    total: int
    """Jumlah baris yang cocok dengan seluruh filter, untuk penomoran halaman."""
    positive_count: int
    negative_count: int
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
    title: str
    type: DocumentType = DocumentType.PDF
    """Asal potongan ini: dokumen PDF atau entri tanya jawab."""
    page: int
    content: str
    rrf_score: float
    raw_scores: dict[str, float]
    """Hanya sumber yang benar-benar menemukan chunk ini yang punya kunci."""
    ranks: dict[str, int]
    neighbor_of: str | None = None
    """`chunk_id` sumbernya bila ini potongan lanjutan (`RETRIEVAL_NEIGHBORS`).
    Potongan lanjutan ikut karena sumbernya, jadi memang tidak punya skor."""


class GateVerdictOut(BaseModel):
    """Vonis gerbang pada uji coba ini; None bila JEV mati dan saringan aturan
    meloloskan pesannya, atau alur berhenti sebelum gerbang."""

    label: str
    confidence: float
    blocked: bool
    error: str | None = None
    """Galat atau lewat tenggat; saat itu pesan diteruskan (fail-open)."""
    source: Literal["jev", "rules"] = "jev"
    """`rules` = saringan aturan, tanpa model; keyakinannya selalu 1."""


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
    """None bila alur berhenti sebelum ambang dinilai: konseling (FR-7), sapaan,
    atau diblokir gerbang JEV."""
    llm_called: bool = False
    refusal_source: Literal["threshold", "llm"] | None = None
    """Hanya untuk `refusal`: `threshold` = ambang FR-3 menolak dan LLM tidak
    dipanggil; `llm` = lolos ambang, tetapi LLM menilai isinya tidak menjawab."""
    rejection_source: Literal["jev", "rules", "llm"] | None = None
    """Hanya untuk `rejected`: gerbang JEV, saringan aturan, atau LLM penjawab
    yang membalas `[DI_LUAR_TOPIK]`."""
    gate: GateVerdictOut | None = None
    thresholds: ThresholdValues
    contacts: list[ContactOut]
    escalated: bool
    attachments: list[AttachmentOut]
    """Daftar dari tool yang tampil di bawah jawaban mahasiswa (docs/tool-call.md §10a)."""
    latency_ms: int
    langsmith_run_id: str | None = None
    """Akar trace uji coba ini. None saat tracing mati (FR-8)."""
    turn_id: str | None = None
    """Giliran uji coba ini di halaman Log (tab Graf). None bila log SQLite mati."""


# --- AD-5 -------------------------------------------------------------------


class DailyVolume(BaseModel):
    date: date
    count: int


class TopicCount(BaseModel):
    topic: str
    count: int


class KindBreakdown(BaseModel):
    answer: int
    refusal: int
    support: int
    smalltalk: int
    """Sapaan dan basa-basi. Bukan pertanyaan administrasi, jadi jangan
    dibandingkan dengan `answer` seolah keduanya setara."""
    rejected: int
    """Dihentikan gerbang JEV (nonsense, manipulasi, di luar topik)."""


class ProgramStat(BaseModel):
    """Pertanyaan dari satu prodi, menurut profil NIM penanya."""

    code: str
    name: str
    """Nama prodi; kodenya sendiri bila sudah tidak ada di `app.prodi`."""
    level: str | None
    question_count: int
    refusal_count: int
    """Yang ditolak (`kind = refusal`): celah dokumen untuk prodi ini."""


class IntakeYearStat(BaseModel):
    """Pertanyaan dari satu angkatan, menurut profil NIM penanya."""

    intake_year: int
    question_count: int
    refusal_count: int


class Stats(BaseModel):
    since: date
    until: date
    total_conversations: int
    total_questions: int
    kind_breakdown: KindBreakdown
    feedback_count: int
    positive_feedback_ratio: float | None
    unanswered_ratio: float | None
    daily_volume: list[DailyVolume]
    top_topics: list[TopicCount]
    running_cost_usd: float
    messages_without_cost_estimate: int
    """Jawaban yang memanggil LLM tetapi modelnya belum punya tarif di
    `costs.PRICES_PER_MTOK`. Lebih dari nol berarti `running_cost_usd` kurang."""
    latency_p95_ms: int | None
    questions_with_profile: int
    """Pertanyaan yang penanyanya mengisi NIM. Penyebut kedua rincian di
    bawah -- bukan `total_questions`, yang juga memuat penanya tanpa NIM."""
    program_breakdown: list[ProgramStat]
    """Semua prodi terdaftar, juga yang nol; terbanyak lebih dulu."""
    intake_year_breakdown: list[IntakeYearStat]
    """Hanya angkatan yang pernah bertanya; terbaru lebih dulu."""


class CostByModel(BaseModel):
    type: Literal["llm_chat", "embedding_chat"]
    model: str
    call_count: int
    input_tokens: int
    output_tokens: int
    tokens: int
    cost_usd: float


class DailyCost(BaseModel):
    date: date
    call_count: int
    llm_tokens: int
    embed_tokens: int
    llm_cost_usd: float
    embedding_cost_usd: float
    cost_usd: float


class Costs(BaseModel):
    since: date
    until: date
    llm_call_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    llm_cost_usd: float
    chat_embed_tokens: int
    chat_embed_cost_usd: float
    chat_embed_count: int
    llm_calls_without_cost: int
    chat_embeds_without_cost: int
    model_breakdown: list[CostByModel]
    daily_costs: list[DailyCost]


# --- FR-9 -------------------------------------------------------------------


class KillSwitchRequest(BaseModel):
    engaged: bool
    reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _alasan_wajib_saat_menyalakan(self) -> KillSwitchRequest:
        if self.engaged and not (self.reason or "").strip():
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
    name: str | None = None
    unit: str | None = None
    """Wajib untuk staf/dosen: membatasi dokumen yang dapat dikelola."""
    created_at: datetime | None = None
    last_login_at: datetime | None = None


class AdminUserCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    email: EmailStr
    role: AdminRole
    name: str | None = Field(default=None, max_length=200)
    unit: str | None = Field(default=None, max_length=200)

    @field_validator("unit")
    @classmethod
    def _rapikan(cls, v: str | None) -> str | None:
        return _rapikan_unit(v)


class AdminUserUpdate(BaseModel):
    """Hanya field yang dikirim yang diubah."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str | None = Field(default=None, max_length=200)
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


def _nama_unit(v: str | None) -> str | None:
    """Rapikan spasi; tolak nama yang tidak dapat dibawa di path URL."""
    v = _rapikan_unit(v)
    if v is not None and "/" in v:
        raise ValueError("nama unit tidak boleh memuat '/'")
    return v


class AdminUnit(BaseModel):
    """Satu unit layanan, termasuk yang nonaktif, beserta pemakaiannya."""

    name: str
    description: str | None = None
    sort_order: int
    is_active: bool
    document_count: int
    account_count: int


class AdminUnitCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=500)
    sort_order: int | None = Field(default=None, ge=0, le=10000)
    """Kosong = diletakkan paling akhir di menu."""

    @field_validator("name")
    @classmethod
    def _rapikan(cls, v: str) -> str:
        return _nama_unit(v) or v


class AdminUnitUpdate(BaseModel):
    """Hanya field yang dikirim yang diubah."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=500)
    sort_order: int | None = Field(default=None, ge=0, le=10000)
    is_active: bool | None = None

    @field_validator("name")
    @classmethod
    def _rapikan(cls, v: str | None) -> str | None:
        return _nama_unit(v)

    @model_validator(mode="after")
    def _field_wajib_tidak_boleh_null(self) -> AdminUnitUpdate:
        for nama in ("name", "sort_order", "is_active"):
            if nama in self.model_fields_set and getattr(self, nama) is None:
                raise ValueError(f"{nama} tidak boleh kosong")
        return self


# --- Kunci sematan (khusus superadmin) ------------------------------------


class EmbedKey(BaseModel):
    """Satu kunci sematan beserta pemakaiannya."""

    key: str
    """`emb_` + 24 karakter. Bukan rahasia: tertulis di kode sumber situs penyemat."""
    name: str
    allowed_origins: list[str]
    """Kosong = situs mana pun boleh memakai kunci ini."""
    is_active: bool
    created_by: str | None = None
    created_at: datetime
    questions_30d: int
    last_used_at: datetime | None = None
    embed_code: str | None = None
    """Baris `<script>` siap tempel; None bila `PORTAL_URL` belum diisi."""


def _daftar_asal(v: list[str] | None) -> list[str] | None:
    return None if v is None else rapikan_daftar_asal(v)


class EmbedKeyCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=200)
    allowed_origins: list[str] = Field(default_factory=list)
    """Alamat situs lengkap tanpa path, mis. `https://pmb.instiki.ac.id`. Kosong =
    situs mana pun. Garis miring akhir, huruf besar, dan port bawaan dirapikan."""

    @field_validator("allowed_origins")
    @classmethod
    def _rapikan_asal(cls, v: list[str]) -> list[str]:
        return _daftar_asal(v) or []


class EmbedKeyUpdate(BaseModel):
    """Hanya field yang dikirim yang diubah."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str | None = Field(default=None, min_length=1, max_length=200)
    allowed_origins: list[str] | None = None
    is_active: bool | None = None

    @field_validator("allowed_origins")
    @classmethod
    def _rapikan_asal(cls, v: list[str] | None) -> list[str] | None:
        return _daftar_asal(v)

    @model_validator(mode="after")
    def _field_wajib_tidak_boleh_null(self) -> EmbedKeyUpdate:
        for nama in ("name", "allowed_origins", "is_active"):
            if nama in self.model_fields_set and getattr(self, nama) is None:
                raise ValueError(f"{nama} tidak boleh kosong")
        return self


class AdminUserCreated(BaseModel):
    user: AdminUser
    temporary_password: str
    """Ditampilkan sekali. Serahkan lewat jalur aman; pengguna menggantinya sendiri."""


class TemporaryPassword(BaseModel):
    temporary_password: str


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=72)

    @field_validator("new_password")
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
    chat_daily_limit: int
    """Pertanyaan per hari dari semua sumber; terlampaui = kill switch. 0 = tanpa batas."""


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
    chat_daily_limit: int | None = Field(default=None, ge=0, le=1_000_000)

    def perubahan(self) -> dict[str, float | None]:
        """Field yang benar-benar dikirim; `None` = kembalikan ke nilai `.env`.

        `exclude_unset` memisahkan "tidak disebut" dari "disebut sebagai null";
        keduanya terlihat sama pada model yang seluruh field-nya opsional.
        """
        return self.model_dump(exclude_unset=True)


class RuntimeConfig(BaseModel):
    """Isi halaman Konfigurasi: yang berlaku sekarang, asalnya, dan jejaknya."""

    values: RuntimeConfigValues
    """Yang dipakai layanan saat ini."""
    env_values: RuntimeConfigValues
    """Yang tertulis di `.env` server. Tombol "kembalikan" menuju ke sini."""
    overridden: list[str]
    """Nama field yang sedang ditimpa dari dashboard."""
    chat_model: str
    embed_model: str
    base_url: str | None = None
    """Endpoint OpenAI-compatible; kosong berarti OpenAI resmi."""
    api_key_set: bool
    """Kunci API-nya sendiri tidak pernah dikirim ke peramban."""
    updated_at: datetime | None = None
    updated_by: str | None = None
    warning: str | None = None
    """Terisi bila nilai tersimpan tidak dapat dipakai (mis. `.env` berubah
    sehingga kombinasinya melanggar aturan) dan layanan sementara kembali ke
    `.env`."""
