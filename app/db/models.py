"""Model SQLAlchemy 2.0 -- skema PRD §10.

Satu database untuk metadata, vektor, log, dan akun admin. Full-text search
memakai `tsvector` bawaan Postgres; tidak ada Elasticsearch.

Kolom `embedding` memakai tipe `Vector` dari pgvector. Dimensinya harus sama
dengan dimensi keluaran `EMBED_MODEL` -- mengganti model embedding berarti
migrasi kolom DAN re-index seluruh dokumen (PRD §10).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import StrEnum

from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

try:  # pgvector opsional saat menjalankan unit test tanpa DB
    from pgvector.sqlalchemy import Vector
except ModuleNotFoundError:  # pragma: no cover - hanya jalur degradasi
    Vector = None  # type: ignore[assignment]

EMBEDDING_DIM = 1024


class JenisDokumen(StrEnum):
    """Asal isi satu baris `documents`.

    `pdf`: berkas resmi yang diunggah admin; isinya hidup di penyimpanan objek
    dan `file_path` menunjuk ke sana.
    `tanya_jawab`: satu pasang pertanyaan-jawaban yang diketik admin langsung di
    dashboard, tanpa berkas sama sekali.

    Keduanya sengaja berbagi satu tabel: filter dokumen aktif (FR-2), masa
    berlaku, unit, dan chunking berlaku sama persis, sehingga retrieval tidak
    perlu tahu bedanya. Yang berbeda hanya dari mana isinya berasal dan
    bagaimana admin menyuntingnya.
    """

    PDF = "pdf"
    TANYA_JAWAB = "tanya_jawab"


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    judul: Mapped[str] = mapped_column(String(500), nullable=False)
    """Untuk entri tanya jawab: pertanyaannya sendiri. Judul inilah yang muncul
    sebagai sumber pada sitasi yang dilihat mahasiswa (FE-2)."""
    unit: Mapped[str] = mapped_column(String(200), nullable=False)
    jenis: Mapped[str] = mapped_column(
        String(20),
        default=JenisDokumen.PDF,
        server_default=JenisDokumen.PDF.value,
        nullable=False,
    )
    file_path: Mapped[str | None] = mapped_column(String(1000))
    """Kunci objek di penyimpanan. NULL untuk entri tanya jawab: tidak ada berkas
    yang bisa dibuka, dan `GET /api/documents/{id}/file` menolaknya 404."""
    nama_file: Mapped[str | None] = mapped_column(String(255))
    """Nama asli unggahan untuk nama tab browser dan berkas unduhan.

    `file_path` tetap memakai key berbasis UUID agar nama pengguna tidak menjadi
    bagian dari lokasi objek. NULL untuk entri tanya jawab dan baris lama.
    """
    jawaban: Mapped[str | None] = mapped_column(Text)
    """Jawaban entri tanya jawab, apa adanya seperti diketik admin. Ini sumber
    kebenarannya yang dapat disunting; chunk hanyalah turunannya -- sejajar
    dengan PDF, yang sumbernya berkas asli dan chunk-nya hasil ekstraksi."""
    tahun_berlaku: Mapped[int | None] = mapped_column(Integer)
    valid_until: Mapped[date | None] = mapped_column(Date)
    """NULL = berlaku tanpa batas. Diperiksa di setiap retrieval (FR-2)."""
    uploaded_by: Mapped[str | None] = mapped_column(String(255))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_documents_aktif", "is_active", "valid_until"),
        CheckConstraint(
            "jenis IN ('pdf', 'tanya_jawab')",
            name="ck_documents_jenis",
        ),
        # Satu tabel untuk dua jenis isi hanya aman bila setiap baris lengkap
        # menurut jenisnya: PDF tanpa berkas akan membuat kartu sitasi buntu,
        # dan entri tanya jawab tanpa jawaban tidak dapat disunting kembali.
        CheckConstraint(
            "(jenis = 'pdf' AND file_path IS NOT NULL AND jawaban IS NULL)"
            " OR (jenis = 'tanya_jawab' AND file_path IS NULL AND jawaban IS NOT NULL)",
            name="ck_documents_isi_sesuai_jenis",
        ),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    konten: Mapped[str] = mapped_column(Text, nullable=False)
    halaman: Mapped[int] = mapped_column(Integer, nullable=False)
    urutan: Mapped[int] = mapped_column(Integer, nullable=False)
    if Vector is not None:
        embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    tsv: Mapped[str | None] = mapped_column(TSVECTOR)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        # Index HNSW WAJIB dideklarasikan di sini walau dibuat oleh migrasi.
        # Tanpa deklarasi, `alembic revision --autogenerate` menganggapnya index
        # yang harus dihapus dan menghasilkan DROP INDEX -- vector search tetap
        # jalan, tetapi diam-diam berubah menjadi sequential scan.
        # Operator class harus cocok dengan operator `<=>` (cosine) di retriever.
        *(
            (
                Index(
                    "ix_chunks_embedding_hnsw",
                    "embedding",
                    postgresql_using="hnsw",
                    postgresql_ops={"embedding": "vector_cosine_ops"},
                ),
            )
            if Vector is not None
            else ()
        ),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    user_hash: Mapped[str | None] = mapped_column(String(64))
    """Hash anonim (PRD §11). Tidak boleh dapat dikembalikan ke identitas."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = _uuid_pk()
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    konten: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[list[uuid.UUID] | None] = mapped_column(
        ARRAY(UUID(as_uuid=True))
    )
    top_score: Mapped[float | None] = mapped_column()
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    langsmith_run_id: Mapped[str | None] = mapped_column(String(64))
    """Penghubung ke trace LangSmith -- menjawab 'kenapa jawaban ini buruk'."""
    meta: Mapped[dict | None] = mapped_column(JSONB)
    """Flag eskalasi, topik risiko, estimasi biaya token (FR-6, FR-8)."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
        Index("ix_messages_created_at", "created_at"),
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = _uuid_pk()
    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    helpful: Mapped[bool] = mapped_column(Boolean, nullable=False)
    catatan: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Unanswered(Base):
    """Pertanyaan yang ditolak FR-3. Sumber utama perbaikan sistem (AD-4)."""

    __tablename__ = "unanswered"

    id: Mapped[uuid.UUID] = _uuid_pk()
    pertanyaan: Mapped[str] = mapped_column(Text, nullable=False)
    top_score: Mapped[float | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL")
    )
    """Jawaban penolakan yang mencatat baris ini. SET NULL, bukan CASCADE: log
    percakapan boleh dibersihkan, sinyal perbaikan AD-4 tidak ikut hilang."""

    __table_args__ = (Index("ix_unanswered_created_at", "created_at"),)


class Admin(Base):
    """Akun dashboard. Level akses: `app.admin.permissions`."""

    __tablename__ = "admins"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(50), default="admin", server_default="admin", nullable=False
    )
    """`staf`, `admin`, atau `superadmin`."""
    nama: Mapped[str | None] = mapped_column(String(200))
    unit: Mapped[str | None] = mapped_column(String(200))
    """Wajib untuk staf/dosen: membatasi dokumen yang dapat dikelola."""
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default=true(), nullable=False
    )
    password_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    """Token yang terbit sebelum waktu ini ditolak."""
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("role IN ('staf', 'admin', 'superadmin')", name="ck_admins_role"),
        CheckConstraint(
            "role <> 'staf' OR (unit IS NOT NULL AND btrim(unit) <> '')",
            name="ck_admins_staf_unit",
        ),
        # Login dan pencarian akun memakai lower(email); tanpa index unik ini
        # "Admin@kampus.ac.id" dan "admin@kampus.ac.id" bisa menjadi dua akun.
        Index("ix_admins_email_lower", text("lower(email)"), unique=True),
    )


class RuntimeConfigEntry(Base):
    """Nilai `.env` yang boleh ditimpa dari dashboard, satu baris per parameter.

    Hanya parameter yang benar-benar ditimpa yang punya baris di sini: tidak ada
    baris berarti "ikut `.env`". Dengan begitu mengubah `.env` lalu restart tetap
    berlaku untuk parameter yang belum pernah disentuh dari dashboard, dan
    "kembalikan ke nilai .env" cukup menghapus barisnya.

    Nilainya disimpan sebagai teks, sama seperti asalnya di `.env`, dan di-parse
    ulang oleh `Settings` -- satu tempat validasi untuk kedua sumber.
    """

    __tablename__ = "runtime_config"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    """Nama field `app.config.Settings`, mis. `retrieval_top_n`. Daftar yang
    boleh ditimpa: `app.admin.runtime_config.DAPAT_DIUBAH`."""
    value: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_by: Mapped[str | None] = mapped_column(String(255))
    """Email admin yang mengubah, untuk jejak audit di halaman Konfigurasi."""


class OperasiPemakaian(StrEnum):
    """Apa yang memicu satu panggilan embedding di luar percakapan mahasiswa."""

    INGEST = "ingest"
    """Dokumen atau entri tanya jawab baru diindeks untuk pertama kalinya."""
    REINDEX = "reindex"
    """Entri tanya jawab yang disunting: chunk lama dibuang, embedding dihitung
    ulang. Menyunting entri yang sama berulang kali membayar penuh setiap kali,
    dan sebelum tabel ini hal itu sama sekali tidak meninggalkan jejak."""


class UsageLog(Base):
    """Buku biaya panggilan model yang tidak punya baris pesan untuk ditumpangi.

    Biaya chat menumpang `messages.meta`, tetapi embedding saat ingestion terjadi
    ketika tidak ada mahasiswa yang bertanya sama sekali -- tidak ada baris yang
    bisa dititipi. Tanpa tabel ini, halaman Biaya AD-5 diam-diam hanya melaporkan
    sebagian dari yang benar-benar dibelanjakan.

    Sengaja append-only: tidak ada jalur yang memperbarui atau menghapus barisnya.
    Buku biaya yang bisa berubah surut tidak dapat dipakai menjawab "bulan lalu
    habis berapa".
    """

    __tablename__ = "usage_log"

    id: Mapped[uuid.UUID] = _uuid_pk()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    operasi: Mapped[str] = mapped_column(String(20), nullable=False)
    """Nilai `OperasiPemakaian`."""
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    """Model yang DIMINTA, sama seperti `EMBED_MODEL` -- kunci yang cocok dengan
    `costs.PRICES_PER_MTOK`."""
    model_dilaporkan: Mapped[str | None] = mapped_column(String(200))
    """Nama menurut respons penyedia, diisi hanya bila berbeda dari `model`."""
    tokens: Mapped[int | None] = mapped_column(Integer)
    """NULL bila endpoint tidak melaporkan pemakaian -- bukan berarti nol token."""
    biaya_usd: Mapped[float | None] = mapped_column()
    """Disimpan tanpa pembulatan. Satu batch embedding bisa berharga $0,000002;
    membulatkannya per baris membuat totalnya nol. Lihat
    `costs.estimate_input_cost`."""
    biaya_sumber: Mapped[str | None] = mapped_column(String(20))
    """`provider` (angka penyedia) atau `estimasi` (hitungan kita). Tanpa penanda
    ini, dua angka berdasar berbeda dalam satu kolom tidak dapat ditafsirkan."""
    is_byok: Mapped[bool | None] = mapped_column(Boolean)
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL")
    )
    """SET NULL, bukan CASCADE -- alasan yang sama dengan `unanswered.message_id`:
    menghapus dokumen tidak boleh mengubah laporan biaya bulan yang sudah lewat."""
    keterangan: Mapped[str | None] = mapped_column(String(500))
    """Judul dokumen pada saat panggilan terjadi, supaya barisnya tetap terbaca
    setelah `document_id` menjadi NULL."""

    __table_args__ = (
        CheckConstraint("operasi IN ('ingest', 'reindex')", name="ck_usage_log_operasi"),
        CheckConstraint(
            "biaya_sumber IS NULL OR biaya_sumber IN ('provider', 'estimasi')",
            name="ck_usage_log_biaya_sumber_nilai",
        ),
        # Angka biaya tanpa asal-usul tidak dapat ditafsirkan, dan asal-usul tanpa
        # angka tidak ada artinya. Keduanya ada, atau keduanya tidak ada.
        CheckConstraint(
            "(biaya_usd IS NULL) = (biaya_sumber IS NULL)",
            name="ck_usage_log_biaya_lengkap",
        ),
        Index("ix_usage_log_created_at", "created_at"),
    )
