"""Skema awal -- PRD §10.

Revision ID: 0001
Revises:
Create Date: 2026-09-10
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1024
"""Harus sama dengan dimensi keluaran EMBED_MODEL. Mengubah nilai ini berarti
re-index seluruh dokumen, bukan sekadar migrasi kolom (PRD §10)."""

FTS_CONFIG = "indonesian"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("judul", sa.String(500), nullable=False),
        sa.Column("unit", sa.String(200), nullable=False),
        sa.Column("file_path", sa.String(1000), nullable=False),
        sa.Column("tahun_berlaku", sa.Integer()),
        sa.Column("valid_until", sa.Date()),
        sa.Column("uploaded_by", sa.String(255)),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.create_index("ix_documents_aktif", "documents", ["is_active", "valid_until"])

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("konten", sa.Text(), nullable=False),
        sa.Column("halaman", sa.Integer(), nullable=False),
        sa.Column("urutan", sa.Integer(), nullable=False),
        sa.Column("tsv", postgresql.TSVECTOR()),
    )
    op.execute(f"ALTER TABLE chunks ADD COLUMN embedding vector({EMBEDDING_DIM})")
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")

    # HNSW dibuat lewat SQL mentah karena butuh operator class cosine.
    # `vector_cosine_ops` harus cocok dengan operator `<=>` di retriever;
    # index dengan operator class lain akan diabaikan optimizer secara diam-diam.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("user_hash", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_conversations_session_id", "conversations", ["session_id"])

    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("konten", sa.Text(), nullable=False),
        sa.Column("retrieved_chunk_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True))),
        sa.Column("top_score", sa.Float()),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("langsmith_run_id", sa.String(64)),
        sa.Column("meta", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])

    op.create_table(
        "feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "message_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("helpful", sa.Boolean(), nullable=False),
        sa.Column("catatan", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_feedback_message_id", "feedback", ["message_id"])

    op.create_table(
        "unanswered",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("pertanyaan", sa.Text(), nullable=False),
        sa.Column("top_score", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("resolved", sa.Boolean(), nullable=False, server_default=sa.false()),
    )

    op.create_table(
        "admins",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="editor"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # tsv diisi otomatis, supaya tidak ada jalur penulisan yang bisa lupa
    # mengisinya dan diam-diam mematikan separuh retrieval hibrida.
    op.execute(
        f"""
        CREATE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
        BEGIN
            NEW.tsv := to_tsvector('{FTS_CONFIG}', COALESCE(NEW.konten, ''));
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER trg_chunks_tsv BEFORE INSERT OR UPDATE OF konten ON chunks "
        "FOR EACH ROW EXECUTE FUNCTION chunks_tsv_update()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_tsv ON chunks")
    op.execute("DROP FUNCTION IF EXISTS chunks_tsv_update()")
    op.drop_table("admins")
    op.drop_table("unanswered")
    op.drop_table("feedback")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("chunks")
    op.drop_table("documents")
