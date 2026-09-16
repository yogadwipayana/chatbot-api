"""Tautkan unanswered ke pesan jawaban, index untuk statistik (AD-4, AD-5).

- `unanswered.message_id` menghubungkan pertanyaan yang ditolak ke jawaban
  penolakannya, sehingga dapat ditelusuri ke percakapan dan trace-nya.
- Index `created_at` pada `messages` dan `unanswered`: seluruh query AD-5 dan
  filter `sejak` AD-4 menyaring berdasarkan rentang tanggal.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-15
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "unanswered",
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "unanswered_message_id_fkey",
        "unanswered",
        "messages",
        ["message_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_messages_created_at", "messages", ["created_at"])
    op.create_index("ix_unanswered_created_at", "unanswered", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_unanswered_created_at", table_name="unanswered")
    op.drop_index("ix_messages_created_at", table_name="messages")
    op.drop_constraint("unanswered_message_id_fkey", "unanswered", type_="foreignkey")
    op.drop_column("unanswered", "message_id")
