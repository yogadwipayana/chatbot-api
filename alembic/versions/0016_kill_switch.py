"""Tabel `kill_switch`: kill switch tetap menyala setelah restart.

Sebelumnya status kill switch hanya ada di memori proses. Deploy, crash, atau
restart kontainer menyalakan lagi layanan chat yang dimatikan superadmin karena
insiden, tanpa siapa pun menyadarinya. Satu baris (`id` = 1) berarti layanan
dimatikan; tanpa baris berarti hidup.

Selama DB pengembangan masih sama dengan DB produksi: jangan jalankan `alembic
upgrade` dari lokal sebelum image server memuat revisi ini. Kalau tidak,
restart kontainer server berikutnya macet di "Can't locate revision 0016".
Sampai tabelnya ada, aplikasi tetap jalan dengan kill switch di memori saja
(`app.security.killswitch.simpan_status`).

Revision ID: 0016
Revises: 0015
Create Date: 2026-10-09
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "kill_switch",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("engaged_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("engaged_by", sa.String(255)),
        sa.CheckConstraint("id = 1", name="ck_kill_switch_satu_baris"),
    )


def downgrade() -> None:
    op.drop_table("kill_switch")
