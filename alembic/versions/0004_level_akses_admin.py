"""Level akses akun dashboard: staf/dosen, admin, superadmin.

- `nama`, `unit`: unit wajib untuk staf/dosen dan membatasi dokumen yang dapat
  dikelolanya.
- `is_active`: menonaktifkan akun tanpa menghapus jejaknya.
- `password_changed_at`: token yang terbit sebelumnya ditolak, sehingga
  mengatur ulang kata sandi juga mengakhiri sesi lama.
- `last_login_at`: untuk menemukan akun yang tidak pernah dipakai.
- Level lama `editor` (bawaan migrasi 0001) menjadi `admin`.
- Index unik `lower(email)`: login tidak peka huruf besar, jadi email yang hanya
  berbeda huruf besar tidak boleh menjadi dua akun.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-16
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("admins", sa.Column("nama", sa.String(200)))
    op.add_column("admins", sa.Column("unit", sa.String(200)))
    op.add_column(
        "admins",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column("admins", sa.Column("password_changed_at", sa.DateTime(timezone=True)))
    op.add_column("admins", sa.Column("last_login_at", sa.DateTime(timezone=True)))

    op.execute(
        "UPDATE admins SET role = 'admin' WHERE role NOT IN ('staf', 'admin', 'superadmin')"
    )
    op.alter_column("admins", "role", server_default="admin")
    op.create_check_constraint(
        "ck_admins_role", "admins", "role IN ('staf', 'admin', 'superadmin')"
    )
    op.create_check_constraint(
        "ck_admins_staf_unit",
        "admins",
        "role <> 'staf' OR (unit IS NOT NULL AND btrim(unit) <> '')",
    )
    op.create_index("ix_admins_email_lower", "admins", [sa.text("lower(email)")], unique=True)


def downgrade() -> None:
    op.drop_index("ix_admins_email_lower", table_name="admins")
    op.drop_constraint("ck_admins_staf_unit", "admins", type_="check")
    op.drop_constraint("ck_admins_role", "admins", type_="check")
    op.alter_column("admins", "role", server_default="editor")
    op.execute("UPDATE admins SET role = 'editor' WHERE role IN ('staf', 'admin')")
    for kolom in ("last_login_at", "password_changed_at", "is_active", "unit", "nama"):
        op.drop_column("admins", kolom)
