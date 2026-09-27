"""Nama tabel, kolom, constraint, dan kunci `messages.meta` ke bahasa Inggris.

Nilai data TIDAK berubah (`tanya_jawab`, `staf`, `estimasi`, nama unit, dst.);
yang berganti hanya nama yang tersimpan di skema dan di kunci JSONB.

- Tabel `unanswered` -> `unanswered_questions`, beserta PK, FK, dan indeksnya.
- Kolom: `units` (`name`, `description`, `sort_order`), `documents` (`title`,
  `type`, `original_filename`, `answer`, `effective_year`), `chunks`
  (`content`, `page`, `position`), `embed_keys` (`key`, `name`,
  `allowed_origins`, `created_by`), `messages.content`, `feedback.comment`,
  `unanswered_questions.question`, `admins.name`.
- CHECK constraint dan indeks yang namanya memuat kata Indonesia. Isi CHECK
  tidak perlu ditulis ulang: Postgres menyimpannya sebagai ekspresi terurai
  yang merujuk kolom, bukan teks, sehingga ikut berganti bersama RENAME COLUMN.
  Foreign key ke `units.name` dan `embed_keys.key` pun demikian.
- Fungsi trigger `chunks_tsv_update()` WAJIB dibuat ulang: badan plpgsql
  disimpan sebagai teks dan baru diurai saat dijalankan, jadi `NEW.konten`
  tidak ikut berganti -- setiap INSERT chunk akan gagal. Trigger
  `trg_chunks_tsv` ikut dibuat ulang dengan `UPDATE OF content`.
- Kunci `messages.meta` yang ada diganti nama. Baris yang tidak memuat
  kuncinya dibiarkan, tidak ditambahi kunci kosong.

Downgrade mengembalikan seluruhnya, termasuk fungsi trigger versi `konten`.

Revision ID: 0013
Revises: 0012
Create Date: 2026-09-27
"""

from __future__ import annotations

from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None

FTS_CONFIG = "indonesian"

KOLOM: tuple[tuple[str, str, str], ...] = (
    ("units", "nama", "name"),
    ("units", "deskripsi", "description"),
    ("units", "urutan", "sort_order"),
    ("documents", "judul", "title"),
    ("documents", "jenis", "type"),
    ("documents", "nama_file", "original_filename"),
    ("documents", "jawaban", "answer"),
    ("documents", "tahun_berlaku", "effective_year"),
    ("chunks", "konten", "content"),
    ("chunks", "halaman", "page"),
    ("chunks", "urutan", "position"),
    ("embed_keys", "kunci", "key"),
    ("embed_keys", "nama", "name"),
    ("embed_keys", "asal_diizinkan", "allowed_origins"),
    ("embed_keys", "dibuat_oleh", "created_by"),
    ("messages", "konten", "content"),
    ("feedback", "catatan", "comment"),
    ("unanswered_questions", "pertanyaan", "question"),
    ("admins", "nama", "name"),
)
"""(tabel, lama, baru). Tabel memakai nama sesudah upgrade: tabel `unanswered`
sudah diganti nama sebelum kolomnya disentuh (dan sesudahnya saat downgrade)."""

CONSTRAINT: tuple[tuple[str, str, str], ...] = (
    ("documents", "ck_documents_jenis", "ck_documents_type"),
    ("documents", "ck_documents_isi_sesuai_jenis", "ck_documents_content_matches_type"),
    ("admins", "ck_admins_staf_unit", "ck_admins_staff_unit"),
    ("unanswered_questions", "unanswered_pkey", "unanswered_questions_pkey"),
    (
        "unanswered_questions",
        "unanswered_message_id_fkey",
        "unanswered_questions_message_id_fkey",
    ),
)
"""Mengganti nama constraint PK ikut mengganti nama indeks di belakangnya."""

INDEKS: tuple[tuple[str, str], ...] = (
    ("ix_documents_aktif", "ix_documents_active"),
    ("ix_unanswered_created_at", "ix_unanswered_questions_created_at"),
)

KUNCI_META: tuple[tuple[str, str], ...] = (
    ("topik", "topics"),
    ("sensitivitas", "sensitivity"),
    ("llm_dipanggil", "llm_called"),
    ("biaya_usd", "llm_cost_usd"),
    ("embed_dipanggil", "embed_called"),
    ("embed_biaya_usd", "embed_cost_usd"),
    ("embed_biaya_sumber", "embed_cost_source"),
    ("gate_biaya_usd", "gate_cost_usd"),
)
"""Kunci `messages.meta` (lama, baru). `biaya_usd` selalu berarti biaya LLM saja."""


def _fungsi_tsv(kolom: str) -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_tsv ON chunks")
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
        BEGIN
            NEW.tsv := to_tsvector('{FTS_CONFIG}', COALESCE(NEW.{kolom}, ''));
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        f"CREATE TRIGGER trg_chunks_tsv BEFORE INSERT OR UPDATE OF {kolom} ON chunks "
        "FOR EACH ROW EXECUTE FUNCTION chunks_tsv_update()"
    )


def _ganti_kunci_meta(pasangan: tuple[tuple[str, str], ...]) -> None:
    # Kunci ditulis langsung sebagai literal: semuanya konstanta ASCII di atas,
    # dan parameter terikat di dalam `-`/`->`/`jsonb_build_object` tidak dapat
    # ditebak tipenya oleh asyncpg. `meta -> 'k' IS NOT NULL` benar juga untuk
    # kunci yang ada dengan nilai JSON null.
    for lama, baru in pasangan:
        op.execute(
            f"UPDATE messages SET meta = (meta - '{lama}')"
            f" || jsonb_build_object('{baru}', meta -> '{lama}')"
            f" WHERE jsonb_typeof(meta) = 'object' AND meta -> '{lama}' IS NOT NULL"
        )


def upgrade() -> None:
    op.rename_table("unanswered", "unanswered_questions")
    for tabel, lama, baru in CONSTRAINT:
        op.execute(f'ALTER TABLE {tabel} RENAME CONSTRAINT "{lama}" TO "{baru}"')
    for lama, baru in INDEKS:
        op.execute(f'ALTER INDEX "{lama}" RENAME TO "{baru}"')
    for tabel, lama, baru in KOLOM:
        op.alter_column(tabel, lama, new_column_name=baru)
    _fungsi_tsv("content")
    _ganti_kunci_meta(KUNCI_META)


def downgrade() -> None:
    _ganti_kunci_meta(tuple((baru, lama) for lama, baru in KUNCI_META))
    for tabel, lama, baru in reversed(KOLOM):
        op.alter_column(tabel, baru, new_column_name=lama)
    _fungsi_tsv("konten")
    for lama, baru in reversed(INDEKS):
        op.execute(f'ALTER INDEX "{baru}" RENAME TO "{lama}"')
    for tabel, lama, baru in reversed(CONSTRAINT):
        op.execute(f'ALTER TABLE {tabel} RENAME CONSTRAINT "{baru}" TO "{lama}"')
    op.rename_table("unanswered_questions", "unanswered")
