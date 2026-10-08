"""Judul dokumen di `chunks.tsv` turun dari bobot C ke D (T36).

Judul ada di SETIAP potongan dokumennya. Dengan bobot C (dua kali isi), satu
kata judul yang umum cukup untuk mengangkat seluruh dokumen: "apa saja jenis
sertifikasi di instiki?" memenuhi fulltext dengan potongan "PEDOMAN SERTIFIKASI
INSTIKI New", dan jawaban FAQ-nya jatuh ke peringkat fulltext 43 lalu keluar
dari top-5. Begitu juga TRANSKRIP KEMAHASISWAAN di bawah "BUKU PEDOMAN
PENGHARGAAN PRESTASI DAN BEASISWA MAHASISWA" ("kapan beasiswa KIP dibuka?",
"siapa PIC beasiswa?").

Evaluasi 41 pertanyaan dengan filter unit (2026-10-08), MRR top-5 hibrida:
tanpa judul 0,825; bobot C 0,708; bobot D 0,846; judul separuh isi 0,801.
Bobot D memulihkan semua kasus di atas dan tetap menolong pertanyaan yang
menyebut nama dokumen ("larangan menurut kode etik"), alasan 0014 dibuat.

Rumus `tsv` lain tidak berubah (tanya jawab tetap tanpa judul, trigger judul
0014 tetap). Semua potongan diisi ulang di upgrade dan downgrade.

Revision ID: 0015
Revises: 0014
Create Date: 2026-10-08
"""

from __future__ import annotations

from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None

FTS_CONFIG = "indonesian"


def _fungsi_chunks(bobot_judul: str) -> None:
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
        BEGIN
            NEW.tsv :=
                setweight(to_tsvector('{FTS_CONFIG}', COALESCE((
                    SELECT CASE WHEN d.type = 'tanya_jawab' THEN '' ELSE d.title END
                    FROM documents d WHERE d.id = NEW.document_id
                ), '')), '{bobot_judul}')
                || to_tsvector('{FTS_CONFIG}', COALESCE(NEW.content, ''));
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    # `content = content` memicu `trg_chunks_tsv`, jadi rumus `tsv` tetap
    # hanya ditulis di fungsi trigger (lihat 0014).
    op.execute("UPDATE chunks SET content = content")


def upgrade() -> None:
    _fungsi_chunks("D")


def downgrade() -> None:
    _fungsi_chunks("C")
