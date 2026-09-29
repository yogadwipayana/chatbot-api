"""Judul dokumen ikut diindeks di `chunks.tsv` (bobot C).

Pertanyaan yang menyebut nama dokumen -- "apa saja larangan bagi mahasiswa
menurut kode etik?" -- tidak bisa memanfaatkan nama itu, karena `tsv` hanya
berisi isi potongan. Potongan "Pasal 10 Mahasiswa INSTIKI dilarang" kalah dari
potongan Kode Etik lain yang kebetulan memuat kata "kode etik", dan tanpa
filter unit bahkan kalah dari Buku SKP yang memuat pasal yang sama
(peringkat fulltext 48 dari 305). Dengan judul ikut diindeks, semua potongan
Kode Etik cocok dengan "kode etik", dan yang membedakan tinggal "larangan":
peringkat 3. Evaluasi 18 pertanyaan (2026-09-29): bobot C sama baiknya dengan
A (MRR semua unit 0,641 -> 0,695) dengan dorongan paling kecil, jadi kata
judul yang umum ("mahasiswa", "sertifikasi") paling sedikit mengacaukan.

- Isi tetap bobot D (bawaan `to_tsvector`), judul bobot C: di `ts_rank`
  bawaan, kecocokan judul bernilai dua kali kecocokan isi.
- Entri tanya jawab dikecualikan: judulnya adalah pertanyaan itu sendiri,
  yang sudah ada di isi ("Pertanyaan: ..."), jadi akan terhitung dua kali.
- Trigger `trg_documents_title_tsv` menghitung ulang `tsv` semua potongan
  sebuah dokumen saat judul atau jenisnya berubah (admin mengubah "Judul
  resmi"). Caranya `UPDATE chunks SET content = content`, yang memicu
  `trg_chunks_tsv` -- satu rumus `tsv`, bukan dua salinan yang bisa menyimpang.
- Semua potongan yang ada diisi ulang di upgrade dan downgrade.

Revision ID: 0014
Revises: 0013
Create Date: 2026-09-29
"""

from __future__ import annotations

from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None

FTS_CONFIG = "indonesian"
BOBOT_JUDUL = "C"


def _fungsi_chunks(badan: str) -> None:
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
        BEGIN
            {badan}
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )


def _isi_ulang_tsv() -> None:
    # `content = content` memicu `trg_chunks_tsv` (UPDATE OF content) untuk
    # setiap baris, jadi rumus `tsv` hanya ditulis di fungsi trigger.
    op.execute("UPDATE chunks SET content = content")


def upgrade() -> None:
    _fungsi_chunks(
        f"""
            NEW.tsv :=
                setweight(to_tsvector('{FTS_CONFIG}', COALESCE((
                    SELECT CASE WHEN d.type = 'tanya_jawab' THEN '' ELSE d.title END
                    FROM documents d WHERE d.id = NEW.document_id
                ), '')), '{BOBOT_JUDUL}')
                || to_tsvector('{FTS_CONFIG}', COALESCE(NEW.content, ''));"""
    )
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_tsv ON chunks")
    op.execute(
        "CREATE TRIGGER trg_chunks_tsv BEFORE INSERT OR UPDATE OF content, document_id "
        "ON chunks FOR EACH ROW EXECUTE FUNCTION chunks_tsv_update()"
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION documents_title_tsv_update() RETURNS trigger AS $$
        BEGIN
            UPDATE chunks SET content = content WHERE document_id = NEW.id;
            RETURN NULL;
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER trg_documents_title_tsv AFTER UPDATE OF title, type ON documents "
        "FOR EACH ROW WHEN (OLD.title IS DISTINCT FROM NEW.title "
        "OR OLD.type IS DISTINCT FROM NEW.type) "
        "EXECUTE FUNCTION documents_title_tsv_update()"
    )
    _isi_ulang_tsv()


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_documents_title_tsv ON documents")
    op.execute("DROP FUNCTION IF EXISTS documents_title_tsv_update()")
    _fungsi_chunks(f"NEW.tsv := to_tsvector('{FTS_CONFIG}', COALESCE(NEW.content, ''));")
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_tsv ON chunks")
    op.execute(
        "CREATE TRIGGER trg_chunks_tsv BEFORE INSERT OR UPDATE OF content ON chunks "
        "FOR EACH ROW EXECUTE FUNCTION chunks_tsv_update()"
    )
    _isi_ulang_tsv()
