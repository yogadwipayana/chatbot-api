"""Fixture test integrasi -- butuh PostgreSQL 16 + pgvector.

Jalankan dengan:
    docker compose up -d postgres
    TEST_DATABASE_URL=postgresql+asyncpg://chatbot:chatbot@localhost:5432/chatbot_test \
        pytest tests/integration -m integration

Tanpa `TEST_DATABASE_URL`, seluruh berkas di direktori ini dilewati
(lihat `pytest_collection_modifyitems` di tests/conftest.py).
"""

from __future__ import annotations

import os
import uuid
from datetime import date, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import Base

DIM = 8
"""Dimensi kecil untuk test; embedding sungguhan 1024 hanya memperlambat."""


def vektor(*nilai: float) -> str:
    """Bentuk literal vektor pgvector, dipadkan sampai DIM."""
    penuh = list(nilai) + [0.0] * (DIM - len(nilai))
    return "[" + ",".join(str(v) for v in penuh) + "]"


def _pastikan_database_uji(url: str) -> None:
    """Fixture `engine` menjalankan DROP ALL -- tolak apa pun yang bukan database uji.

    Satu salah ketik TEST_DATABASE_URL tidak boleh menghapus database aplikasi.
    """
    from sqlalchemy.engine import make_url

    from app.config import get_settings

    uji = make_url(url)
    aplikasi = make_url(str(get_settings().database_url))
    if (uji.host, uji.port, uji.database) == (aplikasi.host, aplikasi.port, aplikasi.database):
        pytest.exit("TEST_DATABASE_URL sama dengan DATABASE_URL aplikasi.", returncode=2)
    if "test" not in (uji.database or "").lower():
        pytest.exit(
            f"TEST_DATABASE_URL menunjuk database '{uji.database}'. Test integrasi "
            "menghapus seluruh tabel; nama database uji wajib mengandung 'test'.",
            returncode=2,
        )


@pytest_asyncio.fixture
async def engine():
    url = os.environ["TEST_DATABASE_URL"]
    _pastikan_database_uji(url)
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        # Kolom embedding dibuat ulang dengan dimensi kecil khusus test.
        await conn.execute(text("ALTER TABLE chunks DROP COLUMN IF EXISTS embedding"))
        await conn.execute(text(f"ALTER TABLE chunks ADD COLUMN embedding vector({DIM})"))
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s


@pytest_asyncio.fixture
async def seed(session):
    """Isi tiga dokumen: aktif, kedaluwarsa, dan dinonaktifkan."""
    ids: dict[str, uuid.UUID] = {}

    async def tambah(
        judul: str,
        konten: str,
        *,
        embedding: str,
        is_active: bool = True,
        valid_until: date | None = None,
        halaman: int = 1,
    ) -> uuid.UUID:
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        await session.execute(
            text(
                "INSERT INTO documents (id, judul, unit, file_path, is_active, valid_until,"
                " updated_at) VALUES (:id, :judul, 'Biro Akademik', :path, :aktif,"
                " :valid_until, now())"
            ),
            {
                "id": doc_id,
                "judul": judul,
                "path": f"storage/documents/{doc_id}.pdf",
                "aktif": is_active,
                "valid_until": valid_until,
            },
        )
        await session.execute(
            text(
                "INSERT INTO chunks (id, document_id, konten, halaman, urutan, embedding, tsv)"
                " VALUES (:id, :doc, :konten, :halaman, 0, (:emb)::vector,"
                " to_tsvector('indonesian', :konten))"
            ),
            {
                "id": chunk_id,
                "doc": doc_id,
                "konten": konten,
                "halaman": halaman,
                "emb": embedding,
            },
        )
        ids[judul] = chunk_id
        return chunk_id

    await tambah(
        "Panduan Akademik 2025",
        "Pengisian KRS dilakukan pada minggu pertama setiap semester.",
        embedding=vektor(1, 0, 0),
    )
    await tambah(
        "Panduan Akademik 2019",
        "Pengisian KRS dilakukan lewat loket tata usaha.",
        embedding=vektor(0.99, 0.01, 0),
        valid_until=date.today() - timedelta(days=1),
    )
    await tambah(
        "Draf Panduan Internal",
        "Pengisian KRS versi draf yang belum disahkan.",
        embedding=vektor(0.98, 0.02, 0),
        is_active=False,
    )
    await session.commit()
    return ids


@pytest.fixture
def embed_query():
    """Embedding palsu: selalu mengarah ke vektor dokumen paling relevan."""

    async def embed(_: str) -> list[float]:
        return [1.0, 0.0] + [0.0] * (DIM - 2)

    return embed
