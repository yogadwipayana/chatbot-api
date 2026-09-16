"""Verifikasi DATABASE_URL -- hanya membaca, tidak mengubah apa pun.

Memeriksa hal-hal yang dibutuhkan sebelum `alembic upgrade head` dijalankan:
koneksi, versi server, ketersediaan pgvector, konfigurasi text search
`indonesian` (tanpanya separuh retrieval hibrida gagal diam-diam), dan status
migrasi.

    python -m scripts.check_db
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.rag.retriever import FTS_CONFIG


def baris(label: str, nilai: object) -> None:
    print(f"  {label:<22} {nilai}")


async def skalar(conn, sql: str, **params):
    return (await conn.execute(text(sql), params)).scalar()


async def jalankan() -> int:
    settings = get_settings()
    url = make_url(str(settings.database_url))
    print("Konfigurasi")
    baris("database", url.render_as_string(hide_password=True))

    engine = create_async_engine(url, connect_args={"timeout": 10})
    masalah: list[str] = []
    try:
        async with engine.connect() as conn:
            versi = await skalar(conn, "SHOW server_version")
            vector_ada = await skalar(
                conn,
                "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'",
            )
            vector_terpasang = await skalar(
                conn, "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            fts = await skalar(
                conn, "SELECT 1 FROM pg_ts_config WHERE cfgname = :n", n=FTS_CONFIG
            )
            tabel_migrasi = await skalar(conn, "SELECT to_regclass('public.alembic_version')")
            revisi = (
                await skalar(conn, "SELECT version_num FROM alembic_version")
                if tabel_migrasi
                else None
            )
    except Exception as exc:
        print(f"\nGAGAL terhubung: {type(exc).__name__}: {str(exc)[:300]}")
        print(_petunjuk(str(exc)))
        return 1
    finally:
        await engine.dispose()

    print("\nServer")
    baris("versi PostgreSQL", versi)
    if int(str(versi).split(".")[0]) < 16:
        masalah.append("PRD §10 menetapkan PostgreSQL 16 atau lebih baru.")

    baris("pgvector tersedia", vector_ada or "TIDAK")
    baris("pgvector terpasang", vector_terpasang or "belum (dipasang oleh migrasi)")
    if not vector_ada:
        masalah.append("Ekstensi pgvector tidak tersedia di server; migrasi akan gagal.")

    baris(f"text search '{FTS_CONFIG}'", "ada" if fts else "TIDAK ADA")
    if not fts:
        masalah.append(
            f"Konfigurasi text search '{FTS_CONFIG}' tidak ada; jalur fulltext retrieval "
            "akan gagal. Lihat FTS_CONFIG di app/rag/retriever.py."
        )

    baris("revisi migrasi", revisi or "belum dimigrasi")

    print()
    if masalah:
        for m in masalah:
            print(f"MASALAH: {m}")
        return 1
    if not revisi:
        print("Koneksi OK. Langkah berikutnya: alembic upgrade head")
    else:
        print("Database siap dipakai.")
    return 0


def _petunjuk(pesan: str) -> str:
    rendah = pesan.lower()
    if "password authentication failed" in rendah:
        return "Petunjuk: username atau password di DATABASE_URL salah."
    if "does not exist" in rendah and "database" in rendah:
        return "Petunjuk: nama database di DATABASE_URL belum dibuat."
    if "refused" in rendah or "connect call failed" in rendah:
        return "Petunjuk: PostgreSQL tidak berjalan atau host/port salah."
    if "timeout" in rendah:
        return "Petunjuk: host tidak terjangkau (firewall, VPN, atau alamat salah)."
    if "ssl" in rendah:
        return "Petunjuk: server mewajibkan SSL; tambahkan ?ssl=require di DATABASE_URL."
    return ""


def main() -> int:
    return asyncio.run(jalankan())


if __name__ == "__main__":
    sys.exit(main())
