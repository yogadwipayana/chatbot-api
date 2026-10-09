"""Migrasi saat kontainer mulai, tanpa restart-loop bila database lebih baru.

`alembic upgrade head` gagal dengan "Can't locate revision ..." bila database
sudah dimigrasi oleh kode yang lebih baru dari image ini -- mis. migrasi yang
dijalankan dari mesin pengembang ke database yang sama, sebelum image server
di-deploy ulang. Kontainer lalu berhenti sebelum uvicorn jalan, restart policy
menyalakannya lagi, dan layanan mati sampai ada yang turun tangan (2026-10-08:
database di 0015, image 0014).

Di sini revisi database dibandingkan dulu dengan revisi yang dikenal image:

- belum ada tabel `alembic_version`, atau revisinya lebih lama: `upgrade head`;
- sudah di head: tidak ada yang dijalankan;
- tidak dikenal image (database lebih baru): migrasi DILEWATI dengan peringatan
  yang jelas, lalu aplikasi tetap jalan. Migrasi proyek ini menambah tabel atau
  kolom, jadi kode lama umumnya tetap berfungsi; yang perlu dilakukan adalah
  deploy image yang lebih baru, bukan mematikan layanan.

Dipanggil `docker-entrypoint.sh` (dan `start.sh`): `python -m app.db.migrasi`.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from enum import StrEnum
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError

from alembic import command

logger = logging.getLogger("app.migrasi")

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


class Rencana(StrEnum):
    NAIKKAN = "naikkan"
    TERBARU = "terbaru"
    DB_LEBIH_BARU = "db-lebih-baru"


def rencana(revisi_db: list[str], script: ScriptDirectory) -> Rencana:
    """Apa yang harus dilakukan untuk revisi database ini."""
    if not revisi_db:
        return Rencana.NAIKKAN
    for revisi in revisi_db:
        try:
            script.get_revision(revisi)
        except CommandError:
            return Rencana.DB_LEBIH_BARU
    if set(revisi_db) == set(script.get_heads()):
        return Rencana.TERBARU
    return Rencana.NAIKKAN


async def baca_revisi_db() -> list[str]:
    """Isi `alembic_version`; kosong bila tabelnya belum ada (database baru)."""
    from sqlalchemy import text
    from sqlalchemy.exc import ProgrammingError
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.config import get_settings

    engine = create_async_engine(str(get_settings().database_url))
    try:
        async with engine.connect() as conn:
            try:
                hasil = await conn.execute(text("SELECT version_num FROM alembic_version"))
            except ProgrammingError:
                return []
            return [baris[0] for baris in hasil]
    finally:
        await engine.dispose()


def main(baca=None, naikkan=None) -> int:
    """`baca` dan `naikkan` diganti di test; bawaannya database dan Alembic sungguhan."""
    config = Config(str(ALEMBIC_INI))
    script = ScriptDirectory.from_config(config)
    revisi_db = asyncio.run((baca or baca_revisi_db)())
    langkah = rencana(revisi_db, script)
    head = ", ".join(script.get_heads())

    if langkah is Rencana.DB_LEBIH_BARU:
        logger.warning(
            "Database di revisi %s, lebih baru dari image ini (head %s). Migrasi DILEWATI "
            "dan layanan tetap dijalankan; deploy image yang memuat revisi itu.",
            ", ".join(revisi_db),
            head,
        )
        return 0
    if langkah is Rencana.TERBARU:
        logger.info("Database sudah di head (%s); tidak ada migrasi.", head)
        return 0

    logger.info("Menaikkan database %s -> %s", ", ".join(revisi_db) or "(kosong)", head)
    (naikkan or (lambda: command.upgrade(config, "head")))()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
