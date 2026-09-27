"""Batas pertanyaan harian global: jaring pengaman terakhir untuk kuota model.

Batas laju per IP menahan satu sumber; ia tidak menahan banjir dari banyak IP
sekaligus. Batas harian menghitung semua pertanyaan dari mana pun, dan begitu
terlampaui menyalakan kill switch (`app.deps.batas_harian`) -- kerugian satu
hari paling banyak sebesar batas itu, dan dashboard menampilkan spanduk kill
switch di setiap halaman sampai superadmin menyalakan layanan lagi.

Hitungannya di memori proses, sejalan dengan kill switch, tetapi dimulai dari
log Postgres setiap kali hari berganti atau proses baru mulai: restart di
tengah hari tidak mengembalikan jatah yang sudah terpakai.

"Hari" dihitung Postgres dalam zona `TIMEZONE`, sama seperti statistik
(`app/admin/stats.py`), bukan oleh `zoneinfo` Python -- yang butuh paket
`tzdata` di Windows dan di image yang tidak memuat basis data zona waktu.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

OLEH_BATAS_HARIAN = "batas harian otomatis"
"""`engaged_by` kill switch, supaya admin tahu bukan orang yang mematikannya."""


def alasan_batas_harian(batas: int) -> str:
    return (
        f"Batas harian {batas} pertanyaan tercapai. Naikkan batas harian di halaman "
        "Konfigurasi sebelum menyalakan layanan lagi -- kalau tidak, layanan mati "
        "lagi pada pertanyaan berikutnya. Jatahnya kembali penuh besok."
    )


Pemuat = Callable[[str], Awaitable[tuple[int, datetime]]]
"""zona -> (pertanyaan yang sudah tercatat hari ini, awal hari berikutnya dalam UTC)."""


class PenghitungHarian:
    """Jumlah pertanyaan hari ini, menurut zona waktu `TIMEZONE`."""

    def __init__(
        self, muat: Pemuat, *, clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    ) -> None:
        self._muat = muat
        self._clock = clock
        self._zona: str | None = None
        self._berakhir: datetime | None = None
        self._jumlah = 0

    def _basi(self, zona: str) -> bool:
        return self._zona != zona or self._berakhir is None or self._clock() >= self._berakhir

    async def tambah(self, zona: str) -> int:
        """Hitung satu pertanyaan lagi; kembalikan jumlah hari ini termasuk yang ini."""
        if self._basi(zona):
            jumlah, berakhir = await self._muat(zona)
            # Permintaan lain yang bersamaan mungkin sudah memuat lebih dulu;
            # jangan timpa hitungannya dengan angka yang lebih lama.
            if self._basi(zona):
                self._zona, self._berakhir, self._jumlah = zona, berakhir, jumlah
        self._jumlah += 1
        return self._jumlah


_HARI_INI_SQL = """
    WITH hari AS (SELECT date_trunc('day', now() AT TIME ZONE :tz) AS awal)
    SELECT
        (SELECT count(*) FROM messages
         WHERE role = 'user' AND created_at >= (SELECT awal FROM hari) AT TIME ZONE :tz)
            AS jumlah,
        ((SELECT awal FROM hari) + interval '1 day') AT TIME ZONE :tz AS berakhir
"""


async def hitung_dari_log(zona: str) -> tuple[int, datetime]:
    """Pertanyaan mahasiswa yang tercatat sejak tengah malam, dan tengah malam berikutnya."""
    from sqlalchemy import text

    from app.db.session import SessionLocal

    async with SessionLocal() as session:
        row = (await session.execute(text(_HARI_INI_SQL), {"tz": zona})).one()
        return int(row.jumlah or 0), row.berakhir


_penghitung = PenghitungHarian(hitung_dari_log)


def get_daily_counter() -> PenghitungHarian:
    """Dependency FastAPI; instans tunggal per proses. Di-override di test."""
    return _penghitung
