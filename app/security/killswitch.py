"""Kill switch (FR-9, PRD §14 'Kill switch diuji').

Pemilik sistem harus bisa mematikan layanan chat dengan cepat saat insiden,
tanpa menunggu deploy ulang. Statusnya disimpan di proses dan diubah lewat
`POST /api/admin/kill-switch`. `KILL_SWITCH_ENABLED=true` menyalakannya sejak
proses dimulai (lihat `app.main.lifespan`) -- jalan cadangan bila dashboard
admin sendiri tidak bisa diakses.

Status yang berlaku dibaca dari memori proses, jadi satu proses uvicorn,
satu kill switch. Setiap perubahan juga ditulis ke tabel `kill_switch` dan
dimuat ulang saat proses mulai (`pulihkan`), supaya deploy, crash, atau restart
kontainer tidak diam-diam menyalakan lagi layanan yang dimatikan karena
insiden. Dockerfile menjalankan satu worker; bila kelak dijalankan dengan
beberapa worker, setiap permintaan harus membaca tabel itu. Kalau tidak, admin
hanya mematikan satu worker dan sisanya tetap menjawab.

Dashboard admin tetap hidup saat kill switch aktif -- justru saat itulah admin
perlu masuk untuk melihat apa yang terjadi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.risk import KONTAK_FRONT_OFFICE

logger = logging.getLogger(__name__)


@dataclass
class KillSwitch:
    engaged: bool = False
    reason: str | None = None
    engaged_at: datetime | None = field(default=None)
    engaged_by: str | None = None
    """Siapa yang menyalakan. Insiden harus dapat diaudit: alasan DAN pelakunya."""

    def engage(self, reason: str, by: str | None = None) -> None:
        """Matikan layanan chat.

        Dipanggil ulang saat sudah aktif hanya memperbarui alasan dan pelaku;
        `engaged_at` tetap menunjuk awal insiden, supaya durasi gangguan tidak
        terhapus hanya karena catatannya dilengkapi.
        """
        if not reason.strip():
            raise ValueError("alasan mematikan layanan wajib diisi")
        if not self.engaged:
            self.engaged_at = datetime.now(UTC)
        self.engaged = True
        self.reason = reason.strip()
        self.engaged_by = by

    def release(self) -> None:
        self.engaged = False
        self.reason = None
        self.engaged_at = None
        self.engaged_by = None

    @property
    def message(self) -> str:
        """Pesan yang ditampilkan ke mahasiswa saat layanan dimatikan."""
        return (
            "Layanan chat sedang dinonaktifkan sementara. "
            f"Silakan hubungi Front Office INSTIKI: {KONTAK_FRONT_OFFICE}."
        )


class KillSwitchStore(Protocol):
    async def load(self) -> KillSwitch | None: ...

    async def save(self, switch: KillSwitch) -> None: ...


class SqlKillSwitchStore:
    """Tabel `kill_switch`: satu baris saat menyala, kosong saat layanan hidup."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def load(self) -> KillSwitch | None:
        row = (
            (
                await self.session.execute(
                    text("SELECT reason, engaged_at, engaged_by FROM kill_switch WHERE id = 1")
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        return KillSwitch(
            engaged=True,
            reason=row["reason"],
            engaged_at=row["engaged_at"],
            engaged_by=row["engaged_by"],
        )

    async def save(self, switch: KillSwitch) -> None:
        try:
            if switch.engaged:
                await self.session.execute(
                    text(
                        "INSERT INTO kill_switch (id, reason, engaged_at, engaged_by)"
                        " VALUES (1, :reason, :engaged_at, :engaged_by)"
                        " ON CONFLICT (id) DO UPDATE SET"
                        " reason = EXCLUDED.reason,"
                        " engaged_at = EXCLUDED.engaged_at,"
                        " engaged_by = EXCLUDED.engaged_by"
                    ),
                    {
                        "reason": switch.reason,
                        "engaged_at": switch.engaged_at,
                        "engaged_by": switch.engaged_by,
                    },
                )
            else:
                await self.session.execute(text("DELETE FROM kill_switch"))
            await self.session.commit()
        except Exception:
            # Sesi permintaan masih dipakai sesudah ini (mis. log chat):
            # jangan tinggalkan transaksi yang gagal.
            await self.session.rollback()
            raise


async def simpan_status(switch: KillSwitch, store: KillSwitchStore) -> None:
    """Tulis status saat ini ke database; galat dicatat, tidak dilempar.

    Status di memori sudah berubah dan berlaku saat itu juga. Mematikan layanan
    saat insiden tidak boleh gagal hanya karena penyimpanannya gagal -- yang
    hilang hanyalah ketahanannya melewati restart, dan itu harus terlihat di
    Log aplikasi.
    """
    try:
        await store.save(switch)
    except Exception:
        logger.exception(
            "Status kill switch (%s) gagal disimpan ke database; "
            "status ini tidak akan bertahan setelah restart",
            "menyala" if switch.engaged else "mati",
        )


async def pulihkan(switch: KillSwitch, store: KillSwitchStore) -> bool:
    """Saat proses mulai: salin kill switch tersimpan ke memori.

    True bila ada yang dipulihkan. `engaged_at` dan pelakunya ikut dipulihkan,
    supaya durasi insiden di dashboard tidak mulai dari nol setiap restart.
    """
    tersimpan = await store.load()
    if tersimpan is None:
        return False
    switch.engaged = True
    switch.reason = tersimpan.reason
    switch.engaged_at = tersimpan.engaged_at
    switch.engaged_by = tersimpan.engaged_by
    return True


_switch = KillSwitch()


def get_kill_switch() -> KillSwitch:
    """Dependency FastAPI; instans tunggal per proses."""
    return _switch
