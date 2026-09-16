"""Kill switch (FR-9, PRD §14 'Kill switch diuji').

Pemilik sistem harus bisa mematikan layanan chat dengan cepat saat insiden,
tanpa menunggu deploy ulang. Statusnya disimpan di proses dan diubah lewat
`POST /api/admin/kill-switch`. `KILL_SWITCH_ENABLED=true` menyalakannya sejak
proses dimulai (lihat `app.main.lifespan`) -- jalan cadangan bila dashboard
admin sendiri tidak bisa diakses.

Status di memori proses berarti satu proses uvicorn, satu kill switch.
Dockerfile menjalankan satu worker; bila kelak dijalankan dengan beberapa
worker, status ini harus dipindah ke Postgres. Kalau tidak, admin hanya
mematikan satu worker dan sisanya tetap menjawab.

Dashboard admin tetap hidup saat kill switch aktif -- justru saat itulah admin
perlu masuk untuk melihat apa yang terjadi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


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
            "Silakan hubungi Biro Administrasi Akademik pada jam kerja."
        )


_switch = KillSwitch()


def get_kill_switch() -> KillSwitch:
    """Dependency FastAPI; instans tunggal per proses."""
    return _switch
