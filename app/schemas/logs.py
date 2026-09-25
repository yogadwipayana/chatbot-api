"""Skema endpoint `/api/admin/logs/*` (halaman Log di dashboard, `logs.md`)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RentangLog = Literal["24h", "7d"]
"""7 hari adalah batas atas: log yang lebih tua sudah dihapus (LOG_RETENTION_DAYS)."""

StatusGiliran = Literal["ok", "error", "dibatalkan"]
LevelLog = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class NodeStat(BaseModel):
    node: str
    jumlah: int
    p50_ms: float | None
    p95_ms: float | None
    error: int


class ExitPoint(BaseModel):
    node: str
    """Node terakhir yang berjalan, yaitu tempat giliran berhenti."""
    jumlah: int


class HourlyLogStat(BaseModel):
    jam: str
    """Awal jam, UTC, mis. `2026-09-25T03:00:00Z`."""
    giliran: int
    p95_total_ms: float | None
    giliran_error: int
    log_error: int


class LogSummary(BaseModel):
    sejak: str
    sampai: str
    jumlah_giliran: int
    p50_total_ms: float | None
    p95_total_ms: float | None
    giliran_error: int
    rasio_error: float
    diblokir_jev: int
    rasio_diblokir_jev: float
    log_error: int
    """Log ERROR ke atas di rentang ini. Untuk role admin, log audit tidak dihitung."""
    per_node: list[NodeStat]
    titik_keluar: list[ExitPoint]
    per_jam: list[HourlyLogStat]


class TurnOut(BaseModel):
    turn_id: str
    waktu: str
    endpoint: str
    session_id: str | None
    message_id: str | None
    """Tautan ke `messages` di Postgres; teks percakapan hanya ada di sana."""
    unit: str | None
    hasil: str | None
    node_terakhir: str | None
    total_ms: int | None
    ttft_ms: int | None
    """Waktu sampai token pertama; hanya untuk `chat_stream` yang memanggil LLM."""
    status: StatusGiliran
    langsmith_run_id: str | None


class TurnPage(BaseModel):
    total: int
    items: list[TurnOut]


class NodeRunOut(BaseModel):
    node: str
    urutan: int
    mulai: str
    durasi_ms: float
    status: Literal["ok", "error"]
    error_tipe: str | None
    error_pesan: str | None
    detail: dict[str, Any]


class AppLogOut(BaseModel):
    id: int
    waktu: str
    level: str
    logger: str
    pesan: str
    lokasi: str | None
    traceback: str | None
    turn_id: str | None


class TurnDetail(TurnOut):
    nodes: list[NodeRunOut]
    logs: list[AppLogOut]
    """Log yang muncul selama giliran ini berjalan."""


class AppLogPage(BaseModel):
    total: int
    items: list[AppLogOut]
    loggers: list[str] = Field(
        description="Nama logger yang ada di rentang ini, untuk pilihan filter."
    )
