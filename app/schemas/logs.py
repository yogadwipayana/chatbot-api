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
    count: int
    p50_ms: float | None
    p95_ms: float | None
    error: int


class ExitPoint(BaseModel):
    node: str
    """Node terakhir yang berjalan, yaitu tempat giliran berhenti."""
    count: int


class HourlyLogStat(BaseModel):
    hour: str
    """Awal jam, UTC, mis. `2026-09-25T03:00:00Z`."""
    turn_count: int
    p95_total_ms: float | None
    error_turn_count: int
    error_log_count: int


class LogSummary(BaseModel):
    since: str
    until: str
    turn_count: int
    p50_total_ms: float | None
    p95_total_ms: float | None
    error_turn_count: int
    error_ratio: float
    cancelled_turn_count: int
    """Giliran `dibatalkan`: mahasiswa menghentikan jawaban atau menutup panel.
    Tidak ikut `error_turn_count` -- pembatalan bukan galat server."""
    jev_blocked_count: int
    jev_blocked_ratio: float
    error_log_count: int
    """Log ERROR ke atas di rentang ini. Untuk role admin, log audit tidak dihitung."""
    per_node: list[NodeStat]
    exit_points: list[ExitPoint]
    per_hour: list[HourlyLogStat]


class TurnOut(BaseModel):
    turn_id: str
    timestamp: str
    endpoint: str
    session_id: str | None
    message_id: str | None
    """Tautan ke `messages` di Postgres; teks percakapan hanya ada di sana."""
    unit: str | None
    outcome: str | None
    last_node: str | None
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
    position: int
    started_at: str
    duration_ms: float
    status: Literal["ok", "error"]
    error_type: str | None
    error_message: str | None
    detail: dict[str, Any]


class AppLogOut(BaseModel):
    id: int
    timestamp: str
    level: str
    logger: str
    message: str
    location: str | None
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
