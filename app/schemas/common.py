"""Skema yang dipakai lintas router."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Error(BaseModel):
    """Galat dengan kalimat siap tampil."""

    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    chat_enabled: bool
    """False berarti kill switch aktif; proses tetap sehat."""
    kill_switch_reason: str | None = None
