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
    tracing_enabled: bool = False
    """Apakah trace FR-8 benar-benar terkirim ke LangSmith.

    Tracing yang mati tidak menjatuhkan satu pun permintaan, jadi ia tidak
    pernah muncul sebagai insiden: yang terjadi hanya kolom `langsmith_run_id`
    yang kosong terus-menerus, dan baru disadari saat ada jawaban buruk yang
    perlu ditelusuri dan jejaknya ternyata tidak pernah ada."""
