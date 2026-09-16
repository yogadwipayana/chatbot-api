"""Health check untuk Uptime Kuma (PRD §10)."""

from __future__ import annotations

from fastapi import APIRouter

from app.deps import KillSwitchDep
from app.schemas.common import HealthResponse

router = APIRouter(tags=["sistem"])


@router.get("/health", response_model=HealthResponse)
async def health(switch: KillSwitchDep) -> HealthResponse:
    """200 selama proses hidup, termasuk saat kill switch aktif.

    Kill switch bukan kegagalan sistem -- ia keputusan operator. Melaporkannya
    sebagai unhealthy akan membuat monitoring membanjiri alert saat insiden
    justru sedang ditangani.
    """
    return HealthResponse(
        status="ok",
        chat_enabled=not switch.engaged,
        kill_switch_reason=switch.reason,
    )
