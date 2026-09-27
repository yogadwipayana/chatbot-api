"""Statistik dan kill switch (AD-5, FR-9)."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, status

from app.admin.permissions import AdminRole
from app.admin.stats import compute_costs, compute_stats, today
from app.deps import (
    BaseSettingsDep,
    CurrentAdminDep,
    KillSwitchDep,
    SessionDep,
    require_admin,
    require_role,
)
from app.schemas.admin import Costs, KillSwitchRequest, KillSwitchState, Stats
from app.schemas.common import Error

audit = logging.getLogger("app.audit")

router = APIRouter(
    prefix="/api/admin",
    tags=["admin-ops"],
    dependencies=[Depends(require_admin)],
    responses={401: {"model": Error}},
)

RENTANG_DEFAULT_HARI = 30
RENTANG_MAKS_HARI = 366


@router.get(
    "/stats",
    response_model=Stats,
    dependencies=[Depends(require_role(AdminRole.ADMIN))],
    responses={403: {"model": Error}},
)
async def admin_stats(
    session: SessionDep,
    settings: BaseSettingsDep,
    since: date | None = None,
    until: date | None = None,
) -> Stats:
    """AD-5. Tanpa parameter: 30 hari terakhir sampai hari ini (zona `TIMEZONE`)."""
    until = until or await today(session, settings.timezone)
    since = since or until - timedelta(days=RENTANG_DEFAULT_HARI - 1)
    if since > until:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Tanggal awal tidak boleh setelah tanggal akhir.",
        )
    if (until - since).days >= RENTANG_MAKS_HARI:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Rentang tanggal paling panjang {RENTANG_MAKS_HARI} hari.",
        )
    return Stats(
        **await compute_stats(session, since=since, until=until, timezone=settings.timezone)
    )


@router.get(
    "/costs",
    response_model=Costs,
    dependencies=[Depends(require_role(AdminRole.ADMIN))],
    responses={403: {"model": Error}},
)
async def admin_costs(
    session: SessionDep,
    settings: BaseSettingsDep,
    since: date | None = None,
    until: date | None = None,
) -> Costs:
    """Rincian token dan estimasi biaya dari `messages.meta`."""
    until = until or await today(session, settings.timezone)
    since = since or until - timedelta(days=RENTANG_DEFAULT_HARI - 1)
    if since > until:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Tanggal awal tidak boleh setelah tanggal akhir.",
        )
    if (until - since).days >= RENTANG_MAKS_HARI:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Rentang tanggal paling panjang {RENTANG_MAKS_HARI} hari.",
        )
    return Costs(
        **await compute_costs(session, since=since, until=until, timezone=settings.timezone)
    )


@router.get("/kill-switch", response_model=KillSwitchState)
def get_kill_switch_state(switch: KillSwitchDep) -> KillSwitchState:
    """Semua level boleh membaca: banner "layanan dimatikan" tampil untuk semua admin."""
    return _state(switch)


@router.post(
    "/kill-switch",
    response_model=KillSwitchState,
    dependencies=[Depends(require_role(AdminRole.SUPERADMIN))],
    responses={403: {"model": Error}},
)
def set_kill_switch(
    payload: KillSwitchRequest, switch: KillSwitchDep, admin: CurrentAdminDep
) -> KillSwitchState:
    """FR-9. Setiap perubahan dicatat ke log audit beserta pelakunya."""
    pelaku = admin.email
    if payload.engaged:
        switch.engage(payload.reason or "", by=pelaku)
        audit.warning("Kill switch DINYALAKAN oleh %s: %s", pelaku, switch.reason)
    else:
        if switch.engaged:
            audit.warning("Kill switch dimatikan oleh %s", pelaku)
        switch.release()
    return _state(switch)


def _state(switch) -> KillSwitchState:
    return KillSwitchState(
        engaged=switch.engaged,
        reason=switch.reason,
        engaged_at=switch.engaged_at,
        engaged_by=switch.engaged_by,
    )
