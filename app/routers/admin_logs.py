"""Halaman Log dashboard: performa per node, giliran chat, dan log aplikasi.

Membaca SQLite log (`app/observability/logstore.py`), bukan Postgres. Minimal
role admin. Log `app.audit` hanya untuk superadmin, disaring di query -- role
admin yang meminta `logger=app.audit` mendapat daftar kosong, bukan 403, supaya
keberadaan log yang disembunyikan tidak terbaca dari beda respons.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.admin.permissions import AdminRole, CurrentAdmin
from app.deps import get_log_store, require_admin, require_role
from app.schemas.common import Error
from app.schemas.logs import (
    AppLogPage,
    LevelLog,
    LogSummary,
    RentangLog,
    StatusGiliran,
    TurnDetail,
    TurnPage,
)

router = APIRouter(
    prefix="/api/admin/logs",
    tags=["admin-log"],
    dependencies=[Depends(require_admin)],
    responses={401: {"model": Error}, 403: {"model": Error}},
)

AdminDep = Annotated[CurrentAdmin, Depends(require_role(AdminRole.ADMIN))]
LogStoreDep = Annotated[Any, Depends(get_log_store)]
RentangQuery = Annotated[RentangLog, Query(alias="range", description="`24h` atau `7d`.")]

_DURASI = {"24h": timedelta(hours=24), "7d": timedelta(days=7)}


def _sejak(rentang: str) -> datetime:
    return datetime.now(UTC) - _DURASI[rentang]


def _boleh_audit(admin: CurrentAdmin) -> bool:
    return admin.at_least(AdminRole.SUPERADMIN)


# Endpoint sinkron: FastAPI menjalankannya di threadpool, jadi sqlite3 yang
# memblokir tidak menahan event loop.


@router.get("/summary", response_model=LogSummary)
def log_summary(admin: AdminDep, store: LogStoreDep, rentang: RentangQuery = "24h") -> Any:
    sampai = datetime.now(UTC)
    return store.ringkasan(sampai - _DURASI[rentang], sampai, audit=_boleh_audit(admin))


@router.get("/turns", response_model=TurnPage)
def list_turns(
    admin: AdminDep,
    store: LogStoreDep,
    rentang: RentangQuery = "24h",
    hasil: str | None = None,
    status_: Annotated[StatusGiliran | None, Query(alias="status")] = None,
    unit: str | None = None,
    node_terakhir: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Any:
    total, items = store.daftar_giliran(
        _sejak(rentang),
        hasil=hasil,
        status=status_,
        unit=unit,
        node_terakhir=node_terakhir,
        limit=limit,
        offset=offset,
    )
    return {"total": total, "items": items}


@router.get("/turns/{turn_id}", response_model=TurnDetail, responses={404: {"model": Error}})
def get_turn(turn_id: str, admin: AdminDep, store: LogStoreDep) -> Any:
    detail = store.detail_giliran(turn_id, audit=_boleh_audit(admin))
    if detail is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Giliran tidak ditemukan. Log yang lebih tua dari masa simpan sudah dihapus.",
        )
    return detail


@router.get("/app", response_model=AppLogPage)
def list_app_logs(
    admin: AdminDep,
    store: LogStoreDep,
    rentang: RentangQuery = "24h",
    level: Annotated[LevelLog, Query(description="Level minimum.")] = "INFO",
    logger: Annotated[
        str | None, Query(description="Nama logger; turunannya ikut, mis. `app.rag`.")
    ] = None,
    q: Annotated[str | None, Query(max_length=200, description="Cari di isi pesan.")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Any:
    audit = _boleh_audit(admin)
    sejak = _sejak(rentang)
    total, items = store.daftar_log(
        sejak,
        audit=audit,
        level_min=logging.getLevelNamesMapping()[level],
        logger=logger,
        cari=q,
        limit=limit,
        offset=offset,
    )
    return {"total": total, "items": items, "loggers": store.daftar_logger(sejak, audit=audit)}
