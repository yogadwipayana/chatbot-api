"""Halaman Log dashboard: performa per node, giliran chat, graf, dan log aplikasi.

Membaca SQLite log (`app/observability/logstore.py`), bukan Postgres. Minimal
role admin. Log `app.audit` hanya untuk superadmin, disaring di query -- role
admin yang meminta `logger=app.audit` mendapat daftar kosong, bukan 403, supaya
keberadaan log yang disembunyikan tidak terbaca dari beda respons.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.admin.permissions import AdminRole, CurrentAdmin
from app.deps import get_log_store, require_admin, require_role
from app.schemas.common import Error
from app.schemas.logs import (
    AppLogPage,
    JalurGiliran,
    LevelLog,
    LogSummary,
    PipelineGraph,
    RentangLog,
    StatusGiliran,
    TurnDetail,
    TurnPage,
    TurnTrace,
)

router = APIRouter(
    prefix="/api/admin/logs",
    tags=["admin-logs"],
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
    outcome: str | None = None,
    status_: Annotated[StatusGiliran | None, Query(alias="status")] = None,
    unit: str | None = None,
    last_node: str | None = None,
    endpoint: JalurGiliran | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Any:
    total, items = store.daftar_giliran(
        _sejak(rentang),
        outcome=outcome,
        status=status_,
        unit=unit,
        last_node=last_node,
        endpoint=endpoint,
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


@router.get(
    "/turns/{turn_id}/trace", response_model=TurnTrace, responses={404: {"model": Error}}
)
def get_turn_trace(turn_id: str, admin: AdminDep, store: LogStoreDep) -> Any:
    """Input/output setiap node dan panggilan di dalamnya (tab Graf).

    404 bila gilirannya tidak ada, sudah lewat masa simpan, atau berjalan saat
    LOG_NODE_IO mati. Waktu dan status node ada di `GET /turns/{turn_id}`.
    """
    rekaman = store.rekaman_giliran(turn_id)
    if rekaman is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Rekaman input/output tidak ada untuk giliran ini.",
        )
    return {"turn_id": turn_id, **rekaman}


@lru_cache(maxsize=1)
def _topologi() -> dict[str, Any]:
    """Bentuk graf dari LangGraph sendiri, supaya diagram di dashboard tidak
    tertinggal saat node atau sisi berubah. `xray` membuka subgraph `cari`
    menjadi `cari:rewrite` dan `cari:retrieve`; namanya dipisah menjadi `id`
    (sama dengan `node_runs.node`) dan `group`."""
    from app.rag.graph import build_graph

    graf = build_graph().get_graph(xray=True)

    def pisah(ident: str) -> tuple[str, str | None]:
        group, _, nama = ident.rpartition(":")
        return nama, group or None

    return {
        "nodes": [
            {"id": nama, "group": group}
            for nama, group in (pisah(ident) for ident in graf.nodes)
        ],
        "edges": [
            {
                "source": pisah(e.source)[0],
                "target": pisah(e.target)[0],
                "conditional": e.conditional,
                "label": e.data if isinstance(e.data, str) else None,
            }
            for e in graf.edges
        ],
    }


@router.get("/graph", response_model=PipelineGraph)
def get_pipeline_graph(admin: AdminDep) -> Any:
    """Node dan sisi pipeline chat, untuk diagram tab Graf."""
    return _topologi()


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
