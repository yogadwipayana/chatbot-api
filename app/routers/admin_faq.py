"""Kelola entri tanya jawab: jawaban siap pakai tanpa berkas PDF.

Jalan tercepat menutup pertanyaan tak terjawab (AD-4): admin mengetik satu
pertanyaan beserta jawabannya, dan chatbot langsung memakainya pada pertanyaan
berikutnya -- tanpa menunggu ada yang menyusun dokumen resmi lebih dulu.

Aturan aksesnya persis sama dengan dokumen: semua level boleh masuk, staf/dosen
dibatasi pada unitnya sendiri, admin dan superadmin mengelola semua unit.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.admin import faq as repo
from app.deps import (
    CurrentAdminDep,
    SessionDep,
    SettingsDep,
    UnitDirectoryDep,
    get_embeddings,
    pastikan_unit,
    require_admin,
    unit_terdaftar,
)
from app.routers.common import terjemahkan_galat_ai
from app.schemas.admin import FaqEntry, FaqEntryCreate, FaqEntryUpdate, FaqPage
from app.schemas.common import Error

router = APIRouter(
    prefix="/api/admin/faq",
    tags=["admin-tanya-jawab"],
    dependencies=[Depends(require_admin)],
    responses={401: {"model": Error}, 403: {"model": Error}},
)

TIDAK_DITEMUKAN = "Entri tanya jawab tidak ditemukan."
APA = "entri tanya jawab"

Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0)]


async def _entri_milik(session, admin, entry_id: uuid.UUID) -> dict[str, Any]:
    row = await repo.get_entry(session, entry_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_DITEMUKAN)
    pastikan_unit(admin, row["unit"], apa=APA)
    return row


@router.get("", response_model=FaqPage)
async def list_faq(
    admin: CurrentAdminDep,
    session: SessionDep,
    include_inactive: bool = False,
    limit: Limit = 50,
    offset: Offset = 0,
) -> FaqPage:
    rows, total = await repo.list_entries(
        session,
        include_inactive=include_inactive,
        limit=limit,
        offset=offset,
        unit=admin.unit_scope,
    )
    return FaqPage(items=[FaqEntry.model_validate(r) for r in rows], total=total)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=FaqEntry,
    responses={422: {"model": Error}, 502: {"model": Error}},
)
async def create_faq(
    payload: FaqEntryCreate,
    admin: CurrentAdminDep,
    session: SessionDep,
    settings: SettingsDep,
    units: UnitDirectoryDep,
    embeddings: Any = Depends(get_embeddings),
) -> FaqEntry:
    """Simpan entri baru. Embedding dihitung saat itu juga, seperti unggah dokumen."""
    unit = await unit_terdaftar(units, payload.unit)
    pastikan_unit(admin, unit, apa=APA)
    with terjemahkan_galat_ai(APA):
        row = await repo.create_entry(
            session,
            pertanyaan=payload.pertanyaan,
            jawaban=payload.jawaban,
            unit=unit,
            valid_until=payload.valid_until,
            uploaded_by=admin.email,
            embeddings=embeddings,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    return FaqEntry.model_validate(row)


@router.patch(
    "/{entry_id}",
    response_model=FaqEntry,
    responses={404: {"model": Error}, 422: {"model": Error}, 502: {"model": Error}},
)
async def update_faq(
    entry_id: uuid.UUID,
    payload: FaqEntryUpdate,
    admin: CurrentAdminDep,
    session: SessionDep,
    settings: SettingsDep,
    units: UnitDirectoryDep,
    embeddings: Any = Depends(get_embeddings),
) -> FaqEntry:
    """Mengubah pertanyaan atau jawaban langsung mengindeks ulang entri ini."""
    await _entri_milik(session, admin, entry_id)
    changes = payload.model_dump(exclude_unset=True)
    if "unit" in changes:
        changes["unit"] = await unit_terdaftar(units, changes["unit"])
        # Staf juga tidak boleh memindahkan entrinya ke unit lain.
        pastikan_unit(admin, changes["unit"], apa=APA)

    with terjemahkan_galat_ai(APA):
        row = await repo.update_entry(
            session,
            entry_id,
            changes,
            embeddings=embeddings,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_DITEMUKAN)
    return FaqEntry.model_validate(row)


@router.delete(
    "/{entry_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={404: {"model": Error}},
)
async def delete_faq(
    entry_id: uuid.UUID, admin: CurrentAdminDep, session: SessionDep
) -> Response:
    """Permanen. Untuk sekadar menghentikan pemakaiannya, pakai `is_active: false`."""
    await _entri_milik(session, admin, entry_id)
    if not await repo.delete_entry(session, entry_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_DITEMUKAN)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
