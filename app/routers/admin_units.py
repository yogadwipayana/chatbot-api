"""Kelola daftar unit layanan (tabel `units`, khusus superadmin).

Daftar ini menentukan menu unit chatbot, isian unit di seluruh dashboard, dan
pembatasan akses staf. Karena itu unit tidak dihapus -- hanya dinonaktifkan --
dan penggantian nama mengalir ke dokumen dan akun lewat `ON UPDATE CASCADE`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.admin.permissions import AdminRole
from app.deps import CurrentAdminDep, UnitDirectoryDep, require_role
from app.schemas.admin import AdminUnit, AdminUnitCreate, AdminUnitUpdate
from app.schemas.common import Error
from app.units import DuplicateUnitError, UnitRecord

audit = logging.getLogger("app.audit")

router = APIRouter(
    prefix="/api/admin/units",
    tags=["admin-unit"],
    dependencies=[Depends(require_role(AdminRole.SUPERADMIN))],
    responses={401: {"model": Error}, 403: {"model": Error}},
)

TIDAK_DITEMUKAN = "Unit tidak ditemukan."


def unit_out(unit: UnitRecord) -> AdminUnit:
    return AdminUnit(
        nama=unit.nama,
        deskripsi=unit.deskripsi,
        urutan=unit.urutan,
        is_active=unit.is_active,
        jumlah_dokumen=unit.jumlah_dokumen,
        jumlah_akun=unit.jumlah_akun,
    )


def _bentrok(nama: str) -> HTTPException:
    return HTTPException(
        status.HTTP_409_CONFLICT,
        f"Unit '{nama}' sudah ada (huruf besar-kecil dan spasi tidak dibedakan).",
    )


@router.get("", response_model=list[AdminUnit])
async def list_admin_units(units: UnitDirectoryDep) -> list[AdminUnit]:
    """Semua unit, termasuk yang nonaktif, dalam urutan menu."""
    return [unit_out(u) for u in await units.semua()]


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=AdminUnit,
    responses={409: {"model": Error}, 422: {"model": Error}},
)
async def create_unit(
    payload: AdminUnitCreate, actor: CurrentAdminDep, units: UnitDirectoryDep
) -> AdminUnit:
    """Unit baru langsung aktif: tampil di menu chatbot dan dapat dipilih di dashboard."""
    try:
        unit = await units.buat(
            nama=payload.nama, deskripsi=payload.deskripsi or None, urutan=payload.urutan
        )
    except DuplicateUnitError as exc:
        raise _bentrok(payload.nama) from exc
    audit.warning("Unit %s dibuat oleh %s", unit.nama, actor.email)
    return unit_out(unit)


@router.patch(
    "/{nama}",
    response_model=AdminUnit,
    responses={404: {"model": Error}, 409: {"model": Error}, 422: {"model": Error}},
)
async def update_unit(
    nama: str, payload: AdminUnitUpdate, actor: CurrentAdminDep, units: UnitDirectoryDep
) -> AdminUnit:
    """Ubah nama, deskripsi, urutan, atau status aktif.

    Nama baru ikut tersimpan di setiap dokumen dan akun unit itu. Menonaktifkan
    menyembunyikannya dari menu chatbot dan dari pilihan isian baru; dokumen
    dan akunnya tetap ada.
    """
    changes = payload.model_dump(exclude_unset=True)
    if "deskripsi" in changes:
        changes["deskripsi"] = changes["deskripsi"] or None
    try:
        unit = await units.ubah(nama, changes)
    except DuplicateUnitError as exc:
        raise _bentrok(changes["nama"]) from exc
    if unit is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_DITEMUKAN)
    audit.warning("Unit %s diubah oleh %s: %s", nama, actor.email, changes)
    return unit_out(unit)
