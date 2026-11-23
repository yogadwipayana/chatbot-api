"""Kelola daftar unit layanan dan kontak PIC-nya (khusus superadmin).

Daftar ini menentukan menu unit chatbot, isian unit di seluruh dashboard, dan
pembatasan akses staf. Karena itu unit tidak dihapus -- hanya dinonaktifkan --
dan penggantian nama mengalir ke dokumen, akun, dan kontak lewat
`ON UPDATE CASCADE`.

Kontak PIC (tabel `unit_contacts`) tampil di banner eskalasi dan teks
penolakan widget mahasiswa (FE-3) begitu disimpan, tanpa deploy. Kontak
berkata kunci hanya tampil bila pertanyaannya menyebut salah satu kata kunci
itu (`app.rag.risk.pilih_kontak`). Jangan mengisinya dengan kontak yang tidak
berasal dari kampus.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.admin.permissions import AdminRole
from app.deps import CurrentAdminDep, UnitDirectoryDep, require_role
from app.schemas.admin import (
    AdminUnit,
    AdminUnitContact,
    AdminUnitContactCreate,
    AdminUnitContactUpdate,
    AdminUnitCreate,
    AdminUnitUpdate,
)
from app.schemas.common import Error
from app.units import (
    CONTACT_CHANNELS,
    ContactRecord,
    DuplicateContactError,
    DuplicateUnitError,
    UnitRecord,
)

audit = logging.getLogger("app.audit")

router = APIRouter(
    prefix="/api/admin/units",
    tags=["admin-units"],
    dependencies=[Depends(require_role(AdminRole.SUPERADMIN))],
    responses={401: {"model": Error}, 403: {"model": Error}},
)

TIDAK_DITEMUKAN = "Unit tidak ditemukan."
KONTAK_TIDAK_DITEMUKAN = "Kontak tidak ditemukan."
TANPA_KANAL = "Isi minimal satu dari WhatsApp, telepon, atau email."


def unit_out(unit: UnitRecord) -> AdminUnit:
    return AdminUnit(
        name=unit.name,
        description=unit.description,
        sort_order=unit.sort_order,
        is_active=unit.is_active,
        document_count=unit.document_count,
        account_count=unit.account_count,
        contact_count=unit.contact_count,
    )


def contact_out(kontak: ContactRecord) -> AdminUnitContact:
    return AdminUnitContact(
        id=kontak.id,
        label=kontak.label,
        phone=kontak.phone,
        whatsapp=kontak.whatsapp,
        email=kontak.email,
        service_hours=kontak.service_hours,
        sort_order=kontak.sort_order,
        display=kontak.display,
        keywords=list(kontak.keywords),
        updated_at=kontak.updated_at,
        updated_by=kontak.updated_by,
    )


def _ringkas(kontak: ContactRecord) -> str:
    """Isi kontak untuk log audit: teks yang tampil, dan kata kuncinya bila ada."""
    if not kontak.keywords:
        return kontak.display
    return f"{kontak.display} [kata kunci: {', '.join(kontak.keywords)}]"


def _label_bentrok(label: str) -> HTTPException:
    return HTTPException(
        status.HTTP_409_CONFLICT,
        f"Kontak '{label}' sudah ada di unit ini (huruf besar-kecil tidak dibedakan).",
    )


async def _unit_ada(units: Any, name: str) -> UnitRecord:
    unit = await units.ambil(name)
    if unit is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_DITEMUKAN)
    return unit


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
            name=payload.name,
            description=payload.description or None,
            sort_order=payload.sort_order,
        )
    except DuplicateUnitError as exc:
        raise _bentrok(payload.name) from exc
    audit.warning("Unit %s dibuat oleh %s", unit.name, actor.email)
    return unit_out(unit)


@router.patch(
    "/{name}",
    response_model=AdminUnit,
    responses={404: {"model": Error}, 409: {"model": Error}, 422: {"model": Error}},
)
async def update_unit(
    name: str, payload: AdminUnitUpdate, actor: CurrentAdminDep, units: UnitDirectoryDep
) -> AdminUnit:
    """Ubah nama, deskripsi, urutan, atau status aktif.

    Nama baru ikut tersimpan di setiap dokumen dan akun unit itu. Menonaktifkan
    menyembunyikannya dari menu chatbot dan dari pilihan isian baru; dokumen
    dan akunnya tetap ada.
    """
    changes = payload.model_dump(exclude_unset=True)
    if "description" in changes:
        changes["description"] = changes["description"] or None
    try:
        unit = await units.ubah(name, changes)
    except DuplicateUnitError as exc:
        raise _bentrok(changes["name"]) from exc
    if unit is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_DITEMUKAN)
    audit.warning("Unit %s diubah oleh %s: %s", name, actor.email, changes)
    return unit_out(unit)


# --- Kontak PIC -----------------------------------------------------------------


@router.get(
    "/{name}/contacts",
    response_model=list[AdminUnitContact],
    responses={404: {"model": Error}},
)
async def list_unit_contacts(name: str, units: UnitDirectoryDep) -> list[AdminUnitContact]:
    """Kontak PIC unit dalam urutan tampil di widget."""
    await _unit_ada(units, name)
    return [contact_out(k) for k in await units.kontak(name)]


@router.post(
    "/{name}/contacts",
    status_code=status.HTTP_201_CREATED,
    response_model=AdminUnitContact,
    responses={404: {"model": Error}, 409: {"model": Error}, 422: {"model": Error}},
)
async def create_unit_contact(
    name: str,
    payload: AdminUnitContactCreate,
    actor: CurrentAdminDep,
    units: UnitDirectoryDep,
) -> AdminUnitContact:
    """Langsung tampil di widget untuk unit ini, menggantikan Front Office bila
    sebelumnya unit ini belum punya kontak."""
    await _unit_ada(units, name)
    data = payload.model_dump()
    try:
        kontak = await units.tambah_kontak(name, data, by=actor.email)
    except DuplicateContactError as exc:
        raise _label_bentrok(payload.label) from exc
    audit.warning(
        "Kontak %s unit %s ditambah oleh %s: %s",
        kontak.label,
        name,
        actor.email,
        _ringkas(kontak),
    )
    return contact_out(kontak)


@router.patch(
    "/{name}/contacts/{contact_id}",
    response_model=AdminUnitContact,
    responses={404: {"model": Error}, 409: {"model": Error}, 422: {"model": Error}},
)
async def update_unit_contact(
    name: str,
    contact_id: uuid.UUID,
    payload: AdminUnitContactUpdate,
    actor: CurrentAdminDep,
    units: UnitDirectoryDep,
) -> AdminUnitContact:
    """Hanya field yang dikirim yang diubah; `null` atau "" mengosongkan kanal."""
    await _unit_ada(units, name)
    lama = await units.ambil_kontak(name, contact_id)
    if lama is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, KONTAK_TIDAK_DITEMUKAN)
    changes = payload.model_dump(exclude_unset=True)
    if not any(changes.get(k, getattr(lama, k)) for k in CONTACT_CHANNELS):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, TANPA_KANAL)
    try:
        kontak = await units.ubah_kontak(name, contact_id, changes, by=actor.email)
    except DuplicateContactError as exc:
        raise _label_bentrok(changes.get("label", lama.label)) from exc
    if kontak is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, KONTAK_TIDAK_DITEMUKAN)
    audit.warning(
        "Kontak %s unit %s diubah oleh %s: %s -> %s",
        lama.label,
        name,
        actor.email,
        _ringkas(lama),
        _ringkas(kontak),
    )
    return contact_out(kontak)


@router.delete(
    "/{name}/contacts/{contact_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": Error}},
)
async def delete_unit_contact(
    name: str, contact_id: uuid.UUID, actor: CurrentAdminDep, units: UnitDirectoryDep
) -> Response:
    """Unit yang kehilangan kontak terakhirnya kembali memakai Front Office."""
    await _unit_ada(units, name)
    lama = await units.ambil_kontak(name, contact_id)
    if lama is None or not await units.hapus_kontak(name, contact_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, KONTAK_TIDAK_DITEMUKAN)
    audit.warning(
        "Kontak %s unit %s dihapus oleh %s: %s", lama.label, name, actor.email, _ringkas(lama)
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
