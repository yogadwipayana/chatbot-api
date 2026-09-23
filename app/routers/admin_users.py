"""Kelola akun dashboard dan level aksesnya (khusus superadmin)."""

from __future__ import annotations

import logging
import secrets
import uuid
from datetime import UTC, datetime

import anyio
from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.admin.accounts import Account, DuplicateEmailError
from app.admin.permissions import (
    STAF_NEEDS_UNIT,
    AdminRole,
    check_account_change,
    check_account_delete,
    normalize_unit,
)
from app.deps import (
    AccountStoreDep,
    CurrentAdminDep,
    UnitDirectoryDep,
    require_role,
    unit_terdaftar,
)
from app.schemas.admin import (
    AdminUser,
    AdminUserCreate,
    AdminUserCreated,
    AdminUserUpdate,
    TemporaryPassword,
)
from app.schemas.common import Error
from app.security.auth import hash_password

audit = logging.getLogger("app.audit")

router = APIRouter(
    prefix="/api/admin/users",
    tags=["admin-pengguna"],
    dependencies=[Depends(require_role(AdminRole.SUPERADMIN))],
    responses={401: {"model": Error}, 403: {"model": Error}},
)

TIDAK_DITEMUKAN = "Akun tidak ditemukan."


def user_out(account: Account) -> AdminUser:
    return AdminUser(
        id=str(account.id),
        email=account.email,
        role=account.role,
        is_active=account.is_active,
        nama=account.nama,
        unit=account.unit,
        created_at=account.created_at,
        last_login_at=account.last_login_at,
    )


def temporary_password() -> str:
    """20 karakter acak; lolos syarat minimal 12 karakter `hash_password`."""
    return secrets.token_urlsafe(15)


async def _target(store, user_id: uuid.UUID) -> Account:
    account = await store.get(user_id)
    if account is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_DITEMUKAN)
    return account


def _konflik(pesan: str) -> HTTPException:
    return HTTPException(status.HTTP_409_CONFLICT, pesan)


@router.get("", response_model=list[AdminUser])
async def list_users(store: AccountStoreDep) -> list[AdminUser]:
    return [user_out(a) for a in await store.list()]


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=AdminUserCreated,
    responses={409: {"model": Error}, 422: {"model": Error}},
)
async def create_user(
    payload: AdminUserCreate,
    actor: CurrentAdminDep,
    store: AccountStoreDep,
    units: UnitDirectoryDep,
) -> AdminUserCreated:
    """Buat akun dengan kata sandi sementara yang ditampilkan SEKALI.

    Superadmin menyerahkannya lewat jalur aman, lalu pemilik akun menggantinya
    sendiri. Superadmin tidak pernah memilih kata sandi orang lain, sehingga
    tidak ada yang mengetahui kata sandi akhir selain pemiliknya.
    """
    unit = await unit_terdaftar(units, payload.unit) if payload.unit else None
    if payload.role is AdminRole.STAF and not normalize_unit(unit):
        raise _konflik(STAF_NEEDS_UNIT)

    sandi = temporary_password()
    password_hash = await anyio.to_thread.run_sync(hash_password, sandi)
    try:
        account = await store.create(
            email=str(payload.email).lower(),
            nama=payload.nama or None,
            role=payload.role,
            unit=unit,
            password_hash=password_hash,
        )
    except DuplicateEmailError as exc:
        raise _konflik("Email ini sudah dipakai akun lain.") from exc

    audit.warning(
        "Akun %s (%s, unit %s) dibuat oleh %s", account.email, account.role, unit, actor.email
    )
    return AdminUserCreated(user=user_out(account), password_sementara=sandi)


@router.patch(
    "/{user_id}",
    response_model=AdminUser,
    responses={404: {"model": Error}, 409: {"model": Error}, 422: {"model": Error}},
)
async def update_user(
    user_id: uuid.UUID,
    payload: AdminUserUpdate,
    actor: CurrentAdminDep,
    store: AccountStoreDep,
    units: UnitDirectoryDep,
) -> AdminUser:
    """Ubah nama, level, unit, atau status aktif. Berlaku pada permintaan berikutnya."""
    target = await _target(store, user_id)
    changes = payload.model_dump(exclude_unset=True)
    for kolom in ("nama", "unit"):
        if kolom in changes:
            changes[kolom] = changes[kolom] or None
    if changes.get("unit"):
        changes["unit"] = await unit_terdaftar(units, changes["unit"])

    alasan = check_account_change(
        actor, target, changes, await store.count_active_superadmins()
    )
    if alasan:
        raise _konflik(alasan)

    updated = await store.update(user_id, changes)
    if updated is None:  # pragma: no cover - dihapus di antara dua query
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_DITEMUKAN)
    audit.warning("Akun %s diubah oleh %s: %s", target.email, actor.email, changes)
    return user_out(updated)


@router.post(
    "/{user_id}/reset-password",
    response_model=TemporaryPassword,
    responses={404: {"model": Error}, 409: {"model": Error}},
)
async def reset_user_password(
    user_id: uuid.UUID, actor: CurrentAdminDep, store: AccountStoreDep
) -> TemporaryPassword:
    """Bangkitkan kata sandi sementara baru. Semua sesi akun itu langsung berakhir."""
    target = await _target(store, user_id)
    if target.id == actor.id:
        raise _konflik("Untuk akun Anda sendiri, gunakan menu Ganti kata sandi.")

    sandi = temporary_password()
    password_hash = await anyio.to_thread.run_sync(hash_password, sandi)
    await store.set_password(target.id, password_hash, datetime.now(UTC))
    audit.warning("Kata sandi %s diatur ulang oleh %s", target.email, actor.email)
    return TemporaryPassword(password_sementara=sandi)


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={404: {"model": Error}, 409: {"model": Error}},
)
async def delete_user(
    user_id: uuid.UUID, actor: CurrentAdminDep, store: AccountStoreDep
) -> Response:
    """Permanen. Untuk menghentikan akses sementara, nonaktifkan saja.

    Dokumen yang pernah diunggah akun ini tidak ikut terhapus; `uploaded_by`
    tetap mencatat email pengunggahnya.
    """
    target = await _target(store, user_id)
    alasan = check_account_delete(actor, target, await store.count_active_superadmins())
    if alasan:
        raise _konflik(alasan)
    await store.delete(target.id)
    audit.warning("Akun %s dihapus oleh %s", target.email, actor.email)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
