"""Autentikasi dashboard admin (AD-1) dan akun milik sendiri."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import anyio
from fastapi import APIRouter, HTTPException, Request, status

from app.deps import AccountStoreDep, BaseSettingsDep, CurrentAdminDep, LoginLimiterDep
from app.routers.admin_users import user_out
from app.schemas.admin import AdminUser, LoginRequest, PasswordChange, TokenResponse
from app.schemas.common import Error
from app.security.auth import (
    create_access_token,
    dummy_password_hash,
    hash_password,
    verify_password,
)
from app.security.ratelimit import client_ip

router = APIRouter(prefix="/api/admin", tags=["admin-auth"])

LOGIN_GAGAL = "Email atau kata sandi salah"
TERLALU_BANYAK = "Terlalu banyak percobaan masuk yang gagal. Coba lagi dalam beberapa menit."
AKUN_NONAKTIF = "Akun ini sudah dinonaktifkan. Hubungi superadmin."


def _terlalu_banyak(tunggu: float) -> HTTPException:
    return HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS,
        TERLALU_BANYAK,
        headers={"Retry-After": str(max(1, math.ceil(tunggu)))},
    )


def _token(email: str, role: str, settings) -> TokenResponse:
    token = create_access_token(
        email,
        settings.admin_jwt_secret.get_secret_value(),
        ttl_minutes=settings.admin_token_ttl_minutes,
        role=role,
    )
    return TokenResponse(access_token=token, token_type="bearer")


@router.post(
    "/login",
    response_model=TokenResponse,
    responses={401: {"model": Error}, 403: {"model": Error}, 429: {"model": Error}},
)
async def admin_login(
    payload: LoginRequest,
    request: Request,
    store: AccountStoreDep,
    settings: BaseSettingsDep,
    limiter: LoginLimiterDep,
) -> TokenResponse:
    """Tukar email + kata sandi dengan JWT.

    Balasan gagal tidak membedakan email tak terdaftar dari kata sandi salah,
    dan waktunya pun disamakan lewat bcrypt pembanding. Kegagalan dihitung per
    IP dan per email: per IP menahan penebakan massal, per email menahan
    penebakan satu akun dari banyak IP.

    Akun nonaktif baru diberi tahu SETELAH kata sandinya terbukti benar, jadi
    status itu tidak bocor ke orang yang sekadar menebak email.
    """
    email = payload.email.strip().lower()
    kunci = (f"ip:{client_ip(request)}", f"email:{email}")

    tunggu = max((limiter.retry_after(k) or 0.0) for k in kunci)
    if tunggu > 0:
        raise _terlalu_banyak(tunggu)

    account = await store.find_by_email(email)

    # bcrypt sengaja lambat (~250 ms); dijalankan di threadpool agar tidak
    # membekukan event loop untuk permintaan mahasiswa yang sedang berjalan.
    pembanding = (
        account.password_hash
        if account
        else await anyio.to_thread.run_sync(dummy_password_hash)
    )
    cocok = await anyio.to_thread.run_sync(verify_password, payload.password, pembanding)

    if account is None or not cocok:
        for k in kunci:
            limiter.record_failure(k)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, LOGIN_GAGAL)

    for k in kunci:
        limiter.reset(k)

    if not account.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, AKUN_NONAKTIF)

    await store.record_login(account.id)
    return _token(account.email, account.role.value, settings)


@router.get("/me", response_model=AdminUser, responses={401: {"model": Error}})
async def get_me(admin: CurrentAdminDep, store: AccountStoreDep) -> AdminUser:
    """Akun yang sedang masuk, beserta level dan unitnya menurut database saat ini."""
    account = await store.get(admin.id)
    if account is None:  # pragma: no cover - dihapus di antara dua query
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Silakan masuk kembali.")
    return user_out(account)


@router.post(
    "/me/password",
    response_model=TokenResponse,
    responses={400: {"model": Error}, 401: {"model": Error}, 429: {"model": Error}},
)
async def change_my_password(
    payload: PasswordChange,
    admin: CurrentAdminDep,
    store: AccountStoreDep,
    settings: BaseSettingsDep,
    limiter: LoginLimiterDep,
) -> TokenResponse:
    """Ganti kata sandi sendiri, mis. setelah menerima kata sandi sementara.

    Semua sesi lain akun ini ikut berakhir. Sesi yang dipakai untuk mengganti
    menerima token baru di balasan, jadi tidak perlu masuk ulang.
    """
    kunci = f"password:{admin.id}"
    tunggu = limiter.retry_after(kunci)
    if tunggu:
        raise _terlalu_banyak(tunggu)

    account = await store.get(admin.id)
    assert account is not None
    if not await anyio.to_thread.run_sync(
        verify_password, payload.password_lama, account.password_hash
    ):
        limiter.record_failure(kunci)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Kata sandi lama salah.")
    limiter.reset(kunci)

    if payload.password_baru == payload.password_lama:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Kata sandi baru harus berbeda dari kata sandi lama."
        )

    password_hash = await anyio.to_thread.run_sync(hash_password, payload.password_baru)
    await store.set_password(account.id, password_hash, datetime.now(UTC))
    return _token(account.email, account.role.value, settings)
