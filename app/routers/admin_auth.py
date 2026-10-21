"""Autentikasi dashboard admin (AD-1) dan akun milik sendiri.

Setiap login (berhasil, gagal, terkunci), penggantian kata sandi, dan percobaan
yang gagal dicatat ke log `app.audit` beserta IP-nya (H31). Tanpa catatan itu,
tebakan kata sandi massal hanya terlihat sebagai deretan 401 di konsol uvicorn,
dan halaman Log dashboard tidak pernah menunjukkannya. Kata sandi tidak pernah
ikut dicatat.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

import anyio
from fastapi import APIRouter, HTTPException, Request, status

from app.config import Settings
from app.deps import (
    AccountStoreDep,
    BaseSettingsDep,
    CurrentAdminDep,
    LoginLimiterDep,
    SesiAdminDep,
)
from app.routers.admin_users import user_out
from app.schemas.admin import AdminUser, LoginRequest, PasswordChange, TokenResponse
from app.schemas.common import Error
from app.security.auth import (
    alasan_sandi_lemah,
    create_access_token,
    dummy_password_hash,
    hash_password,
    verify_password,
)
from app.security.ratelimit import client_ip

audit = logging.getLogger("app.audit")

router = APIRouter(prefix="/api/admin", tags=["admin-auth"])

LOGIN_GAGAL = "Email atau kata sandi salah"
TERLALU_BANYAK = "Terlalu banyak percobaan masuk yang gagal. Coba lagi dalam beberapa menit."
AKUN_NONAKTIF = "Akun ini sudah dinonaktifkan. Hubungi superadmin."
SILAKAN_MASUK_KEMBALI = "Silakan masuk kembali."


def _terlalu_banyak(tunggu: float) -> HTTPException:
    return HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS,
        TERLALU_BANYAK,
        headers={"Retry-After": str(max(1, math.ceil(tunggu)))},
    )


def _token(
    email: str, role: str, settings: Settings, *, masuk: datetime | None = None
) -> TokenResponse:
    """Token sesi dashboard (H27).

    Berlaku `ADMIN_IDLE_TIMEOUT_MINUTES` dan diperpanjang dashboard selama admin
    berinteraksi, tetapi tidak pernah melewati `ADMIN_TOKEN_TTL_MINUTES` sejak
    `masuk`. Kosong berarti sesi baru dimulai sekarang (login, ganti kata sandi).
    """
    token = create_access_token(
        email,
        settings.admin_jwt_secret.get_secret_value(),
        ttl_minutes=settings.admin_idle_timeout_minutes,
        max_age_minutes=settings.admin_token_ttl_minutes,
        auth_time=masuk,
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
    ip = client_ip(request, settings.client_ip_header)
    kunci = (f"ip:{ip}", f"email:{email}")

    tunggu = max((limiter.retry_after(k) or 0.0) for k in kunci)
    if tunggu > 0:
        # Tidak dicatat per percobaan: selama terkunci, setiap percobaan ditolak
        # tanpa memeriksa kata sandi, dan saat mulai terkunci sudah tercatat.
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
        audit.warning(
            "Login gagal untuk %s dari %s (%s)",
            email,
            ip,
            "email tidak terdaftar" if account is None else "kata sandi salah",
        )
        if any(limiter.retry_after(k) for k in kunci):
            audit.warning(
                "Login untuk %s dari %s dikunci sementara: terlalu banyak kegagalan",
                email,
                ip,
            )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, LOGIN_GAGAL)

    for k in kunci:
        limiter.reset(k)

    if not account.is_active:
        audit.warning("Login ditolak untuk %s dari %s: akun nonaktif", email, ip)
        raise HTTPException(status.HTTP_403_FORBIDDEN, AKUN_NONAKTIF)

    await store.record_login(account.id)
    audit.info("Login %s (%s) dari %s", account.email, account.role.value, ip)
    return _token(account.email, account.role.value, settings)


@router.post(
    "/token/refresh",
    response_model=TokenResponse,
    responses={401: {"model": Error}},
)
async def refresh_token(sesi: SesiAdminDep, settings: BaseSettingsDep) -> TokenResponse:
    """Token baru untuk sesi yang sama, dipanggil dashboard selama admin aktif (H27).

    Token admin hanya berlaku `ADMIN_IDLE_TIMEOUT_MINUTES`. Dashboard memintanya
    diperpanjang selama admin masih berinteraksi, sehingga sesi yang ditinggal
    berakhir sendiri, juga token yang tersalin dari komputer bersama. Waktu
    login aslinya ikut terbawa: perpanjangan tidak pernah membuat sesi melewati
    `ADMIN_TOKEN_TTL_MINUTES` sejak login. Pemeriksaan akun dan kata sandi sama
    dengan endpoint admin lain, jadi akun nonaktif tidak dapat memperpanjang.
    """
    admin = sesi.admin
    return _token(admin.email, admin.role.value, settings, masuk=sesi.masuk)


@router.get("/me", response_model=AdminUser, responses={401: {"model": Error}})
async def get_me(admin: CurrentAdminDep, store: AccountStoreDep) -> AdminUser:
    """Akun yang sedang masuk, beserta level dan unitnya menurut database saat ini."""
    account = await store.get(admin.id)
    if account is None:  # pragma: no cover - dihapus di antara dua query
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, SILAKAN_MASUK_KEMBALI)
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

    Kata sandi baru yang mudah ditebak ditolak (H28, `alasan_sandi_lemah`):
    sandi umum, kata dasar seperti "instiki" atau "password" dengan angka di
    belakangnya, urutan, pengulangan, dan yang sebagian besar berisi email akun.
    """
    kunci = f"password:{admin.id}"
    tunggu = limiter.retry_after(kunci)
    if tunggu:
        raise _terlalu_banyak(tunggu)

    account = await store.get(admin.id)
    if account is None:  # dihapus di antara dua query
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, SILAKAN_MASUK_KEMBALI)
    if not await anyio.to_thread.run_sync(
        verify_password, payload.current_password, account.password_hash
    ):
        limiter.record_failure(kunci)
        audit.warning("Ganti kata sandi %s gagal: kata sandi lama salah", account.email)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Kata sandi lama salah.")
    limiter.reset(kunci)

    if payload.new_password == payload.current_password:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Kata sandi baru harus berbeda dari kata sandi lama."
        )
    alasan = alasan_sandi_lemah(payload.new_password, email=account.email)
    if alasan:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Kata sandi baru terlalu mudah ditebak: {alasan}. Pakai frasa panjang "
            "yang hanya Anda ketahui, mis. beberapa kata acak dipisah tanda hubung.",
        )

    password_hash = await anyio.to_thread.run_sync(hash_password, payload.new_password)
    await store.set_password(account.id, password_hash, datetime.now(UTC))
    audit.warning("Kata sandi %s diganti sendiri; sesi lain akun ini berakhir", account.email)
    return _token(account.email, account.role.value, settings)
