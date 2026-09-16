"""Autentikasi admin (AD-1, FR-9)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import bcrypt
import jwt

ALGORITHM = "HS256"

MIN_SECRET_BYTES = 32
"""RFC 7518 §3.2: kunci HMAC-SHA256 minimal sepanjang keluaran hash.

Kunci yang lebih pendek membuat token admin dapat ditebak dengan brute force.
PyJWT hanya memperingatkan; di sini dijadikan kegagalan supaya konfigurasi
yang lemah tidak pernah sampai ke produksi.
"""


def _pastikan_secret_kuat(secret: str) -> None:
    panjang = len(secret.encode("utf-8"))
    if panjang < MIN_SECRET_BYTES:
        raise ValueError(
            f"secret JWT terlalu pendek: {panjang} byte, minimal {MIN_SECRET_BYTES}. "
            "Bangkitkan dengan: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
        )


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("kata sandi admin minimal 12 karakter")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


@lru_cache(maxsize=1)
def dummy_password_hash() -> str:
    """Hash pembanding saat email yang dicoba tidak terdaftar.

    Tanpa ini, login untuk email yang tidak ada selesai jauh lebih cepat (tidak
    ada bcrypt), sehingga waktu respons membocorkan email mana yang punya akun
    admin -- persis yang ingin dicegah oleh pesan galat yang seragam.
    """
    return bcrypt.hashpw(b"pembanding-waktu-login-admin", bcrypt.gensalt()).decode("utf-8")


def create_access_token(
    subject: str, secret: str, *, ttl_minutes: int = 480, role: str = "editor"
) -> str:
    _pastikan_secret_kuat(secret)
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=ttl_minutes),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, secret: str) -> dict[str, Any]:
    """Raises `jwt.PyJWTError` bila token kedaluwarsa atau tanda tangannya salah."""
    return jwt.decode(token, secret, algorithms=[ALGORITHM])
