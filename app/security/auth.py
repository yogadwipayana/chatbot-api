"""Autentikasi admin (AD-1, FR-9)."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import bcrypt
import jwt

ALGORITHM = "HS256"

ISSUER = "chatbot-administrasi-api"
AUDIENCE = "chatbot-administrasi-admin"
"""`iss`/`aud` token admin (H30). Rahasia JWT kelak bisa dipakai bersama layanan
lain; tanpa keduanya, token yang ditandatangani untuk keperluan lain dengan
rahasia yang sama ikut diterima sebagai sesi dashboard."""

KLAIM_WAJIB = ("sub", "iat", "exp", "iss", "aud")
"""PyJWT hanya memeriksa `exp` bila klaimnya ADA. Token tanpa `exp` berlaku
selamanya, dan token tanpa `iat` lolos pemeriksaan "terbit sebelum kata sandi
diganti" bila kata sandinya belum pernah diganti."""

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
    subject: str,
    secret: str,
    *,
    ttl_minutes: float = 480,
    role: str = "editor",
    auth_time: datetime | None = None,
    max_age_minutes: float | None = None,
) -> str:
    """Token admin yang berlaku `ttl_minutes` sejak sekarang.

    `auth_time` adalah saat admin terakhir memasukkan kata sandi; kosong berarti
    sekarang (login). Token perpanjangan membawa `auth_time` login aslinya, dan
    `max_age_minutes` memotong `exp` supaya sesi tidak pernah melewati
    `auth_time + max_age_minutes`, berapa kali pun diperpanjang (H27).
    """
    _pastikan_secret_kuat(secret)
    now = datetime.now(UTC)
    mulai = auth_time or now
    exp = now + timedelta(minutes=ttl_minutes)
    if max_age_minutes is not None:
        exp = min(exp, mulai + timedelta(minutes=max_age_minutes))
    payload = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": exp,
        "auth_time": int(mulai.timestamp()),
        "iss": ISSUER,
        "aud": AUDIENCE,
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, secret: str) -> dict[str, Any]:
    """Raises `jwt.PyJWTError` bila token kedaluwarsa, tanda tangannya salah,
    klaim wajibnya kurang, atau `iss`/`aud`-nya bukan milik dashboard ini."""
    return jwt.decode(
        token,
        secret,
        algorithms=[ALGORITHM],
        audience=AUDIENCE,
        issuer=ISSUER,
        options={"require": list(KLAIM_WAJIB)},
    )


def waktu_masuk(payload: dict[str, Any]) -> datetime:
    """Saat admin memasukkan kata sandi untuk sesi token ini.

    Token lama tanpa `auth_time` memakai `iat`-nya: ia memang belum pernah
    diperpanjang."""
    nilai = payload.get("auth_time", payload.get("iat", 0))
    return datetime.fromtimestamp(int(nilai), UTC)


# --- Kekuatan kata sandi (H28) ---------------------------------------------

_DASAR_UMUM = frozenset(
    {
        "abc",
        "abcd",
        "admin",
        "administrator",
        "asdf",
        "asdfghjkl",
        "asisten",
        "bali",
        "bismillah",
        "changeme",
        "chatbot",
        "dashboard",
        "default",
        "denpasar",
        "dragon",
        "football",
        "iloveyou",
        "indonesia",
        "instiki",
        "instikiacid",
        "katasandi",
        "letmein",
        "login",
        "master",
        "monkey",
        "passw",
        "password",
        "princess",
        "qwerty",
        "qwertyuiop",
        "rahasia",
        "sandi",
        "sayang",
        "shadow",
        "stiki",
        "stikiindonesia",
        "stikom",
        "sunshine",
        "superadmin",
        "test",
        "testing",
        "trustno",
        "user",
        "welcome",
        "zxcvbnm",
    }
)
"""Kata dasar sandi yang paling sering ditebak lebih dulu, termasuk nama kampus.

Dibandingkan dengan inti sandi: huruf kecil, tanpa angka/simbol di awal dan
akhirnya, dan dengan ejaan leet dikembalikan. Akibatnya "Instiki2026!",
"P@ssw0rd1234", dan "adminadmin12" ditolak, sedangkan frasa panjang yang
kebetulan memuat kata ini, mis. "kopi-bali-pagi-hari", tetap boleh."""

_SANDI_UMUM = frozenset(
    {
        "1q2w3e4r5t6y",
        "1q2w3e4r5t6y7u8i",
        "1qaz2wsx3edc",
        "1qaz2wsx3edc4rfv",
        "a1b2c3d4e5f6",
        "q1w2e3r4t5y6",
        "zaq12wsxcde3",
        "zaq1xsw2cde3",
    }
)
"""Pola papan ketik 12 karakter ke atas yang tidak tertangkap aturan lain."""

_URUTAN = (
    "0123456789012345678901234567890123456789",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
    "qwertyuiopasdfghjklzxcvbnmqwertyuiopasdfghjklzxcvbnm",
)
_LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)
_PINGGIR = re.compile(r"^[\d\W_]+|[\d\W_]+$")


def _hanya_kata_dasar(huruf: str) -> bool:
    """`huruf` seluruhnya terdiri dari satu kata dasar umum yang diulang."""
    return any(len(d) >= 3 and not huruf.replace(d, "") for d in _DASAR_UMUM)


def alasan_sandi_lemah(sandi: str, *, email: str | None = None) -> str | None:
    """Alasan `sandi` terlalu mudah ditebak, atau None bila boleh dipakai.

    NIST SP 800-63B §5.1.1.2: bukan aturan komposisi (wajib huruf besar, angka),
    melainkan menolak sandi yang ada di daftar sandi umum, berulang atau berurutan,
    dan yang diturunkan dari nama layanan atau nama pengguna. Panjang minimal 12
    tetap ditegakkan `hash_password`.
    """
    kecil = sandi.casefold()
    if len(set(kecil)) <= 3 or re.fullmatch(r"(.+?)\1+", kecil):
        return "terlalu banyak pengulangan"
    rapat = re.sub(r"[^a-z0-9]", "", kecil)
    if len(rapat) >= 6 and any(rapat in u or rapat in u[::-1] for u in _URUTAN):
        return "berupa urutan huruf, angka, atau papan ketik"
    if rapat in _SANDI_UMUM:
        return "termasuk sandi yang paling umum"

    inti = _PINGGIR.sub("", kecil)
    for calon in (inti, inti.translate(_LEET)):
        huruf = re.sub(r"[^a-z]", "", calon)
        if huruf and _hanya_kata_dasar(huruf):
            return "berupa kata yang paling sering ditebak (mis. password, admin, instiki)"

    if email:
        # Hanya bila nama email menjadi sebagian besar isi sandi: frasa panjang
        # yang kebetulan memuat "admin" tidak lebih mudah ditebak karenanya.
        lokal = re.sub(r"[^a-z]", "", email.casefold().split("@", 1)[0])
        huruf = re.sub(r"[^a-z]", "", kecil)
        if len(lokal) >= 4 and lokal in huruf and len(lokal) * 2 >= len(huruf):
            return "memuat alamat email akun ini"
    return None
