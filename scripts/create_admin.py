"""Buat atau atur ulang akun dashboard dari server (AD-1).

    python -m scripts.create_admin admin@instiki.ac.id --role superadmin
    python -m scripts.create_admin keuangan@instiki.ac.id --role staf --unit Keuangan
    python -m scripts.create_admin admin@instiki.ac.id --reset

Dipakai untuk akun superadmin PERTAMA, atau untuk memulihkan akses bila semua
superadmin terkunci. Akun lain sebaiknya dibuat lewat menu Admin di dashboard
oleh superadmin.

Kata sandi dibangkitkan acak dan ditampilkan SEKALI -- simpan di pengelola kata
sandi, lalu serahkan lewat jalur yang aman. Pemilik akun sebaiknya segera
menggantinya lewat menu Ganti kata sandi.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
import uuid
from datetime import UTC, datetime

from email_validator import EmailNotValidError, validate_email
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.admin.permissions import AdminRole, normalize_unit
from app.config import get_settings
from app.security.auth import hash_password
from app.units import SqlUnitDirectory


async def jalankan(args: argparse.Namespace, email: str) -> int:
    engine = create_async_engine(str(get_settings().database_url))
    sandi = secrets.token_urlsafe(15)
    try:
        async with engine.begin() as conn:
            ada = (
                await conn.execute(
                    text("SELECT role, unit FROM admins WHERE lower(email) = :email"),
                    {"email": email},
                )
            ).first()
            if ada and not args.reset:
                print(f"Akun {email} sudah ada. Pakai --reset untuk membuat kata sandi baru.")
                return 1
            if not ada and args.reset:
                print(f"Akun {email} belum ada; tidak ada yang dapat di-reset.")
                return 1

            role = args.role or (ada.role if ada else AdminRole.ADMIN.value)
            unit = args.unit if args.unit is not None else (ada.unit if ada else None)
            if role == AdminRole.STAF and not normalize_unit(unit):
                print("Level staf wajib disertai --unit.")
                return 1
            if unit:
                units = SqlUnitDirectory(conn)
                resmi = await units.resolve(unit)
                if resmi is None:
                    pilihan = ", ".join(u.nama for u in await units.list())
                    print(f"Unit '{unit}' tidak terdaftar. Pilih salah satu: {pilihan}.")
                    return 1
                unit = resmi

            password_hash = hash_password(sandi)
            if ada:
                await conn.execute(
                    text(
                        "UPDATE admins SET password_hash = :hash, role = :role, unit = :unit,"
                        " nama = coalesce(:nama, nama), is_active = true,"
                        " password_changed_at = :waktu WHERE lower(email) = :email"
                    ),
                    {
                        "hash": password_hash,
                        "role": role,
                        "unit": unit,
                        "nama": args.nama,
                        "waktu": datetime.now(UTC),
                        "email": email,
                    },
                )
            else:
                await conn.execute(
                    text(
                        "INSERT INTO admins (id, email, nama, role, unit, password_hash)"
                        " VALUES (:id, :email, :nama, :role, :unit, :hash)"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "email": email,
                        "nama": args.nama,
                        "role": role,
                        "unit": unit,
                        "hash": password_hash,
                    },
                )
    finally:
        await engine.dispose()

    print(f"Akun {'di-reset dan diaktifkan' if args.reset else 'dibuat'}: {email} ({role})")
    print(f"Kata sandi (hanya ditampilkan sekali): {sandi}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Buat atau atur ulang akun dashboard.")
    parser.add_argument("email")
    parser.add_argument(
        "--role", choices=[r.value for r in AdminRole], default=None, help="default: admin"
    )
    parser.add_argument(
        "--unit", default=None, help="wajib untuk --role staf; nama dari tabel units"
    )
    parser.add_argument("--nama", default=None)
    parser.add_argument(
        "--reset", action="store_true", help="kata sandi baru, aktifkan kembali akun"
    )
    args = parser.parse_args()

    try:
        email = validate_email(args.email, check_deliverability=False).normalized.lower()
    except EmailNotValidError as exc:
        parser.error(f"email tidak sah: {exc}")

    return asyncio.run(jalankan(args, email))


if __name__ == "__main__":
    sys.exit(main())
