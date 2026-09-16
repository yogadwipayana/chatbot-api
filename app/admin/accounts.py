"""Akun dashboard admin (tabel `admins`).

Diakses lewat `AccountStore`, bukan SQL langsung di router, supaya test API
dapat memakai penyimpanan di memori dan membuktikan aturan akses tanpa
PostgreSQL.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.permissions import AdminRole, CurrentAdmin


@dataclass(frozen=True)
class Account:
    id: uuid.UUID
    email: str
    role: AdminRole
    password_hash: str
    is_active: bool = True
    nama: str | None = None
    unit: str | None = None
    password_changed_at: datetime | None = None
    """Token yang terbit sebelum waktu ini ditolak."""
    created_at: datetime | None = None
    last_login_at: datetime | None = None

    def as_current(self) -> CurrentAdmin:
        return CurrentAdmin(
            id=self.id, email=self.email, role=self.role, unit=self.unit, nama=self.nama
        )


class DuplicateEmailError(ValueError):
    """Email sudah dipakai akun lain (tidak peka huruf besar)."""


EDITABLE_FIELDS = ("nama", "role", "unit", "is_active")


class AccountStore(Protocol):
    async def find_by_email(self, email: str) -> Account | None: ...

    async def get(self, account_id: uuid.UUID) -> Account | None: ...

    async def list(self) -> list[Account]: ...

    async def create(
        self,
        *,
        email: str,
        nama: str | None,
        role: AdminRole,
        unit: str | None,
        password_hash: str,
    ) -> Account: ...

    async def update(
        self, account_id: uuid.UUID, changes: dict[str, Any]
    ) -> Account | None: ...

    async def set_password(
        self, account_id: uuid.UUID, password_hash: str, changed_at: datetime
    ) -> None: ...

    async def delete(self, account_id: uuid.UUID) -> bool: ...

    async def count_active_superadmins(self) -> int: ...

    async def record_login(self, account_id: uuid.UUID) -> None: ...


_KOLOM = (
    "id, email, nama, role, unit, is_active, password_hash, password_changed_at,"
    " created_at, last_login_at"
)


def _account(row: Any) -> Account:
    data = dict(row)
    data["role"] = AdminRole(data["role"])
    return Account(**data)


class SqlAccountStore:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def _one(self, sql: str, params: dict[str, Any]) -> Account | None:
        row = (await self.session.execute(text(sql), params)).mappings().first()
        return _account(row) if row else None

    async def find_by_email(self, email: str) -> Account | None:
        return await self._one(
            f"SELECT {_KOLOM} FROM admins WHERE lower(email) = lower(:email)",
            {"email": email.strip()},
        )

    async def get(self, account_id: uuid.UUID) -> Account | None:
        return await self._one(
            f"SELECT {_KOLOM} FROM admins WHERE id = :id", {"id": account_id}
        )

    async def list(self) -> list[Account]:
        rows = await self.session.execute(
            text(
                f"SELECT {_KOLOM} FROM admins ORDER BY"
                " CASE role WHEN 'superadmin' THEN 0 WHEN 'admin' THEN 1 ELSE 2 END,"
                " lower(email)"
            )
        )
        return [_account(r) for r in rows.mappings()]

    async def create(
        self,
        *,
        email: str,
        nama: str | None,
        role: AdminRole,
        unit: str | None,
        password_hash: str,
    ) -> Account:
        if await self.find_by_email(email):
            raise DuplicateEmailError(email)
        try:
            account = await self._one(
                "INSERT INTO admins (id, email, nama, role, unit, password_hash, is_active)"
                " VALUES (:id, :email, :nama, :role, :unit, :hash, true)"
                f" RETURNING {_KOLOM}",
                {
                    "id": uuid.uuid4(),
                    "email": email,
                    "nama": nama,
                    "role": role.value,
                    "unit": unit,
                    "hash": password_hash,
                },
            )
        except IntegrityError:
            await self.session.rollback()
            # Balapan dua pembuatan akun dengan email sama; selain itu galat sungguhan.
            if await self.find_by_email(email):
                raise DuplicateEmailError(email) from None
            raise
        await self.session.commit()
        assert account is not None
        return account

    async def update(self, account_id: uuid.UUID, changes: dict[str, Any]) -> Account | None:
        kolom = [k for k in EDITABLE_FIELDS if k in changes]
        if not kolom:
            return await self.get(account_id)
        params = {k: changes[k] for k in kolom}
        if "role" in params:
            params["role"] = AdminRole(params["role"]).value
        account = await self._one(
            f"UPDATE admins SET {', '.join(f'{k} = :{k}' for k in kolom)}"
            f" WHERE id = :id RETURNING {_KOLOM}",
            {**params, "id": account_id},
        )
        if account is None:
            await self.session.rollback()
            return None
        await self.session.commit()
        return account

    async def set_password(
        self, account_id: uuid.UUID, password_hash: str, changed_at: datetime
    ) -> None:
        # Waktu dari aplikasi, bukan now() database: pembandingnya adalah `iat`
        # token yang juga dibuat aplikasi. Selisih jam server database dapat
        # membuat token yang baru diterbitkan langsung dianggap usang.
        await self.session.execute(
            text(
                "UPDATE admins SET password_hash = :hash, password_changed_at = :waktu"
                " WHERE id = :id"
            ),
            {"hash": password_hash, "waktu": changed_at, "id": account_id},
        )
        await self.session.commit()

    async def delete(self, account_id: uuid.UUID) -> bool:
        row = (
            await self.session.execute(
                text("DELETE FROM admins WHERE id = :id RETURNING id"), {"id": account_id}
            )
        ).first()
        if row is None:
            await self.session.rollback()
            return False
        await self.session.commit()
        return True

    async def count_active_superadmins(self) -> int:
        return (
            await self.session.execute(
                text("SELECT count(*) FROM admins WHERE role = 'superadmin' AND is_active")
            )
        ).scalar_one()

    async def record_login(self, account_id: uuid.UUID) -> None:
        await self.session.execute(
            text("UPDATE admins SET last_login_at = now() WHERE id = :id"), {"id": account_id}
        )
        await self.session.commit()
