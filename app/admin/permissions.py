"""Level akses dashboard admin.

Tiga level bertingkat; setiap level mencakup semua hak level di bawahnya.

| Level | Hak |
|---|---|
| `staf` (Staf/Dosen) | Dokumen unitnya, uji coba jawaban, lihat pertanyaan tak terjawab |
| `admin` | + kelola dokumen semua unit, tandai pertanyaan selesai, statistik |
| `superadmin` | + kill switch layanan chat, kelola akun dashboard |

Level dibaca dari database pada SETIAP permintaan (`app.deps.require_admin`),
bukan dari isi token. Menurunkan level atau menonaktifkan akun berlaku saat itu
juga, tidak menunggu token berumur 8 jam kedaluwarsa.

Sengaja tanpa impor pihak ketiga: aturan akses harus dapat dibaca dan diuji
tanpa FastAPI maupun database.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class AdminRole(StrEnum):
    STAF = "staf"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"


ROLE_LEVEL: dict[AdminRole, int] = {
    AdminRole.STAF: 1,
    AdminRole.ADMIN: 2,
    AdminRole.SUPERADMIN: 3,
}

ROLE_LABELS: dict[AdminRole, str] = {
    AdminRole.STAF: "Staf/Dosen",
    AdminRole.ADMIN: "Admin",
    AdminRole.SUPERADMIN: "Superadmin",
}


def normalize_unit(unit: str | None) -> str:
    """Bentuk pembanding nama unit: huruf kecil, spasi dirapikan.

    Sejak ada tabel `units`, yang tersimpan selalu ejaan resmi. Normalisasi ini
    tetap dipakai untuk mencocokkan ketikan admin ("keuangan") dengan ejaan
    resmi ("Keuangan") -- lihat `app.units.cocokkan`.
    """
    return " ".join((unit or "").split()).casefold()


@dataclass(frozen=True)
class CurrentAdmin:
    """Akun yang sedang memanggil API, sesuai isi database saat permintaan ini."""

    id: uuid.UUID
    email: str
    role: AdminRole
    unit: str | None = None
    nama: str | None = None

    def at_least(self, minimum: AdminRole) -> bool:
        return ROLE_LEVEL[self.role] >= ROLE_LEVEL[minimum]

    @property
    def unit_scope(self) -> str | None:
        """Unit yang membatasi akses dokumen, atau None bila boleh semua unit.

        Staf tanpa unit (dicegah constraint database, tetapi tetap dijaga di
        sini) mendapat lingkup kosong -- tidak ada dokumen -- bukan semua unit.
        """
        if self.role is not AdminRole.STAF:
            return None
        return self.unit or ""

    def can_manage_unit(self, unit: str | None) -> bool:
        scope = self.unit_scope
        if scope is None:
            return True
        return bool(normalize_unit(scope)) and normalize_unit(scope) == normalize_unit(unit)


class AccountLike(Protocol):
    id: uuid.UUID
    role: AdminRole
    unit: str | None
    is_active: bool


SELF_CHANGE = "Anda tidak dapat mengubah level atau menonaktifkan akun Anda sendiri."
SELF_DELETE = "Anda tidak dapat menghapus akun Anda sendiri."
LAST_SUPERADMIN = "Harus selalu ada setidaknya satu superadmin aktif."
STAF_NEEDS_UNIT = (
    "Staf/dosen wajib memiliki unit, karena aksesnya dibatasi pada dokumen unit tersebut."
)


def _menyingkirkan_superadmin(
    target: AccountLike, role_baru: AdminRole, aktif_baru: bool
) -> bool:
    return (
        AdminRole(target.role) is AdminRole.SUPERADMIN
        and target.is_active
        and (role_baru is not AdminRole.SUPERADMIN or not aktif_baru)
    )


def check_account_change(
    actor: CurrentAdmin,
    target: AccountLike,
    changes: Mapping[str, Any],
    active_superadmins: int,
) -> str | None:
    """Alasan perubahan akun ditolak, atau None bila boleh.

    Dua aturan pertama mencegah dashboard terkunci tanpa ada yang dapat
    mengelolanya: superadmin tidak dapat menurunkan level atau menonaktifkan
    dirinya sendiri, dan superadmin aktif terakhir tidak dapat disingkirkan.
    """
    role_lama = AdminRole(target.role)
    role_baru = AdminRole(changes.get("role") or role_lama)
    aktif_baru = bool(changes.get("is_active", target.is_active))
    unit_baru = changes.get("unit", target.unit)

    if target.id == actor.id and (role_baru is not role_lama or not aktif_baru):
        return SELF_CHANGE
    if _menyingkirkan_superadmin(target, role_baru, aktif_baru) and active_superadmins <= 1:
        return LAST_SUPERADMIN
    if role_baru is AdminRole.STAF and not normalize_unit(unit_baru):
        return STAF_NEEDS_UNIT
    return None


def check_account_delete(
    actor: CurrentAdmin, target: AccountLike, active_superadmins: int
) -> str | None:
    """Alasan penghapusan akun ditolak, atau None bila boleh."""
    if target.id == actor.id:
        return SELF_DELETE
    if _menyingkirkan_superadmin(target, AdminRole.STAF, False) and active_superadmins <= 1:
        return LAST_SUPERADMIN
    return None
