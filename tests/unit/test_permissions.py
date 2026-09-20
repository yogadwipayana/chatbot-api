"""Aturan level akses dashboard -- tanpa FastAPI, tanpa database."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest

from app.admin.permissions import (
    LAST_SUPERADMIN,
    SELF_CHANGE,
    SELF_DELETE,
    STAF_NEEDS_UNIT,
    AdminRole,
    CurrentAdmin,
    check_account_change,
    check_account_delete,
    normalize_unit,
)

STAF, ADMIN, SUPER = AdminRole.STAF, AdminRole.ADMIN, AdminRole.SUPERADMIN


@dataclass
class Akun:
    role: AdminRole
    unit: str | None = None
    is_active: bool = True
    id: uuid.UUID = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.id = self.id or uuid.uuid4()


def pelaku(role: AdminRole = SUPER, unit: str | None = None, id_: uuid.UUID | None = None):
    return CurrentAdmin(id=id_ or uuid.uuid4(), email="x@instiki.ac.id", role=role, unit=unit)


class TestLevel:
    @pytest.mark.parametrize(
        "role,minimum,boleh",
        [
            (STAF, STAF, True),
            (STAF, ADMIN, False),
            (STAF, SUPER, False),
            (ADMIN, STAF, True),
            (ADMIN, ADMIN, True),
            (ADMIN, SUPER, False),
            (SUPER, SUPER, True),
        ],
    )
    def test_bertingkat(self, role, minimum, boleh):
        assert pelaku(role).at_least(minimum) is boleh


class TestLingkupUnit:
    def test_normalisasi_huruf_besar_dan_spasi(self):
        assert normalize_unit("  Biro   KEUANGAN ") == normalize_unit("biro keuangan")

    def test_staf_hanya_unitnya(self):
        staf = pelaku(STAF, unit="Biro Keuangan")
        assert staf.can_manage_unit("biro  keuangan")
        assert not staf.can_manage_unit("Biro Administrasi Akademik")
        assert staf.unit_scope == "Biro Keuangan"

    @pytest.mark.parametrize("role", [ADMIN, SUPER])
    def test_admin_ke_atas_semua_unit(self, role):
        admin = pelaku(role, unit="Biro Keuangan")
        assert admin.unit_scope is None
        assert admin.can_manage_unit("Unit Mana Pun")

    @pytest.mark.parametrize("unit", [None, "", "   "])
    def test_staf_tanpa_unit_tidak_mendapat_apa_pun(self, unit):
        """Data rusak tidak boleh berubah menjadi akses ke semua unit."""
        staf = pelaku(STAF, unit=unit)
        assert staf.unit_scope is not None
        assert not staf.can_manage_unit("Biro Keuangan")
        assert not staf.can_manage_unit(unit)


class TestPerubahanAkun:
    def test_superadmin_tidak_bisa_menurunkan_levelnya_sendiri(self):
        diri = Akun(SUPER)
        assert (
            check_account_change(pelaku(id_=diri.id), diri, {"role": ADMIN}, 2) == SELF_CHANGE
        )

    def test_superadmin_tidak_bisa_menonaktifkan_dirinya(self):
        diri = Akun(SUPER)
        assert (
            check_account_change(pelaku(id_=diri.id), diri, {"is_active": False}, 2)
            == SELF_CHANGE
        )

    def test_mengubah_nama_sendiri_boleh(self):
        diri = Akun(SUPER)
        assert check_account_change(pelaku(id_=diri.id), diri, {"nama": "Baru"}, 1) is None

    @pytest.mark.parametrize("perubahan", [{"role": ADMIN}, {"is_active": False}])
    def test_superadmin_aktif_terakhir_tidak_bisa_disingkirkan(self, perubahan):
        assert check_account_change(pelaku(), Akun(SUPER), perubahan, 1) == LAST_SUPERADMIN

    def test_superadmin_lain_boleh_disingkirkan_bila_masih_ada(self):
        assert check_account_change(pelaku(), Akun(SUPER), {"role": ADMIN}, 2) is None

    def test_superadmin_nonaktif_tidak_dihitung_sebagai_yang_terakhir(self):
        assert (
            check_account_change(pelaku(), Akun(SUPER, is_active=False), {"role": ADMIN}, 1)
            is None
        )

    def test_menjadi_staf_wajib_unit(self):
        assert (
            check_account_change(pelaku(), Akun(ADMIN), {"role": STAF}, 1) == STAF_NEEDS_UNIT
        )
        assert (
            check_account_change(pelaku(), Akun(ADMIN), {"role": STAF, "unit": "Biro X"}, 1)
            is None
        )

    def test_mengosongkan_unit_staf_ditolak(self):
        assert (
            check_account_change(pelaku(), Akun(STAF, unit="Biro X"), {"unit": None}, 1)
            == STAF_NEEDS_UNIT
        )


class TestHapusAkun:
    def test_diri_sendiri(self):
        diri = Akun(SUPER)
        assert check_account_delete(pelaku(id_=diri.id), diri, 2) == SELF_DELETE

    def test_superadmin_aktif_terakhir(self):
        assert check_account_delete(pelaku(), Akun(SUPER), 1) == LAST_SUPERADMIN

    @pytest.mark.parametrize("role", [STAF, ADMIN])
    def test_akun_lain_boleh(self, role):
        assert check_account_delete(pelaku(), Akun(role, unit="Biro X"), 1) is None
