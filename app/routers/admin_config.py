"""Setelan retrieval dan chunking dari dashboard (FR-1, FR-2, FR-3).

Hanya superadmin: nilainya berlaku untuk setiap pertanyaan mahasiswa, dan
ambang yang keliru membuat chatbot menolak menjawab hal yang sebenarnya ada di
dokumen. Pasangannya adalah kotak uji coba AD-6 -- setel di sini, buktikan di
sana.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

from app.admin.permissions import AdminRole
from app.admin.runtime_config import (
    DAPAT_DIUBAH,
    NilaiTersimpan,
    bangun,
    keluhan,
    pesan_pertama,
    sebagai_teks,
    terapkan,
)
from app.config import Settings
from app.deps import (
    BaseSettingsDep,
    CurrentAdminDep,
    RuntimeConfigStoreDep,
    require_admin,
    require_role,
)
from app.schemas.admin import RuntimeConfig, RuntimeConfigUpdate, RuntimeConfigValues
from app.schemas.common import Error

audit = logging.getLogger("app.audit")

router = APIRouter(
    prefix="/api/admin",
    tags=["admin-operasional"],
    dependencies=[Depends(require_admin), Depends(require_role(AdminRole.SUPERADMIN))],
    responses={401: {"model": Error}, 403: {"model": Error}},
)


@router.get("/config", response_model=RuntimeConfig)
async def get_config(base: BaseSettingsDep, store: RuntimeConfigStoreDep) -> RuntimeConfig:
    """Nilai yang berlaku sekarang, beserta nilai `.env` sebagai pembandingnya."""
    return susun(base, await store.load())


@router.patch(
    "/config", response_model=RuntimeConfig, responses={422: {"model": Error}}
)
async def update_config(
    payload: RuntimeConfigUpdate,
    base: BaseSettingsDep,
    store: RuntimeConfigStoreDep,
    admin: CurrentAdminDep,
) -> RuntimeConfig:
    """Ubah sebagian parameter. Berlaku untuk permintaan berikutnya, tanpa restart.

    Nilai yang sama dengan `.env` -- atau `null` -- menghapus penimpaannya,
    sehingga "kembalikan ke nilai server" untuk satu parameter tidak butuh
    endpoint tersendiri.
    """
    diminta = payload.perubahan()
    tersimpan = await store.load()
    if not diminta:
        return susun(base, tersimpan)

    # Penghapusan (nilai == .env) dan penyimpanan disiapkan bersamaan supaya
    # yang divalidasi persis keadaan setelah permintaan ini dijalankan.
    perubahan: dict[str, str | None] = {
        k: None if v is None or v == getattr(base, k) else sebagai_teks(v)
        for k, v in diminta.items()
    }
    sesudah = {k: v.value for k, v in tersimpan.items()}
    for kunci, nilai in perubahan.items():
        if nilai is None:
            sesudah.pop(kunci, None)
        else:
            sesudah[kunci] = nilai

    try:
        bangun(base, sesudah)
    except ValidationError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, pesan_pertama(exc)
        ) from exc

    await store.replace(perubahan, by=admin.email)
    audit.info(
        "Konfigurasi diubah oleh %s: %s",
        admin.email,
        ", ".join(
            f"{k}={v if v is not None else '(ikut .env)'}" for k, v in perubahan.items()
        ),
    )
    return susun(base, await store.load())


@router.delete("/config", response_model=RuntimeConfig)
async def reset_config(
    base: BaseSettingsDep, store: RuntimeConfigStoreDep, admin: CurrentAdminDep
) -> RuntimeConfig:
    """Hapus seluruh penimpaan: semua parameter kembali mengikuti `.env`.

    Jalan keluar saat penyetelan membuat jawaban makin buruk dan tidak jelas
    lagi nilai mana yang diubah.
    """
    if await store.load():
        await store.clear()
        audit.warning("Konfigurasi dikembalikan ke nilai .env oleh %s", admin.email)
    return susun(base, await store.load())


def susun(base: Settings, tersimpan: dict[str, NilaiTersimpan]) -> RuntimeConfig:
    terbaru = max(tersimpan.values(), key=lambda n: n.updated_at, default=None)
    return RuntimeConfig(
        nilai=_nilai(terapkan(base, tersimpan)),
        nilai_env=_nilai(base),
        diubah=sorted(k for k in tersimpan if k in DAPAT_DIUBAH),
        chat_model=base.chat_model,
        embed_model=base.embed_model,
        base_url=base.base_url,
        # Kunci API tidak pernah meninggalkan server; halaman Konfigurasi hanya
        # perlu tahu sudah diisi atau belum untuk menjelaskan kegagalan model.
        api_key_terisi=base.kunci_api() is not None,
        diperbarui_at=terbaru.updated_at if terbaru else None,
        diperbarui_oleh=terbaru.updated_by if terbaru else None,
        peringatan=keluhan(base, tersimpan),
    )


def _nilai(settings: Settings) -> RuntimeConfigValues:
    return RuntimeConfigValues(**{k: getattr(settings, k) for k in DAPAT_DIUBAH})
