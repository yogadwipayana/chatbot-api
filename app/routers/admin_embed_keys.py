"""Kelola kunci sematan (tabel `embed_keys`, khusus superadmin).

Setiap situs lain yang memasang asisten punya kuncinya sendiri, sehingga satu
situs dapat dibatasi ke domainnya, dinonaktifkan, atau dihapus tanpa menyentuh
situs lain -- dan tanpa build ulang portal. Lihat `app/embed_keys.py`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.admin.permissions import AdminRole
from app.config import Settings
from app.deps import BaseSettingsDep, CurrentAdminDep, EmbedKeyStoreDep, require_role
from app.embed_keys import KunciSematan, kode_sematan
from app.schemas.admin import EmbedKey, EmbedKeyCreate, EmbedKeyUpdate
from app.schemas.common import Error

audit = logging.getLogger("app.audit")

router = APIRouter(
    prefix="/api/admin/embed-keys",
    tags=["admin-embed-keys"],
    dependencies=[Depends(require_role(AdminRole.SUPERADMIN))],
    responses={401: {"model": Error}, 403: {"model": Error}},
)

TIDAK_DITEMUKAN = "Kunci sematan tidak ditemukan."


def embed_key_out(kunci: KunciSematan, settings: Settings) -> EmbedKey:
    portal = settings.url_portal()
    return EmbedKey(
        key=kunci.key,
        name=kunci.name,
        allowed_origins=kunci.allowed_origins,
        is_active=kunci.is_active,
        created_by=kunci.created_by,
        created_at=kunci.created_at,
        questions_30d=kunci.questions_30d,
        last_used_at=kunci.last_used_at,
        embed_code=kode_sematan(portal, kunci.key) if portal else None,
    )


@router.get("", response_model=list[EmbedKey])
async def list_embed_keys(
    store: EmbedKeyStoreDep, settings: BaseSettingsDep
) -> list[EmbedKey]:
    """Semua kunci, termasuk yang nonaktif, beserta pemakaian 30 hari terakhir."""
    return [embed_key_out(k, settings) for k in await store.semua()]


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=EmbedKey,
    responses={422: {"model": Error}},
)
async def create_embed_key(
    payload: EmbedKeyCreate,
    actor: CurrentAdminDep,
    store: EmbedKeyStoreDep,
    settings: BaseSettingsDep,
) -> EmbedKey:
    """Kunci baru langsung aktif. Kuncinya dibuat server, bukan dipilih admin."""
    kunci = await store.buat(
        name=payload.name, allowed_origins=payload.allowed_origins, oleh=actor.email
    )
    audit.warning(
        "Kunci sematan %s (%s) dibuat oleh %s untuk %s",
        kunci.key,
        kunci.name,
        actor.email,
        kunci.allowed_origins or "semua situs",
    )
    return embed_key_out(kunci, settings)


@router.patch(
    "/{key}",
    response_model=EmbedKey,
    responses={404: {"model": Error}, 422: {"model": Error}},
)
async def update_embed_key(
    key: str,
    payload: EmbedKeyUpdate,
    actor: CurrentAdminDep,
    store: EmbedKeyStoreDep,
    settings: BaseSettingsDep,
) -> EmbedKey:
    """Ubah nama, daftar situs, atau status aktif.

    Berlaku saat panel dibuka berikutnya; kunci yang dinonaktifkan juga menolak
    pertanyaan dari panel yang sedang terbuka. Kuncinya sendiri tidak berubah,
    jadi situs penyemat tidak perlu mengganti kodenya.
    """
    changes = payload.model_dump(exclude_unset=True)
    hasil = await store.ubah(key, changes)
    if hasil is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_DITEMUKAN)
    audit.warning("Kunci sematan %s diubah oleh %s: %s", key, actor.email, changes)
    return embed_key_out(hasil, settings)


@router.delete(
    "/{key}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={404: {"model": Error}},
)
async def delete_embed_key(
    key: str, actor: CurrentAdminDep, store: EmbedKeyStoreDep
) -> Response:
    """Hapus permanen. Situs yang masih memasangnya harus diberi kunci baru.

    Percakapan dari situs ini tetap tersimpan, hanya tidak lagi tertaut ke
    kunci mana pun. Untuk menghentikan sementara, nonaktifkan saja.
    """
    if not await store.hapus(key):
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_DITEMUKAN)
    audit.warning("Kunci sematan %s dihapus oleh %s", key, actor.email)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
