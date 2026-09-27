"""Pemeriksaan kunci sematan untuk portal (`client/src/proxy.ts`)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.deps import KUNCI_TIDAK_BERLAKU, EmbedKeyStoreDep
from app.schemas.chat import EmbedKeyInfo
from app.schemas.common import Error

router = APIRouter(prefix="/api/embed", tags=["embed"])


@router.get(
    "/keys/{key}", response_model=EmbedKeyInfo, responses={404: {"model": Error}}
)
async def get_embed_key(key: str, store: EmbedKeyStoreDep) -> EmbedKeyInfo:
    """Situs yang boleh memuat panel untuk kunci ini, atau 404 bila tidak berlaku.

    Dipanggil server portal setiap kali `/embed` dimuat, bukan oleh peramban:
    hasilnya menjadi header `frame-ancestors`, jadi perubahan dari dashboard
    berlaku saat panel dibuka berikutnya tanpa build ulang portal.

    Publik, karena isinya memang tidak rahasia -- kunci tertulis di kode sumber
    situs penyemat, dan daftar situsnya dapat dilihat siapa pun yang membuka
    situs-situs itu. Tidak tunduk pada kill switch: panelnya tetap dimuat dan
    menyampaikan pesan penutupan saat mahasiswa bertanya.
    """
    info = await store.aktif(key)
    if info is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, KUNCI_TIDAK_BERLAKU)
    return EmbedKeyInfo(allowed_origins=info.allowed_origins)
