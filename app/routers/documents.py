"""Akses PDF sumber untuk verifikasi sitasi (FE-2)."""

from __future__ import annotations

import uuid
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy import text

from app.deps import SessionDep, StorageDep
from app.rag.filters import active_document_clause
from app.schemas.common import Error
from app.storage import ObjectNotFound

router = APIRouter(prefix="/api", tags=["dokumen"])

TIDAK_TERSEDIA = "Dokumen tidak ditemukan atau sudah tidak berlaku."

_BERKAS_AKTIF_SQL = text(
    "SELECT d.file_path, d.judul FROM documents d"
    f" WHERE d.id = :id AND {active_document_clause('d')}"
)


@router.get(
    "/documents/{document_id}/file",
    response_class=Response,
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "Berkas PDF (penyimpanan lokal)",
        },
        307: {"description": "Pengalihan ke object storage (S3 / R2)"},
        404: {"model": Error},
    },
)
async def get_document_file(
    document_id: uuid.UUID, session: SessionDep, storage: StorageDep
) -> Response:
    """Tujuan klik kartu sitasi. Hanya dokumen yang masih terambil retrieval.

    Filter yang dipakai sama dengan retriever (`is_active` DAN `valid_until`),
    supaya tautan pada percakapan lama tidak diam-diam menyajikan dokumen yang
    sudah kedaluwarsa.
    """
    row = (await session.execute(_BERKAS_AKTIF_SQL, {"id": document_id})).mappings().first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_TERSEDIA)

    url = storage.url_for(row["file_path"])
    if url is not None:
        # Presigned URL kedaluwarsa; peramban tidak boleh menyimpan pengalihannya.
        return RedirectResponse(
            url,
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Cache-Control": "no-store"},
        )

    try:
        isi = await storage.load(row["file_path"])
    except ObjectNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_TERSEDIA) from exc

    nama = quote(f"{row['judul']}.pdf")
    return Response(
        isi,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{nama}"},
    )
