"""Kelola dokumen sumber (AD-2, AD-3).

Semua level boleh masuk. Staf/dosen dibatasi pada dokumen unitnya sendiri:
daftar hanya berisi unitnya, dan membuka, mengubah, menghapus, atau mengunggah
dokumen unit lain ditolak 403. Admin dan superadmin mengelola semua unit.
"""

from __future__ import annotations

import logging
import re
import tempfile
import uuid
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)

from app.admin import documents as repo
from app.admin.permissions import CurrentAdmin
from app.deps import (
    CurrentAdminDep,
    SessionDep,
    SettingsDep,
    StorageDep,
    get_embeddings,
    require_admin,
)
from app.ingestion.embedder import EmbeddingDimensionError
from app.ingestion.loader import ScannedPdfError, UnreadablePdfError
from app.ingestion.pipeline import EmptyDocumentError, ingest_document
from app.schemas.admin import Chunk, Document, DocumentPage, DocumentUpdate, IngestionResult
from app.schemas.common import Error
from app.storage import StorageError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin/documents",
    tags=["admin-dokumen"],
    dependencies=[Depends(require_admin)],
    responses={401: {"model": Error}, 403: {"model": Error}},
)

TIDAK_DITEMUKAN = "Dokumen tidak ditemukan."
PDF_MAGIC = b"%PDF-"
BACA_PER = 1024 * 1024

Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0)]


def pastikan_unit(admin: CurrentAdmin, unit: str | None) -> None:
    """403 bila staf/dosen menyentuh dokumen unit lain."""
    if not admin.can_manage_unit(unit):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"Dokumen ini milik unit lain. Akun Anda hanya dapat mengelola dokumen "
            f"unit {admin.unit}.",
        )


async def _dokumen_milik(
    session, admin: CurrentAdmin, document_id: uuid.UUID
) -> dict[str, Any]:
    row = await repo.get_document(session, document_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_DITEMUKAN)
    pastikan_unit(admin, row["unit"])
    return row


@router.get("", response_model=DocumentPage)
async def list_documents(
    admin: CurrentAdminDep,
    session: SessionDep,
    include_inactive: bool = False,
    only_stale: bool = Query(False, description="Saring hanya dokumen yang perlu ditinjau."),
    limit: Limit = 50,
    offset: Offset = 0,
) -> DocumentPage:
    """AD-2. Dokumen yang perlu ditinjau diurutkan lebih dulu, bukan disembunyikan."""
    rows, total, jumlah_stale = await repo.list_documents(
        session,
        include_inactive=include_inactive,
        only_stale=only_stale,
        limit=limit,
        offset=offset,
        unit=admin.unit_scope,
    )
    return DocumentPage(
        items=[Document.model_validate(r) for r in rows],
        total=total,
        jumlah_stale=jumlah_stale,
    )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=IngestionResult,
    responses={413: {"model": Error}, 422: {"model": Error}, 502: {"model": Error}},
)
async def upload_document(
    admin: CurrentAdminDep,
    settings: SettingsDep,
    session: SessionDep,
    storage: StorageDep,
    file: Annotated[UploadFile, File(description="Berkas PDF.")],
    judul: Annotated[str, Form(min_length=3, max_length=500)],
    unit: Annotated[str, Form(min_length=2, max_length=200)],
    tahun_berlaku: Annotated[int | None, Form(ge=2000, le=2100)] = None,
    valid_until: Annotated[date | None, Form()] = None,
    embeddings: Any = Depends(get_embeddings),
) -> IngestionResult:
    """AD-3. Sinkron: unggah, ekstraksi, chunking, embedding, simpan.

    Setiap galat yang bisa diperbaiki admin dikembalikan sebagai kalimat
    non-teknis (PRD §9). Galat layanan luar (penyimpanan, API AI) menjadi 502
    dengan saran mencoba lagi; rinciannya hanya masuk log server.
    """
    judul, unit = judul.strip(), unit.strip()
    if not admin.can_manage_unit(unit):
        # Diperiksa sebelum berkas dibaca: tidak ada gunanya memproses PDF
        # puluhan megabita yang toh akan ditolak.
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"Akun Anda hanya dapat mengunggah dokumen untuk unit {admin.unit}.",
        )
    if len(judul) < 3:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Judul minimal 3 karakter.")
    if len(unit) < 2:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Nama unit minimal 2 karakter."
        )

    batas = settings.max_upload_mb * 1024 * 1024
    nama = nama_berkas_aman(file.filename)

    with tempfile.TemporaryDirectory(prefix="unggah-") as folder:
        path = Path(folder) / nama
        ukuran = 0
        with path.open("wb") as keluaran:
            while potongan := await file.read(BACA_PER):
                ukuran += len(potongan)
                if ukuran > batas:
                    raise HTTPException(
                        status.HTTP_413_CONTENT_TOO_LARGE,
                        f"Berkas terlalu besar. Ukuran maksimum {settings.max_upload_mb} MB.",
                    )
                keluaran.write(potongan)

        if ukuran == 0:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"'{nama}' kosong.")
        with path.open("rb") as masukan:
            if not masukan.read(1024).lstrip().startswith(PDF_MAGIC):
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    f"'{nama}' bukan berkas PDF. Unggah dokumen dalam format PDF.",
                )

        try:
            hasil = await ingest_document(
                session,
                path=path,
                judul=judul,
                unit=unit,
                embeddings=embeddings,
                storage=storage,
                tahun_berlaku=tahun_berlaku,
                valid_until=valid_until,
                uploaded_by=admin.email,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        except (ScannedPdfError, UnreadablePdfError, EmptyDocumentError) as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        except StorageError as exc:
            logger.exception("Gagal menyimpan berkas dokumen '%s'", nama)
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                "Berkas tidak dapat disimpan ke penyimpanan dokumen. "
                "Coba lagi beberapa saat lagi.",
            ) from exc
        except EmbeddingDimensionError as exc:
            logger.error("Ingestion '%s' gagal: %s", nama, exc)
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                "Dokumen tidak dapat diproses karena pengaturan model AI tidak sesuai. "
                "Hubungi pengelola teknis.",
            ) from exc
        except Exception as exc:
            if not _galat_layanan_ai(exc):
                raise
            logger.exception("Layanan embedding gagal saat memproses '%s'", nama)
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                "Layanan AI untuk memproses dokumen sedang tidak dapat dihubungi. "
                "Coba lagi beberapa saat lagi.",
            ) from exc

    return IngestionResult(
        document_id=str(hasil.document_id),
        jumlah_halaman=hasil.jumlah_halaman,
        jumlah_chunk=hasil.jumlah_chunk,
    )


@router.get("/{document_id}", response_model=Document, responses={404: {"model": Error}})
async def get_document(
    document_id: uuid.UUID, admin: CurrentAdminDep, session: SessionDep
) -> Document:
    return Document.model_validate(await _dokumen_milik(session, admin, document_id))


@router.patch("/{document_id}", response_model=Document, responses={404: {"model": Error}})
async def update_document(
    document_id: uuid.UUID,
    payload: DocumentUpdate,
    admin: CurrentAdminDep,
    session: SessionDep,
) -> Document:
    """Menonaktifkan dokumen TIDAK menghapus chunk-nya, sehingga dapat dikembalikan."""
    await _dokumen_milik(session, admin, document_id)
    changes = payload.model_dump(exclude_unset=True)
    if "unit" in changes:
        # Staf juga tidak boleh "memindahkan" dokumennya ke unit lain.
        pastikan_unit(admin, changes["unit"])
    row = await repo.update_document(session, document_id, changes)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_DITEMUKAN)
    return Document.model_validate(row)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={404: {"model": Error}},
)
async def delete_document(
    document_id: uuid.UUID,
    admin: CurrentAdminDep,
    session: SessionDep,
    storage: StorageDep,
) -> Response:
    """Permanen. Baris database dihapus lebih dulu, berkasnya menyusul.

    Urutan yang sama dengan alasan yang sama seperti ingestion: berkas yatim di
    penyimpanan hanya memakan tempat, sedangkan baris yang menunjuk berkas yang
    sudah hilang membuat kartu sitasi mahasiswa rusak.
    """
    await _dokumen_milik(session, admin, document_id)
    key = await repo.delete_document(session, document_id)
    if key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, TIDAK_DITEMUKAN)
    try:
        await storage.delete(key)
    except StorageError:
        logger.exception(
            "Dokumen %s terhapus dari database, tetapi berkas '%s' gagal dihapus",
            document_id,
            key,
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{document_id}/chunks", response_model=list[Chunk], responses={404: {"model": Error}}
)
async def list_document_chunks(
    document_id: uuid.UUID,
    admin: CurrentAdminDep,
    session: SessionDep,
    limit: Limit = 50,
    offset: Offset = 0,
) -> list[Chunk]:
    await _dokumen_milik(session, admin, document_id)
    rows = await repo.list_chunks(session, document_id, limit=limit, offset=offset)
    return [Chunk.model_validate(r) for r in rows]


def nama_berkas_aman(nama: str | None) -> str:
    """Nama berkas unggahan, tanpa unsur lintasan dan karakter terlarang.

    Dipakai hanya untuk berkas sementara dan kalimat galat
    (`'Panduan Akademik 2025.pdf' tampaknya hasil scan...`) -- bukan sebagai
    kunci penyimpanan, yang selalu diturunkan dari `document_id`.
    """
    dasar = Path((nama or "").replace("\\", "/")).name
    dasar = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", dasar).strip(" .")
    return dasar[:150] or "dokumen.pdf"


def _galat_layanan_ai(exc: BaseException) -> bool:
    try:
        import openai
    except ModuleNotFoundError:  # pragma: no cover - langchain-openai selalu membawanya
        return False
    return isinstance(exc, openai.APIError)
