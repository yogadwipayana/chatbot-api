"""Antarmuka penyimpanan objek.

Dokumen PDF sumber tidak lagi wajib berada di disk server. Lapisan tipis ini
memisahkan "di mana berkas disimpan" dari kode yang memakainya, sehingga
backend dapat diganti lewat konfigurasi tanpa menyentuh pipeline ingestion
maupun endpoint yang menyajikan PDF untuk verifikasi sitasi (FE-2).

Dua implementasi: `LocalStorage` (disk, default pengembangan dan test) dan
`S3Storage` (S3, Cloudflare R2, MinIO, Backblaze B2 -- apa pun yang bicara
protokol S3).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

DOCUMENT_PREFIX = "documents"
"""Awalan kunci objek untuk PDF sumber."""


class StorageError(RuntimeError):
    """Kegagalan penyimpanan yang bukan salah pemanggil."""


class ObjectNotFound(StorageError):
    """Kunci tidak ada di penyimpanan."""


def document_key(document_id: str, *, prefix: str = DOCUMENT_PREFIX) -> str:
    """Kunci objek untuk satu dokumen.

    Deterministik dari `documents.id`, sehingga berkas dapat ditemukan kembali
    tanpa menyimpan lintasan penuh -- dan berpindah backend penyimpanan tidak
    memaksa migrasi kolom `file_path`.

    Nama asli berkas sengaja tidak dipakai sebagai kunci: judul dokumen kampus
    kerap mengandung spasi, tanda kurung, dan huruf beraksen yang menghasilkan
    kunci objek bermasalah di sebagian penyedia.
    """
    document_id = document_id.strip()
    if not document_id:
        raise ValueError("document_id kosong")
    if "/" in document_id or "\\" in document_id:
        raise ValueError(
            f"document_id tidak boleh mengandung pemisah lintasan: {document_id!r}"
        )
    prefix = prefix.strip("/")
    return f"{prefix}/{document_id}.pdf" if prefix else f"{document_id}.pdf"


@runtime_checkable
class ObjectStorage(Protocol):
    """Kontrak minimum yang dipakai aplikasi."""

    async def save(
        self, key: str, data: bytes, *, content_type: str = "application/pdf"
    ) -> str:
        """Simpan objek. Menimpa bila kunci sudah ada. Return: kunci."""
        ...

    async def load(self, key: str) -> bytes:
        """Baca objek utuh.

        Raises:
            ObjectNotFound: kunci tidak ada.
        """
        ...

    async def delete(self, key: str) -> None:
        """Hapus objek. Tidak melempar bila kunci memang sudah tidak ada."""
        ...

    async def exists(self, key: str) -> bool: ...

    def url_for(self, key: str, *, expires_in: int | None = None) -> str | None:
        """URL yang dapat dibuka peramban secara langsung, atau None.

        None berarti backend tidak bisa melayani peramban sendiri (mis. disk
        lokal) sehingga API harus mengalirkan isinya. Bila mengembalikan URL,
        endpoint FE-2 cukup mengarahkan peramban ke sana -- PDF panduan
        akademik bisa puluhan megabita, dan menyalurkannya lewat proses API
        hanya menghabiskan memori dan koneksi tanpa manfaat.
        """
        ...
