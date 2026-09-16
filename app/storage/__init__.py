"""Penyimpanan objek untuk dokumen PDF sumber."""

from __future__ import annotations

from app.config import Settings
from app.storage.base import (
    DOCUMENT_PREFIX,
    ObjectNotFound,
    ObjectStorage,
    StorageError,
    document_key,
)
from app.storage.local import LocalStorage

__all__ = [
    "DOCUMENT_PREFIX",
    "LocalStorage",
    "ObjectNotFound",
    "ObjectStorage",
    "StorageError",
    "build_storage",
    "document_key",
]


def build_storage(settings: Settings) -> ObjectStorage:
    """Rakit backend penyimpanan sesuai `STORAGE_BACKEND`.

    `S3Storage` diimpor di dalam fungsi agar boto3 tidak perlu terpasang untuk
    menjalankan sistem dengan penyimpanan lokal -- termasuk saat unit test.
    """
    if settings.storage_backend == "local":
        return LocalStorage(settings.storage_dir)

    from app.storage.s3 import S3Storage

    # Validator di Settings sudah memastikan ketiganya terisi bila backend s3.
    return S3Storage(
        bucket=settings.s3_bucket,
        access_key_id=settings.s3_access_key_id.get_secret_value(),
        secret_access_key=settings.s3_secret_access_key.get_secret_value(),
        endpoint_url=settings.s3_endpoint_url,
        region=settings.s3_region,
        addressing_style=settings.s3_addressing_style,
        checksum_compat=settings.s3_checksum_compat,
        public_base_url=settings.s3_public_base_url,
        presign_ttl=settings.s3_presign_ttl_seconds,
    )
