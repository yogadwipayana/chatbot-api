"""Penyimpanan objek lewat protokol S3.

Bekerja dengan AWS S3, **Cloudflare R2**, MinIO, dan Backblaze B2 -- semuanya
lewat kelas yang sama, dibedakan hanya oleh `endpoint_url` dan kredensial.

## Cloudflare R2

    S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
    S3_REGION=auto
    S3_ACCESS_KEY_ID=<dari R2 API token>
    S3_SECRET_ACCESS_KEY=<dari R2 API token>
    S3_BUCKET=<nama bucket>

Tiga hal yang membuat R2 berbeda dari S3, dan semuanya sudah ditangani di sini:

1. **Region wajib `auto`.** R2 tidak punya region; SDK tetap menuntut nilai.
   Nilai kosong dan `us-east-1` di-alias ke `auto`, tetapi tulis `auto` saja
   supaya tidak menyesatkan pembaca berikutnya.

2. **Checksum.** Sejak botocore 1.36 default `request_checksum_calculation`
   adalah `when_supported`, yang menempelkan checksum CRC32 pada setiap unggahan
   sebagai FULL_OBJECT. R2 hanya mendukung CRC32 sebagai COMPOSITE, sehingga
   unggahan bisa ditolak. `checksum_compat=True` (default) menurunkannya ke
   `when_required`. Aman juga untuk AWS S3 -- checksum tetap dikirim saat
   operasinya memang mensyaratkan.

3. **URL publik.** Endpoint `r2.cloudflarestorage.com` adalah endpoint API,
   bukan untuk peramban. Untuk menyajikan PDF ke mahasiswa (FE-2), pakai
   presigned URL (default) atau isi `S3_PUBLIC_BASE_URL` dengan domain kustom
   / URL pengembangan r2.dev bila bucket-nya memang publik.
"""

from __future__ import annotations

import functools
from typing import Any
from urllib.parse import quote

import anyio

from app.storage.base import ObjectNotFound, StorageError

DEFAULT_PRESIGN_TTL = 900
"""15 menit. Cukup untuk membuka PDF, cukup pendek supaya tautan yang bocor
lewat riwayat peramban atau grup WhatsApp tidak berlaku lama."""


class S3Storage:
    """Penyimpanan objek S3-compatible.

    Klien boto3 dibuat sekali dan dipakai ulang. Panggilan jaringan dijalankan
    di threadpool -- boto3 sinkron, dan memanggilnya langsung dari endpoint
    async akan membekukan event loop untuk seluruh permintaan lain.
    """

    def __init__(
        self,
        *,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        endpoint_url: str | None = None,
        region: str = "auto",
        addressing_style: str = "auto",
        checksum_compat: bool = True,
        public_base_url: str | None = None,
        presign_ttl: int = DEFAULT_PRESIGN_TTL,
    ) -> None:
        if not bucket:
            raise ValueError("nama bucket kosong")
        if not access_key_id or not secret_access_key:
            raise ValueError("kredensial S3 belum diisi")

        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None
        self.presign_ttl = presign_ttl
        self.client = self._build_client(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            endpoint_url=endpoint_url,
            region=region,
            addressing_style=addressing_style,
            checksum_compat=checksum_compat,
        )

    @staticmethod
    def _build_client(
        *,
        access_key_id: str,
        secret_access_key: str,
        endpoint_url: str | None,
        region: str,
        addressing_style: str,
        checksum_compat: bool,
    ) -> Any:
        import boto3
        from botocore.config import Config

        opsi: dict[str, Any] = {
            "signature_version": "s3v4",
            "s3": {"addressing_style": addressing_style},
            "retries": {"max_attempts": 3, "mode": "standard"},
        }
        if checksum_compat:
            # Lihat catatan (2) di docstring modul. Tanpa ini, unggahan ke R2
            # dapat ditolak karena checksum CRC32 FULL_OBJECT yang tidak didukung.
            opsi["request_checksum_calculation"] = "when_required"
            opsi["response_checksum_validation"] = "when_supported"

        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(**opsi),
        )

    # --- operasi ---------------------------------------------------------
    async def save(
        self, key: str, data: bytes, *, content_type: str = "application/pdf"
    ) -> str:
        await self._jalankan(
            functools.partial(
                self.client.put_object,
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        )
        return key

    async def load(self, key: str) -> bytes:
        respons = await self._jalankan(
            functools.partial(self.client.get_object, Bucket=self.bucket, Key=key),
            key=key,
        )
        return await anyio.to_thread.run_sync(respons["Body"].read)

    async def delete(self, key: str) -> None:
        # S3 tidak menganggap menghapus kunci yang tidak ada sebagai galat,
        # jadi perilakunya sudah sesuai kontrak `ObjectStorage`.
        await self._jalankan(
            functools.partial(self.client.delete_object, Bucket=self.bucket, Key=key)
        )

    async def exists(self, key: str) -> bool:
        try:
            await self._jalankan(
                functools.partial(self.client.head_object, Bucket=self.bucket, Key=key),
                key=key,
            )
        except ObjectNotFound:
            return False
        return True

    def url_for(self, key: str, *, expires_in: int | None = None) -> str | None:
        """URL yang dapat dibuka peramban.

        Bila `public_base_url` diisi, kembalikan URL publik apa adanya --
        pilihan yang tepat untuk bucket yang memang dipublikasikan lewat domain
        kustom, karena tautannya stabil dan dapat di-cache CDN.

        Bila tidak, bangkitkan presigned URL. Ini murni perhitungan tanda tangan
        di sisi klien: tidak ada permintaan jaringan, jadi aman dipanggil dari
        konteks async tanpa threadpool.
        """
        if self.public_base_url:
            return f"{self.public_base_url}/{quote(key)}"

        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in or self.presign_ttl,
            )
        except Exception as exc:  # pragma: no cover - jalur galat penyedia
            raise StorageError(f"gagal membuat presigned URL untuk {key}: {exc}") from exc

    # --- pembantu --------------------------------------------------------
    @staticmethod
    async def _jalankan(panggilan, *, key: str | None = None):
        """Jalankan panggilan boto3 di threadpool, terjemahkan galatnya.

        Kode 404/NoSuchKey/NotFound dipetakan ke `ObjectNotFound` supaya
        pemanggil tidak perlu mengimpor botocore, dan supaya galat "berkas
        tidak ada" tidak tertukar dengan "kredensial salah" -- keduanya
        muncul sebagai `ClientError` yang sama di boto3.
        """
        from botocore.exceptions import ClientError

        try:
            return await anyio.to_thread.run_sync(panggilan)
        except ClientError as exc:
            kode = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if kode in {"404", "NoSuchKey", "NotFound"} or status == 404:
                raise ObjectNotFound(f"objek tidak ditemukan: {key}") from exc
            raise StorageError(f"galat penyimpanan objek ({kode or status}): {exc}") from exc
