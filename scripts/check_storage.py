"""Verifikasi kredensial penyimpanan objek terhadap bucket sungguhan.

Test otomatis memakai `moto`, yang meniru AWS S3 -- perbedaan perilaku
Cloudflare R2 tidak tertangkap di sana. Script ini melakukan satu putaran penuh
terhadap bucket yang benar-benar dikonfigurasi: tulis, baca, buat URL, hapus.

    python -m scripts.check_storage

Jalankan setelah mengisi `.env` dan sebelum menyatakan konfigurasi selesai.
Objek uji ditulis dengan awalan `_healthcheck/` dan dihapus kembali di akhir.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import uuid

from app.config import get_settings
from app.storage import build_storage

MUATAN = b"%PDF-1.7\ncek penyimpanan\n%%EOF"


def baris(label: str, nilai: object) -> None:
    print(f"  {label:<24} {nilai}")


async def jalankan() -> int:
    settings = get_settings()

    print("Konfigurasi")
    baris("backend", settings.storage_backend)
    if settings.storage_backend == "local":
        baris("direktori", settings.storage_dir)
        print("\nBackend lokal tidak butuh kredensial. Tidak ada yang perlu diverifikasi.")
        print("Untuk produksi, set STORAGE_BACKEND=s3 (lihat README bagian Penyimpanan).")
        return 0

    baris("bucket", settings.s3_bucket)
    baris("endpoint", settings.s3_endpoint_url or "(AWS S3 bawaan)")
    baris("region", settings.s3_region)
    baris("penyedia", "Cloudflare R2" if settings.is_r2 else "S3-compatible")
    baris("checksum_compat", settings.s3_checksum_compat)
    baris("url publik", settings.s3_public_base_url or "(presigned URL)")

    if settings.is_r2 and not settings.s3_checksum_compat:
        print(
            "\n  PERINGATAN: endpoint R2 dengan S3_CHECKSUM_COMPAT=false. "
            "Unggahan kemungkinan besar ditolak; lihat README."
        )

    storage = build_storage(settings)
    kunci = f"_healthcheck/{uuid.uuid4()}.pdf"
    print("\nUji putaran penuh")

    try:
        await storage.save(kunci, MUATAN, content_type="application/pdf")
        baris("tulis", "OK")

        kembali = await storage.load(kunci)
        if kembali != MUATAN:
            baris("baca", "GAGAL - isi tidak sama")
            return 1
        baris("baca", f"OK ({len(kembali)} byte)")

        baris("ada", await storage.exists(kunci))

        url = storage.url_for(kunci)
        baris("url", (url[:78] + "...") if url and len(url) > 78 else url)

        await storage.delete(kunci)
        baris("hapus", "OK" if not await storage.exists(kunci) else "GAGAL - masih ada")
    except Exception as exc:
        print(f"\nGAGAL: {type(exc).__name__}: {exc}")
        print(_petunjuk(settings, exc))
        # Usahakan tidak meninggalkan objek uji walau gagal di tengah.
        with contextlib.suppress(Exception):
            await storage.delete(kunci)
        return 1

    print("\nPenyimpanan objek siap dipakai.")
    return 0


def _petunjuk(settings, exc: Exception) -> str:
    pesan = str(exc)
    if "InvalidAccessKeyId" in pesan or "SignatureDoesNotMatch" in pesan:
        return (
            "\nPetunjuk: S3_ACCESS_KEY_ID atau S3_SECRET_ACCESS_KEY salah. Untuk R2, "
            "keduanya berasal dari R2 API token -- bukan dari API token Cloudflare biasa."
        )
    if "NoSuchBucket" in pesan:
        return (
            f"\nPetunjuk: bucket '{settings.s3_bucket}' tidak ada, "
            "atau token tidak dapat melihatnya."
        )
    if "AccessDenied" in pesan:
        return (
            "\nPetunjuk: token ada tetapi izinnya kurang. "
            "R2 butuh izin 'Object Read & Write'."
        )
    if "checksum" in pesan.lower() or "CRC" in pesan:
        return "\nPetunjuk: set S3_CHECKSUM_COMPAT=true. R2 menolak CRC32 FULL_OBJECT."
    if "EndpointConnectionError" in type(exc).__name__ or "Could not connect" in pesan:
        return f"\nPetunjuk: endpoint tidak terjangkau: {settings.s3_endpoint_url}"
    return ""


def main() -> int:
    return asyncio.run(jalankan())


if __name__ == "__main__":
    sys.exit(main())
