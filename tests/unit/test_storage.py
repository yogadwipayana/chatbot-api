"""Penyimpanan objek -- disk lokal dan S3/R2.

Test S3 memakai `moto`, yang mensimulasikan API S3 di dalam proses. Tidak ada
permintaan jaringan dan tidak ada kredensial sungguhan yang dibutuhkan.

Yang TIDAK dapat dibuktikan di sini: apakah R2 sungguhan menerima permintaan
kita. moto meniru AWS S3, bukan R2, sehingga perbedaan perilaku R2 -- terutama
soal checksum -- tidak akan tertangkap. Karena itu ada `TestKonfigurasiR2` yang
memeriksa klien dirakit dengan opsi yang benar, dan `scripts/check_storage.py`
untuk verifikasi terhadap bucket sungguhan.
"""

from __future__ import annotations

import pytest

from app.storage.base import ObjectNotFound, StorageError, content_disposition, document_key
from app.storage.local import LocalStorage

PDF = b"%PDF-1.7\nisi dokumen panduan akademik\n%%EOF"


class TestKunciObjek:
    def test_bentuk_kunci(self):
        assert document_key("3f1a2b3c") == "documents/3f1a2b3c.pdf"

    def test_awalan_dapat_diganti(self):
        assert document_key("abc", prefix="arsip/2025") == "arsip/2025/abc.pdf"

    def test_awalan_kosong(self):
        assert document_key("abc", prefix="") == "abc.pdf"

    def test_garis_miring_berlebih_dirapikan(self):
        assert document_key("abc", prefix="/dokumen/") == "dokumen/abc.pdf"

    def test_deterministik(self):
        assert document_key("abc") == document_key("abc")

    def test_id_kosong_ditolak(self):
        with pytest.raises(ValueError, match="kosong"):
            document_key("   ")

    @pytest.mark.parametrize("jahat", ["../rahasia", "a/b", "a\\b"])
    def test_id_dengan_pemisah_lintasan_ditolak(self, jahat):
        """Kunci ikut membentuk lintasan berkas dan URL; jangan biarkan lolos."""
        with pytest.raises(ValueError, match="pemisah lintasan"):
            document_key(jahat)


class TestNamaBerkas:
    def test_content_disposition_mempertahankan_nama_utf8(self):
        header = content_disposition("Panduan Keuangan (2026).pdf")
        assert header.startswith('inline; filename="Panduan Keuangan (2026).pdf";')
        assert "filename*=UTF-8''Panduan%20Keuangan%20%282026%29.pdf" in header

    def test_content_disposition_menolak_pemisah_dan_header_injection(self):
        header = content_disposition("../rahasia\r\nX-Leak: true")
        assert ".._rahasiaX-Leak: true" in header
        assert "\r" not in header and "\n" not in header


class TestLocalStorage:
    @pytest.fixture
    def storage(self, tmp_path) -> LocalStorage:
        return LocalStorage(tmp_path / "dokumen")

    async def test_simpan_lalu_baca(self, storage):
        kunci = await storage.save(document_key("abc"), PDF)
        assert await storage.load(kunci) == PDF

    async def test_direktori_akar_dibuat_otomatis(self, tmp_path):
        akar = tmp_path / "belum" / "ada"
        LocalStorage(akar)
        assert akar.is_dir()

    async def test_subdirektori_kunci_dibuat_otomatis(self, storage):
        kunci = await storage.save("arsip/2025/abc.pdf", PDF)
        assert await storage.exists(kunci)

    async def test_menimpa_isi_lama(self, storage):
        kunci = document_key("abc")
        await storage.save(kunci, PDF)
        await storage.save(kunci, b"%PDF-1.7 versi baru")
        assert await storage.load(kunci) == b"%PDF-1.7 versi baru"

    async def test_tidak_meninggalkan_berkas_sementara(self, storage, tmp_path):
        """Penulisan lewat berkas .part lalu rename; sisa .part berarti bocor."""
        await storage.save(document_key("abc"), PDF)
        assert list((tmp_path / "dokumen").rglob("*.part")) == []

    async def test_membaca_kunci_tak_ada(self, storage):
        with pytest.raises(ObjectNotFound):
            await storage.load(document_key("tidak-ada"))

    async def test_exists(self, storage):
        assert await storage.exists(document_key("abc")) is False
        await storage.save(document_key("abc"), PDF)
        assert await storage.exists(document_key("abc")) is True

    async def test_hapus(self, storage):
        kunci = await storage.save(document_key("abc"), PDF)
        await storage.delete(kunci)
        assert await storage.exists(kunci) is False

    async def test_hapus_kunci_tak_ada_tidak_melempar(self, storage):
        """Kontrak `ObjectStorage`: menghapus yang sudah tidak ada bukan galat."""
        await storage.delete(document_key("tidak-pernah-ada"))

    def test_url_selalu_none(self, storage):
        """Disk lokal tidak dapat diakses peramban; API harus mengalirkannya."""
        assert storage.url_for(document_key("abc")) is None

    @pytest.mark.parametrize("jahat", ["../../etc/passwd", "../keluar.pdf"])
    async def test_kunci_keluar_dari_akar_ditolak(self, storage, jahat):
        with pytest.raises(StorageError, match="keluar dari direktori"):
            await storage.save(jahat, PDF)


class TestKonfigurasiR2:
    """Merakit klien tanpa memanggil jaringan, lalu memeriksa opsinya.

    Inilah yang menangkap tiga kesalahan konfigurasi R2 paling umum: region
    salah, endpoint tidak dipakai, dan checksum bawaan botocore yang ditolak R2.
    """

    @pytest.fixture
    def r2(self):
        from app.storage.s3 import S3Storage

        return S3Storage(
            bucket="dokumen-kampus",
            access_key_id="akses",
            secret_access_key="rahasia",
            endpoint_url="https://akun123.r2.cloudflarestorage.com",
            region="auto",
        )

    def test_endpoint_dipakai(self, r2):
        assert "r2.cloudflarestorage.com" in r2.client.meta.endpoint_url

    def test_region_auto(self, r2):
        assert r2.client.meta.region_name == "auto"

    def test_signature_v4(self, r2):
        assert r2.client.meta.config.signature_version == "s3v4"

    def test_checksum_compat_menurunkan_ke_when_required(self, r2):
        """Default botocore `when_supported` menempelkan CRC32 FULL_OBJECT pada
        setiap unggahan. R2 hanya mendukung CRC32 sebagai COMPOSITE, sehingga
        default itu dapat membuat unggahan ditolak."""
        assert r2.client.meta.config.request_checksum_calculation == "when_required"

    def test_checksum_compat_dimatikan_mengembalikan_default_botocore(self):
        """Mematikan mode kompatibilitas mengembalikan perilaku bawaan botocore.

        Nilai `when_supported` inilah yang bermasalah dengan R2 -- test ini
        merekamnya secara eksplisit, sehingga kalau botocore suatu saat
        mengubah defaultnya, kita tahu dari sini.
        """
        from app.storage.s3 import S3Storage

        s3 = S3Storage(
            bucket="b",
            access_key_id="a",
            secret_access_key="s",
            region="ap-southeast-1",
            checksum_compat=False,
        )
        assert s3.client.meta.config.request_checksum_calculation == "when_supported"

    def test_aws_s3_asli_tanpa_endpoint_kustom(self):
        from app.storage.s3 import S3Storage

        s3 = S3Storage(
            bucket="b", access_key_id="a", secret_access_key="s", region="ap-southeast-1"
        )
        assert "amazonaws.com" in s3.client.meta.endpoint_url

    def test_addressing_style_path_untuk_minio(self):
        from app.storage.s3 import S3Storage

        minio = S3Storage(
            bucket="b",
            access_key_id="a",
            secret_access_key="s",
            endpoint_url="http://localhost:9000",
            addressing_style="path",
        )
        assert minio.client.meta.config.s3["addressing_style"] == "path"

    def test_bucket_kosong_ditolak(self):
        from app.storage.s3 import S3Storage

        with pytest.raises(ValueError, match="bucket"):
            S3Storage(bucket="", access_key_id="a", secret_access_key="s")

    @pytest.mark.parametrize("kunci,rahasia", [("", "s"), ("a", "")])
    def test_kredensial_kosong_ditolak(self, kunci, rahasia):
        from app.storage.s3 import S3Storage

        with pytest.raises(ValueError, match="[Kk]redensial"):
            S3Storage(bucket="b", access_key_id=kunci, secret_access_key=rahasia)


class TestUrlPublikDanPresigned:
    def test_presigned_url_dibangkitkan(self):
        from app.storage.s3 import S3Storage

        s3 = S3Storage(
            bucket="dokumen",
            access_key_id="a",
            secret_access_key="s",
            endpoint_url="https://akun123.r2.cloudflarestorage.com",
        )
        url = s3.url_for("documents/abc.pdf")
        assert url.startswith("https://")
        assert "X-Amz-Signature" in url
        assert "X-Amz-Expires" in url

    def test_masa_berlaku_presigned_dihormati(self):
        from app.storage.s3 import S3Storage

        s3 = S3Storage(
            bucket="dokumen", access_key_id="a", secret_access_key="s", presign_ttl=60
        )
        assert "X-Amz-Expires=60" in s3.url_for("documents/abc.pdf")

    def test_masa_berlaku_dapat_ditimpa_per_panggilan(self):
        from app.storage.s3 import S3Storage

        s3 = S3Storage(bucket="d", access_key_id="a", secret_access_key="s")
        assert "X-Amz-Expires=120" in s3.url_for("k.pdf", expires_in=120)

    def test_base_url_publik_mengalahkan_presigned(self):
        """Domain kustom menghasilkan tautan stabil yang dapat di-cache CDN;
        presigned URL berubah setiap kali dan tidak dapat di-cache."""
        from app.storage.s3 import S3Storage

        s3 = S3Storage(
            bucket="d",
            access_key_id="a",
            secret_access_key="s",
            public_base_url="https://dokumen.kampus.ac.id",
        )
        assert s3.url_for("documents/abc.pdf") == (
            "https://dokumen.kampus.ac.id/documents/abc.pdf"
        )

    def test_garis_miring_akhir_pada_base_url_tidak_menggandakan(self):
        from app.storage.s3 import S3Storage

        s3 = S3Storage(
            bucket="d",
            access_key_id="a",
            secret_access_key="s",
            public_base_url="https://dokumen.kampus.ac.id/",
        )
        assert "//documents" not in s3.url_for("documents/abc.pdf")

    def test_kunci_dengan_karakter_khusus_di_url_encode(self):
        from app.storage.s3 import S3Storage

        s3 = S3Storage(
            bucket="d",
            access_key_id="a",
            secret_access_key="s",
            public_base_url="https://dokumen.kampus.ac.id",
        )
        assert " " not in s3.url_for("documents/ada spasi.pdf")
