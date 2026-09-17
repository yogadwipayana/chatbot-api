"""Operasi S3 sungguhan terhadap server S3 tiruan (`moto`).

Melengkapi `test_storage.py` yang hanya memeriksa perakitan klien. Di sini
permintaan benar-benar dikirim dan diproses -- hanya saja lawan bicaranya
simulasi dalam proses, bukan jaringan.

Batasnya jujur: moto meniru AWS S3. Perbedaan perilaku Cloudflare R2 tidak
tertangkap di sini. Untuk memverifikasi bucket R2 sungguhan, jalankan
`python -m scripts.check_storage`.
"""

from __future__ import annotations

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from moto import mock_aws  # noqa: E402

from app.storage.base import ObjectNotFound, document_key  # noqa: E402
from app.storage.s3 import S3Storage  # noqa: E402

BUCKET = "dokumen-kampus"
PDF = b"%PDF-1.7\nPanduan Akademik 2025\n%%EOF"


@pytest.fixture
def storage(monkeypatch):
    """S3Storage terhubung ke S3 tiruan, dengan bucket yang sudah dibuat."""
    # moto menolak kredensial dari lingkungan nyata; paksa nilai dummy.
    for nama in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(nama, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield S3Storage(
            bucket=BUCKET,
            access_key_id="testing",
            secret_access_key="testing",
            region="us-east-1",
        )


class TestOperasiDasar:
    async def test_simpan_lalu_baca(self, storage):
        kunci = await storage.save(document_key("abc"), PDF)
        assert await storage.load(kunci) == PDF

    async def test_save_mengembalikan_kunci(self, storage):
        kunci = document_key("abc")
        assert await storage.save(kunci, PDF) == kunci

    async def test_content_type_tersimpan(self, storage):
        """Peramban hanya menampilkan PDF inline bila Content-Type benar;
        tanpa ini kartu sitasi FE-2 akan memicu unduhan, bukan tampilan."""
        kunci = await storage.save(document_key("abc"), PDF)
        objek = storage.client.get_object(Bucket=BUCKET, Key=kunci)
        assert objek["ContentType"] == "application/pdf"

    async def test_nama_file_tersimpan_sebagai_content_disposition(self, storage):
        kunci = await storage.save(
            document_key("nama-asli"),
            PDF,
            content_disposition='inline; filename="Panduan UKT.pdf"',
        )
        objek = storage.client.get_object(Bucket=BUCKET, Key=kunci)
        assert objek["ContentDisposition"] == 'inline; filename="Panduan UKT.pdf"'

    async def test_menimpa_isi_lama(self, storage):
        kunci = document_key("abc")
        await storage.save(kunci, PDF)
        await storage.save(kunci, b"%PDF-1.7 revisi")
        assert await storage.load(kunci) == b"%PDF-1.7 revisi"

    async def test_exists_benar(self, storage):
        kunci = document_key("abc")
        assert await storage.exists(kunci) is False
        await storage.save(kunci, PDF)
        assert await storage.exists(kunci) is True

    async def test_hapus(self, storage):
        kunci = await storage.save(document_key("abc"), PDF)
        await storage.delete(kunci)
        assert await storage.exists(kunci) is False

    async def test_hapus_kunci_tak_ada_tidak_melempar(self, storage):
        await storage.delete(document_key("tidak-pernah-ada"))

    async def test_beberapa_dokumen_tidak_saling_menimpa(self, storage):
        a = await storage.save(document_key("aaa"), b"%PDF A")
        b = await storage.save(document_key("bbb"), b"%PDF B")
        assert await storage.load(a) == b"%PDF A"
        assert await storage.load(b) == b"%PDF B"


class TestPemetaanGalat:
    async def test_kunci_tak_ada_menjadi_ObjectNotFound(self, storage):
        """Pemanggil tidak perlu mengimpor botocore untuk membedakan
        'berkas tidak ada' dari 'kredensial salah'."""
        with pytest.raises(ObjectNotFound):
            await storage.load(document_key("tidak-ada"))

    async def test_pesan_galat_menyebut_kunci(self, storage):
        with pytest.raises(ObjectNotFound, match="documents/hilang.pdf"):
            await storage.load("documents/hilang.pdf")

    async def test_exists_pada_kunci_tak_ada_tidak_melempar(self, storage):
        assert await storage.exists("documents/tidak-ada.pdf") is False


class TestIntegrasiPipeline:
    """Pipeline ingestion menyimpan berkas ke penyimpanan objek, bukan ke disk."""

    async def test_kunci_objek_deterministik_dari_document_id(self, storage):
        document_id = "3f1a2b3c-4d5e-6f70-8192-a3b4c5d6e7f8"
        kunci = await storage.save(document_key(document_id), PDF)
        assert kunci == f"documents/{document_id}.pdf"
        assert await storage.exists(document_key(document_id))

    async def test_pembersihan_setelah_kegagalan(self, storage):
        """Pipeline menghapus objek bila pencatatan ke DB gagal, supaya bucket
        tidak menumpuk objek yatim."""
        kunci = await storage.save(document_key("gagal"), PDF)
        await storage.delete(kunci)
        assert await storage.exists(kunci) is False
