"""`GET /api/documents/{id}/file` (FE-2) dengan penyimpanan disk lokal.

Endpoint ini publik dan tanpa batas laju, sedangkan PDF panduan bisa puluhan
megabita: isinya harus dialirkan dari disk (`FileResponse`), bukan dimuat utuh
ke memori setiap permintaan (H36). Penyimpanan S3 tidak lewat sini -- ia
mengalihkan ke presigned URL (`tests/unit/test_storage.py`).
"""

from __future__ import annotations

import uuid

import pytest

from app.deps import get_storage
from app.storage import document_key
from app.storage.local import LocalStorage

PDF = b"%PDF-1.7\nisi dokumen panduan akademik\n%%EOF"
ID_ADA = uuid.UUID("3f6c7a1e-2b4d-4c8e-9a0f-1d2e3c4b5a69")
ID_HILANG = uuid.UUID("9e8d7c6b-5a49-4382-b1c0-ffeeddccbbaa")


class _Hasil:
    def __init__(self, row):
        self._row = row

    def mappings(self):
        return self

    def first(self):
        return self._row


class _Sesi:
    """Satu baris `documents` aktif untuk `ID_ADA`; `ID_HILANG` menunjuk kunci
    yang tidak pernah disimpan; id lain tidak ada."""

    async def execute(self, _sql, params):
        if params["id"] == ID_ADA:
            return _Hasil(
                {
                    "file_path": document_key(str(ID_ADA)),
                    "original_filename": "Panduan Akademik 2025.pdf",
                    "title": "Panduan Akademik 2025",
                }
            )
        if params["id"] == ID_HILANG:
            return _Hasil(
                {
                    "file_path": document_key(str(ID_HILANG)),
                    "original_filename": None,
                    "title": "Hilang",
                }
            )
        return _Hasil(None)


@pytest.fixture
async def storage(tmp_path) -> LocalStorage:
    storage = LocalStorage(tmp_path / "dokumen")
    await storage.save(document_key(str(ID_ADA)), PDF)
    return storage


@pytest.fixture
def client(make_client, storage):
    client = make_client([], session=_Sesi())
    client.app.dependency_overrides[get_storage] = lambda: storage
    return client


def test_pdf_dialirkan_dari_disk_tanpa_load(client, monkeypatch):
    async def jangan_dipanggil(self, key):
        raise AssertionError("load() memuat seluruh PDF ke memori; pakai path_for()")

    monkeypatch.setattr(LocalStorage, "load", jangan_dipanggil)
    r = client.get(f"/api/documents/{ID_ADA}/file")
    assert r.status_code == 200
    assert r.content == PDF
    assert r.headers["content-type"] == "application/pdf"
    assert r.headers["content-length"] == str(len(PDF))
    assert r.headers["content-disposition"].startswith('inline; filename="Panduan Akademik')
    # Header keamanan H23 tetap terpasang pada balasan berkas.
    assert r.headers["x-content-type-options"] == "nosniff"


def test_berkas_hilang_404_bukan_500(client):
    r = client.get(f"/api/documents/{ID_HILANG}/file")
    assert r.status_code == 404


def test_dokumen_tak_dikenal_404(client):
    assert client.get(f"/api/documents/{uuid.uuid4()}/file").status_code == 404


def test_penyimpanan_tanpa_path_for_tetap_dilayani(make_client, storage):
    """Backend lain tanpa `path_for` memakai `load()` seperti sebelumnya."""

    class TanpaPathFor:
        def url_for(self, key, **_):
            return None

        async def load(self, key):
            return await storage.load(key)

    client = make_client([], session=_Sesi())
    client.app.dependency_overrides[get_storage] = lambda: TanpaPathFor()
    r = client.get(f"/api/documents/{ID_ADA}/file")
    assert r.status_code == 200
    assert r.content == PDF
