"""Penyimpanan di disk server.

Default untuk pengembangan dan test: tidak butuh kredensial, tidak butuh
jaringan. Untuk produksi lihat `S3Storage` -- penyimpanan lokal berarti berkas
ikut hilang bila kontainer diganti, dan tidak bisa dibagi antar instans.
"""

from __future__ import annotations

from pathlib import Path

import anyio

from app.storage.base import ObjectNotFound, StorageError


class LocalStorage:
    """Menyimpan objek sebagai berkas di bawah satu direktori akar."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        """Ubah kunci menjadi lintasan, menolak yang keluar dari akar.

        Kunci berasal dari `document_key()` sehingga semestinya aman, tetapi
        pemeriksaan ini tetap ada: satu jalur pemanggilan baru yang meneruskan
        kunci dari input pengguna akan langsung menjadi path traversal.
        """
        calon = (self.root / key).resolve()
        if not calon.is_relative_to(self.root):
            raise StorageError(f"kunci keluar dari direktori penyimpanan: {key!r}")
        return calon

    async def save(
        self, key: str, data: bytes, *, content_type: str = "application/pdf"
    ) -> str:
        path = self._path(key)
        await anyio.to_thread.run_sync(lambda: path.parent.mkdir(parents=True, exist_ok=True))
        # Tulis ke berkas sementara lalu ganti nama: pembaca tidak pernah
        # melihat berkas setengah tertulis bila proses mati di tengah jalan.
        sementara = path.with_suffix(path.suffix + ".part")

        def tulis() -> None:
            sementara.write_bytes(data)
            sementara.replace(path)

        await anyio.to_thread.run_sync(tulis)
        return key

    async def load(self, key: str) -> bytes:
        path = self._path(key)
        try:
            return await anyio.to_thread.run_sync(path.read_bytes)
        except FileNotFoundError as exc:
            raise ObjectNotFound(f"objek tidak ditemukan: {key}") from exc

    async def delete(self, key: str) -> None:
        path = self._path(key)
        await anyio.to_thread.run_sync(lambda: path.unlink(missing_ok=True))

    async def exists(self, key: str) -> bool:
        return await anyio.to_thread.run_sync(self._path(key).is_file)

    def url_for(self, key: str, *, expires_in: int | None = None) -> str | None:
        """Selalu None -- disk lokal tidak dapat diakses peramban secara langsung.

        Endpoint FE-2 harus mengalirkan isinya sendiri.
        """
        return None
