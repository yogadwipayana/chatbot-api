"""Parameter `.env` yang dapat disetel dari dashboard.

Menyetel ambang penolakan (FR-3) dan ukuran potongan (FR-1) adalah pekerjaan
berulang: uji coba di AD-6, geser sedikit, uji lagi. Selama nilainya hanya ada
di `.env`, setiap putaran menuntut akses server dan restart -- yang berarti
pemilik sistem tidak dapat melakukannya sendiri.

Yang disimpan di database hanyalah nilai yang benar-benar ditimpa. Tidak ada
baris berarti "ikut `.env`", sehingga:

- mengubah `.env` tetap berlaku untuk parameter yang belum pernah disentuh,
- "kembalikan ke nilai .env" cukup menghapus barisnya, bukan menebak nilai awal.

Nilai disimpan sebagai teks persis seperti di `.env` dan di-parse ulang oleh
`Settings`, jadi validasinya (termasuk aturan antar-field seperti
`chunk_overlap < chunk_size`) hanya ditulis satu kali.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings

log = logging.getLogger(__name__)

DAPAT_DIUBAH: tuple[str, ...] = (
    "retrieval_candidates",
    "retrieval_top_n",
    "rrf_k",
    "rrf_weight_vector",
    "rrf_weight_fulltext",
    "vector_threshold",
    "lexical_threshold",
    "chunk_size",
    "chunk_overlap",
)
"""Field `Settings` yang boleh ditimpa dari dashboard.

Sengaja hanya parameter retrieval dan chunking: keduanya disetel dengan
mencoba, dan kekeliruan paling jauh membuat jawaban lebih sering ditolak --
dapat dikembalikan dalam satu klik.

Di luar daftar ini tetap lewat `.env` + restart. Nama model (`CHAT_MODEL`,
`EMBED_MODEL`) bukan sekadar angka: mengganti model embedding menuntut
re-index seluruh dokumen (`app.db.models.EMBEDDING_DIM`), jadi ia bukan
setelan yang boleh diubah sambil layanan berjalan. Kredensial, CORS, dan
batas unggah adalah urusan pengelola server, bukan admin konten.

Batas atas-bawah tiap field ada di `app.schemas.admin.RuntimeConfigUpdate`;
`tests/unit/test_config.py` menjaga kedua daftar tetap sama.
"""

BERLAKU_SETELAH_INGEST_ULANG = frozenset({"chunk_size", "chunk_overlap"})
"""Perubahan di sini hanya mengenai dokumen yang diproses SETELAHNYA.

Dokumen yang sudah terlanjur dipecah tidak ikut berubah sampai diunggah
ulang. Dashboard menyebutkan ini supaya admin tidak menunggu perubahan yang
tidak akan datang."""


@dataclass(frozen=True)
class NilaiTersimpan:
    value: str
    updated_at: datetime
    updated_by: str | None


class RuntimeConfigStore(Protocol):
    async def load(self) -> dict[str, NilaiTersimpan]: ...

    async def replace(self, changes: Mapping[str, str | None], *, by: str) -> None: ...

    async def clear(self) -> None: ...


class SqlRuntimeConfigStore:
    """Tabel `runtime_config`. Dibaca sekali per permintaan yang membutuhkannya.

    Tidak di-cache di memori proses dengan sengaja: dashboard dan layanan chat
    dapat berjalan di beberapa worker, dan setelan yang baru berlaku "entah
    kapan" di sebagian worker adalah bug yang mahal untuk didiagnosis. Tabelnya
    paling banyak berisi sembilan baris dengan primary key, jadi bacaannya
    jauh lebih murah daripada satu langkah retrieval.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def load(self) -> dict[str, NilaiTersimpan]:
        rows = await self.session.execute(
            text("SELECT key, value, updated_at, updated_by FROM runtime_config")
        )
        return {
            r["key"]: NilaiTersimpan(r["value"], r["updated_at"], r["updated_by"])
            for r in rows.mappings()
        }

    async def replace(self, changes: Mapping[str, str | None], *, by: str) -> None:
        """Simpan nilai baru; `None` menghapus baris (kembali mengikuti `.env`)."""
        for key, value in changes.items():
            if value is None:
                await self.session.execute(
                    text("DELETE FROM runtime_config WHERE key = :key"), {"key": key}
                )
                continue
            await self.session.execute(
                text(
                    "INSERT INTO runtime_config (key, value, updated_at, updated_by)"
                    " VALUES (:key, :value, now(), :by)"
                    " ON CONFLICT (key) DO UPDATE SET"
                    " value = EXCLUDED.value,"
                    " updated_at = EXCLUDED.updated_at,"
                    " updated_by = EXCLUDED.updated_by"
                ),
                {"key": key, "value": value, "by": by},
            )
        await self.session.commit()

    async def clear(self) -> None:
        await self.session.execute(text("DELETE FROM runtime_config"))
        await self.session.commit()


def sebagai_teks(nilai: Any) -> str:
    """Bentuk simpan satu nilai. Sama seperti yang ditulis orang di `.env`."""
    return str(nilai)


def bangun(base: Settings, overrides: Mapping[str, str]) -> Settings:
    """`Settings` efektif = `.env` + nilai yang ditimpa.

    Melempar `ValidationError` bila kombinasinya tidak sah -- dipakai endpoint
    PATCH untuk menolak sebelum menyimpan. Perhatikan bahwa yang divalidasi
    adalah hasil gabungannya, bukan hanya nilai yang dikirim: `chunk_overlap`
    yang sah sendirian bisa melanggar `chunk_size` yang sudah tersimpan.
    """
    dipakai = {k: v for k, v in overrides.items() if k in DAPAT_DIUBAH}
    if not dipakai:
        return base
    return Settings.model_validate({**base.model_dump(), **dipakai})


def terapkan(base: Settings, tersimpan: Mapping[str, NilaiTersimpan]) -> Settings:
    """Versi `bangun` untuk jalur permintaan: tidak pernah melempar.

    Baris yang tidak dapat dipakai (mis. `.env` berubah sehingga kombinasinya
    melanggar aturan antar-field) membuat seluruh nilai simpanan diabaikan dan
    layanan kembali memakai `.env`. Tetap menjawab dengan setelan yang masuk
    akal lebih baik daripada 500 di setiap pertanyaan mahasiswa; halaman
    Konfigurasi menampilkan peringatan yang sama lewat `keluhan()`.
    """
    try:
        return bangun(base, {k: v.value for k, v in tersimpan.items()})
    except ValidationError as exc:
        log.error(
            "Konfigurasi tersimpan diabaikan, memakai .env: %s",
            pesan_pertama(exc),
        )
        return base


def keluhan(base: Settings, tersimpan: Mapping[str, NilaiTersimpan]) -> str | None:
    """Alasan nilai simpanan tidak dapat dipakai, atau None bila sehat."""
    try:
        bangun(base, {k: v.value for k, v in tersimpan.items()})
    except ValidationError as exc:
        return pesan_pertama(exc)
    return None


def pesan_pertama(exc: ValidationError) -> str:
    """Kalimat galat pertama, tanpa jejak teknis Pydantic (PRD §9)."""
    galat = exc.errors()[0]
    pesan = str(galat.get("msg", "")).removeprefix("Value error, ")
    lokasi = ".".join(str(b) for b in galat.get("loc", ()))
    return f"{lokasi}: {pesan}" if lokasi and lokasi not in pesan else pesan
