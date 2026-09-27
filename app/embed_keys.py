"""Kunci sematan: satu kunci untuk setiap situs lain yang memasang asisten.

Situs penyemat menempel `<script src=".../embed.js" data-key="emb_...">`. Saat
panel dibuka, portal (`client/src/proxy.ts`) menanyakan kunci itu ke
`GET /api/embed/keys/{key}` lalu memasang `frame-ancestors` dari
`allowed_origins`; setiap pertanyaan dari panel membawa kunci yang sama di header
`X-Embed-Key` (`app.deps.kunci_sematan`).

Kunci BUKAN rahasia: ia tertulis di kode sumber situs penyemat dan siapa pun
dapat menyalinnya. Yang membatasi pemakaiannya:

- `allowed_origins` -- ditegakkan peramban, bukan server: situs di luar daftar
  tidak dapat menampilkan panelnya, kunci salinan pun tidak berguna di sana.
  Kosong = situs mana pun, sama dengan `EMBED_ALLOWED_ORIGINS` kosong dulu.
- `is_active` -- mencabut satu situs tanpa menyentuh situs lain.
"""

from __future__ import annotations

import re
import secrets
import string
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

AWALAN = "emb_"
PANJANG_ACAK = 24
"""24 karakter alfanumerik = ~143 bit. Kunci tidak dirahasiakan, tetapi tetap
tidak boleh dapat ditebak: menebak kunci situs lain berarti memakai kuotanya."""
_ALFABET = string.ascii_letters + string.digits
_BENTUK_KUNCI = re.compile(rf"^{AWALAN}[A-Za-z0-9]{{{PANJANG_ACAK}}}$")

MAKS_ASAL = 20
_LABEL_HOST = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_PORT_BAWAAN = {"http": 80, "https": 443}

EDITABLE_FIELDS = ("name", "allowed_origins", "is_active")


def buat_kunci() -> str:
    return AWALAN + "".join(secrets.choice(_ALFABET) for _ in range(PANJANG_ACAK))


def bentuk_sah(kunci: str | None) -> bool:
    """Kunci yang bentuknya saja sudah salah tidak perlu ditanyakan ke database."""
    return bool(kunci) and _BENTUK_KUNCI.fullmatch(kunci or "") is not None


def normalisasi_asal(nilai: str) -> str:
    """Asal situs dalam bentuk yang dipahami CSP `frame-ancestors`, atau ValueError.

    Yang diterima hanya asal -- skema, domain, dan port bila bukan bawaan --
    tanpa path: `frame-ancestors` membandingkan asal situs induk, jadi path
    `https://pmb.instiki.ac.id/daftar` tidak pernah cocok dengan apa pun dan
    panelnya ditolak tanpa pesan yang jelas. Garis miring akhir, huruf besar,
    dan port bawaan (`:443`) dibuang supaya satu situs tidak tercatat dua kali.

    Wildcard subdomain ala CSP diterima di label pertama: `https://*.instiki.ac.id`
    mencakup semua subdomain, tetapi TIDAK `https://instiki.ac.id` sendiri.
    """
    teks = nilai.strip()
    if not teks:
        raise ValueError("Alamat situs tidak boleh kosong.")
    try:
        bagian = urlsplit(teks)
        port = bagian.port
    except ValueError:
        raise ValueError(f"'{teks}' bukan alamat situs yang sah.") from None

    skema = bagian.scheme.lower()
    if skema not in _PORT_BAWAAN:
        raise ValueError(
            f"'{teks}' harus diawali https:// (atau http:// untuk pengembangan)."
        )
    if (
        bagian.username
        or bagian.password
        or bagian.query
        or bagian.fragment
        or bagian.path not in ("", "/")
    ):
        raise ValueError(
            f"'{teks}' harus berupa alamat situs saja, tanpa path -- "
            "mis. https://pmb.instiki.ac.id."
        )

    host = bagian.hostname or ""
    label = host.split(".")
    wildcard = label[0] == "*"
    sisa = label[1:] if wildcard else label
    label_sah = bool(host) and all(_LABEL_HOST.fullmatch(x) for x in sisa)
    # `https://*.id` sah menurut CSP, tetapi berarti seluruh domain .id.
    if not label_sah or (wildcard and len(sisa) < 2):
        raise ValueError(f"'{teks}' bukan nama domain yang sah.")

    akhiran = f":{port}" if port is not None and port != _PORT_BAWAAN[skema] else ""
    return f"{skema}://{host}{akhiran}"


def rapikan_daftar_asal(daftar: list[str]) -> list[str]:
    """Normalisasi setiap asal dan buang duplikatnya, urutan pertama dipertahankan."""
    hasil = list(dict.fromkeys(normalisasi_asal(a) for a in daftar))
    if len(hasil) > MAKS_ASAL:
        raise ValueError(f"Paling banyak {MAKS_ASAL} alamat situs untuk satu kunci.")
    return hasil


def kode_sematan(url_portal: str, kunci: str) -> str:
    """Baris yang ditempel pemilik situs, persis seperti yang ditampilkan dashboard."""
    return f'<script src="{url_portal}/embed.js" data-key="{kunci}" async></script>'


@dataclass(frozen=True)
class KunciSematan:
    """Satu baris `embed_keys`, beserta statistik pemakaiannya."""

    key: str
    name: str
    allowed_origins: list[str]
    is_active: bool
    created_by: str | None
    created_at: datetime
    questions_30d: int = 0
    """Pertanyaan mahasiswa dari situs ini, 30 hari terakhir."""
    last_used_at: datetime | None = None
    """Pertanyaan terakhir dari situs ini; None = belum pernah dipakai."""


# Pertanyaan dihitung dari baris `messages` milik pengguna, bukan dari
# percakapan: satu percakapan bisa berisi sepuluh pertanyaan, dan yang memakai
# kuota model adalah pertanyaannya.
_SEMUA_SQL = """
    SELECT k.key, k.name, k.allowed_origins, k.is_active, k.created_by, k.created_at,
           p.questions_30d, p.last_used_at
    FROM embed_keys k
    LEFT JOIN LATERAL (
        SELECT count(*) FILTER (WHERE m.created_at > now() - interval '30 days')
                   AS questions_30d,
               max(m.created_at) AS last_used_at
        FROM conversations c
        JOIN messages m ON m.conversation_id = c.id AND m.role = 'user'
        WHERE c.embed_key = k.key
    ) p ON true
"""

_KOLOM_SET = {
    "name": "name = :name",
    "allowed_origins": "allowed_origins = CAST(:allowed_origins AS text[])",
    "is_active": "is_active = :is_active",
}


def _dari_baris(row: Any) -> KunciSematan:
    data = dict(row)
    data["allowed_origins"] = list(data.get("allowed_origins") or [])
    data["questions_30d"] = int(data.get("questions_30d") or 0)
    return KunciSematan(**data)


class SqlEmbedKeyStore:
    """Tabel `embed_keys`, langsung dari database. Di-override di test."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def aktif(self, kunci: str) -> KunciSematan | None:
        """Kunci yang masih berlaku, tanpa statistik -- dipanggil setiap pertanyaan."""
        if not bentuk_sah(kunci):
            return None
        row = (
            (
                await self.session.execute(
                    text(
                        "SELECT key, name, allowed_origins, is_active, created_by,"
                        " created_at FROM embed_keys WHERE key = :key AND is_active"
                    ),
                    {"key": kunci},
                )
            )
            .mappings()
            .first()
        )
        return _dari_baris(row) if row else None

    async def semua(self) -> list[KunciSematan]:
        rows = await self.session.execute(text(_SEMUA_SQL + " ORDER BY k.created_at, k.key"))
        return [_dari_baris(r) for r in rows.mappings()]

    async def ambil(self, kunci: str) -> KunciSematan | None:
        sql = text(_SEMUA_SQL + " WHERE k.key = :key")
        row = (await self.session.execute(sql, {"key": kunci})).mappings().first()
        return _dari_baris(row) if row else None

    async def buat(self, *, name: str, allowed_origins: list[str], oleh: str) -> KunciSematan:
        kunci = buat_kunci()
        await self.session.execute(
            text(
                "INSERT INTO embed_keys (key, name, allowed_origins, is_active, created_by)"
                " VALUES (:key, :name, CAST(:allowed_origins AS text[]), true, :created_by)"
            ),
            {
                "key": kunci,
                "name": name,
                "allowed_origins": allowed_origins,
                "created_by": oleh,
            },
        )
        await self.session.commit()
        hasil = await self.ambil(kunci)
        assert hasil is not None
        return hasil

    async def ubah(self, kunci: str, changes: dict[str, Any]) -> KunciSematan | None:
        """`changes` memakai nama kolom (`EDITABLE_FIELDS`), sama dengan field API."""
        kolom = [k for k in EDITABLE_FIELDS if k in changes]
        if not kolom:
            return await self.ambil(kunci)
        row = (
            await self.session.execute(
                text(
                    f"UPDATE embed_keys SET {', '.join(_KOLOM_SET[k] for k in kolom)}"
                    " WHERE key = :key RETURNING key"
                ),
                {**{k: changes[k] for k in kolom}, "key": kunci},
            )
        ).first()
        if row is None:
            await self.session.rollback()
            return None
        await self.session.commit()
        return await self.ambil(kunci)

    async def hapus(self, kunci: str) -> bool:
        """Percakapan dari situs ini tetap ada; penandanya menjadi NULL."""
        row = (
            await self.session.execute(
                text("DELETE FROM embed_keys WHERE key = :key RETURNING key"),
                {"key": kunci},
            )
        ).first()
        await self.session.commit()
        return row is not None
