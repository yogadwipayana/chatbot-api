"""Hapus percakapan uji (tes manual, uji A/B, uji browser) dari database.

    python -m scripts.hapus_data_uji                         # daftar sesi, tanpa menghapus
    python -m scripts.hapus_data_uji --awalan jev-uji-       # pratinjau yang akan dihapus
    python -m scripts.hapus_data_uji --awalan jev-uji- --sesi <id> --hapus
    python -m scripts.hapus_data_uji --semua --hapus --konfirmasi HAPUS-SEMUA   # sebelum rilis

Database pengembangan dan produksi saat ini SATU database (kontainer `pg` di
host server; lewat Tailscale maupun `localhost` di server). Setiap giliran uji
ikut terbaca di statistik admin, dan setiap penolakan uji muncul di daftar
*Pertanyaan tak terjawab* seolah celah dokumen sungguhan.

Tanpa `--hapus` skrip hanya menampilkan pratinjau. Yang dihapus per sesi
terpilih, dalam SATU transaksi:

1. `unanswered_questions` yang menunjuk pesan sesi itu. FK-nya `ON DELETE SET
   NULL`: tanpa langkah ini, entri AD-4 tertinggal tanpa induk dan tetap tampil.
2. `conversations` sesi itu. `messages` dan `feedback` ikut terhapus (CASCADE).

Dokumen, unit, akun admin, kunci sematan, dan konfigurasi tidak pernah
disentuh. Log aplikasi SQLite (`log/app.db`) ada di mesin masing-masing, bukan
di database bersama, jadi dibiarkan. Data `seed_demo` punya perintahnya sendiri
(`python -m scripts.seed_demo --hapus`).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

KONFIRMASI_SEMUA = "HAPUS-SEMUA"
MIN_PANJANG_AWALAN = 4
"""Awalan pendek ("a", "8") bisa mencocokkan sesi mahasiswa sungguhan tanpa
disadari; untuk menghapus semuanya ada `--semua`, yang wajib dikonfirmasi."""


@dataclass(frozen=True)
class Target:
    sesi: tuple[str, ...] = ()
    awalan: tuple[str, ...] = ()
    semua: bool = False

    @property
    def kosong(self) -> bool:
        return not (self.sesi or self.awalan or self.semua)

    def kondisi(self) -> tuple[str, dict[str, object]]:
        """Klausa WHERE atas alias `c` (conversations) beserta parameternya."""
        if self.semua:
            return "TRUE", {}
        return (
            "(c.session_id = ANY(CAST(:sesi AS text[]))"
            " OR c.session_id LIKE ANY(CAST(:pola AS text[])))",
            {"sesi": list(self.sesi), "pola": [pola_awalan(a) for a in self.awalan]},
        )


def pola_awalan(awalan: str) -> str:
    """Pola LIKE untuk awalan; `%` dan `_` di awalan diperlakukan harfiah.

    Tanpa ini `--awalan uji_` juga mencocokkan "uji-", "ujiX", dst."""
    harfiah = awalan.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return harfiah + "%"


def periksa_argumen(target: Target, *, hapus: bool, konfirmasi: str | None) -> str | None:
    """Pesan galat, atau None bila kombinasi argumen aman dijalankan."""
    pendek = [a for a in target.awalan if len(a) < MIN_PANJANG_AWALAN]
    if pendek:
        return (
            f"awalan terlalu pendek: {pendek} (minimal {MIN_PANJANG_AWALAN} karakter). "
            "Untuk semua sesi pakai --semua."
        )
    if target.semua and (target.sesi or target.awalan):
        return "--semua tidak bisa digabung dengan --sesi/--awalan"
    if hapus and target.kosong:
        return "--hapus butuh target: --sesi, --awalan, atau --semua"
    if hapus and target.semua and konfirmasi != KONFIRMASI_SEMUA:
        return f"--semua --hapus butuh --konfirmasi {KONFIRMASI_SEMUA}"
    return None


# --- Baca ------------------------------------------------------------------


async def ringkasan(session: AsyncSession, target: Target, zona: str) -> list[dict]:
    """Satu baris per sesi: jumlah percakapan, pesan, entri AD-4, umpan balik."""
    kondisi, params = target.kondisi() if not target.kosong else ("TRUE", {})
    hasil = await session.execute(
        text(
            f"""
            WITH t AS (SELECT c.id, c.session_id FROM conversations c WHERE {kondisi})
            SELECT t.session_id,
                   count(DISTINCT t.id) AS percakapan,
                   count(DISTINCT m.id) FILTER (WHERE m.role = 'user') AS pertanyaan,
                   count(DISTINCT m.id) FILTER (WHERE m.role = 'assistant') AS jawaban,
                   count(DISTINCT u.id) AS tak_terjawab,
                   count(DISTINCT f.id) AS umpan_balik,
                   to_char(min(m.created_at) AT TIME ZONE :zona, 'YYYY-MM-DD HH24:MI') AS awal,
                   to_char(max(m.created_at) AT TIME ZONE :zona, 'YYYY-MM-DD HH24:MI')
                       AS akhir,
                   (array_agg(DISTINCT left(m.content, 60))
                        FILTER (WHERE m.role = 'user'))[1:2] AS contoh
            FROM t
            LEFT JOIN messages m ON m.conversation_id = t.id
            LEFT JOIN unanswered_questions u ON u.message_id = m.id
            LEFT JOIN feedback f ON f.message_id = m.id
            GROUP BY t.session_id
            ORDER BY min(m.created_at) NULLS FIRST
            """
        ),
        {**params, "zona": zona},
    )
    return [dict(r) for r in hasil.mappings()]


async def jumlah_yatim(session: AsyncSession) -> int:
    """Entri AD-4 yang pesannya sudah tidak ada (sisa penghapusan tanpa langkah 1)."""
    return (
        await session.execute(
            text("SELECT count(*) FROM unanswered_questions WHERE message_id IS NULL")
        )
    ).scalar_one()


# --- Hapus -----------------------------------------------------------------


async def hapus(session: AsyncSession, target: Target, *, yatim: bool) -> dict[str, int]:
    """Hapus sesi terpilih dalam satu transaksi. Urutannya penting (lihat docstring modul)."""
    kondisi, params = target.kondisi()
    tak_terjawab = await session.execute(
        text(
            "DELETE FROM unanswered_questions WHERE message_id IN ("
            " SELECT m.id FROM messages m JOIN conversations c ON c.id = m.conversation_id"
            f" WHERE {kondisi})"
        ),
        params,
    )
    percakapan = await session.execute(
        text(f"DELETE FROM conversations c WHERE {kondisi}"), params
    )
    hasil = {
        "tak_terjawab": tak_terjawab.rowcount,
        "percakapan": percakapan.rowcount,
        "yatim": 0,
    }
    if yatim:
        sisa = await session.execute(
            text("DELETE FROM unanswered_questions WHERE message_id IS NULL")
        )
        hasil["yatim"] = sisa.rowcount
    await session.commit()
    return hasil


# --- CLI -------------------------------------------------------------------


def _cetak(baris: list[dict]) -> None:
    if not baris:
        print("  (tidak ada sesi yang cocok)")
        return
    for b in baris:
        print(
            f"- {b['session_id']}: {b['percakapan']} percakapan, "
            f"{b['pertanyaan']} pertanyaan, "
            f"{b['jawaban']} jawaban, {b['tak_terjawab']} tak terjawab, "
            f"{b['umpan_balik']} umpan balik | {b['awal']} s.d. {b['akhir']}"
        )
        for contoh in b["contoh"] or []:
            print(f"      contoh: {contoh!r}")


async def jalankan(args: argparse.Namespace, target: Target) -> int:
    settings = get_settings()
    url = make_url(str(settings.database_url))
    zona = settings.timezone
    print(
        f"Database: {url.database} @ {url.host}:{url.port} "
        f"(ENVIRONMENT={settings.environment}, jam dalam {zona})"
    )

    engine = create_async_engine(url)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with maker() as session:
            baris = await ringkasan(session, target, zona)
            yatim = await jumlah_yatim(session)
            if target.kosong:
                print("Semua sesi (tanpa menghapus; pilih dengan --sesi/--awalan/--semua):")
                _cetak(baris)
                print(f"Entri tak terjawab tanpa pesan (yatim): {yatim}")
                return 0

            print("Akan dihapus:" if args.hapus else "Pratinjau (tambahkan --hapus):")
            _cetak(baris)
            kolom = ("percakapan", "jawaban", "tak_terjawab")
            total = {k: sum(b[k] for b in baris) for k in kolom}
            print(
                f"Total: {total['percakapan']} percakapan, {total['jawaban']} jawaban, "
                f"{total['tak_terjawab']} entri tak terjawab"
                + (f", plus {yatim} entri yatim" if args.yatim else "")
            )
            if not args.hapus:
                return 0

            dihapus = await hapus(session, target, yatim=args.yatim)
            print(
                f"Dihapus: {dihapus['percakapan']} percakapan (pesan dan umpan balik ikut), "
                f"{dihapus['tak_terjawab']} entri tak terjawab, {dihapus['yatim']} entri yatim"
            )
        async with maker() as session:
            sisa = await ringkasan(session, target, zona)
            print(f"Pemeriksaan ulang: {len(sisa)} sesi target tersisa")
            return 0 if not sisa else 1
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Hapus percakapan uji dari database.")
    parser.add_argument("--sesi", action="append", default=[], help="session_id persis")
    parser.add_argument("--awalan", action="append", default=[], help="awalan session_id")
    parser.add_argument("--semua", action="store_true", help="SEMUA percakapan (pra-rilis)")
    parser.add_argument("--yatim", action="store_true", help="juga entri tak terjawab yatim")
    parser.add_argument("--hapus", action="store_true", help="benar-benar menghapus")
    parser.add_argument(
        "--konfirmasi", help=f"wajib '{KONFIRMASI_SEMUA}' untuk --semua --hapus"
    )
    args = parser.parse_args()

    target = Target(sesi=tuple(args.sesi), awalan=tuple(args.awalan), semua=args.semua)
    galat = periksa_argumen(target, hapus=args.hapus, konfirmasi=args.konfirmasi)
    if galat:
        parser.error(galat)
    return asyncio.run(jalankan(args, target))


if __name__ == "__main__":
    sys.exit(main())
