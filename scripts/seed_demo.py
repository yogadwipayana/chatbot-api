"""Isi database pengembangan dengan data contoh untuk dashboard admin.

    python -m scripts.seed_demo --dokumen   # 4 PDF contoh lewat pipeline ingestion sungguhan
    python -m scripts.seed_demo --log       # 30 hari percakapan, penolakan, umpan balik
    python -m scripts.seed_demo --hapus     # hapus seluruh data contoh di atas

`--dokumen` memanggil API embedding (EMBED_MODEL) dan menulis ke penyimpanan
objek sungguhan (STORAGE_BACKEND), sehingga sekaligus menguji keduanya ujung ke
ujung. Biayanya beberapa ribu token embedding.

Data contoh diberi penanda supaya dapat dihapus tanpa menyentuh data asli:
dokumen lewat `uploaded_by = 'seed-demo'`, percakapan lewat awalan `session_id`
`seed-demo-`. Menolak berjalan saat ENVIRONMENT=production.

Isi dokumen dan pertanyaan di sini FIKTIF. Jangan dibiarkan di database yang
dipakai mahasiswa.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.observability.chatlog import SENSITIVE_PLACEHOLDER
from app.observability.costs import estimate_cost
from app.rag import risk, sensitive
from app.rag.chain import (
    DEFAULT_FALLBACK_CONTACT,
    REFUSAL_TEMPLATE,
    SUPPORT_TEMPLATE,
    render_contacts,
)

PENANDA_DOKUMEN = "seed-demo"
AWALAN_SESI = "seed-demo-"
HARI_LOG = 30
MODEL_CONTOH = "gpt-4o-mini"
ZONA_KAMPUS = timezone(timedelta(hours=7))
"""Hanya untuk menyebar jam contoh di jam kerja; statistik memakai TIMEZONE."""


# --- Dokumen ---------------------------------------------------------------


@dataclass(frozen=True)
class DokumenContoh:
    judul: str
    unit: str
    tahun_berlaku: int | None
    valid_until: date | None
    halaman: tuple[str, ...]
    usang_bulan: int | None = None
    """Mundurkan `updated_at` sekian bulan, untuk memicu badge usia AD-2."""


DOKUMEN = (
    DokumenContoh(
        judul="Panduan Akademik 2026 (Contoh)",
        unit="Biro Administrasi Akademik",
        tahun_berlaku=2026,
        valid_until=date(2027, 8, 31),
        halaman=(
            "BAB I. PENGISIAN KARTU RENCANA STUDI (KRS)\n\n"
            "Pengisian KRS semester ganjil 2026/2027 dibuka pada 3 sampai 14 Agustus 2026 "
            "melalui portal akademik. Mahasiswa wajib berkonsultasi dengan dosen wali "
            "sebelum mengajukan KRS. KRS yang belum disetujui dosen wali sampai batas akhir "
            "dianggap tidak mengambil mata kuliah.\n\n"
            "Jumlah SKS maksimum ditentukan oleh IP semester sebelumnya: IP 3,00 atau lebih "
            "boleh mengambil paling banyak 24 SKS; IP 2,50 sampai 2,99 paling banyak 21 SKS; "
            "IP di bawah 2,50 paling banyak 18 SKS.\n\n"
            "Perubahan KRS berupa tambah atau batal mata kuliah dapat dilakukan paling lambat "
            "dua minggu setelah perkuliahan dimulai.",
            "BAB II. CUTI AKADEMIK\n\n"
            "Mahasiswa dapat mengajukan cuti akademik paling lama dua semester selama masa "
            "studi, dan tidak boleh diambil dua semester berturut-turut kecuali dengan izin "
            "Wakil Rektor Bidang Akademik.\n\n"
            "Persyaratan pengajuan cuti: (1) telah menempuh paling sedikit dua semester; "
            "(2) tidak memiliki tunggakan UKT; (3) mengisi formulir cuti yang ditandatangani "
            "dosen wali dan ketua program studi; (4) menyerahkan formulir ke Biro "
            "Administrasi Akademik paling lambat satu minggu sebelum pengisian KRS "
            "ditutup.\n\n"
            "Selama cuti mahasiswa tidak dikenai UKT, dan masa cuti tidak dihitung dalam "
            "batas masa studi.",
            "BAB III. YUDISIUM DAN WISUDA\n\n"
            "Syarat mengikuti yudisium: (1) lulus seluruh mata kuliah wajib dengan jumlah "
            "paling sedikit 144 SKS; (2) IPK paling rendah 2,00 tanpa nilai E; (3) telah "
            "menyerahkan naskah skripsi yang disahkan; (4) bebas pinjaman perpustakaan; "
            "(5) memiliki sertifikat TOEFL atau setara dengan skor paling rendah 450.\n\n"
            "Pendaftaran wisuda dilakukan melalui portal akademik setelah dinyatakan lulus "
            "yudisium. Wisuda periode I dilaksanakan pada bulan Maret dan periode II pada "
            "bulan September. Batas akhir pendaftaran wisuda adalah 30 hari sebelum tanggal "
            "pelaksanaan.",
            "BAB IV. LAYANAN SURAT KETERANGAN\n\n"
            "Surat keterangan aktif kuliah diajukan melalui portal akademik pada menu Layanan "
            "Surat. Surat dapat diunduh dalam bentuk dokumen bertanda tangan elektronik "
            "paling lama dua hari kerja setelah pengajuan.\n\n"
            "Surat keterangan lulus diterbitkan oleh Biro Administrasi Akademik setelah "
            "yudisium dan berlaku sampai ijazah diterbitkan. Legalisir ijazah dan transkrip "
            "dilayani pada hari Senin sampai Jumat pukul 08.00 sampai 14.00 dengan membawa "
            "ijazah asli.",
        ),
    ),
    DokumenContoh(
        judul="Ketentuan Pembayaran UKT 2026 (Contoh)",
        unit="Biro Keuangan",
        tahun_berlaku=2026,
        valid_until=date(2027, 1, 31),
        halaman=(
            "KETENTUAN PEMBAYARAN UANG KULIAH TUNGGAL (UKT)\n\n"
            "Pembayaran UKT semester ganjil 2026/2027 dilakukan pada 20 Juli sampai "
            "7 Agustus 2026 melalui bank mitra atau virtual account yang tercantum di "
            "portal akademik. Mahasiswa yang belum membayar UKT sampai batas akhir tidak "
            "dapat mengisi KRS.\n\n"
            "Keterlambatan pembayaran dikenai denda administrasi sebesar 5 persen dari "
            "nominal UKT. Pembayaran setelah 21 Agustus 2026 tidak dilayani dan mahasiswa "
            "dianggap tidak aktif pada semester berjalan.",
            "PENGAJUAN KERINGANAN DAN ANGSURAN UKT\n\n"
            "Mahasiswa yang mengalami kesulitan ekonomi dapat mengajukan penurunan "
            "golongan UKT atau pembayaran secara angsuran. Pengajuan dilakukan paling "
            "lambat 10 Juli 2026 dengan melampirkan surat keterangan penghasilan orang "
            "tua, kartu keluarga, dan surat pernyataan yang diketahui dosen wali.\n\n"
            "Angsuran UKT dapat dibayar paling banyak dalam dua tahap: tahap pertama "
            "50 persen sebelum pengisian KRS dan tahap kedua paling lambat "
            "30 September 2026.",
        ),
    ),
    DokumenContoh(
        judul="Kalender Akademik 2024/2025 (Contoh)",
        unit="Biro Administrasi Akademik",
        tahun_berlaku=2024,
        valid_until=date(2025, 8, 31),
        halaman=(
            "KALENDER AKADEMIK TAHUN AKADEMIK 2024/2025\n\n"
            "Pengisian KRS semester ganjil: 5 sampai 16 Agustus 2024. Perkuliahan "
            "semester ganjil: 2 September sampai 20 Desember 2024. Ujian akhir semester "
            "ganjil: 6 sampai 17 Januari 2025.\n\n"
            "Pengisian KRS semester genap: 3 sampai 14 Februari 2025. Perkuliahan semester "
            "genap: 3 Maret sampai 20 Juni 2025. Ujian akhir semester genap: 30 Juni sampai "
            "11 Juli 2025. Wisuda periode II: 20 September 2025.",
        ),
    ),
    DokumenContoh(
        judul="Tata Tertib Mahasiswa (Contoh)",
        unit="Bagian Kemahasiswaan",
        tahun_berlaku=2025,
        valid_until=None,
        usang_bulan=8,
        halaman=(
            "TATA TERTIB DAN SANKSI\n\n"
            "Mahasiswa wajib menjaga nama baik kampus, menaati peraturan akademik, dan "
            "menghormati sivitas akademika. Pelanggaran tata tertib dikelompokkan menjadi "
            "pelanggaran ringan, sedang, dan berat.\n\n"
            "Sanksi pelanggaran ringan berupa teguran lisan atau tertulis. Sanksi pelanggaran "
            "sedang berupa skorsing paling lama satu semester. Sanksi pelanggaran berat, "
            "termasuk plagiarisme skripsi dan pemalsuan dokumen, berupa pemberhentian sebagai "
            "mahasiswa.\n\n"
            "Mahasiswa yang dikenai sanksi berhak mengajukan keberatan secara tertulis kepada "
            "Dekan paling lambat 14 hari kerja setelah keputusan diterima.",
        ),
    ),
)


def buat_pdf(path: Path, halaman: tuple[str, ...]) -> None:
    import pymupdf

    doc = pymupdf.open()
    try:
        for isi in halaman:
            page = doc.new_page(width=595, height=842)  # A4
            sisa = page.insert_textbox(
                pymupdf.Rect(56, 56, 539, 786), isi, fontsize=11, fontname="helv"
            )
            if sisa < 0:
                raise RuntimeError(f"teks halaman contoh tidak muat: {isi[:40]!r}")
        doc.save(path)
    finally:
        doc.close()


async def seed_dokumen(maker: async_sessionmaker[AsyncSession]) -> None:
    from app.ingestion.pipeline import ingest_document
    from app.rag.providers import build_embeddings
    from app.storage import build_storage

    settings = get_settings()
    embeddings = build_embeddings(settings)
    storage = build_storage(settings)

    print("Dokumen contoh:")
    for dok in DOKUMEN:
        async with maker() as session:
            ada = (
                await session.execute(
                    text("SELECT 1 FROM documents WHERE judul = :judul AND uploaded_by = :p"),
                    {"judul": dok.judul, "p": PENANDA_DOKUMEN},
                )
            ).scalar()
            if ada:
                print(f"  = {dok.judul}: sudah ada, dilewati")
                continue

            with tempfile.TemporaryDirectory(prefix="seed-") as folder:
                # Judul boleh mengandung "/" ("2024/2025"), yang di nama berkas
                # berarti subfolder yang tidak ada.
                path = Path(folder) / f"{dok.judul.replace('/', '-')}.pdf"
                buat_pdf(path, dok.halaman)
                hasil = await ingest_document(
                    session,
                    path=path,
                    judul=dok.judul,
                    unit=dok.unit,
                    embeddings=embeddings,
                    storage=storage,
                    tahun_berlaku=dok.tahun_berlaku,
                    valid_until=dok.valid_until,
                    uploaded_by=PENANDA_DOKUMEN,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )

            if dok.usang_bulan:
                await session.execute(
                    text(
                        "UPDATE documents SET updated_at = now() - make_interval(months => :m)"
                        " WHERE id = :id"
                    ),
                    {"m": dok.usang_bulan, "id": hasil.document_id},
                )
                await session.commit()

        print(f"  + {dok.judul}: {hasil.jumlah_halaman} halaman, {hasil.jumlah_chunk} chunk")


# --- Log percakapan --------------------------------------------------------

PERTANYAAN_DIJAWAB = (
    "Kapan pengisian KRS semester ganjil dibuka?",
    "Berapa SKS maksimal kalau IP saya 3,2?",
    "Bagaimana cara mengajukan cuti kuliah?",
    "Apa saja syarat yudisium?",
    "Kapan batas akhir pembayaran UKT semester ganjil?",
    "UKT bisa dicicil tidak?",
    "Cara minta surat keterangan aktif kuliah gimana?",
    "Syarat wisuda apa aja?",
    "Denda telat bayar UKT berapa persen?",
    "Legalisir ijazah bisa jam berapa?",
    "Perubahan KRS paling lambat kapan?",
    "Sanksi plagiarisme skripsi apa?",
)


@dataclass(frozen=True)
class KelompokDitolak:
    bobot: int
    variasi: tuple[str, ...]
    resolved: bool = False


KELOMPOK_DITOLAK = (
    KelompokDitolak(
        10,
        (
            "Bagaimana cara daftar beasiswa prestasi?",
            "cara daftar beasiswa prestasi gimana",
            "Pendaftaran beasiswa prestasi kapan?",
            "daftar beasiswa prestasi dimana min?",
        ),
    ),
    KelompokDitolak(
        6,
        (
            "Syarat pindah program studi apa saja?",
            "Bisa pindah program studi tidak?",
            "cara pindah program studi",
        ),
    ),
    KelompokDitolak(5, ("Kapan pendaftaran KKN dibuka?", "pendaftaran KKN kapan min")),
    KelompokDitolak(
        3,
        ("Lupa password portal akademik gimana?", "password portal akademik lupa"),
        resolved=True,
    ),
    KelompokDitolak(2, ("Parkir motor mahasiswa di gedung mana?",)),
    KelompokDitolak(2, ("Jadwal bus kampus jam berapa?",)),
    KelompokDitolak(1, ("Wifi perpustakaan passwordnya apa?",)),
)

PERTANYAAN_SENSITIF = (
    "saya stres berat skripsi tidak selesai-selesai",
    "rasanya mau menyerah kuliah",
    "saya depresi karena nilai jelek terus",
)

_PESAN_SQL = text(
    """
    INSERT INTO messages
        (id, conversation_id, role, konten, top_score, latency_ms, meta, created_at)
    VALUES
        (:id, :cid, :role, :konten, :top_score, :latency, CAST(:meta AS jsonb), :t)
    """
)


async def tulis_percakapan(
    session: AsyncSession, rng: random.Random, mulai: datetime, n: int, hitung: Counter
) -> None:
    cid = uuid.uuid4()
    await session.execute(
        text("INSERT INTO conversations (id, session_id, created_at) VALUES (:id, :sid, :t)"),
        {"id": cid, "sid": AWALAN_SESI + uuid.uuid4().hex[:12], "t": mulai},
    )

    t = mulai
    for _ in range(n):
        jenis = rng.choices(("answer", "refusal", "support"), weights=(78, 19, 3))[0]
        hitung[jenis] += 1
        kelompok: KelompokDitolak | None = None

        if jenis == "answer":
            pertanyaan = rng.choice(PERTANYAAN_DIJAWAB)
            topik = risk.detect(pertanyaan)
            kontak = topik.contacts
            latency, top_score = rng.randint(1400, 4300), rng.uniform(0.45, 0.88)
            token_masuk, token_keluar = rng.randint(1500, 3200), rng.randint(100, 420)
            teks = (
                "(Jawaban contoh untuk demo dashboard) Informasinya tercantum di "
                f"dokumen resmi [Panduan Akademik 2026 (Contoh), hal. {rng.randint(1, 4)}]."
            )
            meta_llm = {
                "llm_dipanggil": True,
                "model": MODEL_CONTOH,
                "input_tokens": token_masuk,
                "output_tokens": token_keluar,
                "biaya_usd": estimate_cost(MODEL_CONTOH, token_masuk, token_keluar).usd,
            }
            daftar_topik, tingkat = [x.value for x in topik.topics], "none"
        elif jenis == "refusal":
            kelompok = rng.choices(
                KELOMPOK_DITOLAK, weights=[k.bobot for k in KELOMPOK_DITOLAK]
            )[0]
            pertanyaan = rng.choice(kelompok.variasi)
            topik = risk.detect(pertanyaan)
            kontak = topik.contacts or (DEFAULT_FALLBACK_CONTACT,)
            latency, top_score = rng.randint(350, 950), rng.uniform(0.06, 0.31)
            teks = REFUSAL_TEMPLATE.format(contacts=render_contacts(kontak))
            meta_llm = {
                "llm_dipanggil": False,
                "model": None,
                "input_tokens": None,
                "output_tokens": None,
                "biaya_usd": None,
            }
            daftar_topik, tingkat = [x.value for x in topik.topics], "none"
        else:
            pertanyaan = rng.choice(PERTANYAAN_SENSITIF)
            penilaian = sensitive.detect(pertanyaan)
            kontak = penilaian.contacts
            latency, top_score = rng.randint(15, 60), None
            teks = SUPPORT_TEMPLATE.format(contacts=render_contacts(kontak))
            meta_llm = {
                "llm_dipanggil": False,
                "model": None,
                "input_tokens": None,
                "output_tokens": None,
                "biaya_usd": None,
            }
            daftar_topik, tingkat = [], penilaian.level.value

        meta = {
            "kind": jenis,
            "escalated": bool(kontak),
            "topik": daftar_topik,
            "sensitivitas": tingkat,
            "rewritten_query": None,
            "seed": True,
            **meta_llm,
        }

        dijawab = t + timedelta(milliseconds=latency)
        jawaban_id = uuid.uuid4()
        await session.execute(
            _PESAN_SQL,
            {
                "id": uuid.uuid4(),
                "cid": cid,
                "role": "user",
                "konten": SENSITIVE_PLACEHOLDER if jenis == "support" else pertanyaan,
                "top_score": None,
                "latency": None,
                "meta": None,
                "t": t,
            },
        )
        await session.execute(
            _PESAN_SQL,
            {
                "id": jawaban_id,
                "cid": cid,
                "role": "assistant",
                "konten": teks,
                "top_score": top_score,
                "latency": latency,
                "meta": json.dumps(meta),
                "t": dijawab,
            },
        )

        if kelompok is not None:
            await session.execute(
                text(
                    "INSERT INTO unanswered (id, pertanyaan, top_score, created_at, resolved,"
                    " message_id) VALUES (:id, :q, :skor, :t, :resolved, :mid)"
                ),
                {
                    "id": uuid.uuid4(),
                    "q": pertanyaan,
                    "skor": top_score,
                    "t": t,
                    "resolved": kelompok.resolved,
                    "mid": jawaban_id,
                },
            )

        peluang = {"answer": 0.45, "refusal": 0.25}.get(jenis, 0.0)
        if rng.random() < peluang:
            membantu = rng.random() < (0.82 if jenis == "answer" else 0.3)
            await session.execute(
                text(
                    "INSERT INTO feedback (id, message_id, helpful, created_at)"
                    " VALUES (:id, :mid, :helpful, :t)"
                ),
                {
                    "id": uuid.uuid4(),
                    "mid": jawaban_id,
                    "helpful": membantu,
                    "t": dijawab + timedelta(seconds=20),
                },
            )
            hitung["feedback"] += 1

        t = dijawab + timedelta(minutes=rng.randint(1, 5))


async def seed_log(maker: async_sessionmaker[AsyncSession]) -> None:
    rng = random.Random(20260915)
    async with maker() as session:
        ada = (
            await session.execute(
                text("SELECT 1 FROM conversations WHERE session_id LIKE :a LIMIT 1"),
                {"a": AWALAN_SESI + "%"},
            )
        ).scalar()
        if ada:
            print("Log contoh sudah ada. Jalankan --hapus dulu bila ingin membuat ulang.")
            return

        sekarang = datetime.now(UTC)
        hari_ini = sekarang.astimezone(ZONA_KAMPUS).date()
        hitung: Counter = Counter()
        for mundur in range(HARI_LOG - 1, -1, -1):
            tanggal = hari_ini - timedelta(days=mundur)
            rata2 = 14 if tanggal.weekday() < 5 else 5
            target = max(0, round(rng.gauss(rata2, 4)))
            dibuat = 0
            while dibuat < target:
                mulai = datetime(
                    tanggal.year,
                    tanggal.month,
                    tanggal.day,
                    rng.randint(7, 20),
                    rng.randint(0, 59),
                    tzinfo=ZONA_KAMPUS,
                )
                n = min(rng.choice((1, 1, 1, 2, 2, 3)), target - dibuat)
                dibuat += n
                if mulai > sekarang:
                    continue
                await tulis_percakapan(session, rng, mulai, n, hitung)
        await session.commit()

    print(
        f"Log contoh {HARI_LOG} hari: {hitung['answer']} dijawab, "
        f"{hitung['refusal']} ditolak, {hitung['support']} sensitif, "
        f"{hitung['feedback']} umpan balik"
    )


# --- Hapus -----------------------------------------------------------------


async def hapus(maker: async_sessionmaker[AsyncSession]) -> None:
    from app.storage import build_storage

    async with maker() as session:
        tak_terjawab = await session.execute(
            text(
                "DELETE FROM unanswered WHERE message_id IN ("
                " SELECT m.id FROM messages m JOIN conversations c ON c.id = m.conversation_id"
                " WHERE c.session_id LIKE :a)"
            ),
            {"a": AWALAN_SESI + "%"},
        )
        percakapan = await session.execute(
            text("DELETE FROM conversations WHERE session_id LIKE :a"),
            {"a": AWALAN_SESI + "%"},
        )
        berkas = (
            (
                await session.execute(
                    text("DELETE FROM documents WHERE uploaded_by = :p RETURNING file_path"),
                    {"p": PENANDA_DOKUMEN},
                )
            )
            .scalars()
            .all()
        )
        await session.commit()

    if berkas:
        storage = build_storage(get_settings())
        for key in berkas:
            await storage.delete(key)

    print(
        f"Dihapus: {percakapan.rowcount} percakapan contoh, "
        f"{tak_terjawab.rowcount} pertanyaan tak terjawab, {len(berkas)} dokumen contoh"
    )


async def jalankan(args: argparse.Namespace) -> int:
    engine = create_async_engine(str(get_settings().database_url))
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        if args.hapus:
            await hapus(maker)
        if args.dokumen:
            await seed_dokumen(maker)
        if args.log:
            await seed_log(maker)
    finally:
        await engine.dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Data contoh untuk dashboard admin.")
    parser.add_argument("--dokumen", action="store_true", help="4 PDF contoh lewat ingestion")
    parser.add_argument("--log", action="store_true", help=f"{HARI_LOG} hari log percakapan")
    parser.add_argument("--hapus", action="store_true", help="hapus seluruh data contoh")
    args = parser.parse_args()
    if not (args.dokumen or args.log or args.hapus):
        parser.error("pilih minimal satu: --dokumen, --log, --hapus")
    if get_settings().environment == "production":
        print("Menolak berjalan: ENVIRONMENT=production.")
        return 2
    return asyncio.run(jalankan(args))


if __name__ == "__main__":
    sys.exit(main())
