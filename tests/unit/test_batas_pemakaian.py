"""Batas laju (sesi, IP, kunci sematan), sumber IP, dan penghitung batas harian."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from starlette.requests import Request

from app.security.batas_harian import PenghitungHarian
from app.security.ratelimit import BatasLaju, SlidingWindowLimiter, client_ip, parse_batas


class Jam:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class TestParseBatas:
    @pytest.mark.parametrize(
        "teks,hasil",
        [
            ("20/minute", BatasLaju(20, 60)),
            (" 60/Minutes ", BatasLaju(60, 60)),
            ("5/second", BatasLaju(5, 1)),
            ("100/hour", BatasLaju(100, 3600)),
            ("3000/day", BatasLaju(3000, 86400)),
        ],
    )
    def test_bentuk_env(self, teks, hasil):
        assert parse_batas(teks) == hasil

    @pytest.mark.parametrize("teks", ["", "0", "0/minute"])
    def test_nol_berarti_tanpa_batas(self, teks):
        assert parse_batas(teks) is None

    @pytest.mark.parametrize(
        "teks", ["20", "20/menit", "dua puluh/minute", "-1/minute", "20/"]
    )
    def test_salah_ketik_ditolak(self, teks):
        """Diam-diam menjadi tanpa batas justru membuka celah yang ingin ditutup."""
        with pytest.raises(ValueError):
            parse_batas(teks)


class TestJendelaBergeser:
    @pytest.fixture
    def jam(self) -> Jam:
        return Jam()

    @pytest.fixture
    def limiter(self, jam) -> SlidingWindowLimiter:
        return SlidingWindowLimiter(clock=jam)

    def test_ditolak_setelah_penuh_dan_tahu_berapa_lama(self, limiter, jam):
        batas = [("ip:a", BatasLaju(2, 60))]
        assert limiter.coba(batas) is None
        jam.t += 10
        assert limiter.coba(batas) is None
        jam.t += 5
        # Yang tertua tercatat 15 detik lalu; terbuka lagi 45 detik lagi.
        assert limiter.coba(batas) == pytest.approx(45)

    def test_jendela_bergeser(self, limiter, jam):
        batas = [("ip:a", BatasLaju(1, 60))]
        assert limiter.coba(batas) is None
        jam.t += 60
        assert limiter.coba(batas) is None

    def test_yang_ditolak_tidak_menghabiskan_jatah(self, limiter, jam):
        """Pengunjung yang menunggu sesuai Retry-After harus benar-benar dilayani."""
        batas = [("ip:a", BatasLaju(1, 60))]
        limiter.coba(batas)
        for _ in range(5):
            jam.t += 1
            assert limiter.coba(batas) is not None
        jam.t = 1000.0 + 60
        assert limiter.coba(batas) is None

    def test_semua_kunci_atau_tidak_sama_sekali(self, limiter):
        penuh = ("sesi:x", BatasLaju(1, 60))
        limiter.coba([penuh])
        ip = ("ip:a", BatasLaju(1, 60))
        assert limiter.coba([ip, penuh]) is not None
        # IP-nya tidak ikut tercatat saat permintaan itu ditolak karena sesinya.
        assert limiter.coba([ip]) is None

    def test_kunci_berbeda_tidak_saling_memakan(self, limiter):
        assert limiter.coba([("ip:a", BatasLaju(1, 60))]) is None
        assert limiter.coba([("ip:b", BatasLaju(1, 60))]) is None

    def test_kunci_basi_dibersihkan(self, jam):
        limiter = SlidingWindowLimiter(clock=jam)
        limiter.SWEEP_ABOVE = 3
        for i in range(3):
            limiter.coba([(f"sesi:{i}", BatasLaju(1, 60))])
        jam.t += 61
        limiter.coba([("sesi:baru", BatasLaju(1, 60))])
        assert list(limiter._hits) == ["sesi:baru"]


def permintaan(headers: dict[str, str], peer: str = "10.0.0.1") -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "client": (peer, 1234),
        }
    )


class TestSumberIp:
    def test_xff_memakai_entri_terakhir(self):
        """Entri pertama ditulis pengirim; yang terakhir ditulis proxy kita."""
        r = permintaan({"X-Forwarded-For": "6.6.6.6, 203.0.113.9"})
        assert client_ip(r) == "203.0.113.9"

    def test_header_cloudflare(self):
        r = permintaan({"CF-Connecting-IP": "203.0.113.9", "X-Forwarded-For": "6.6.6.6"})
        assert client_ip(r, "CF-Connecting-IP") == "203.0.113.9"

    def test_tanpa_header_memakai_koneksi(self):
        assert client_ip(permintaan({}), "X-Forwarded-For") == "10.0.0.1"

    def test_header_kosong_berarti_koneksi_langsung(self):
        """API tanpa proxy: header apa pun dari pengirim diabaikan."""
        r = permintaan({"X-Forwarded-For": "6.6.6.6"})
        assert client_ip(r, "") == "10.0.0.1"


class TestPenghitungHarian:
    @pytest.fixture
    def sekarang(self) -> list[datetime]:
        return [datetime(2026, 9, 26, 10, 0, tzinfo=UTC)]

    @pytest.fixture
    def dimuat(self) -> list[str]:
        return []

    @pytest.fixture
    def penghitung(self, sekarang, dimuat) -> PenghitungHarian:
        async def muat(zona: str) -> tuple[int, datetime]:
            dimuat.append(zona)
            # Anggap tengah malam berikutnya 17.00 UTC, dan hari ini sudah ada 40.
            return 40, datetime(2026, 9, 26, 17, 0, tzinfo=UTC)

        return PenghitungHarian(muat, clock=lambda: sekarang[0])

    async def test_mulai_dari_log_lalu_dihitung_di_memori(self, penghitung, dimuat):
        assert await penghitung.tambah("Asia/Jakarta") == 41
        assert await penghitung.tambah("Asia/Jakarta") == 42
        assert dimuat == ["Asia/Jakarta"]

    async def test_dimuat_ulang_saat_hari_berganti(self, penghitung, dimuat, sekarang):
        await penghitung.tambah("Asia/Jakarta")
        sekarang[0] += timedelta(hours=7)
        assert await penghitung.tambah("Asia/Jakarta") == 41
        assert len(dimuat) == 2

    async def test_dimuat_ulang_saat_zona_diganti(self, penghitung, dimuat):
        await penghitung.tambah("Asia/Jakarta")
        await penghitung.tambah("Asia/Makassar")
        assert dimuat == ["Asia/Jakarta", "Asia/Makassar"]
