"""Rate limiting per sesi, per IP, dan per kunci sematan (FR-9).

Tiga kunci berbeda, karena masing-masing hanya menahan satu jenis masalah:

- sesi (`X-Session-Id`) membatasi satu tab. Menahan tab yang macet mengirim
  ulang, tetapi tidak menahan siapa pun yang sengaja: Postman cukup mengganti
  atau menghapus header ini.
- IP membatasi satu jaringan. Inilah yang menahan skrip dan Postman -- IP tidak
  dapat diganti semudah header. IP kampus dipakai bersama ratusan mahasiswa,
  jadi batasnya tidak boleh terlalu ketat.
- kunci sematan membatasi satu situs penyemat, supaya satu situs yang ramai
  (atau disalahgunakan) tidak menghabiskan kuota milik semua.

Semuanya disimpan di memori proses, sejalan dengan kill switch: cukup untuk
satu worker uvicorn, dan sengaja tanpa Redis (PRD §10).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from fastapi import Request


def client_ip(request: Request, header: str = "X-Forwarded-For") -> str:
    """IP pengunjung menurut sumber yang ditetapkan `CLIENT_IP_HEADER`.

    - `X-Forwarded-For` (bawaan, di balik SATU reverse proxy seperti Caddy):
      entri TERAKHIR, yaitu alamat yang dilihat proxy itu sendiri. Entri
      pertama ditulis pengirim dan dapat dipalsukan -- dengan entri acak di
      setiap permintaan, batas per IP tidak pernah tercapai.
    - nama header lain (`CF-Connecting-IP` di balik Cloudflare, `X-Real-IP`):
      nilainya apa adanya.
    - kosong: alamat koneksi TCP, untuk API yang langsung menghadap internet.

    Header apa pun hanya dapat dipercaya bila API TIDAK dapat dicapai tanpa
    melewati proxy yang menulisnya.
    """
    if header:
        nilai = request.headers.get(header)
        if nilai:
            if header.lower() == "x-forwarded-for":
                return nilai.split(",")[-1].strip() or "unknown"
            return nilai.strip()
    return request.client.host if request.client else "unknown"


_SATUAN_DETIK = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
}


@dataclass(frozen=True)
class BatasLaju:
    jumlah: int
    detik: float


def parse_batas(teks: str) -> BatasLaju | None:
    """`"20/minute"` -> 20 permintaan per 60 detik; kosong atau `0` -> tanpa batas.

    Satuan: second, minute, hour, day (boleh jamak). Bentuk ini sama dengan
    yang sudah tertulis di `.env` sejak awal (`RATE_LIMIT_PER_SESSION`).
    """
    t = teks.strip().lower()
    if t in ("", "0"):
        return None
    jumlah, _, satuan = t.partition("/")
    satuan = satuan.strip().removesuffix("s")
    if not jumlah.strip().isdigit() or satuan not in _SATUAN_DETIK:
        raise ValueError(
            f"batas laju {teks!r} tidak sah; tulis seperti 20/minute "
            "(satuan: second, minute, hour, day), atau 0 untuk mematikan"
        )
    angka = int(jumlah)
    return None if angka == 0 else BatasLaju(angka, float(_SATUAN_DETIK[satuan]))


class SlidingWindowLimiter:
    """Batas laju jendela bergeser untuk beberapa kunci sekaligus.

    Satu permintaan dicatat di SEMUA kuncinya, atau tidak sama sekali: yang
    ditolak tidak ikut menghabiskan jatah, jadi pengunjung yang menunggu sesuai
    `Retry-After` benar-benar dilayani setelahnya.
    """

    SWEEP_ABOVE = 10_000
    """Bersihkan kunci basi bila jumlah kunci melewati ini, supaya sesi dan IP
    acak dari pemindai tidak membuat memori tumbuh tanpa batas."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._hits: dict[str, tuple[float, deque[float]]] = {}

    def coba(self, batas: Sequence[tuple[str, BatasLaju]]) -> float | None:
        """Catat satu permintaan, atau kembalikan detik tunggu bila ada kunci yang penuh."""
        now = self._clock()
        tunggu = 0.0
        for kunci, b in batas:
            hits = self._prune(kunci, now, b.detik)
            if hits is not None and len(hits) >= b.jumlah:
                # Terbuka lagi saat permintaan ke-(len - jumlah + 1) dari yang
                # tertua keluar dari jendela.
                penentu = hits[len(hits) - b.jumlah]
                tunggu = max(tunggu, b.detik - (now - penentu))
        if tunggu > 0:
            return tunggu
        if len(self._hits) >= self.SWEEP_ABOVE:
            self._sweep(now)
        for kunci, b in batas:
            self._hits.setdefault(kunci, (b.detik, deque()))[1].append(now)
        return None

    def _prune(self, kunci: str, now: float, jendela: float) -> deque[float] | None:
        entri = self._hits.get(kunci)
        if entri is None:
            return None
        hits = entri[1]
        while hits and now - hits[0] >= jendela:
            hits.popleft()
        if not hits:
            del self._hits[kunci]
            return None
        return hits

    def _sweep(self, now: float) -> None:
        for kunci, (jendela, _) in list(self._hits.items()):
            self._prune(kunci, now, jendela)


_chat_limiter = SlidingWindowLimiter()


def get_chat_limiter() -> SlidingWindowLimiter:
    """Dependency FastAPI untuk endpoint mahasiswa; instans tunggal per proses.
    Di-override di test."""
    return _chat_limiter


class FailureLimiter:
    """Batasi percobaan GAGAL per kunci dalam jendela waktu bergeser (AD-1).

    Hanya kegagalan yang dihitung: admin yang salah ketik sekali lalu berhasil
    tidak ikut terkunci, dan keberhasilan mengosongkan hitungannya.

    Disimpan di memori proses, sejalan dengan kill switch: cukup untuk satu
    worker uvicorn, dan sengaja tanpa Redis (PRD §10).
    """

    SWEEP_ABOVE = 10_000
    """Bersihkan kunci basi bila jumlah kunci melewati ini, supaya IP acak dari
    pemindai tidak membuat memori tumbuh tanpa batas."""

    def __init__(
        self,
        max_failures: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_failures < 1:
            raise ValueError(f"max_failures harus >= 1, diberi {max_failures}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds harus > 0, diberi {window_seconds}")
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self._clock = clock
        self._failures: dict[str, deque[float]] = {}

    def retry_after(self, key: str) -> float | None:
        """Detik sampai kunci boleh mencoba lagi, atau None bila tidak diblokir."""
        now = self._clock()
        hits = self._prune(key, now)
        if hits is None or len(hits) < self.max_failures:
            return None
        # Kunci terbuka lagi saat jumlah kegagalan turun di bawah batas, yaitu
        # ketika kegagalan ke-(len - max + 1) dari yang tertua kedaluwarsa.
        penentu = hits[len(hits) - self.max_failures]
        return self.window_seconds - (now - penentu)

    def record_failure(self, key: str) -> None:
        now = self._clock()
        hits = self._prune(key, now)
        if hits is None:
            if len(self._failures) >= self.SWEEP_ABOVE:
                self._sweep(now)
            hits = self._failures.setdefault(key, deque())
        hits.append(now)

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)

    def _prune(self, key: str, now: float) -> deque[float] | None:
        hits = self._failures.get(key)
        if hits is None:
            return None
        while hits and now - hits[0] >= self.window_seconds:
            hits.popleft()
        if not hits:
            del self._failures[key]
            return None
        return hits

    def _sweep(self, now: float) -> None:
        for key in list(self._failures):
            self._prune(key, now)


LOGIN_MAX_FAILURES = 10
LOGIN_WINDOW_SECONDS = 15 * 60

_login_limiter = FailureLimiter(LOGIN_MAX_FAILURES, LOGIN_WINDOW_SECONDS)


def get_login_limiter() -> FailureLimiter:
    """Dependency FastAPI; instans tunggal per proses. Di-override di test."""
    return _login_limiter
