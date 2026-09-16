"""Rate limiting per session dan per IP (FR-9).

Dua kunci berbeda: session_id membatasi satu tab browser, IP membatasi satu
jaringan. Keduanya perlu -- session_id ada di localStorage dan gampang
dihapus, sementara IP kampus dipakai bersama ratusan mahasiswa sehingga tidak
boleh terlalu ketat sendirian.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from fastapi import Request


def session_key(request: Request) -> str:
    """Kunci limiter berbasis header session; jatuh ke IP bila tidak ada."""
    session_id = request.headers.get("X-Session-Id")
    if session_id:
        return f"session:{session_id}"
    return f"ip:{client_ip(request)}"


def client_ip(request: Request) -> str:
    """IP asli di balik reverse proxy Caddy.

    Hanya entri pertama `X-Forwarded-For` yang dipakai, dan hanya karena Caddy
    di depan menulis ulang header ini. Bila proxy diganti, tinjau kembali --
    header ini dapat dipalsukan jika tidak ada proxy tepercaya.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


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
