"""Penulis log aplikasi ke SQLite, perekam node LangGraph, dan konfigurasi logging.

Jalur tulisnya: kode aplikasi -> antrean di memori -> satu thread latar belakang
-> `LogStore.tulis` per batch. Permintaan mahasiswa tidak pernah menunggu disk,
dan kegagalan menulis log tidak pernah menggagalkan jawaban -- prinsip yang sama
dengan `chatlog.py`. Antrean yang penuh membuang baris baru alih-alih menahan
permintaan: kehilangan log lebih murah daripada layanan yang tersendat.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler

from app.observability.costs import try_estimate_cost
from app.observability.logstore import LogStore, waktu_iso

logger = logging.getLogger(__name__)

_internal = logging.getLogger("logstore")
"""Galat penulis sendiri. Sengaja di luar hierarki `app`: bila dicatat ke
`app.*`, galat menulis SQLite akan masuk antrean SQLite yang sama."""

turn_id_var: ContextVar[str | None] = ContextVar("turn_id", default=None)
"""Giliran chat yang sedang berjalan. Log yang muncul di tengah giliran ikut
membawa `turn_id`, sehingga dashboard bisa melompat dari error ke gilirannya."""

FORMAT_KONSOL = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
PESAN_MAKS = 500


class LogWriter:
    """Antrean + satu thread yang menulis ke SQLite dan membuang baris lama."""

    def __init__(
        self,
        store: LogStore,
        *,
        retention_days: int,
        max_antrean: int = 10_000,
        batch: int = 500,
        jeda_hapus_detik: float = 3600,
    ) -> None:
        self.store = store
        self.retention = timedelta(days=retention_days)
        self.batch = batch
        self.jeda_hapus = jeda_hapus_detik
        self.dibuang = 0
        """Baris yang dibuang karena antrean penuh."""
        self._antrean: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue(
            max_antrean
        )
        self._thread: threading.Thread | None = None

    def kirim(self, jenis: str, data: dict[str, Any]) -> None:
        """Titipkan satu baris; tidak pernah memblokir dan tidak pernah melempar."""
        try:
            self._antrean.put_nowait((jenis, data))
        except queue.Full:
            self.dibuang += 1

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._jalan, name="log-writer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Tulis sisa antrean lalu berhenti."""
        if self._thread is None:
            return
        self._antrean.put(None)
        self._thread.join(timeout)
        self._thread = None

    def _jalan(self) -> None:
        hapus_berikutnya = 0.0
        berhenti = False
        while not berhenti:
            if time.monotonic() >= hapus_berikutnya:
                self._hapus_lama()
                hapus_berikutnya = time.monotonic() + self.jeda_hapus
            try:
                item = self._antrean.get(timeout=1.0)
            except queue.Empty:
                continue
            batch: list[tuple[str, dict[str, Any]]] = []
            while True:
                if item is None:
                    berhenti = True
                    break
                batch.append(item)
                if len(batch) >= self.batch:
                    break
                try:
                    item = self._antrean.get_nowait()
                except queue.Empty:
                    break
            self._tulis(batch)

    def _tulis(self, batch: list[tuple[str, dict[str, Any]]]) -> None:
        if not batch:
            return
        try:
            self.store.tulis(batch)
        except Exception:
            _internal.exception(
                "Gagal menulis %d baris log ke %s", len(batch), self.store.path
            )

    def _hapus_lama(self) -> None:
        try:
            self.store.hapus_lama(datetime.now(UTC) - self.retention)
        except Exception:
            _internal.exception("Gagal menghapus log lama di %s", self.store.path)


class SQLiteLogHandler(logging.Handler):
    """`logging.Handler` yang meneruskan record ke `LogWriter`."""

    def __init__(self, writer: LogWriter, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.writer = writer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tb = None
            if record.exc_info:
                tb = "".join(traceback.format_exception(*record.exc_info))
            self.writer.kirim(
                "app",
                {
                    "waktu": waktu_iso(datetime.fromtimestamp(record.created, UTC)),
                    "level": record.levelname,
                    "levelno": record.levelno,
                    "logger": record.name,
                    "pesan": record.getMessage(),
                    "lokasi": f"{record.module}:{record.lineno}",
                    "traceback": tb,
                    # Dibaca saat emit, yaitu di task yang menulis log -- di situ
                    # contextvar giliran masih terpasang.
                    "turn_id": turn_id_var.get(),
                },
            )
        except Exception:
            self.handleError(record)


_TANDA = "_pandu_applog"
"""Penanda handler buatan modul ini, supaya konfigurasi ulang (mis. reload
uvicorn di proses yang sama) mengganti handler lama alih-alih menumpuknya."""


def configure_logging(level: str, writer: LogWriter | None) -> None:
    """Pasang handler untuk logger `app`: konsol berformat lengkap + SQLite.

    Tanpa ini logger `app.*` jatuh ke handler cadangan Python: INFO hilang, dan
    WARNING tercetak tanpa waktu, level, maupun nama logger. Hanya `app` yang
    disentuh; `uvicorn.*` tetap memakai konfigurasinya sendiri, dan `propagate`
    dibiarkan True supaya `caplog` pytest tetap bekerja.
    """
    app_logger = logging.getLogger("app")
    for h in list(app_logger.handlers):
        if getattr(h, _TANDA, False):
            app_logger.removeHandler(h)
            h.close()

    konsol = logging.StreamHandler(sys.stderr)
    konsol.setFormatter(logging.Formatter(FORMAT_KONSOL))
    handlers: list[logging.Handler] = [konsol]
    if writer is not None:
        handlers.append(SQLiteLogHandler(writer))
    for h in handlers:
        setattr(h, _TANDA, True)
        app_logger.addHandler(h)
    app_logger.setLevel(level.upper())


_writer_aktif: LogWriter | None = None


def mulai_log(settings: Any) -> LogWriter:
    """Nyalakan penulis SQLite dan konfigurasi logging. Dipanggil di lifespan."""
    global _writer_aktif
    hentikan_log()
    writer = LogWriter(
        LogStore(settings.log_db_path), retention_days=settings.log_retention_days
    )
    writer.start()
    configure_logging(settings.log_level, writer)
    _writer_aktif = writer
    return writer


def hentikan_log() -> None:
    global _writer_aktif
    if _writer_aktif is not None:
        _writer_aktif.stop()
        _writer_aktif = None


def writer_aktif() -> LogWriter | None:
    """Penulis yang sedang berjalan, atau None (mis. di test yang tanpa lifespan)."""
    return _writer_aktif


# --- Node LangGraph --------------------------------------------------------


def _nilai(x: Any) -> Any:
    return getattr(x, "value", x)


def ringkas_node(node: str, keluaran: Any) -> dict[str, Any]:
    """Isi kolom `detail` untuk satu node. Tidak pernah memuat teks mahasiswa."""
    out = keluaran if isinstance(keluaran, dict) else {}
    if node == "sanitize":
        return {"panjang": len(out.get("clean") or "")}
    if node == "sensitive":
        s = out.get("sensitivity")
        if s is None:
            return {}
        return {"level": _nilai(s.level), "dialihkan": bool(s.bypasses_rag)}
    if node == "smalltalk":
        return {"ditangani": "outcome" in out}
    if node == "jev_gate":
        g = out.get("gate")
        if g is None:
            return {"dimatikan": True}
        return {
            "label": _nilai(g.label),
            "confidence": g.confidence,
            "blocked": g.blocked,
            "biaya_usd": g.cost_usd,
            "error": g.error,
        }
    if node == "rewrite":
        return {"query_berubah": out.get("rewritten") is not None}
    if node == "retrieve":
        return {"jumlah_dokumen": len(out.get("documents") or [])}
    if node == "validate_context":
        d = out.get("decision")
        if d is None:
            return {}
        return {
            "keputusan": _nilai(d.decision),
            "alasan": _nilai(d.reason),
            "top_score": d.top_score,
            "top_rerank_score": d.top_rerank_score,
        }
    if node == "refuse":
        o = out.get("outcome")
        return {"jumlah_kontak": len(o.contacts) if o is not None else 0}
    return {}


@dataclass
class _NodeBerjalan:
    node: str
    urutan: int
    mulai: datetime
    t0: float


class NodeRecorder(AsyncCallbackHandler):
    """Callback LangGraph: satu catatan per node, disimpan di memori.

    Nama node dibaca dari `metadata.langgraph_node`. Router bersyarat
    (`selesai_atau_*`, `route_context`) juga membawa metadata itu -- berisi node
    induknya -- tetapi namanya berbeda, jadi ia dilewati. Catatan baru ditulis
    oleh `Giliran` saat giliran selesai, sekaligus dengan baris `turns`-nya.
    """

    run_inline = True

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self._berjalan: dict[uuid.UUID, _NodeBerjalan] = {}
        self._urutan = 0

    async def on_chain_start(
        self,
        serialized: Any,
        inputs: Any,
        *,
        run_id: uuid.UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node = (metadata or {}).get("langgraph_node")
        if not node or kwargs.get("name") != node:
            return
        self._urutan += 1
        self._berjalan[run_id] = _NodeBerjalan(
            node, self._urutan, datetime.now(UTC), time.perf_counter()
        )

    async def on_chain_end(self, outputs: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        jalan = self._berjalan.pop(run_id, None)
        if jalan is None:
            return
        try:
            detail = ringkas_node(jalan.node, outputs)
        except Exception:
            detail = {}
        self._simpan(jalan, "ok", detail=detail)

    async def on_chain_error(
        self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any
    ) -> None:
        jalan = self._berjalan.pop(run_id, None)
        if jalan is None:
            return
        self._simpan(
            jalan,
            "error",
            error_tipe=type(error).__name__,
            error_pesan=str(error)[:PESAN_MAKS],
        )

    def _simpan(self, jalan: _NodeBerjalan, status: str, **lain: Any) -> None:
        self.nodes.append(
            {
                "node": jalan.node,
                "urutan": jalan.urutan,
                "mulai": waktu_iso(jalan.mulai),
                "durasi_ms": round((time.perf_counter() - jalan.t0) * 1000, 2),
                "status": status,
                "error_tipe": None,
                "error_pesan": None,
                "detail": {},
                **lain,
            }
        )

    @property
    def node_terakhir(self) -> str | None:
        if not self.nodes:
            return None
        return max(self.nodes, key=lambda n: n["urutan"])["node"]


# --- Satu giliran chat ------------------------------------------------------


@dataclass
class Giliran:
    """Pengumpul catatan satu giliran chat; ditulis sekali saat `with` selesai."""

    sink: Any
    endpoint: str
    session_id: str | None
    unit: str | None
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    recorder: NodeRecorder = field(default_factory=NodeRecorder)
    waktu: datetime = field(default_factory=lambda: datetime.now(UTC))
    t0: float = field(default_factory=time.perf_counter)
    ttft_ms: int | None = None
    hasil: str | None = None
    message_id: str | None = None
    langsmith_run_id: str | None = None
    llm_call: Any = None

    def token_pertama(self) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = round((time.perf_counter() - self.t0) * 1000)

    def selesai(
        self,
        *,
        hasil: Any,
        message_id: str | None,
        langsmith_run_id: str | None,
        llm_call: Any = None,
    ) -> None:
        self.hasil = _nilai(hasil)
        self.message_id = message_id
        self.langsmith_run_id = langsmith_run_id
        self.llm_call = llm_call

    def _detail_generate(self) -> dict[str, Any]:
        model = getattr(self.llm_call, "model", None)
        usage = getattr(self.llm_call, "usage", None) or {}
        masuk, keluar = usage.get("input_tokens"), usage.get("output_tokens")
        biaya = None
        if model and masuk is not None and keluar is not None:
            estimasi = try_estimate_cost(model, int(masuk), int(keluar))
            biaya = estimasi.usd if estimasi else None
        return {
            "model": model,
            "input_tokens": masuk,
            "output_tokens": keluar,
            "biaya_usd": biaya,
        }

    def tulis(self, status: str) -> None:
        if self.sink is None:
            return
        try:
            for n in self.recorder.nodes:
                if n["node"] == "generate" and n["status"] == "ok":
                    n["detail"] = {**n["detail"], **self._detail_generate()}
                self.sink.kirim("node", {"turn_id": self.turn_id, **n})
            self.sink.kirim(
                "turn",
                {
                    "turn_id": self.turn_id,
                    "waktu": waktu_iso(self.waktu),
                    "endpoint": self.endpoint,
                    "session_id": self.session_id,
                    "message_id": self.message_id,
                    "unit": self.unit,
                    "hasil": self.hasil,
                    "node_terakhir": self.recorder.node_terakhir,
                    "total_ms": round((time.perf_counter() - self.t0) * 1000),
                    "ttft_ms": self.ttft_ms,
                    "status": status,
                    "langsmith_run_id": self.langsmith_run_id,
                },
            )
        except Exception:
            _internal.exception("Gagal mencatat giliran %s", self.turn_id)


@contextlib.contextmanager
def catat_giliran(
    sink: Any, *, endpoint: str, session_id: str | None, unit: str | None
) -> Iterator[Giliran]:
    """Buka satu giliran: pasang `turn_id` untuk log, tulis ringkasannya di akhir.

    Status `ok` bila blok selesai normal, `dibatalkan` bila task dibatalkan
    (mahasiswa menutup panel di tengah streaming), `error` untuk galat lainnya --
    yang juga dicatat ke log bersama traceback-nya, supaya error itu dapat
    ditemukan dari gilirannya dan sebaliknya.
    """
    giliran = Giliran(sink=sink, endpoint=endpoint, session_id=session_id, unit=unit)
    token = turn_id_var.set(giliran.turn_id)
    try:
        yield giliran
    except BaseException as exc:
        if isinstance(exc, Exception):
            logger.exception("Giliran chat gagal")
            giliran.tulis("error")
        else:
            giliran.tulis("dibatalkan")
        raise
    else:
        giliran.tulis("ok")
    finally:
        turn_id_var.reset(token)
