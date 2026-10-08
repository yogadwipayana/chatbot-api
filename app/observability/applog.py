"""Penulis log aplikasi ke SQLite, perekam node LangGraph, dan konfigurasi logging.

Jalur tulisnya: kode aplikasi -> antrean di memori -> satu thread latar belakang
-> `LogStore.tulis` per batch. Permintaan mahasiswa tidak pernah menunggu disk,
dan kegagalan menulis log tidak pernah menggagalkan jawaban -- prinsip yang sama
dengan `chatlog.py`. Antrean yang penuh membuang baris baru alih-alih menahan
permintaan: kehilangan log lebih murah daripada layanan yang tersendat.
"""

from __future__ import annotations

import contextlib
import dataclasses
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
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler

from app.observability.costs import try_estimate_cost
from app.observability.logstore import LogStore, waktu_iso
from app.observability.rekaman import ACARA_PANGGILAN

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
                    "timestamp": waktu_iso(datetime.fromtimestamp(record.created, UTC)),
                    "level": record.levelname,
                    "levelno": record.levelno,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "location": f"{record.module}:{record.lineno}",
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
        return {"length": len(out.get("clean") or "")}
    if node == "sensitive":
        s = out.get("sensitivity")
        if s is None:
            return {}
        return {"level": _nilai(s.level), "redirected": bool(s.bypasses_rag)}
    if node == "smalltalk":
        return {"handled": "outcome" in out}
    if node == "rule_gate":
        g = out.get("gate")
        return {"blocked": False} if g is None else {"blocked": True, "label": _nilai(g.label)}
    if node == "jev_gate":
        g = out.get("gate")
        if g is None:
            return {"disabled": True}
        return {
            "label": _nilai(g.label),
            "confidence": g.confidence,
            "blocked": g.blocked,
            "cost_usd": g.cost_usd,
            "error": g.error,
        }
    if node == "rewrite":
        return {"query_rewritten": out.get("rewritten") is not None}
    if node == "retrieve":
        return {"document_count": len(out.get("documents") or [])}
    if node == "validate_context":
        d = out.get("decision")
        if d is None:
            return {}
        return {
            "decision": _nilai(d.decision),
            "reason": _nilai(d.reason),
            "top_score": d.top_score,
            "top_rerank_score": d.top_rerank_score,
        }
    if node == "refuse":
        o = out.get("outcome")
        return {"contact_count": len(o.contacts) if o is not None else 0}
    return {}


@dataclass
class _NodeBerjalan:
    node: str
    urutan: int
    mulai: datetime
    t0: float


@dataclass
class _Panggilan:
    """Satu panggilan di dalam node: LLM, retriever, rantai, tool, atau JEV."""

    id: str
    induk: str | None
    """ID panggilan induk; None bila tergantung langsung di bawah node."""
    posisi: int
    """Posisi node pemiliknya (`node_runs.position`)."""
    nama: str
    jenis: str
    mulai: datetime
    t0: float
    masukan: Any = None
    keluaran: Any = None
    status: str = "berjalan"
    durasi_ms: float | None = None
    galat: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None


WRAPPER_NODES = frozenset({"cari"})
"""Node subgraph yang hanya membungkus langkah lain (`app.rag.graph.SEARCH_NODE`).

Durasinya jumlah langkah di dalamnya; mencatatnya membuat langkah yang sama
terhitung dua kali di halaman Log, dengan nama yang tidak dikenal dashboard."""


@lru_cache(maxsize=1)
def nama_router() -> frozenset[str]:
    """Nama fungsi sisi bersyarat graf (`selesai_atau_*`, `route_context`).

    Dibaca dari graf yang terkompilasi, bukan ditulis ulang di sini, supaya
    router baru tidak diam-diam tercatat sebagai panggilan."""
    from app.rag.graph import build_graph

    return frozenset(
        nama for cabang in build_graph().builder.branches.values() for nama in cabang
    )


def _salin(x: Any) -> Any:
    """Salinan dangkal: loop tool menambah ke daftar pesan yang sama setelah
    giliran LLM pertama, dan tanpa salinan input giliran itu ikut berubah."""
    if isinstance(x, dict):
        return dict(x)
    if isinstance(x, list | tuple):
        return list(x)
    return x


class NodeRecorder(AsyncCallbackHandler):
    """Callback LangGraph: satu catatan per node, disimpan di memori.

    Nama node dibaca dari `metadata.langgraph_node`. Router bersyarat
    (`selesai_atau_*`, `route_context`) juga membawa metadata itu -- berisi node
    induknya -- tetapi namanya berbeda, jadi ia tidak dihitung sebagai node.
    Catatan baru ditulis oleh `Giliran` saat giliran selesai, sekaligus dengan
    baris `turns`-nya.

    Dengan `rekam_io` (LOG_NODE_IO) perekam juga menyimpan bahan tab Graf
    (`rekaman.py`): state yang diterima dan dikembalikan setiap node, keluaran
    router (rute yang diambil), dan pohon panggilan di dalam node. Panggilan
    LLM dan retriever datang dari callback LangChain; JEV dan tool datang lewat
    custom event (`rekaman.catat_panggilan`). Template prompt dilewati: isinya
    sama dengan pesan yang diterima LLM.
    """

    run_inline = True

    def __init__(self, *, rekam_io: bool = False) -> None:
        self.rekam_io = rekam_io
        self.nodes: list[dict[str, Any]] = []
        self._berjalan: dict[uuid.UUID, _NodeBerjalan] = {}
        self._urutan = 0
        self._penentu: str | None = None
        """Node yang mengisi `outcome` -- tempat alur benar-benar berhenti."""
        self.hasil_graf: Any = None
        """`outcome` di state akhir graf; None bila giliran gagal atau dibatalkan."""

        self._akar: uuid.UUID | None = None
        self.masukan_graf: Any = None
        self._pemilik: dict[uuid.UUID, tuple[int, str | None]] = {}
        self._posisi_node: dict[str, int] = {}
        """Nama node -> posisi terakhirnya, untuk custom event yang induknya tidak dikenal."""
        """run_id -> (posisi node, ID panggilan induk). Node sendiri: (posisi, None)."""
        self._router: dict[uuid.UUID, int] = {}
        self._io: dict[int, dict[str, Any]] = {}
        self._panggilan: dict[str, _Panggilan] = {}

    # --- Node dan rantai -------------------------------------------------

    async def on_chain_start(
        self,
        serialized: Any,
        inputs: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node = (metadata or {}).get("langgraph_node")
        nama = kwargs.get("name")
        if node is None:
            # Akar graf. `parent_run_id` tidak dipakai untuk mengenalinya: saat
            # tracing menyala, induknya adalah akar trace LangSmith.
            if self._akar is None:
                self._akar = run_id
                if self.rekam_io:
                    self.masukan_graf = _salin(inputs)
            return
        if nama == node and node not in WRAPPER_NODES:
            self._urutan += 1
            self._berjalan[run_id] = _NodeBerjalan(
                node, self._urutan, datetime.now(UTC), time.perf_counter()
            )
            if self.rekam_io:
                self._pemilik[run_id] = (self._urutan, None)
                self._posisi_node[node] = self._urutan
                self._io[self._urutan] = {"node": node, "input": _salin(inputs)}
            return
        if self.rekam_io:
            self._mulai_panggilan(
                run_id, parent_run_id, nama, inputs, tags=tags, run_type=kwargs.get("run_type")
            )

    async def on_chain_end(self, outputs: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        jalan = self._berjalan.pop(run_id, None)
        if jalan is not None:
            if isinstance(outputs, dict) and outputs.get("outcome") is not None:
                self._penentu = jalan.node
            try:
                detail = ringkas_node(jalan.node, outputs)
            except Exception:
                detail = {}
            self._simpan(jalan, "ok", detail=detail)
            if self.rekam_io:
                self._io[jalan.urutan]["output"] = _salin(outputs)
            return
        if run_id == self._akar:
            if isinstance(outputs, dict):
                self.hasil_graf = outputs.get("outcome")
            return
        posisi = self._router.pop(run_id, None)
        if posisi is not None:
            self._io[posisi]["route"] = outputs
            return
        self._akhiri_panggilan(run_id, outputs)

    async def on_chain_error(
        self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any
    ) -> None:
        jalan = self._berjalan.pop(run_id, None)
        if jalan is None:
            self._router.pop(run_id, None)
            self._akhiri_panggilan(run_id, None, galat=error)
            return
        self._simpan(
            jalan,
            "error",
            error_type=type(error).__name__,
            error_message=str(error)[:PESAN_MAKS],
        )

    def _simpan(self, jalan: _NodeBerjalan, status: str, **lain: Any) -> None:
        self.nodes.append(
            {
                "node": jalan.node,
                "position": jalan.urutan,
                "started_at": waktu_iso(jalan.mulai),
                "duration_ms": round((time.perf_counter() - jalan.t0) * 1000, 2),
                "status": status,
                "error_type": None,
                "error_message": None,
                "detail": {},
                **lain,
            }
        )

    # --- Panggilan di dalam node (hanya bila rekam_io) --------------------

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self.rekam_io:
            return
        params = kwargs.get("invocation_params") or {}
        self._mulai_panggilan(
            run_id,
            parent_run_id,
            kwargs.get("name") or _nama_serial(serialized, "llm"),
            list(messages[0]) if messages else [],
            tags=tags,
            jenis="llm",
            model=(
                params.get("model")
                or params.get("model_name")
                or (metadata or {}).get("ls_model_name")
            ),
        )

    async def on_llm_end(self, response: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        p = self._panggilan.get(str(run_id))
        if p is None:
            return
        generasi = (response.generations or [[None]])[0] or [None]
        pesan = getattr(generasi[0], "message", None)
        usage = getattr(pesan, "usage_metadata", None)
        p.usage = dict(usage) if usage else None
        keluaran = pesan if pesan is not None else getattr(generasi[0], "text", None)
        self._akhiri_panggilan(run_id, keluaran)

    async def on_llm_error(
        self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any
    ) -> None:
        self._akhiri_panggilan(run_id, None, galat=error)

    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self.rekam_io:
            return
        self._mulai_panggilan(
            run_id,
            parent_run_id,
            kwargs.get("name") or _nama_serial(serialized, "retriever"),
            {"query": query},
            tags=tags,
            jenis="retriever",
        )

    async def on_retriever_end(
        self, documents: Any, *, run_id: uuid.UUID, **kwargs: Any
    ) -> None:
        self._akhiri_panggilan(run_id, _salin(documents))

    async def on_retriever_error(
        self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any
    ) -> None:
        self._akhiri_panggilan(run_id, None, galat=error)

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: uuid.UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Panggilan JEV dan tool (`rekaman.catat_panggilan`): sudah selesai saat tiba.

        Induknya dicari lewat `run_id`, lalu lewat `metadata.langgraph_node`. Yang
        kedua dibutuhkan JEV saat tracing menyala: `JevGate.__call__` dibungkus
        `@traceable`, dan langchain-core lalu memakai run LangSmith `jev_classify`
        sebagai induk event -- ID yang tidak pernah lewat callback ini.
        """
        if not self.rekam_io or name != ACARA_PANGGILAN or not isinstance(data, dict):
            return
        pemilik = self._pemilik.get(run_id)
        if pemilik is None:
            posisi_node = self._posisi_node.get((metadata or {}).get("langgraph_node", ""))
            pemilik = (posisi_node, None) if posisi_node is not None else None
        if pemilik is None:
            return
        posisi, induk = pemilik
        durasi = float(data.get("duration_ms") or 0.0)
        akhir = data.get("ended_at") or datetime.now(UTC)
        galat = data.get("error")
        p = _Panggilan(
            id=str(uuid.uuid4()),
            induk=induk,
            posisi=posisi,
            nama=str(data.get("name") or "panggilan"),
            jenis=str(data.get("kind") or "chain"),
            mulai=akhir - timedelta(milliseconds=durasi),
            t0=0.0,
            masukan=data.get("input"),
            keluaran=data.get("output"),
            status="error" if galat else "ok",
            durasi_ms=round(durasi, 2),
            galat=str(galat)[:PESAN_MAKS] if galat else None,
        )
        self._panggilan[p.id] = p

    def _mulai_panggilan(
        self,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None,
        nama: str | None,
        masukan: Any,
        *,
        tags: list[str] | None,
        run_type: str | None = None,
        jenis: str = "chain",
        model: str | None = None,
    ) -> None:
        pemilik = self._pemilik.get(parent_run_id) if parent_run_id else None
        if pemilik is None:
            return
        if run_type == "prompt" or "langsmith:hidden" in (tags or ()):
            return
        posisi, induk = pemilik
        if induk is None and jenis == "chain" and nama in nama_router():
            self._router[run_id] = posisi
            return
        p = _Panggilan(
            id=str(run_id),
            induk=induk,
            posisi=posisi,
            nama=nama or jenis,
            jenis=jenis,
            mulai=datetime.now(UTC),
            t0=time.perf_counter(),
            masukan=_salin(masukan),
            model=model,
        )
        self._panggilan[p.id] = p
        self._pemilik[run_id] = (posisi, p.id)

    def _akhiri_panggilan(
        self, run_id: uuid.UUID, keluaran: Any, *, galat: BaseException | None = None
    ) -> None:
        p = self._panggilan.get(str(run_id))
        if p is None or p.status != "berjalan":
            return
        p.keluaran = keluaran
        p.durasi_ms = round((time.perf_counter() - p.t0) * 1000, 2)
        p.status = "error" if galat is not None else "ok"
        if galat is not None:
            p.galat = f"{type(galat).__name__}: {galat}"[:PESAN_MAKS]

    # --- Hasil -----------------------------------------------------------

    @property
    def node_terakhir(self) -> str | None:
        """Tempat alur berhenti: node yang mengisi `outcome`.

        Bukan sekadar node yang paling akhir dimulai: gerbang JEV berjalan
        paralel dengan pencarian, dan pesan yang diblokirnya tetap melewati
        `validate_context` sebagai titik temu. Tanpa `outcome` (galat atau
        dibatalkan), node terakhir yang dimulai yang dilaporkan.
        """
        if self._penentu is not None:
            return self._penentu
        if not self.nodes:
            return None
        return max(self.nodes, key=lambda n: n["position"])["node"]

    @property
    def teks_bersih(self) -> str | None:
        """Pertanyaan setelah sanitasi, dari keluaran node `sanitize`."""
        for io in self._io.values():
            if io.get("node") == "sanitize" and isinstance(io.get("output"), dict):
                return io["output"].get("clean")
        return None

    def rekaman(self) -> dict[str, Any]:
        """Isi mentah blob `traces` (diserialisasi `rekaman.kemas` di thread penulis)."""
        return {
            "input": self.masukan_graf,
            "output": self.hasil_graf,
            "nodes": [
                {
                    "position": posisi,
                    "node": io.get("node"),
                    "input": io.get("input"),
                    "output": io.get("output"),
                    "route": io.get("route"),
                }
                for posisi, io in sorted(self._io.items())
            ],
            "calls": [_panggilan_keluar(p) for p in _ratakan(list(self._panggilan.values()))],
        }


def _nama_serial(serialized: Any, cadangan: str) -> str:
    s = serialized if isinstance(serialized, dict) else {}
    if s.get("name"):
        return str(s["name"])
    ident = s.get("id")
    return str(ident[-1]) if isinstance(ident, list) and ident else cadangan


def _ratakan(daftar: list[_Panggilan]) -> list[_Panggilan]:
    """Satukan rantai pembungkus dengan satu-satunya LLM di dalamnya.

    `answer_prompt() | llm` yang dipanggil dengan `run_name="generate_answer"`
    menjadi dua run: rantai bernama generate_answer (inputnya variabel template)
    dan ChatOpenAI di dalamnya (inputnya pesan sungguhan; template prompt sudah
    dilewati). LangSmith menampilkan keduanya; di sini cukup satu panggilan LLM
    bernama generate_answer, dengan pesan, model, dan token dari LLM-nya.
    """
    anak: dict[str | None, list[_Panggilan]] = {}
    for p in daftar:
        anak.setdefault(p.induk, []).append(p)
    buang: set[str] = set()
    hasil: list[_Panggilan] = []
    for p in daftar:
        isi = anak.get(p.id, [])
        tunggal = isi[0] if len(isi) == 1 else None
        if (
            p.jenis == "chain"
            and tunggal
            and tunggal.jenis == "llm"
            and not anak.get(tunggal.id)
        ):
            gagal = tunggal.status == "error" and p.status != "error"
            p = dataclasses.replace(
                p,
                jenis="llm",
                masukan=tunggal.masukan,
                keluaran=tunggal.keluaran,
                model=tunggal.model,
                usage=tunggal.usage,
                status="error" if gagal else p.status,
                galat=tunggal.galat if gagal else p.galat,
            )
            buang.add(tunggal.id)
        hasil.append(p)
    return sorted((p for p in hasil if p.id not in buang), key=lambda p: p.mulai)


def _panggilan_keluar(p: _Panggilan) -> dict[str, Any]:
    biaya = None
    if p.model and p.usage:
        masuk, keluar = p.usage.get("input_tokens"), p.usage.get("output_tokens")
        if masuk is not None and keluar is not None:
            estimasi = try_estimate_cost(p.model, int(masuk), int(keluar))
            biaya = estimasi.usd if estimasi else None
    return {
        "id": p.id,
        "parent_id": p.induk,
        "position": p.posisi,
        "name": p.nama,
        "kind": p.jenis,
        "started_at": waktu_iso(p.mulai),
        "duration_ms": p.durasi_ms,
        "status": p.status,
        "error": p.galat,
        "model": p.model,
        "usage": p.usage,
        "cost_usd": biaya,
        "input": p.masukan,
        "output": p.keluaran,
    }


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
    """Diisi sejak awal, bukan saat selesai: trace LangSmith giliran yang gagal
    atau dibatalkan tetap ada, dan justru giliran itulah yang paling perlu
    ditelusuri."""
    llm_call: Any = None
    pertanyaan: str | None = None
    nim: str | None = None
    jawaban: str | None = None

    @property
    def rekam_io(self) -> bool:
        return self.recorder.rekam_io

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
        jawaban: str | None = None,
    ) -> None:
        self.hasil = _nilai(hasil)
        self.message_id = message_id
        self.langsmith_run_id = langsmith_run_id
        self.llm_call = llm_call
        self.jawaban = jawaban

    def _detail_generate(self) -> dict[str, Any]:
        model = getattr(self.llm_call, "model", None)
        usage = getattr(self.llm_call, "usage", None) or {}
        masuk, keluar = usage.get("input_tokens"), usage.get("output_tokens")
        biaya = None
        if model and masuk is not None and keluar is not None:
            estimasi = try_estimate_cost(model, int(masuk), int(keluar))
            biaya = estimasi.usd if estimasi else None
        detail: dict[str, Any] = {
            "model": model,
            "input_tokens": masuk,
            "output_tokens": keluar,
            "cost_usd": biaya,
        }
        # Nama tool saja, tanpa argumen: kolom ini tampil di rincian giliran
        # bahkan saat LOG_NODE_IO mati, dan argumen bisa memuat teks mahasiswa.
        alat = [tc.get("name") for tc in getattr(self.llm_call, "tool_calls", None) or []]
        if alat:
            detail["tools"] = ", ".join(str(a) for a in alat)
        return detail

    def _sensitif(self) -> bool:
        """Giliran FR-7: dialihkan ke konseling, entah selesai atau tidak."""
        return self.hasil == "support" or any(
            n["node"] == "sensitive" and n["detail"].get("redirected")
            for n in self.recorder.nodes
        )

    def _teks_sensitif(self, sensitif: bool) -> list[str]:
        """Teks yang disamarkan di rekaman: pertanyaan FR-7 giliran ini, dan
        pertanyaan FR-7 dari giliran sebelumnya yang ikut di riwayat widget.

        Postgres menyamarkan pertanyaan sensitif saat menyimpannya, tetapi
        widget tetap mengirim teks aslinya sebagai riwayat di giliran
        berikutnya -- ke rewrite, ke JEV, dan ke state setiap node."""
        from app.rag import sensitive as sensitive_module

        teks: list[str] = []
        if sensitif:
            teks += [t for t in (self.pertanyaan, self.recorder.teks_bersih) if t]
        masukan = self.recorder.masukan_graf
        riwayat = masukan.get("history") if isinstance(masukan, dict) else None
        for giliran in riwayat or []:
            isi = getattr(giliran, "konten", None)
            if (
                getattr(giliran, "role", None) == "user"
                and isi
                and sensitive_module.detect(isi).bypasses_rag
            ):
                teks.append(isi)
        return teks

    def tulis(self, status: str) -> None:
        if self.sink is None:
            return
        try:
            for n in self.recorder.nodes:
                if n["node"] == "generate" and n["status"] == "ok":
                    n["detail"] = {**n["detail"], **self._detail_generate()}
                self.sink.kirim("node", {"turn_id": self.turn_id, **n})
            teks: dict[str, Any] = {"nim": None, "question": None, "answer": None}
            if self.rekam_io:
                from app.observability.chatlog import SENSITIVE_PLACEHOLDER

                sensitif = self._sensitif()
                teks = {
                    "nim": self.nim,
                    "question": SENSITIVE_PLACEHOLDER if sensitif else self.pertanyaan,
                    "answer": self.jawaban,
                }
                self.sink.kirim(
                    "trace",
                    {
                        "turn_id": self.turn_id,
                        "timestamp": waktu_iso(self.waktu),
                        "isi": self.recorder.rekaman(),
                        "sensitif": self._teks_sensitif(sensitif),
                        "pengganti": SENSITIVE_PLACEHOLDER,
                    },
                )
            self.sink.kirim(
                "turn",
                {
                    "turn_id": self.turn_id,
                    "timestamp": waktu_iso(self.waktu),
                    "endpoint": self.endpoint,
                    "session_id": self.session_id,
                    "message_id": self.message_id,
                    "unit": self.unit,
                    "outcome": self.hasil,
                    "last_node": self.recorder.node_terakhir,
                    "total_ms": round((time.perf_counter() - self.t0) * 1000),
                    "ttft_ms": self.ttft_ms,
                    "status": status,
                    "langsmith_run_id": self.langsmith_run_id,
                    **teks,
                },
            )
        except Exception:
            _internal.exception("Gagal mencatat giliran %s", self.turn_id)


@contextlib.contextmanager
def catat_giliran(
    sink: Any,
    *,
    endpoint: str,
    session_id: str | None,
    unit: str | None,
    pertanyaan: str | None = None,
    nim: str | None = None,
    langsmith_run_id: str | None = None,
    rekam_io: bool = False,
) -> Iterator[Giliran]:
    """Buka satu giliran: pasang `turn_id` untuk log, tulis ringkasannya di akhir.

    Status `ok` bila blok selesai normal, `dibatalkan` bila task dibatalkan
    (mahasiswa menutup panel di tengah streaming), `error` untuk galat lainnya --
    yang juga dicatat ke log bersama traceback-nya, supaya error itu dapat
    ditemukan dari gilirannya dan sebaliknya.

    `rekam_io` (LOG_NODE_IO): simpan juga teks pertanyaan, jawaban, NIM, dan
    rekaman input/output setiap node untuk tab Graf. Giliran yang gagal atau
    dibatalkan tetap direkam sampai langkah terakhir yang sempat berjalan.
    """
    giliran = Giliran(
        sink=sink,
        endpoint=endpoint,
        session_id=session_id,
        unit=unit,
        recorder=NodeRecorder(rekam_io=rekam_io),
        langsmith_run_id=langsmith_run_id,
        pertanyaan=pertanyaan,
        nim=nim,
    )
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
