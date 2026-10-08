"""Penyimpanan log aplikasi di SQLite (`logs.md`).

Empat tabel: `turns` (satu baris per giliran chat), `node_runs` (satu baris per
node LangGraph per giliran), `app_logs` (log Python logger `app.*`), dan
`traces` (rekaman input/output satu giliran untuk tab Graf, `rekaman.py`).

Sengaja SQLite, bukan Postgres. Log yang paling dibutuhkan -- "gagal mencatat
percakapan ke database" -- justru hilang bila disimpan di Postgres yang sedang
bermasalah; dan belasan baris node per pertanyaan tidak layak dibayar dengan
write jaringan ke database utama.

Sejak tahap 4 (keputusan user 2026-10-08) teks pertanyaan, jawaban, NIM, dan
rekaman input/output node ikut disimpan, supaya tracing LangSmith boleh
dimatikan. Pertanyaan sensitif (FR-7) disamarkan dengan penanda yang sama seperti
di Postgres. `LOG_NODE_IO=false` mengembalikan perilaku lama: hanya metrik dan log.

Semua waktu disimpan sebagai teks UTC berformat tetap (`waktu_iso`), sehingga
perbandingan dan pengurutan string sama dengan perbandingan waktu, dan 13
karakter pertamanya adalah jamnya.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.observability.rekaman import buka, kemas

_internal = logging.getLogger("logstore")
"""Sama dengan `applog._internal`: di luar hierarki `app`, supaya galat menulis
SQLite tidak masuk antrean SQLite yang sama."""

NODE_ORDER = (
    "sanitize",
    "sensitive",
    "smalltalk",
    "rule_gate",
    "jev_gate",
    "rewrite",
    "retrieve",
    "validate_context",
    "refuse",
    "generate",
)
"""Urutan node di graf (`app.rag.graph`), dipakai mengurutkan ringkasan."""

AUDIT_LOGGER = "app.audit"

_TANPA_AUDIT = f"NOT (logger = '{AUDIT_LOGGER}' OR logger LIKE '{AUDIT_LOGGER}.%')"
"""Log audit memuat email admin dan aksi pengelolaan akun: ranah superadmin.
Disaring di SQL, bukan di UI, supaya role `admin` tidak pernah menerimanya
dari endpoint mana pun."""

SKEMA = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id          TEXT PRIMARY KEY,
    timestamp        TEXT NOT NULL,
    endpoint         TEXT NOT NULL,
    session_id       TEXT,
    message_id       TEXT,
    unit             TEXT,
    outcome          TEXT,
    last_node        TEXT,
    total_ms         INTEGER,
    ttft_ms          INTEGER,
    status           TEXT NOT NULL,
    langsmith_run_id TEXT,
    nim              TEXT,
    question         TEXT,
    answer           TEXT
);
CREATE INDEX IF NOT EXISTS ix_turns_timestamp ON turns (timestamp);

CREATE TABLE IF NOT EXISTS node_runs (
    id            INTEGER PRIMARY KEY,
    turn_id       TEXT NOT NULL,
    position      INTEGER NOT NULL,
    node          TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    duration_ms   REAL NOT NULL,
    status        TEXT NOT NULL,
    error_type    TEXT,
    error_message TEXT,
    detail        TEXT
);
CREATE INDEX IF NOT EXISTS ix_node_runs_turn ON node_runs (turn_id);
CREATE INDEX IF NOT EXISTS ix_node_runs_started_at ON node_runs (started_at);

CREATE TABLE IF NOT EXISTS app_logs (
    id        INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    level     TEXT NOT NULL,
    levelno   INTEGER NOT NULL,
    logger    TEXT NOT NULL,
    message   TEXT NOT NULL,
    location  TEXT,
    traceback TEXT,
    turn_id   TEXT
);
CREATE INDEX IF NOT EXISTS ix_app_logs_timestamp ON app_logs (timestamp);
CREATE INDEX IF NOT EXISTS ix_app_logs_turn ON app_logs (turn_id);

CREATE TABLE IF NOT EXISTS traces (
    turn_id   TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    data      BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_traces_timestamp ON traces (timestamp);
"""

VERSI_SKEMA = 2
"""Disimpan di `PRAGMA user_version`. Naikkan setiap kali `SKEMA` berubah.

`CREATE TABLE IF NOT EXISTS` tidak pernah mengubah tabel yang sudah ada, jadi
berkas lama akan tetap memakai kolom lama dan setiap INSERT gagal. Versi 0
(bawaan SQLite, skema berkolom bahasa Indonesia sebelum versi ini ada) dibuang
tabelnya lalu dibuat ulang: isinya log yang memang berumur pendek
(`LOG_RETENTION_DAYS`). Perubahan sesudahnya cukup menambah kolom atau tabel,
jadi dimigrasi di tempat (`_MIGRASI`) dan log yang ada tetap terbaca."""

_TABEL = ("turns", "node_runs", "app_logs", "traces")

_MIGRASI: dict[int, tuple[str, ...]] = {
    2: (
        "ALTER TABLE turns ADD COLUMN nim TEXT",
        "ALTER TABLE turns ADD COLUMN question TEXT",
        "ALTER TABLE turns ADD COLUMN answer TEXT",
    ),
}
"""Versi tujuan -> perintah yang membawa berkas dari versi sebelumnya ke sana.
Tabel baru tidak perlu dicantumkan: `SKEMA` membuatnya dengan IF NOT EXISTS."""

UJI_COBA = "uji_coba"
"""`turns.endpoint` untuk kotak uji coba admin (AD-6). Tercatat supaya bisa
ditelusuri di tab Graf, tetapi tidak dihitung di Performa: itu bukan lalu lintas
mahasiswa."""

_KOLOM = {
    "turn": (
        "turns",
        (
            "turn_id",
            "timestamp",
            "endpoint",
            "session_id",
            "message_id",
            "unit",
            "outcome",
            "last_node",
            "total_ms",
            "ttft_ms",
            "status",
            "langsmith_run_id",
            "nim",
            "question",
            "answer",
        ),
    ),
    "node": (
        "node_runs",
        (
            "turn_id",
            "position",
            "node",
            "started_at",
            "duration_ms",
            "status",
            "error_type",
            "error_message",
            "detail",
        ),
    ),
    "app": (
        "app_logs",
        (
            "timestamp",
            "level",
            "levelno",
            "logger",
            "message",
            "location",
            "traceback",
            "turn_id",
        ),
    ),
    "trace": ("traces", ("turn_id", "timestamp", "data")),
}


def waktu_iso(dt: datetime | None = None) -> str:
    """`2026-09-25T03:12:03.123Z`: UTC, milidetik, panjang selalu sama."""
    dt = (dt or datetime.now(UTC)).astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def persentil(nilai: Sequence[float], p: float) -> float | None:
    """Persentil dengan interpolasi linear; None untuk data kosong."""
    if not nilai:
        return None
    urut = sorted(nilai)
    posisi = (len(urut) - 1) * p
    bawah = int(posisi)
    atas = min(bawah + 1, len(urut) - 1)
    return urut[bawah] + (urut[atas] - urut[bawah]) * (posisi - bawah)


def _bulat(x: float | None) -> float | None:
    return None if x is None else round(x, 1)


def _siapkan_skema(conn: sqlite3.Connection) -> None:
    """Buat tabel; migrasikan atau buang tabel berskema lama (lihat `VERSI_SKEMA`)."""
    versi = conn.execute("PRAGMA user_version").fetchone()[0]
    if versi < 1:
        for tabel in _TABEL:
            conn.execute(f"DROP TABLE IF EXISTS {tabel}")
    else:
        for tujuan in range(versi + 1, VERSI_SKEMA + 1):
            for perintah in _MIGRASI.get(tujuan, ()):
                try:
                    conn.execute(perintah)
                except sqlite3.OperationalError as exc:
                    # Thread penulis dan pembaca dashboard membuka berkas yang
                    # sama; yang kalah cepat mendapati kolomnya sudah ada.
                    if "duplicate column" not in str(exc):
                        raise
    conn.executescript(SKEMA)
    if versi < VERSI_SKEMA:
        # PRAGMA tidak menerima parameter terikat; nilainya konstanta modul.
        conn.execute(f"PRAGMA user_version = {VERSI_SKEMA}")
    conn.commit()


_ADA_REKAMAN = "EXISTS (SELECT 1 FROM traces tr WHERE tr.turn_id = turns.turn_id) AS has_trace"


def _escape_like(teks: str) -> str:
    return teks.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class LogStore:
    """Satu berkas SQLite. Koneksi dibuka per operasi, jadi aman dipakai dari
    thread penulis dan dari threadpool FastAPI sekaligus."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._siap = False
        self._kunci = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        if not self._siap:
            with self._kunci:
                if not self._siap:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    conn = sqlite3.connect(self.path, timeout=5)
                    try:
                        # WAL: pembaca (dashboard) tidak menahan penulis, dan sebaliknya.
                        conn.execute("PRAGMA journal_mode=WAL")
                        _siapkan_skema(conn)
                    finally:
                        conn.close()
                    self._siap = True
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    # --- Tulis ---------------------------------------------------------

    def tulis(self, baris: Iterable[tuple[str, dict[str, Any]]]) -> None:
        """Tulis satu batch `(jenis, data)` dalam satu transaksi.

        `jenis`: `turn`, `node`, `app`, atau `trace`. `detail` pada node boleh
        berupa dict. `trace` membawa `data` (blob jadi) atau `isi` (rekaman
        mentah dari `NodeRecorder.rekaman`) yang dikemas di sini, di thread
        penulis; rekaman yang gagal dikemas dilewati tanpa menggagalkan batch.
        """
        per_jenis: dict[str, list[tuple]] = defaultdict(list)
        for jenis, data in baris:
            _, kolom = _KOLOM[jenis]
            if jenis == "node" and not isinstance(data.get("detail"), str | None):
                data = {**data, "detail": json.dumps(data["detail"], ensure_ascii=False)}
            if jenis == "trace" and data.get("data") is None:
                try:
                    blob = kemas(
                        data.get("isi"),
                        sensitif=data.get("sensitif"),
                        pengganti=data.get("pengganti") or "",
                    )
                except Exception:
                    _internal.exception(
                        "Gagal mengemas rekaman giliran %s", data.get("turn_id")
                    )
                    continue
                data = {**data, "data": blob}
            per_jenis[jenis].append(tuple(data.get(k) for k in kolom))
        if not per_jenis:
            return
        conn = self._connect()
        try:
            with conn:
                for jenis, nilai in per_jenis.items():
                    tabel, kolom = _KOLOM[jenis]
                    tanda = ", ".join("?" for _ in kolom)
                    # INSERT OR REPLACE untuk turns dan traces: giliran yang
                    # sama tidak boleh menjadi dua baris bila tercatat ulang.
                    perintah = "INSERT OR REPLACE" if jenis in ("turn", "trace") else "INSERT"
                    conn.executemany(
                        f"{perintah} INTO {tabel} ({', '.join(kolom)}) VALUES ({tanda})",
                        nilai,
                    )
        finally:
            conn.close()

    def hapus_lama(self, batas: datetime) -> int:
        """Hapus baris sebelum `batas`. Return: jumlah baris terhapus."""
        b = waktu_iso(batas)
        conn = self._connect()
        try:
            with conn:
                n = conn.execute("DELETE FROM turns WHERE timestamp < ?", (b,)).rowcount
                n += conn.execute("DELETE FROM node_runs WHERE started_at < ?", (b,)).rowcount
                n += conn.execute("DELETE FROM app_logs WHERE timestamp < ?", (b,)).rowcount
                n += conn.execute("DELETE FROM traces WHERE timestamp < ?", (b,)).rowcount
            return n
        finally:
            conn.close()

    # --- Baca ----------------------------------------------------------

    def ringkasan(self, sejak: datetime, sampai: datetime, *, audit: bool) -> dict[str, Any]:
        """Angka tab Performa: KPI, durasi per node, titik keluar, tren per jam.

        Giliran uji coba admin (`UJI_COBA`) tidak dihitung, begitu pula node-nya."""
        s = waktu_iso(sejak)
        filter_audit = "" if audit else f" AND {_TANPA_AUDIT}"
        conn = self._connect()
        try:
            turns = conn.execute(
                "SELECT timestamp, total_ms, status, last_node FROM turns"
                " WHERE timestamp >= ? AND endpoint != ?",
                (s, UJI_COBA),
            ).fetchall()
            nodes = conn.execute(
                "SELECT node, duration_ms, status FROM node_runs WHERE started_at >= ?"
                " AND turn_id NOT IN (SELECT turn_id FROM turns WHERE endpoint = ?)",
                (s, UJI_COBA),
            ).fetchall()
            log_error = conn.execute(
                "SELECT substr(timestamp, 1, 13) AS jam, count(*) AS n FROM app_logs"
                f" WHERE timestamp >= ? AND levelno >= 40{filter_audit} GROUP BY jam",
                (s,),
            ).fetchall()
        finally:
            conn.close()

        total = [r["total_ms"] for r in turns if r["total_ms"] is not None]
        jumlah = len(turns)
        gagal = sum(1 for r in turns if r["status"] == "error")
        # Terpisah dari `gagal`: pembatalan bukan galat server. Tetapi mahasiswa
        # yang menghentikan jawaban sering kali sudah kehilangan kesabaran, jadi
        # jumlahnya perlu terlihat di ringkasan, bukan hanya di filter tab Giliran.
        dibatalkan = sum(1 for r in turns if r["status"] == "dibatalkan")
        diblokir_jev = sum(1 for r in turns if r["last_node"] == "jev_gate")

        durasi: dict[str, list[float]] = defaultdict(list)
        error_node: Counter[str] = Counter()
        for r in nodes:
            durasi[r["node"]].append(r["duration_ms"])
            if r["status"] == "error":
                error_node[r["node"]] += 1
        urutan = {n: i for i, n in enumerate(NODE_ORDER)}
        per_node = [
            {
                "node": node,
                "count": len(d),
                "p50_ms": _bulat(persentil(d, 0.5)),
                "p95_ms": _bulat(persentil(d, 0.95)),
                "error": error_node[node],
            }
            for node, d in sorted(durasi.items(), key=lambda kv: urutan.get(kv[0], 99))
        ]

        keluar = Counter(r["last_node"] for r in turns if r["last_node"])
        titik_keluar = [
            {"node": node, "count": n}
            for node, n in sorted(keluar.items(), key=lambda kv: -kv[1])
        ]

        per_jam_total: dict[str, list[float]] = defaultdict(list)
        per_jam_giliran: Counter[str] = Counter()
        per_jam_gagal: Counter[str] = Counter()
        for r in turns:
            jam = r["timestamp"][:13]
            per_jam_giliran[jam] += 1
            if r["total_ms"] is not None:
                per_jam_total[jam].append(r["total_ms"])
            if r["status"] == "error":
                per_jam_gagal[jam] += 1
        per_jam_log = {r["jam"]: r["n"] for r in log_error}

        per_jam = []
        jam = sejak.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        akhir = sampai.astimezone(UTC)
        while jam <= akhir:
            k = jam.strftime("%Y-%m-%dT%H")
            per_jam.append(
                {
                    "hour": f"{k}:00:00Z",
                    "turn_count": per_jam_giliran[k],
                    "p95_total_ms": _bulat(persentil(per_jam_total[k], 0.95)),
                    "error_turn_count": per_jam_gagal[k],
                    "error_log_count": per_jam_log.get(k, 0),
                }
            )
            jam += timedelta(hours=1)

        return {
            "since": waktu_iso(sejak),
            "until": waktu_iso(sampai),
            "turn_count": jumlah,
            "p50_total_ms": _bulat(persentil(total, 0.5)),
            "p95_total_ms": _bulat(persentil(total, 0.95)),
            "error_turn_count": gagal,
            "error_ratio": round(gagal / jumlah, 4) if jumlah else 0.0,
            "cancelled_turn_count": dibatalkan,
            "jev_blocked_count": diblokir_jev,
            "jev_blocked_ratio": round(diblokir_jev / jumlah, 4) if jumlah else 0.0,
            "error_log_count": sum(per_jam_log.values()),
            "per_node": per_node,
            "exit_points": titik_keluar,
            "per_hour": per_jam,
        }

    def daftar_giliran(
        self,
        sejak: datetime,
        *,
        outcome: str | None = None,
        status: str | None = None,
        unit: str | None = None,
        last_node: str | None = None,
        endpoint: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[int, list[dict[str, Any]]]:
        syarat = ["timestamp >= ?"]
        arg: list[Any] = [waktu_iso(sejak)]
        for kolom, nilai in (
            ("outcome", outcome),
            ("status", status),
            ("unit", unit),
            ("last_node", last_node),
            ("endpoint", endpoint),
        ):
            if nilai is not None:
                syarat.append(f"{kolom} = ?")
                arg.append(nilai)
        where = " AND ".join(syarat)
        conn = self._connect()
        try:
            total = conn.execute(f"SELECT count(*) FROM turns WHERE {where}", arg).fetchone()[
                0
            ]
            rows = conn.execute(
                f"SELECT *, {_ADA_REKAMAN} FROM turns WHERE {where}"
                " ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                [*arg, limit, offset],
            ).fetchall()
        finally:
            conn.close()
        return total, [dict(r) for r in rows]

    def detail_giliran(self, turn_id: str, *, audit: bool) -> dict[str, Any] | None:
        filter_audit = "" if audit else f" AND {_TANPA_AUDIT}"
        conn = self._connect()
        try:
            turn = conn.execute(
                f"SELECT *, {_ADA_REKAMAN} FROM turns WHERE turn_id = ?", (turn_id,)
            ).fetchone()
            if turn is None:
                return None
            nodes = conn.execute(
                "SELECT node, position, started_at, duration_ms, status, error_type,"
                " error_message, detail FROM node_runs WHERE turn_id = ? ORDER BY position",
                (turn_id,),
            ).fetchall()
            logs = conn.execute(
                f"SELECT * FROM app_logs WHERE turn_id = ?{filter_audit} ORDER BY id",
                (turn_id,),
            ).fetchall()
        finally:
            conn.close()
        return {
            **dict(turn),
            "nodes": [
                {**dict(n), "detail": json.loads(n["detail"]) if n["detail"] else {}}
                for n in nodes
            ],
            "logs": [dict(r) for r in logs],
        }

    def rekaman_giliran(self, turn_id: str) -> Any | None:
        """Rekaman input/output satu giliran (tab Graf), atau None bila tidak ada."""
        conn = self._connect()
        try:
            baris = conn.execute(
                "SELECT data FROM traces WHERE turn_id = ?", (turn_id,)
            ).fetchone()
        finally:
            conn.close()
        return None if baris is None else buka(baris["data"])

    def daftar_log(
        self,
        sejak: datetime,
        *,
        audit: bool,
        level_min: int = 0,
        logger: str | None = None,
        cari: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[int, list[dict[str, Any]]]:
        syarat = ["timestamp >= ?", "levelno >= ?"]
        arg: list[Any] = [waktu_iso(sejak), level_min]
        if not audit:
            syarat.append(_TANPA_AUDIT)
        if logger:
            syarat.append("(logger = ? OR logger LIKE ? ESCAPE '\\')")
            arg += [logger, _escape_like(logger) + ".%"]
        if cari:
            syarat.append("message LIKE ? ESCAPE '\\'")
            arg.append(f"%{_escape_like(cari)}%")
        where = " AND ".join(syarat)
        conn = self._connect()
        try:
            total = conn.execute(
                f"SELECT count(*) FROM app_logs WHERE {where}", arg
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM app_logs WHERE {where} ORDER BY timestamp DESC, id DESC"
                " LIMIT ? OFFSET ?",
                [*arg, limit, offset],
            ).fetchall()
        finally:
            conn.close()
        return total, [dict(r) for r in rows]

    def daftar_logger(self, sejak: datetime, *, audit: bool) -> list[str]:
        """Nama logger yang muncul di rentang ini, untuk isi filter di dashboard."""
        filter_audit = "" if audit else f" AND {_TANPA_AUDIT}"
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT DISTINCT logger FROM app_logs WHERE timestamp >= ?{filter_audit}"
                " ORDER BY logger",
                (waktu_iso(sejak),),
            ).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]
