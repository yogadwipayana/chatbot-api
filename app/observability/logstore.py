"""Penyimpanan log aplikasi di SQLite (`logs.md`).

Tiga tabel: `turns` (satu baris per giliran chat), `node_runs` (satu baris per
node LangGraph per giliran), dan `app_logs` (log Python logger `app.*`).

Sengaja SQLite, bukan Postgres. Log yang paling dibutuhkan -- "gagal mencatat
percakapan ke database" -- justru hilang bila disimpan di Postgres yang sedang
bermasalah; dan belasan baris node per pertanyaan tidak layak dibayar dengan
write jaringan ke database utama.

Tidak ada teks pertanyaan maupun jawaban di sini. Teks tinggal di Postgres,
lengkap dengan penyamaran pertanyaan sensitif (FR-7); tabel ini hanya berisi
metrik dan log, ditautkan lewat `message_id`.

Semua waktu disimpan sebagai teks UTC berformat tetap (`waktu_iso`), sehingga
perbandingan dan pengurutan string sama dengan perbandingan waktu, dan 13
karakter pertamanya adalah jamnya.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

NODE_ORDER = (
    "sanitize",
    "sensitive",
    "smalltalk",
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
    waktu            TEXT NOT NULL,
    endpoint         TEXT NOT NULL,
    session_id       TEXT,
    message_id       TEXT,
    unit             TEXT,
    hasil            TEXT,
    node_terakhir    TEXT,
    total_ms         INTEGER,
    ttft_ms          INTEGER,
    status           TEXT NOT NULL,
    langsmith_run_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_turns_waktu ON turns (waktu);

CREATE TABLE IF NOT EXISTS node_runs (
    id          INTEGER PRIMARY KEY,
    turn_id     TEXT NOT NULL,
    urutan      INTEGER NOT NULL,
    node        TEXT NOT NULL,
    mulai       TEXT NOT NULL,
    durasi_ms   REAL NOT NULL,
    status      TEXT NOT NULL,
    error_tipe  TEXT,
    error_pesan TEXT,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS ix_node_runs_turn ON node_runs (turn_id);
CREATE INDEX IF NOT EXISTS ix_node_runs_mulai ON node_runs (mulai);

CREATE TABLE IF NOT EXISTS app_logs (
    id        INTEGER PRIMARY KEY,
    waktu     TEXT NOT NULL,
    level     TEXT NOT NULL,
    levelno   INTEGER NOT NULL,
    logger    TEXT NOT NULL,
    pesan     TEXT NOT NULL,
    lokasi    TEXT,
    traceback TEXT,
    turn_id   TEXT
);
CREATE INDEX IF NOT EXISTS ix_app_logs_waktu ON app_logs (waktu);
CREATE INDEX IF NOT EXISTS ix_app_logs_turn ON app_logs (turn_id);
"""

_KOLOM = {
    "turn": (
        "turns",
        (
            "turn_id",
            "waktu",
            "endpoint",
            "session_id",
            "message_id",
            "unit",
            "hasil",
            "node_terakhir",
            "total_ms",
            "ttft_ms",
            "status",
            "langsmith_run_id",
        ),
    ),
    "node": (
        "node_runs",
        (
            "turn_id",
            "urutan",
            "node",
            "mulai",
            "durasi_ms",
            "status",
            "error_tipe",
            "error_pesan",
            "detail",
        ),
    ),
    "app": (
        "app_logs",
        ("waktu", "level", "levelno", "logger", "pesan", "lokasi", "traceback", "turn_id"),
    ),
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
                        conn.executescript(SKEMA)
                        conn.commit()
                    finally:
                        conn.close()
                    self._siap = True
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    # --- Tulis ---------------------------------------------------------

    def tulis(self, baris: Iterable[tuple[str, dict[str, Any]]]) -> None:
        """Tulis satu batch `(jenis, data)` dalam satu transaksi.

        `jenis`: `turn`, `node`, atau `app`. `detail` pada node boleh berupa dict.
        """
        per_jenis: dict[str, list[tuple]] = defaultdict(list)
        for jenis, data in baris:
            _, kolom = _KOLOM[jenis]
            if jenis == "node" and not isinstance(data.get("detail"), str | None):
                data = {**data, "detail": json.dumps(data["detail"], ensure_ascii=False)}
            per_jenis[jenis].append(tuple(data.get(k) for k in kolom))
        if not per_jenis:
            return
        conn = self._connect()
        try:
            with conn:
                for jenis, nilai in per_jenis.items():
                    tabel, kolom = _KOLOM[jenis]
                    tanda = ", ".join("?" for _ in kolom)
                    # INSERT OR REPLACE untuk turns: giliran yang sama tidak
                    # boleh menjadi dua baris bila tercatat ulang.
                    perintah = "INSERT OR REPLACE" if jenis == "turn" else "INSERT"
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
                n = conn.execute("DELETE FROM turns WHERE waktu < ?", (b,)).rowcount
                n += conn.execute("DELETE FROM node_runs WHERE mulai < ?", (b,)).rowcount
                n += conn.execute("DELETE FROM app_logs WHERE waktu < ?", (b,)).rowcount
            return n
        finally:
            conn.close()

    # --- Baca ----------------------------------------------------------

    def ringkasan(self, sejak: datetime, sampai: datetime, *, audit: bool) -> dict[str, Any]:
        """Angka tab Performa: KPI, durasi per node, titik keluar, tren per jam."""
        s = waktu_iso(sejak)
        filter_audit = "" if audit else f" AND {_TANPA_AUDIT}"
        conn = self._connect()
        try:
            turns = conn.execute(
                "SELECT waktu, total_ms, status, node_terakhir FROM turns WHERE waktu >= ?",
                (s,),
            ).fetchall()
            nodes = conn.execute(
                "SELECT node, durasi_ms, status FROM node_runs WHERE mulai >= ?", (s,)
            ).fetchall()
            log_error = conn.execute(
                "SELECT substr(waktu, 1, 13) AS jam, count(*) AS n FROM app_logs"
                f" WHERE waktu >= ? AND levelno >= 40{filter_audit} GROUP BY jam",
                (s,),
            ).fetchall()
        finally:
            conn.close()

        total = [r["total_ms"] for r in turns if r["total_ms"] is not None]
        jumlah = len(turns)
        gagal = sum(1 for r in turns if r["status"] == "error")
        diblokir_jev = sum(1 for r in turns if r["node_terakhir"] == "jev_gate")

        durasi: dict[str, list[float]] = defaultdict(list)
        error_node: Counter[str] = Counter()
        for r in nodes:
            durasi[r["node"]].append(r["durasi_ms"])
            if r["status"] == "error":
                error_node[r["node"]] += 1
        urutan = {n: i for i, n in enumerate(NODE_ORDER)}
        per_node = [
            {
                "node": node,
                "jumlah": len(d),
                "p50_ms": _bulat(persentil(d, 0.5)),
                "p95_ms": _bulat(persentil(d, 0.95)),
                "error": error_node[node],
            }
            for node, d in sorted(durasi.items(), key=lambda kv: urutan.get(kv[0], 99))
        ]

        keluar = Counter(r["node_terakhir"] for r in turns if r["node_terakhir"])
        titik_keluar = [
            {"node": node, "jumlah": n}
            for node, n in sorted(keluar.items(), key=lambda kv: -kv[1])
        ]

        per_jam_total: dict[str, list[float]] = defaultdict(list)
        per_jam_giliran: Counter[str] = Counter()
        per_jam_gagal: Counter[str] = Counter()
        for r in turns:
            jam = r["waktu"][:13]
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
                    "jam": f"{k}:00:00Z",
                    "giliran": per_jam_giliran[k],
                    "p95_total_ms": _bulat(persentil(per_jam_total[k], 0.95)),
                    "giliran_error": per_jam_gagal[k],
                    "log_error": per_jam_log.get(k, 0),
                }
            )
            jam += timedelta(hours=1)

        return {
            "sejak": waktu_iso(sejak),
            "sampai": waktu_iso(sampai),
            "jumlah_giliran": jumlah,
            "p50_total_ms": _bulat(persentil(total, 0.5)),
            "p95_total_ms": _bulat(persentil(total, 0.95)),
            "giliran_error": gagal,
            "rasio_error": round(gagal / jumlah, 4) if jumlah else 0.0,
            "diblokir_jev": diblokir_jev,
            "rasio_diblokir_jev": round(diblokir_jev / jumlah, 4) if jumlah else 0.0,
            "log_error": sum(per_jam_log.values()),
            "per_node": per_node,
            "titik_keluar": titik_keluar,
            "per_jam": per_jam,
        }

    def daftar_giliran(
        self,
        sejak: datetime,
        *,
        hasil: str | None = None,
        status: str | None = None,
        unit: str | None = None,
        node_terakhir: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[int, list[dict[str, Any]]]:
        syarat = ["waktu >= ?"]
        arg: list[Any] = [waktu_iso(sejak)]
        for kolom, nilai in (
            ("hasil", hasil),
            ("status", status),
            ("unit", unit),
            ("node_terakhir", node_terakhir),
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
                f"SELECT * FROM turns WHERE {where} ORDER BY waktu DESC LIMIT ? OFFSET ?",
                [*arg, limit, offset],
            ).fetchall()
        finally:
            conn.close()
        return total, [dict(r) for r in rows]

    def detail_giliran(self, turn_id: str, *, audit: bool) -> dict[str, Any] | None:
        filter_audit = "" if audit else f" AND {_TANPA_AUDIT}"
        conn = self._connect()
        try:
            turn = conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
            if turn is None:
                return None
            nodes = conn.execute(
                "SELECT node, urutan, mulai, durasi_ms, status, error_tipe, error_pesan,"
                " detail FROM node_runs WHERE turn_id = ? ORDER BY urutan",
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
        syarat = ["waktu >= ?", "levelno >= ?"]
        arg: list[Any] = [waktu_iso(sejak), level_min]
        if not audit:
            syarat.append(_TANPA_AUDIT)
        if logger:
            syarat.append("(logger = ? OR logger LIKE ? ESCAPE '\\')")
            arg += [logger, _escape_like(logger) + ".%"]
        if cari:
            syarat.append("pesan LIKE ? ESCAPE '\\'")
            arg.append(f"%{_escape_like(cari)}%")
        where = " AND ".join(syarat)
        conn = self._connect()
        try:
            total = conn.execute(
                f"SELECT count(*) FROM app_logs WHERE {where}", arg
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM app_logs WHERE {where} ORDER BY waktu DESC, id DESC"
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
                f"SELECT DISTINCT logger FROM app_logs WHERE waktu >= ?{filter_audit}"
                " ORDER BY logger",
                (waktu_iso(sejak),),
            ).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]
