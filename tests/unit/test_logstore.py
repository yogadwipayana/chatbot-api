"""SQLite log aplikasi (`app/observability/logstore.py`)."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.observability.logstore import VERSI_SKEMA, LogStore, persentil, waktu_iso

SEKARANG = datetime(2026, 9, 25, 10, 30, tzinfo=UTC)


def turn(turn_id: str, *, menit_lalu: int = 5, **lain) -> tuple[str, dict]:
    data = {
        "turn_id": turn_id,
        "timestamp": waktu_iso(SEKARANG - timedelta(minutes=menit_lalu)),
        "endpoint": "chat",
        "session_id": "sesi-1",
        "message_id": None,
        "unit": None,
        "outcome": "answer",
        "last_node": "generate",
        "total_ms": 1000,
        "ttft_ms": None,
        "status": "ok",
        "langsmith_run_id": None,
    }
    data.update(lain)
    return "turn", data


def node(
    turn_id: str, nama: str, durasi: float, *, position: int = 1, **lain
) -> tuple[str, dict]:
    data = {
        "turn_id": turn_id,
        "position": position,
        "node": nama,
        "started_at": waktu_iso(SEKARANG - timedelta(minutes=5)),
        "duration_ms": durasi,
        "status": "ok",
        "detail": {"contoh": 1},
    }
    data.update(lain)
    return "node", data


def log(logger: str, pesan: str, *, levelno: int = 20, turn_id=None, menit_lalu=5):
    return "app", {
        "timestamp": waktu_iso(SEKARANG - timedelta(minutes=menit_lalu)),
        "level": {20: "INFO", 30: "WARNING", 40: "ERROR"}[levelno],
        "levelno": levelno,
        "logger": logger,
        "message": pesan,
        "location": "modul:1",
        "traceback": None,
        "turn_id": turn_id,
    }


@pytest.fixture
def store(tmp_path) -> LogStore:
    return LogStore(tmp_path / "log" / "app.db")


class TestWaktu:
    def test_format_tetap_dan_utc(self):
        wita = datetime(2026, 9, 25, 11, 0, 0, 5000, tzinfo=UTC).astimezone(
            timezone(timedelta(hours=8))
        )
        assert waktu_iso(wita) == "2026-09-25T11:00:00.005Z"

    def test_urutan_string_sama_dengan_urutan_waktu(self):
        a = waktu_iso(SEKARANG)
        b = waktu_iso(SEKARANG + timedelta(milliseconds=1))
        assert len(a) == len(b) and a < b


class TestPersentil:
    def test_kosong(self):
        assert persentil([], 0.5) is None

    def test_interpolasi(self):
        assert persentil([10, 20, 30, 40], 0.5) == 25
        assert persentil([10], 0.95) == 10
        assert persentil([0, 100], 0.95) == pytest.approx(95)


class TestTulisDanHapus:
    def test_direktori_dan_skema_dibuat_sendiri(self, store):
        store.tulis([turn("t1")])
        assert store.path.exists()
        total, items = store.daftar_giliran(SEKARANG - timedelta(hours=1))
        assert total == 1 and items[0]["turn_id"] == "t1"

    def test_giliran_yang_sama_tidak_berlipat(self, store):
        store.tulis([turn("t1", status="ok")])
        store.tulis([turn("t1", status="error")])
        total, items = store.daftar_giliran(SEKARANG - timedelta(hours=1))
        assert total == 1 and items[0]["status"] == "error"

    def test_hapus_lama(self, store):
        store.tulis(
            [
                turn("lama", menit_lalu=60 * 24 * 8),
                turn("baru"),
                log("app.x", "lama", menit_lalu=60 * 24 * 8),
                log("app.x", "baru"),
                node(
                    "lama", "sanitize", 1, started_at=waktu_iso(SEKARANG - timedelta(days=8))
                ),
            ]
        )
        terhapus = store.hapus_lama(SEKARANG - timedelta(days=7))
        assert terhapus == 3
        total, items = store.daftar_giliran(SEKARANG - timedelta(days=30))
        assert [i["turn_id"] for i in items] == ["baru"]


class TestVersiSkema:
    def test_berkas_berskema_lama_dibuat_ulang(self, tmp_path):
        """`CREATE TABLE IF NOT EXISTS` tidak mengubah tabel lama; tanpa versi
        skema setiap INSERT ke berkas lama gagal karena kolomnya berbeda."""
        path = tmp_path / "app.db"
        with closing(sqlite3.connect(path)) as conn:
            conn.executescript(
                "CREATE TABLE turns (turn_id TEXT PRIMARY KEY, waktu TEXT NOT NULL);"
                "INSERT INTO turns VALUES ('lama', '2026-09-25T10:00:00.000Z');"
            )
        store = LogStore(path)
        store.tulis([turn("t1")])
        _, items = store.daftar_giliran(SEKARANG - timedelta(hours=1))
        assert [i["turn_id"] for i in items] == ["t1"]
        with closing(sqlite3.connect(path)) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == VERSI_SKEMA

    def test_berkas_versi_terkini_tidak_dikosongkan(self, tmp_path):
        path = tmp_path / "app.db"
        LogStore(path).tulis([turn("t1")])
        # Objek baru = persiapan skema berjalan lagi, seperti setelah restart.
        LogStore(path).tulis([turn("t2")])
        assert LogStore(path).daftar_giliran(SEKARANG - timedelta(hours=1))[0] == 2


class TestRingkasan:
    def test_kosong(self, store):
        r = store.ringkasan(SEKARANG - timedelta(hours=2), SEKARANG, audit=True)
        assert r["turn_count"] == 0
        assert r["p95_total_ms"] is None
        assert r["error_ratio"] == 0.0
        assert len(r["per_hour"]) == 3

    def test_kpi_node_dan_titik_keluar(self, store):
        store.tulis(
            [
                turn("t1", total_ms=1000),
                turn("t2", total_ms=3000, status="error"),
                turn("t3", total_ms=200, outcome="rejected", last_node="jev_gate"),
                turn("t4", total_ms=100, outcome="refusal", last_node="refuse"),
                node("t1", "retrieve", 100),
                node("t2", "retrieve", 300, status="error"),
                node("t1", "sanitize", 1),
                log("app.rag.gate", "gagal", levelno=40),
                log("app.audit", "akun dihapus", levelno=40),
            ]
        )
        r = store.ringkasan(SEKARANG - timedelta(hours=1), SEKARANG, audit=True)
        assert r["turn_count"] == 4
        assert r["error_turn_count"] == 1 and r["error_ratio"] == 0.25
        assert r["jev_blocked_count"] == 1 and r["jev_blocked_ratio"] == 0.25
        assert r["error_log_count"] == 2
        # Urutan mengikuti graf, bukan abjad.
        assert [n["node"] for n in r["per_node"]] == ["sanitize", "retrieve"]
        retrieve = r["per_node"][1]
        assert retrieve["count"] == 2 and retrieve["error"] == 1
        assert retrieve["p50_ms"] == 200
        assert r["exit_points"][0] == {"node": "generate", "count": 2}
        jam_isi = [j for j in r["per_hour"] if j["turn_count"]]
        assert len(jam_isi) == 1 and jam_isi[0]["error_turn_count"] == 1

    def test_dibatalkan_dihitung_terpisah_dari_gagal(self, store):
        """Pembatalan bukan galat server, tetapi harus terlihat di ringkasan (T8)."""
        store.tulis(
            [
                turn("t1", total_ms=1000),
                turn("t2", total_ms=3000, status="error"),
                turn("t3", total_ms=40000, status="dibatalkan", last_node="generate"),
            ]
        )
        r = store.ringkasan(SEKARANG - timedelta(hours=1), SEKARANG, audit=True)
        assert r["cancelled_turn_count"] == 1
        assert r["error_turn_count"] == 1

    def test_log_audit_tidak_dihitung_tanpa_hak_audit(self, store):
        store.tulis([log("app.audit", "x", levelno=40), log("app.audit.sub", "y", levelno=40)])
        r = store.ringkasan(SEKARANG - timedelta(hours=1), SEKARANG, audit=False)
        assert r["error_log_count"] == 0


class TestDaftarGiliran:
    def test_filter_dan_paginasi(self, store):
        store.tulis(
            [
                turn(f"t{i}", menit_lalu=i, unit="Keuangan" if i % 2 else None)
                for i in range(1, 6)
            ]
        )
        total, items = store.daftar_giliran(
            SEKARANG - timedelta(hours=1), unit="Keuangan", limit=2, offset=0
        )
        assert total == 3
        assert [i["turn_id"] for i in items] == ["t1", "t3"]  # terbaru dulu

    def test_filter_hasil_dan_node_terakhir(self, store):
        store.tulis(
            [
                turn("t1"),
                turn("t2", outcome="rejected", last_node="jev_gate"),
            ]
        )
        total, items = store.daftar_giliran(
            SEKARANG - timedelta(hours=1), outcome="rejected", last_node="jev_gate"
        )
        assert total == 1 and items[0]["turn_id"] == "t2"
        assert {"timestamp", "outcome", "last_node"} <= items[0].keys()

    def test_rentang_waktu(self, store):
        store.tulis([turn("lama", menit_lalu=120), turn("baru")])
        total, _ = store.daftar_giliran(SEKARANG - timedelta(hours=1))
        assert total == 1


class TestDetailGiliran:
    def test_tidak_ada(self, store):
        assert store.detail_giliran("tidak-ada", audit=True) is None

    def test_node_urut_dan_detail_diurai(self, store):
        store.tulis(
            [
                turn("t1"),
                node("t1", "retrieve", 5, position=2, detail={"document_count": 3}),
                node("t1", "sanitize", 1, position=1),
                log("app.rag.gate", "lambat", turn_id="t1"),
                log("app.audit", "rahasia", turn_id="t1"),
            ]
        )
        d = store.detail_giliran("t1", audit=False)
        assert [n["node"] for n in d["nodes"]] == ["sanitize", "retrieve"]
        assert d["nodes"][1]["detail"] == {"document_count": 3}
        assert d["nodes"][1]["position"] == 2
        kolom = {"started_at", "duration_ms", "error_type", "error_message"}
        assert kolom <= d["nodes"][1].keys()
        assert [x["message"] for x in d["logs"]] == ["lambat"]
        assert d["logs"][0]["location"] == "modul:1"
        assert len(store.detail_giliran("t1", audit=True)["logs"]) == 2


class TestDaftarLog:
    @pytest.fixture
    def terisi(self, store):
        store.tulis(
            [
                log("app.main", "Tracing mati"),
                log("app.rag.gate", "Gerbang JEV gagal", levelno=30),
                log("app.routers.common", "100% gagal_total", levelno=40),
                log("app.audit", "Akun a@b dihapus", levelno=30),
            ]
        )
        return store

    def test_level_minimum(self, terisi):
        total, items = terisi.daftar_log(
            SEKARANG - timedelta(hours=1), audit=True, level_min=30
        )
        assert total == 3
        assert {i["level"] for i in items} == {"WARNING", "ERROR"}

    def test_logger_termasuk_turunan(self, terisi):
        total, _ = terisi.daftar_log(
            SEKARANG - timedelta(hours=1), audit=True, logger="app.rag"
        )
        assert total == 1

    def test_cari_tidak_menganggap_persen_sebagai_wildcard(self, terisi):
        sejak = SEKARANG - timedelta(hours=1)
        assert terisi.daftar_log(sejak, audit=True, cari="100%")[0] == 1
        assert terisi.daftar_log(sejak, audit=True, cari="gagal_total")[0] == 1
        assert terisi.daftar_log(sejak, audit=True, cari="%")[0] == 1

    def test_audit_disembunyikan_tanpa_hak(self, terisi):
        sejak = SEKARANG - timedelta(hours=1)
        _, items = terisi.daftar_log(sejak, audit=False)
        assert "app.audit" not in {i["logger"] for i in items}
        assert terisi.daftar_log(sejak, audit=False, logger="app.audit") == (0, [])
        assert terisi.daftar_log(sejak, audit=False, cari="dihapus")[0] == 0
        assert "app.audit" not in terisi.daftar_logger(sejak, audit=False)
        assert "app.audit" in terisi.daftar_logger(sejak, audit=True)


# --- Tahap 4: teks giliran, rekaman tab Graf, uji coba admin -----------------


def trace(turn_id: str, isi: dict, *, menit_lalu: int = 5) -> tuple[str, dict]:
    return "trace", {
        "turn_id": turn_id,
        "timestamp": waktu_iso(SEKARANG - timedelta(minutes=menit_lalu)),
        "isi": isi,
        "sensitif": [],
        "pengganti": "",
    }


class TestMigrasiV2:
    def test_berkas_v1_dimigrasi_tanpa_kehilangan_log(self, tmp_path):
        """Versi 1 -> 2 hanya menambah kolom dan tabel: log 7 hari tetap terbaca."""
        path = tmp_path / "app.db"
        with closing(sqlite3.connect(path)) as conn:
            conn.executescript(
                "CREATE TABLE turns (turn_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,"
                " endpoint TEXT NOT NULL, session_id TEXT, message_id TEXT, unit TEXT,"
                " outcome TEXT, last_node TEXT, total_ms INTEGER, ttft_ms INTEGER,"
                " status TEXT NOT NULL, langsmith_run_id TEXT);"
                f"INSERT INTO turns (turn_id, timestamp, endpoint, status) VALUES"
                f" ('lama', '{waktu_iso(SEKARANG - timedelta(minutes=9))}', 'chat', 'ok');"
                "PRAGMA user_version = 1;"
            )
        store = LogStore(path)
        store.tulis([turn("baru", question="kapan KRS?", nim="2401010101")])
        _, items = store.daftar_giliran(SEKARANG - timedelta(hours=1))
        assert {i["turn_id"]: i["question"] for i in items} == {
            "baru": "kapan KRS?",
            "lama": None,
        }
        with closing(sqlite3.connect(path)) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == VERSI_SKEMA == 2

    def test_migrasi_bersamaan_tidak_gagal(self, tmp_path):
        """Penulis dan pembaca yang sama-sama melihat versi 1: yang kedua tidak
        boleh gagal karena kolomnya sudah ditambahkan yang pertama."""
        path = tmp_path / "app.db"
        LogStore(path).tulis([turn("t1")])
        with closing(sqlite3.connect(path)) as conn:
            conn.execute("PRAGMA user_version = 1")
            conn.commit()
        LogStore(path).tulis([turn("t2", nim="x")])
        assert LogStore(path).daftar_giliran(SEKARANG - timedelta(hours=1))[0] == 2


class TestRekaman:
    def test_tulis_dan_baca(self, store):
        isi = {"input": {"question": "kapan KRS?"}, "nodes": [], "calls": []}
        store.tulis([turn("t1"), trace("t1", isi)])
        assert store.rekaman_giliran("t1") == isi
        assert store.rekaman_giliran("tidak-ada") is None

    def test_penanda_ada_rekaman(self, store):
        store.tulis([turn("t1"), turn("t2"), trace("t1", {"nodes": []})])
        _, items = store.daftar_giliran(SEKARANG - timedelta(hours=1))
        assert {i["turn_id"]: bool(i["has_trace"]) for i in items} == {"t1": True, "t2": False}
        assert store.detail_giliran("t1", audit=False)["has_trace"]

    def test_rekaman_yang_gagal_dikemas_dilewati(self, store, monkeypatch):
        """Satu rekaman rusak tidak boleh membuang baris lain di batch yang sama."""
        from app.observability import logstore

        def rusak(*_a, **_k):
            raise ValueError("tidak bisa dikemas")

        monkeypatch.setattr(logstore, "kemas", rusak)
        store.tulis([turn("t1"), trace("t1", {"nodes": []})])
        assert store.daftar_giliran(SEKARANG - timedelta(hours=1))[0] == 1
        assert store.rekaman_giliran("t1") is None

    def test_rekaman_lama_ikut_terhapus(self, store):
        store.tulis([trace("lama", {"a": 1}, menit_lalu=60 * 24 * 8), trace("baru", {"a": 2})])
        store.hapus_lama(SEKARANG - timedelta(days=7))
        assert store.rekaman_giliran("lama") is None
        assert store.rekaman_giliran("baru") == {"a": 2}


class TestUjiCoba:
    def test_tidak_dihitung_di_performa(self, store):
        store.tulis(
            [
                turn("mhs", endpoint="chat_stream", total_ms=1000),
                node("mhs", "generate", 900.0),
                turn("admin", endpoint="uji_coba", total_ms=50_000),
                node("admin", "generate", 49_000.0),
            ]
        )
        r = store.ringkasan(SEKARANG - timedelta(hours=1), SEKARANG, audit=True)
        assert r["turn_count"] == 1
        assert r["p95_total_ms"] == 1000
        [generate] = r["per_node"]
        assert generate["count"] == 1 and generate["p95_ms"] == 900

    def test_filter_jalur(self, store):
        store.tulis([turn("mhs", endpoint="chat_stream"), turn("admin", endpoint="uji_coba")])
        total, items = store.daftar_giliran(SEKARANG - timedelta(hours=1), endpoint="uji_coba")
        assert total == 1 and items[0]["turn_id"] == "admin"


def test_saringan_aturan_diurutkan_sesudah_sapaan(store):
    """T55: `rule_gate` dulu tidak ada di NODE_ORDER dan tampil paling bawah."""
    store.tulis(
        [
            turn("t1"),
            node("t1", "generate", 5.0, position=4),
            node("t1", "rule_gate", 1.0, position=3),
            node("t1", "smalltalk", 1.0, position=2),
            node("t1", "jev_gate", 1.0, position=5),
        ]
    )
    r = store.ringkasan(SEKARANG - timedelta(hours=1), SEKARANG, audit=True)
    assert [n["node"] for n in r["per_node"]] == [
        "smalltalk",
        "rule_gate",
        "jev_gate",
        "generate",
    ]
