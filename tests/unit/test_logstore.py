"""SQLite log aplikasi (`app/observability/logstore.py`)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.observability.logstore import LogStore, persentil, waktu_iso

SEKARANG = datetime(2026, 9, 25, 10, 30, tzinfo=UTC)


def turn(turn_id: str, *, menit_lalu: int = 5, **lain) -> tuple[str, dict]:
    data = {
        "turn_id": turn_id,
        "waktu": waktu_iso(SEKARANG - timedelta(minutes=menit_lalu)),
        "endpoint": "chat",
        "session_id": "sesi-1",
        "message_id": None,
        "unit": None,
        "hasil": "answer",
        "node_terakhir": "generate",
        "total_ms": 1000,
        "ttft_ms": None,
        "status": "ok",
        "langsmith_run_id": None,
    }
    data.update(lain)
    return "turn", data


def node(
    turn_id: str, nama: str, durasi: float, *, urutan: int = 1, **lain
) -> tuple[str, dict]:
    data = {
        "turn_id": turn_id,
        "urutan": urutan,
        "node": nama,
        "mulai": waktu_iso(SEKARANG - timedelta(minutes=5)),
        "durasi_ms": durasi,
        "status": "ok",
        "detail": {"contoh": 1},
    }
    data.update(lain)
    return "node", data


def log(logger: str, pesan: str, *, levelno: int = 20, turn_id=None, menit_lalu=5):
    return "app", {
        "waktu": waktu_iso(SEKARANG - timedelta(minutes=menit_lalu)),
        "level": {20: "INFO", 30: "WARNING", 40: "ERROR"}[levelno],
        "levelno": levelno,
        "logger": logger,
        "pesan": pesan,
        "lokasi": "modul:1",
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
                node("lama", "sanitize", 1, mulai=waktu_iso(SEKARANG - timedelta(days=8))),
            ]
        )
        terhapus = store.hapus_lama(SEKARANG - timedelta(days=7))
        assert terhapus == 3
        total, items = store.daftar_giliran(SEKARANG - timedelta(days=30))
        assert [i["turn_id"] for i in items] == ["baru"]


class TestRingkasan:
    def test_kosong(self, store):
        r = store.ringkasan(SEKARANG - timedelta(hours=2), SEKARANG, audit=True)
        assert r["jumlah_giliran"] == 0
        assert r["p95_total_ms"] is None
        assert r["rasio_error"] == 0.0
        assert len(r["per_jam"]) == 3

    def test_kpi_node_dan_titik_keluar(self, store):
        store.tulis(
            [
                turn("t1", total_ms=1000),
                turn("t2", total_ms=3000, status="error"),
                turn("t3", total_ms=200, hasil="rejected", node_terakhir="jev_gate"),
                turn("t4", total_ms=100, hasil="refusal", node_terakhir="refuse"),
                node("t1", "retrieve", 100),
                node("t2", "retrieve", 300, status="error"),
                node("t1", "sanitize", 1),
                log("app.rag.gate", "gagal", levelno=40),
                log("app.audit", "akun dihapus", levelno=40),
            ]
        )
        r = store.ringkasan(SEKARANG - timedelta(hours=1), SEKARANG, audit=True)
        assert r["jumlah_giliran"] == 4
        assert r["giliran_error"] == 1 and r["rasio_error"] == 0.25
        assert r["diblokir_jev"] == 1 and r["rasio_diblokir_jev"] == 0.25
        assert r["log_error"] == 2
        # Urutan mengikuti graf, bukan abjad.
        assert [n["node"] for n in r["per_node"]] == ["sanitize", "retrieve"]
        retrieve = r["per_node"][1]
        assert retrieve["jumlah"] == 2 and retrieve["error"] == 1
        assert retrieve["p50_ms"] == 200
        assert r["titik_keluar"][0] == {"node": "generate", "jumlah": 2}
        jam_isi = [j for j in r["per_jam"] if j["giliran"]]
        assert len(jam_isi) == 1 and jam_isi[0]["giliran_error"] == 1

    def test_log_audit_tidak_dihitung_tanpa_hak_audit(self, store):
        store.tulis([log("app.audit", "x", levelno=40), log("app.audit.sub", "y", levelno=40)])
        r = store.ringkasan(SEKARANG - timedelta(hours=1), SEKARANG, audit=False)
        assert r["log_error"] == 0


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
                node("t1", "retrieve", 5, urutan=2, detail={"jumlah_dokumen": 3}),
                node("t1", "sanitize", 1, urutan=1),
                log("app.rag.gate", "lambat", turn_id="t1"),
                log("app.audit", "rahasia", turn_id="t1"),
            ]
        )
        d = store.detail_giliran("t1", audit=False)
        assert [n["node"] for n in d["nodes"]] == ["sanitize", "retrieve"]
        assert d["nodes"][1]["detail"] == {"jumlah_dokumen": 3}
        assert [x["pesan"] for x in d["logs"]] == ["lambat"]
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
