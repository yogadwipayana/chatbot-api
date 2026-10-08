"""Halaman Log dashboard (`/api/admin/logs/*`) dan pencatatan giliran chat ke SQLite.

Akses minimal admin (staf ditolak) sudah dibangkitkan otomatis dari api.yaml di
`test_admin_roles.py`. Di sini: isi respons, dan bahwa log `app.audit` tidak
pernah sampai ke role admin lewat jalur mana pun.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.config import get_settings
from app.observability.logstore import waktu_iso
from app.observability.rekaman import buka, kemas
from tests.api.conftest import ADMIN_BIASA_EMAIL, ADMIN_EMAIL, STAF_EMAIL

BARU = waktu_iso(datetime.now(UTC) - timedelta(minutes=5))


def _log(logger: str, pesan: str, *, levelno: int = 30, turn_id: str | None = None):
    return (
        "app",
        {
            "timestamp": BARU,
            "level": {20: "INFO", 30: "WARNING", 40: "ERROR"}[levelno],
            "levelno": levelno,
            "logger": logger,
            "message": pesan,
            "location": "modul:1",
            "traceback": None,
            "turn_id": turn_id,
        },
    )


@pytest.fixture
def terisi(log_store):
    log_store.tulis(
        [
            (
                "turn",
                {
                    "turn_id": "t1",
                    "timestamp": BARU,
                    "endpoint": "chat",
                    "session_id": "sesi-uji-12345",
                    "message_id": "m1",
                    "unit": None,
                    "outcome": "answer",
                    "last_node": "generate",
                    "total_ms": 1200,
                    "ttft_ms": None,
                    "status": "ok",
                    "langsmith_run_id": None,
                },
            ),
            (
                "node",
                {
                    "turn_id": "t1",
                    "position": 1,
                    "node": "retrieve",
                    "started_at": BARU,
                    "duration_ms": 300.0,
                    "status": "ok",
                    "detail": {"document_count": 2},
                },
            ),
            _log("app.rag.gate", "Gerbang JEV gagal", turn_id="t1"),
            _log("app.audit", "Akun x@instiki.ac.id dihapus oleh y", turn_id="t1"),
            _log("app.audit", "Kill switch dinyalakan", levelno=40),
        ]
    )
    return log_store


@pytest.fixture
def admin(headers_for):
    return headers_for(ADMIN_BIASA_EMAIL)


@pytest.fixture
def superadmin(headers_for):
    return headers_for(ADMIN_EMAIL)


class TestAkses:
    def test_staf_ditolak(self, client, headers_for):
        r = client.get("/api/admin/logs/summary", headers=headers_for(STAF_EMAIL))
        assert r.status_code == 403

    @pytest.mark.parametrize("email", [ADMIN_BIASA_EMAIL, ADMIN_EMAIL])
    def test_admin_dan_superadmin_boleh(self, client, headers_for, email):
        r = client.get("/api/admin/logs/summary", headers=headers_for(email))
        assert r.status_code == 200


class TestRingkasan:
    def test_kosong_tetap_200(self, client, admin):
        r = client.get("/api/admin/logs/summary?range=7d", headers=admin)
        body = r.json()
        assert body["turn_count"] == 0
        assert len(body["per_hour"]) >= 7 * 24

    def test_rentang_tidak_dikenal_422(self, client, admin):
        assert (
            client.get("/api/admin/logs/summary?range=30d", headers=admin).status_code == 422
        )

    def test_isi(self, client, terisi, superadmin):
        body = client.get("/api/admin/logs/summary", headers=superadmin).json()
        assert body["turn_count"] == 1
        assert body["per_node"][0]["node"] == "retrieve"
        assert body["exit_points"] == [{"node": "generate", "count": 1}]
        assert body["error_log_count"] == 1

    def test_error_audit_tidak_dihitung_untuk_admin(self, client, terisi, admin):
        body = client.get("/api/admin/logs/summary", headers=admin).json()
        assert body["error_log_count"] == 0


class TestGiliran:
    def test_daftar(self, client, terisi, admin):
        body = client.get("/api/admin/logs/turns?status=ok", headers=admin).json()
        assert body["total"] == 1
        assert body["items"][0]["message_id"] == "m1"

    def test_detail(self, client, terisi, superadmin):
        body = client.get("/api/admin/logs/turns/t1", headers=superadmin).json()
        assert body["nodes"][0]["detail"] == {"document_count": 2}
        assert len(body["logs"]) == 2

    def test_detail_tanpa_audit_untuk_admin(self, client, terisi, admin):
        body = client.get("/api/admin/logs/turns/t1", headers=admin).json()
        assert [x["logger"] for x in body["logs"]] == ["app.rag.gate"]

    def test_tidak_ada_404(self, client, admin):
        r = client.get("/api/admin/logs/turns/tidak-ada", headers=admin)
        assert r.status_code == 404
        assert "masa simpan" in r.json()["detail"]


class TestLogAplikasi:
    def test_superadmin_melihat_audit(self, client, terisi, superadmin):
        body = client.get("/api/admin/logs/app?logger=app.audit", headers=superadmin).json()
        assert body["total"] == 2
        assert "app.audit" in body["loggers"]

    @pytest.mark.parametrize(
        "query",
        ["", "?logger=app.audit", "?q=dihapus", "?level=ERROR", "?logger=app"],
    )
    def test_admin_tidak_pernah_menerima_audit(self, client, terisi, admin, query):
        r = client.get(f"/api/admin/logs/app{query}", headers=admin)
        assert r.status_code == 200
        body = r.json()
        assert all(i["logger"] != "app.audit" for i in body["items"])
        assert "app.audit" not in body["loggers"]
        assert "instiki.ac.id" not in r.text

    def test_filter_audit_untuk_admin_kosong_bukan_403(self, client, terisi, admin):
        body = client.get("/api/admin/logs/app?logger=app.audit", headers=admin).json()
        assert body == {"total": 0, "items": [], "loggers": ["app.rag.gate"]}

    def test_level_tidak_dikenal_422(self, client, admin):
        assert client.get("/api/admin/logs/app?level=FATAL", headers=admin).status_code == 422


class TestPencatatanGiliranChat:
    def test_chat_mencatat_giliran_dan_node(self, client, payload, log_sink, chat_logger):
        assert client.post("/api/chat", json=payload).status_code == 200
        [turn] = log_sink.of("turn")
        assert turn["endpoint"] == "chat"
        assert turn["status"] == "ok"
        assert turn["outcome"] == "answer"
        assert turn["last_node"] == "generate"
        assert turn["message_id"] == chat_logger.message_id
        nodes = log_sink.of("node")
        assert {n["turn_id"] for n in nodes} == {turn["turn_id"]}
        assert [n["node"] for n in nodes][-1] == "generate"

    def test_teks_nim_dan_rekaman_ikut_tercatat(self, client, payload, log_sink):
        """Keputusan 2026-10-08 (logs.md tahap 4): teks disimpan untuk tab Graf."""
        client.post("/api/chat", json=payload)
        [turn] = log_sink.of("turn")
        assert turn["question"] == payload["question"]
        assert turn["nim"] == payload["nim"]
        assert turn["answer"]
        [trace] = log_sink.of("trace")
        isi = buka(
            kemas(trace["isi"], sensitif=trace["sensitif"], pengganti=trace["pengganti"])
        )
        assert isi["input"]["question"] == payload["question"]
        assert [n["node"] for n in isi["nodes"]][-1] == "generate"

    def test_log_node_io_mati_tanpa_teks_mahasiswa(self, client, payload, log_sink):
        settings = get_settings().model_copy(update={"log_node_io": False})
        client.app.dependency_overrides[get_settings] = lambda: settings
        client.post("/api/chat", json=payload)
        assert log_sink.of("trace") == []
        assert payload["question"] not in repr(log_sink.rows)
        assert payload["nim"] not in repr(log_sink.rows)

    def test_streaming_mencatat_ttft(self, client, payload, log_sink):
        with client.stream("POST", "/api/chat/stream", json=payload) as r:
            r.read()
        [turn] = log_sink.of("turn")
        assert turn["endpoint"] == "chat_stream"
        assert turn["status"] == "ok"
        assert turn["ttft_ms"] is not None
        assert turn["ttft_ms"] <= turn["total_ms"]

    def test_penolakan_berhenti_di_refuse(
        self, make_client, weak_documents, payload, log_sink
    ):
        make_client(weak_documents).post("/api/chat", json=payload)
        [turn] = log_sink.of("turn")
        assert turn["outcome"] == "refusal"
        assert turn["last_node"] == "refuse"


class TestRekamanDanGraf:
    def test_rekaman_giliran(self, client, log_store, admin):
        isi = {
            "input": {"question": "kapan KRS?"},
            "output": None,
            "nodes": [
                {
                    "position": 1,
                    "node": "sanitize",
                    "input": {"question": "kapan KRS?"},
                    "output": {"clean": "kapan KRS?"},
                    "route": None,
                },
            ],
            "calls": [
                {
                    "id": "c1",
                    "parent_id": None,
                    "position": 1,
                    "name": "generate_answer",
                    "kind": "llm",
                    "started_at": BARU,
                    "duration_ms": 10.0,
                    "status": "ok",
                    "error": None,
                    "model": "m",
                    "usage": {"input_tokens": 1},
                    "cost_usd": None,
                    "input": [{"role": "human", "content": "x"}],
                    "output": {"content": "y"},
                },
            ],
        }
        log_store.tulis([("trace", {"turn_id": "t9", "timestamp": BARU, "isi": isi})])
        r = client.get("/api/admin/logs/turns/t9/trace", headers=admin)
        assert r.status_code == 200
        body = r.json()
        assert body["turn_id"] == "t9"
        assert body["nodes"][0]["output"] == {"clean": "kapan KRS?"}
        assert body["calls"][0]["kind"] == "llm"

    def test_tanpa_rekaman_404(self, client, terisi, admin):
        r = client.get("/api/admin/logs/turns/t1/trace", headers=admin)
        assert r.status_code == 404

    def test_graf_dari_langgraph(self, client, admin):
        r = client.get("/api/admin/logs/graph", headers=admin)
        assert r.status_code == 200
        body = r.json()
        nodes = {n["id"]: n["group"] for n in body["nodes"]}
        assert {"__start__", "__end__", "rule_gate", "jev_gate", "generate"} <= set(nodes)
        assert nodes["rewrite"] == nodes["retrieve"] == "cari"
        assert nodes["generate"] is None
        sisi = {(e["source"], e["target"]): e for e in body["edges"]}
        assert sisi[("rule_gate", "jev_gate")]["conditional"]
        assert sisi[("rule_gate", "rewrite")]["conditional"]
        assert sisi[("rewrite", "retrieve")]["conditional"] is False
        assert sisi[("validate_context", "__end__")]["label"] == "selesai"

    def test_graf_untuk_staf_ditolak(self, client, headers_for):
        r = client.get("/api/admin/logs/graph", headers=headers_for(STAF_EMAIL))
        assert r.status_code == 403


class TestUjiCobaTercatat:
    def test_uji_coba_dicatat_sebagai_jalur_sendiri(self, client, log_sink, superadmin):
        r = client.post(
            "/api/admin/test-query", json={"question": "kapan KRS dibuka?"}, headers=superadmin
        )
        assert r.status_code == 200
        [turn] = log_sink.of("turn")
        assert turn["endpoint"] == "uji_coba"
        assert turn["question"] == "kapan KRS dibuka?"
        assert turn["message_id"] is None
        assert r.json()["turn_id"] == turn["turn_id"]
        assert log_sink.of("trace")[0]["turn_id"] == turn["turn_id"]

    def test_filter_jalur(self, client, log_store, admin):
        log_store.tulis(
            [
                (
                    "turn",
                    {
                        "turn_id": "u1",
                        "timestamp": BARU,
                        "endpoint": "uji_coba",
                        "status": "ok",
                    },
                ),
                (
                    "turn",
                    {
                        "turn_id": "m1",
                        "timestamp": BARU,
                        "endpoint": "chat_stream",
                        "status": "ok",
                    },
                ),
            ]
        )
        r = client.get("/api/admin/logs/turns?endpoint=uji_coba", headers=admin)
        assert [i["turn_id"] for i in r.json()["items"]] == ["u1"]
        ringkas = client.get("/api/admin/logs/summary", headers=admin).json()
        assert ringkas["turn_count"] == 1
