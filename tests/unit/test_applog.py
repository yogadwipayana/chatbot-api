"""Penulis log SQLite, perekam node LangGraph, dan pencatat giliran (`applog.py`)."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import pytest

from app.observability.applog import (
    LogWriter,
    NodeRecorder,
    SQLiteLogHandler,
    catat_giliran,
    configure_logging,
    turn_id_var,
)
from app.observability.logstore import LogStore
from app.rag.chain import run_pipeline
from app.rag.gate import GateLabel, GateVerdict
from tests.fixtures.fakes import FakeLogSink, RecordingLLM

SEJAK = datetime.now(UTC) - timedelta(hours=1)


@pytest.fixture
def store(tmp_path) -> LogStore:
    return LogStore(tmp_path / "app.db")


async def jalankan(question: str, retriever, llm, **lain) -> tuple:
    recorder = NodeRecorder()
    outcome = await run_pipeline(
        question, retriever=retriever, llm_call=llm, callbacks=[recorder], **lain
    )
    return outcome, recorder


class TestNodeRecorder:
    async def test_jawaban_melewati_seluruh_jalur_utama(self, strong_retriever, llm):
        _, rec = await jalankan("kapan pengisian KRS dibuka?", strong_retriever, llm)
        assert [n["node"] for n in rec.nodes] == [
            "sanitize",
            "sensitive",
            "smalltalk",
            "jev_gate",
            "rewrite",
            "retrieve",
            "validate_context",
            "generate",
        ]
        assert [n["urutan"] for n in rec.nodes] == list(range(1, 9))
        assert rec.node_terakhir == "generate"
        assert all(n["status"] == "ok" and n["durasi_ms"] >= 0 for n in rec.nodes)

    async def test_router_bersyarat_tidak_ikut_tercatat(self, strong_retriever, llm):
        _, rec = await jalankan("kapan pengisian KRS dibuka?", strong_retriever, llm)
        assert not any(n["node"].startswith("selesai_atau") for n in rec.nodes)
        assert "route_context" not in {n["node"] for n in rec.nodes}

    async def test_detail_per_node_tanpa_teks_mahasiswa(self, strong_retriever, llm):
        pertanyaan = "kapan pengisian KRS dibuka?"
        _, rec = await jalankan(pertanyaan, strong_retriever, llm)
        detail = {n["node"]: n["detail"] for n in rec.nodes}
        assert detail["sanitize"] == {"panjang": len(pertanyaan)}
        assert detail["jev_gate"] == {"dimatikan": True}
        assert detail["rewrite"] == {"query_berubah": False}
        assert detail["retrieve"] == {"jumlah_dokumen": 2}
        assert detail["validate_context"]["keputusan"] == "proceed"
        assert pertanyaan not in repr(rec.nodes)

    async def test_berhenti_di_node_sensitif(self, strong_retriever, llm):
        _, rec = await jalankan("saya stres dan ingin bunuh diri", strong_retriever, llm)
        assert rec.node_terakhir == "sensitive"
        assert rec.nodes[-1]["detail"]["dialihkan"] is True

    async def test_vonis_jev_tercatat(self, strong_retriever, llm):
        async def gerbang(q, riwayat):
            return GateVerdict(GateLabel.NONSENSE, 0.97, blocked=True, cost_usd=0.0002)

        _, rec = await jalankan("asdf qwer zxcv", strong_retriever, llm, gate_call=gerbang)
        assert rec.node_terakhir == "jev_gate"
        assert rec.nodes[-1]["detail"] == {
            "label": "nonsense",
            "confidence": 0.97,
            "blocked": True,
            "biaya_usd": 0.0002,
            "error": None,
        }

    async def test_node_gagal_tercatat_sebagai_error(self, llm):
        class RetrieverRusak:
            async def ainvoke(self, q, *, unit=None):
                raise RuntimeError("database mati")

        recorder = NodeRecorder()
        with pytest.raises(RuntimeError):
            await run_pipeline(
                "kapan KRS?", retriever=RetrieverRusak(), llm_call=llm, callbacks=[recorder]
            )
        gagal = recorder.nodes[-1]
        assert gagal["node"] == "retrieve"
        assert gagal["status"] == "error"
        assert gagal["error_tipe"] == "RuntimeError"
        assert gagal["error_pesan"] == "database mati"


class TestCatatGiliran:
    def test_status_ok_dan_detail_generate(self):
        sink = FakeLogSink()
        llm = RecordingLLM()
        llm.model = "cx/gpt-5.5"
        llm.usage = {"input_tokens": 1000, "output_tokens": 100}
        with catat_giliran(sink, endpoint="chat", session_id="s", unit="Keuangan") as g:
            assert turn_id_var.get() == g.turn_id
            g.recorder.nodes.append(
                {
                    "node": "generate",
                    "urutan": 1,
                    "mulai": "x",
                    "durasi_ms": 5.0,
                    "status": "ok",
                    "error_tipe": None,
                    "error_pesan": None,
                    "detail": {},
                }
            )
            g.selesai(hasil="answer", message_id="m1", langsmith_run_id=None, llm_call=llm)
        assert turn_id_var.get() is None
        [baris] = sink.of("turn")
        assert baris["status"] == "ok"
        assert baris["hasil"] == "answer"
        assert baris["node_terakhir"] == "generate"
        assert baris["unit"] == "Keuangan"
        [n] = sink.of("node")
        assert n["turn_id"] == g.turn_id
        assert n["detail"]["model"] == "cx/gpt-5.5"
        assert n["detail"]["biaya_usd"] == pytest.approx(0.008)

    def test_galat_dicatat_dan_diteruskan(self, caplog):
        sink = FakeLogSink()
        with (
            caplog.at_level(logging.ERROR, logger="app"),
            pytest.raises(ValueError),
            catat_giliran(sink, endpoint="chat", session_id="s", unit=None),
        ):
            raise ValueError("rusak")
        assert sink.of("turn")[0]["status"] == "error"
        assert "Giliran chat gagal" in caplog.text

    async def test_dibatalkan(self):
        sink = FakeLogSink()

        async def tugas():
            with catat_giliran(sink, endpoint="chat_stream", session_id="s", unit=None):
                await asyncio.sleep(10)

        t = asyncio.create_task(tugas())
        await asyncio.sleep(0)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert sink.of("turn")[0]["status"] == "dibatalkan"

    def test_tanpa_sink_tidak_menulis_apa_pun(self):
        with catat_giliran(None, endpoint="chat", session_id=None, unit=None) as g:
            g.selesai(hasil="answer", message_id=None, langsmith_run_id=None)

    def test_ttft_hanya_token_pertama(self):
        with catat_giliran(None, endpoint="chat_stream", session_id=None, unit=None) as g:
            g.token_pertama()
            pertama = g.ttft_ms
            g.t0 -= 10
            g.token_pertama()
        assert g.ttft_ms == pertama


class TestLogWriter:
    def test_menulis_ke_sqlite_saat_berhenti(self, store):
        writer = LogWriter(store, retention_days=7)
        writer.start()
        with catat_giliran(writer, endpoint="chat", session_id="s", unit=None) as g:
            g.selesai(hasil="refusal", message_id=None, langsmith_run_id=None)
        writer.stop()
        total, items = store.daftar_giliran(SEJAK)
        assert total == 1 and items[0]["hasil"] == "refusal"

    def test_antrean_penuh_membuang_bukan_memblokir(self, store):
        writer = LogWriter(store, retention_days=7, max_antrean=1)
        writer.kirim("app", {})
        writer.kirim("app", {})
        assert writer.dibuang == 1

    def test_galat_menulis_tidak_melempar(self, store):
        """Batch rusak dibuang dan dicatat ke stderr; thread penulis tidak ikut mati."""
        LogWriter(store, retention_days=7)._tulis([("jenis-tak-dikenal", {})])

    def test_baris_lama_dihapus_saat_start(self, store):
        store.tulis(
            [
                (
                    "turn",
                    {
                        "turn_id": "lama",
                        "waktu": "2000-01-01T00:00:00.000Z",
                        "endpoint": "chat",
                        "status": "ok",
                    },
                )
            ]
        )
        writer = LogWriter(store, retention_days=7)
        writer.start()
        writer.stop()
        assert store.daftar_giliran(datetime(1999, 1, 1, tzinfo=UTC))[0] == 0


class TestLogging:
    @pytest.fixture(autouse=True)
    def pulihkan(self):
        app_logger = logging.getLogger("app")
        semula = (list(app_logger.handlers), app_logger.level)
        yield
        configure_logging("INFO", None)
        for h in list(app_logger.handlers):
            if h not in semula[0]:
                app_logger.removeHandler(h)
        app_logger.setLevel(semula[1])

    def test_info_dan_turn_id_masuk_sqlite(self):
        sink = FakeLogSink()
        configure_logging("INFO", sink)  # type: ignore[arg-type]
        token = turn_id_var.set("giliran-1")
        try:
            logging.getLogger("app.main").info("Tracing LangSmith %s", "mati")
        finally:
            turn_id_var.reset(token)
        [baris] = sink.of("app")
        assert baris["pesan"] == "Tracing LangSmith mati"
        assert baris["level"] == "INFO"
        assert baris["logger"] == "app.main"
        assert baris["turn_id"] == "giliran-1"

    def test_traceback_ikut(self):
        sink = FakeLogSink()
        configure_logging("INFO", sink)  # type: ignore[arg-type]
        try:
            raise KeyError("x")
        except KeyError:
            logging.getLogger("app.x").exception("gagal")
        assert "KeyError" in sink.of("app")[0]["traceback"]

    def test_level_minimum_dihormati(self):
        sink = FakeLogSink()
        configure_logging("WARNING", sink)  # type: ignore[arg-type]
        logging.getLogger("app.x").info("tidak tercatat")
        assert sink.of("app") == []

    def test_konfigurasi_ulang_tidak_menumpuk_handler(self):
        configure_logging("INFO", FakeLogSink())  # type: ignore[arg-type]
        configure_logging("INFO", FakeLogSink())  # type: ignore[arg-type]
        milik_kita = [
            h for h in logging.getLogger("app").handlers if getattr(h, "_pandu_applog", False)
        ]
        assert len(milik_kita) == 2  # satu konsol + satu SQLite
        assert sum(isinstance(h, SQLiteLogHandler) for h in milik_kita) == 1
