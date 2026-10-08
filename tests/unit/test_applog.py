"""Penulis log SQLite, perekam node LangGraph, dan pencatat giliran (`applog.py`)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.observability.applog import (
    WRAPPER_NODES,
    LogWriter,
    NodeRecorder,
    SQLiteLogHandler,
    catat_giliran,
    configure_logging,
    turn_id_var,
)
from app.observability.chatlog import SENSITIVE_PLACEHOLDER
from app.observability.logstore import LogStore
from app.observability.rekaman import ACARA_PANGGILAN, buka, catat_panggilan, kemas
from app.rag.chain import run_pipeline
from app.rag.gate import GateLabel, GateVerdict
from app.rag.rewriter import Turn
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
            "rule_gate",
            "jev_gate",
            "rewrite",
            "retrieve",
            "validate_context",
            "generate",
        ]
        assert [n["position"] for n in rec.nodes] == list(range(1, 10))
        assert rec.node_terakhir == "generate"
        assert all(n["status"] == "ok" and n["duration_ms"] >= 0 for n in rec.nodes)

    async def test_router_bersyarat_tidak_ikut_tercatat(self, strong_retriever, llm):
        _, rec = await jalankan("kapan pengisian KRS dibuka?", strong_retriever, llm)
        assert not any(n["node"].startswith("selesai_atau") for n in rec.nodes)
        assert "route_context" not in {n["node"] for n in rec.nodes}

    async def test_pembungkus_subgraph_tidak_ikut_tercatat(self, strong_retriever, llm):
        """rewrite + retrieve tercatat sendiri-sendiri, bukan juga sebagai `cari`."""
        _, rec = await jalankan("kapan pengisian KRS dibuka?", strong_retriever, llm)
        nama = [n["node"] for n in rec.nodes]
        assert not WRAPPER_NODES & set(nama)
        assert nama.count("rewrite") == nama.count("retrieve") == 1

    async def test_detail_per_node_tanpa_teks_mahasiswa(self, strong_retriever, llm):
        pertanyaan = "kapan pengisian KRS dibuka?"
        _, rec = await jalankan(pertanyaan, strong_retriever, llm)
        detail = {n["node"]: n["detail"] for n in rec.nodes}
        assert detail["sanitize"] == {"length": len(pertanyaan)}
        assert detail["jev_gate"] == {"disabled": True}
        assert detail["rewrite"] == {"query_rewritten": False}
        assert detail["retrieve"] == {"document_count": 2}
        assert detail["validate_context"]["decision"] == "proceed"
        assert pertanyaan not in repr(rec.nodes)

    async def test_berhenti_di_node_sensitif(self, strong_retriever, llm):
        _, rec = await jalankan("saya stres dan ingin bunuh diri", strong_retriever, llm)
        assert rec.node_terakhir == "sensitive"
        assert rec.nodes[-1]["detail"]["redirected"] is True

    async def test_vonis_jev_tercatat(self, strong_retriever, llm):
        async def gerbang(q, riwayat, unit=None):
            return GateVerdict(GateLabel.NONSENSE, 0.97, blocked=True, cost_usd=0.0002)

        _, rec = await jalankan(
            "resep rendang padang", strong_retriever, llm, gate_call=gerbang
        )
        # Pencarian berjalan paralel dan validate_context tetap menjadi titik
        # temu, tetapi alur berhenti karena vonis JEV.
        assert rec.node_terakhir == "jev_gate"
        assert "generate" not in {n["node"] for n in rec.nodes}
        detail = {n["node"]: n["detail"] for n in rec.nodes}
        assert detail["jev_gate"] == {
            "label": "nonsense",
            "confidence": 0.97,
            "blocked": True,
            "cost_usd": 0.0002,
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
        assert gagal["error_type"] == "RuntimeError"
        assert gagal["error_message"] == "database mati"


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
                    "position": 1,
                    "started_at": "x",
                    "duration_ms": 5.0,
                    "status": "ok",
                    "error_type": None,
                    "error_message": None,
                    "detail": {},
                }
            )
            g.selesai(hasil="answer", message_id="m1", langsmith_run_id=None, llm_call=llm)
        assert turn_id_var.get() is None
        [baris] = sink.of("turn")
        assert baris["status"] == "ok"
        assert baris["outcome"] == "answer"
        assert baris["last_node"] == "generate"
        assert baris["unit"] == "Keuangan"
        [n] = sink.of("node")
        assert n["turn_id"] == g.turn_id
        assert n["detail"]["model"] == "cx/gpt-5.5"
        assert n["detail"]["cost_usd"] == pytest.approx(0.008)

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
        assert total == 1 and items[0]["outcome"] == "refusal"

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
                        "timestamp": "2000-01-01T00:00:00.000Z",
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
        assert baris["message"] == "Tracing LangSmith mati"
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


# --- Rekaman input/output untuk tab Graf (LOG_NODE_IO) ----------------------


class LLMLangChain:
    """Pengganti `LLMCall` yang memakai runnable LangChain sungguhan (model palsu),
    supaya callback panggilan LLM benar-benar terpancar seperti di produksi."""

    def __init__(self, balasan: str = "Jawaban [Panduan Akademik 2025, hal. 12].") -> None:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        self.chat = FakeListChatModel(responses=[balasan])

    async def __call__(self, wrapped_question: str, documents) -> str:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [("system", "KONTEKS: {context}"), ("human", "{question}")]
        )
        hasil = await (prompt | self.chat).ainvoke(
            {"context": "isi", "question": wrapped_question},
            config={"run_name": "generate_answer"},
        )
        return str(hasil.content)


async def jalankan_rekam(question: str, retriever, llm, **lain) -> tuple:
    recorder = NodeRecorder(rekam_io=True)
    outcome = await run_pipeline(
        question, retriever=retriever, llm_call=llm, callbacks=[recorder], **lain
    )
    return outcome, recorder


class TestRekamanIO:
    async def test_mati_secara_bawaan(self, strong_retriever, llm):
        _, rec = await jalankan("kapan pengisian KRS dibuka?", strong_retriever, llm)
        isi = rec.rekaman()
        assert isi["nodes"] == [] and isi["calls"] == [] and isi["input"] is None

    async def test_input_output_dan_rute_tiap_node(self, strong_retriever, llm):
        _, rec = await jalankan_rekam("kapan pengisian KRS dibuka?", strong_retriever, llm)
        isi = rec.rekaman()
        assert [n["node"] for n in isi["nodes"]] == [n["node"] for n in rec.nodes]
        assert [n["position"] for n in isi["nodes"]] == [n["position"] for n in rec.nodes]
        per_node = {n["node"]: n for n in isi["nodes"]}
        assert isi["input"]["question"] == "kapan pengisian KRS dibuka?"
        assert per_node["sanitize"]["output"] == {"clean": "kapan pengisian KRS dibuka?"}
        assert per_node["generate"]["input"]["clean"] == "kapan pengisian KRS dibuka?"
        # Rute: keluaran router sesudah node; paralel = daftar nama node.
        assert per_node["rule_gate"]["route"] == ["jev_gate", "cari"]
        assert per_node["validate_context"]["route"] == "generate"
        assert per_node["generate"]["route"] is None
        assert isi["output"].text.startswith("Jawaban")
        assert rec.teks_bersih == "kapan pengisian KRS dibuka?"

    async def test_llm_langchain_menjadi_satu_panggilan(self, strong_retriever):
        """Rantai `prompt | model` bernama generate_answer: satu panggilan LLM
        berisi pesan sungguhan, tanpa run template prompt maupun ChatModel lepas."""
        _, rec = await jalankan_rekam(
            "kapan pengisian KRS dibuka?", strong_retriever, LLMLangChain()
        )
        calls = rec.rekaman()["calls"]
        [panggilan] = [c for c in calls if c["kind"] == "llm"]
        posisi_generate = next(n["position"] for n in rec.nodes if n["node"] == "generate")
        assert panggilan["name"] == "generate_answer"
        assert panggilan["position"] == posisi_generate
        assert panggilan["parent_id"] is None
        assert panggilan["status"] == "ok"
        assert [m.type for m in panggilan["input"]] == ["system", "human"]
        assert panggilan["output"].content.startswith("Jawaban")
        assert not any(c["name"] in {"ChatPromptTemplate", "FakeListChatModel"} for c in calls)

    async def test_router_tidak_menjadi_panggilan(self, strong_retriever):
        _, rec = await jalankan_rekam(
            "kapan pengisian KRS dibuka?", strong_retriever, LLMLangChain()
        )
        nama = {c["name"] for c in rec.rekaman()["calls"]}
        assert not any(n.startswith("selesai_atau") for n in nama)
        assert "route_context" not in nama

    async def test_panggilan_jev_lewat_custom_event(self, strong_retriever, llm):
        import time

        async def gerbang(q, riwayat, unit=None):
            mulai = time.perf_counter()
            await catat_panggilan(
                jenis="jev",
                nama="jev_classify",
                masukan={"state": {"pesan_terbaru": q}},
                keluaran={"label": "academic"},
                mulai=mulai,
            )
            return GateVerdict(GateLabel.ACADEMIC, 0.9, blocked=False)

        _, rec = await jalankan_rekam(
            "kapan pengisian KRS dibuka?", strong_retriever, llm, gate_call=gerbang
        )
        [jev] = [c for c in rec.rekaman()["calls"] if c["kind"] == "jev"]
        posisi_jev = next(n["position"] for n in rec.nodes if n["node"] == "jev_gate")
        assert jev["position"] == posisi_jev
        assert jev["input"] == {"state": {"pesan_terbaru": "kapan pengisian KRS dibuka?"}}
        assert jev["status"] == "ok"

    async def test_event_dengan_induk_asing_dicari_lewat_nama_node(self):
        """Tracing menyala: `@traceable` di JEV membuat induk event menjadi run
        LangSmith yang tidak dikenal perekam; node-nya dibaca dari metadata."""
        rec = NodeRecorder(rekam_io=True)
        node_run = uuid.uuid4()
        await rec.on_chain_start({}, {}, run_id=uuid.uuid4(), metadata={}, name="pandu_chat")
        await rec.on_chain_start(
            {},
            {"clean": "q"},
            run_id=node_run,
            metadata={"langgraph_node": "jev_gate"},
            name="jev_gate",
        )
        await rec.on_custom_event(
            ACARA_PANGGILAN,
            {
                "kind": "jev",
                "name": "jev_classify",
                "input": {},
                "output": None,
                "duration_ms": 5.0,
            },
            run_id=uuid.uuid4(),
            metadata={"langgraph_node": "jev_gate"},
        )
        [jev] = rec.rekaman()["calls"]
        assert jev["kind"] == "jev" and jev["position"] == 1 and jev["parent_id"] is None

    async def test_node_gagal_tetap_terekam(self, llm):
        class RetrieverRusak:
            async def ainvoke(self, q, *, unit=None):
                raise RuntimeError("database mati")

        recorder = NodeRecorder(rekam_io=True)
        with pytest.raises(RuntimeError):
            await run_pipeline(
                "kapan KRS?", retriever=RetrieverRusak(), llm_call=llm, callbacks=[recorder]
            )
        per_node = {n["node"]: n for n in recorder.rekaman()["nodes"]}
        assert per_node["retrieve"]["input"]["search_query"] == "kapan KRS?"
        assert per_node["retrieve"]["output"] is None
        assert recorder.rekaman()["output"] is None


def _giliran_dengan_pipeline(sink, pertanyaan: str, **lain):
    """Jalankan pipeline di dalam `catat_giliran` seperti router chat."""

    async def jalan(retriever, llm, history=None):
        with catat_giliran(
            sink,
            endpoint="chat",
            session_id="s",
            unit=None,
            pertanyaan=pertanyaan,
            nim="2401010101",
            rekam_io=lain.get("rekam_io", True),
        ) as g:
            outcome = await run_pipeline(
                pertanyaan,
                retriever=retriever,
                llm_call=llm,
                history=history or [],
                callbacks=[g.recorder],
            )
            g.selesai(
                hasil=outcome.kind,
                message_id=None,
                langsmith_run_id=None,
                llm_call=llm,
                jawaban=outcome.text,
            )
        return g

    return jalan


class TestGiliranMerekam:
    async def test_teks_nim_dan_rekaman_tercatat(self, strong_retriever, llm):
        sink = FakeLogSink()
        await _giliran_dengan_pipeline(sink, "kapan pengisian KRS dibuka?")(
            strong_retriever, llm
        )
        [turn] = sink.of("turn")
        assert turn["question"] == "kapan pengisian KRS dibuka?"
        assert turn["nim"] == "2401010101"
        assert turn["answer"].startswith("Jawaban")
        [trace] = sink.of("trace")
        assert trace["turn_id"] == turn["turn_id"]
        isi = buka(
            kemas(trace["isi"], sensitif=trace["sensitif"], pengganti=trace["pengganti"])
        )
        assert isi["input"]["question"] == "kapan pengisian KRS dibuka?"
        assert isi["nodes"][-1]["node"] == "generate"

    async def test_pertanyaan_sensitif_disamarkan(self, strong_retriever, llm):
        sink = FakeLogSink()
        pertanyaan = "saya stres dan ingin bunuh diri"
        await _giliran_dengan_pipeline(sink, pertanyaan)(strong_retriever, llm)
        [turn] = sink.of("turn")
        assert turn["outcome"] == "support"
        assert turn["question"] == SENSITIVE_PLACEHOLDER
        [trace] = sink.of("trace")
        blob = kemas(trace["isi"], sensitif=trace["sensitif"], pengganti=trace["pengganti"])
        teks = repr(buka(blob))
        assert "bunuh diri" not in teks
        assert SENSITIVE_PLACEHOLDER in teks

    async def test_pertanyaan_sensitif_di_riwayat_disamarkan(self, strong_retriever, llm):
        """Widget mengirim teks asli giliran sensitif sebelumnya sebagai riwayat."""
        sink = FakeLogSink()
        riwayat = [
            Turn("user", "saya stres dan ingin bunuh diri"),
            Turn("assistant", "Kamu tidak sendiri."),
        ]
        await _giliran_dengan_pipeline(sink, "kapan pengisian KRS dibuka?")(
            strong_retriever, llm, history=riwayat
        )
        [turn] = sink.of("turn")
        assert turn["question"] == "kapan pengisian KRS dibuka?"
        [trace] = sink.of("trace")
        blob = kemas(trace["isi"], sensitif=trace["sensitif"], pengganti=trace["pengganti"])
        teks = repr(buka(blob))
        assert "bunuh diri" not in teks
        assert "Kamu tidak sendiri." in teks

    async def test_log_node_io_mati_tanpa_teks(self, strong_retriever, llm):
        sink = FakeLogSink()
        await _giliran_dengan_pipeline(sink, "kapan pengisian KRS dibuka?", rekam_io=False)(
            strong_retriever, llm
        )
        assert sink.of("trace") == []
        [turn] = sink.of("turn")
        assert turn["question"] is None and turn["nim"] is None and turn["answer"] is None
        assert "kapan pengisian KRS dibuka?" not in repr(sink.rows)

    async def test_id_trace_tercatat_walau_dibatalkan(self):
        """T56: trace LangSmith giliran yang dibatalkan tetap ada."""
        sink = FakeLogSink()

        async def tugas():
            with catat_giliran(
                sink, endpoint="chat_stream", session_id="s", unit=None, langsmith_run_id="r1"
            ):
                await asyncio.sleep(10)

        t = asyncio.create_task(tugas())
        await asyncio.sleep(0)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        [turn] = sink.of("turn")
        assert turn["status"] == "dibatalkan"
        assert turn["langsmith_run_id"] == "r1"

    def test_detail_generate_menyebut_nama_tool(self):
        """T54: rincian giliran menunjukkan tool walau LOG_NODE_IO mati."""
        sink = FakeLogSink()
        llm = RecordingLLM()
        llm.model = "cx/gpt-6-luna"
        llm.usage = None
        llm.tool_calls = [
            {"name": "get_mk_diampu_dosen", "args": {"matkul": "BD"}, "ok": True},
            {"name": "get_mk_dosen", "args": {"nama": "x"}, "ok": True},
        ]
        with catat_giliran(sink, endpoint="chat", session_id="s", unit=None) as g:
            g.recorder.nodes.append(
                {
                    "node": "generate",
                    "position": 1,
                    "started_at": "x",
                    "duration_ms": 5.0,
                    "status": "ok",
                    "error_type": None,
                    "error_message": None,
                    "detail": {},
                }
            )
            g.selesai(hasil="answer", message_id=None, langsmith_run_id=None, llm_call=llm)
        [n] = sink.of("node")
        assert n["detail"]["tools"] == "get_mk_diampu_dosen, get_mk_dosen"
        assert "BD" not in repr(n["detail"])
