"""Tool-calling: registry & kelayakan, validasi argumen, handler SADS, loop
agentik, dan integrasi pipeline (docs/tool-call.md).

Tidak ada test yang menyentuh jaringan: LLM dan klien SADS dipalsukan, sama
seperti invarian "LLM tidak dipanggil" pada test pipeline lain.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk

from app.config import Settings
from app.rag.chain import OutcomeKind, run_pipeline
from app.rag.tools import sads
from app.rag.tools.base import (
    ToolArgumentError,
    ToolResult,
    ToolSpec,
    hasil_tool_ke_dokumen,
    validasi_argumen,
)
from app.rag.tools.loop import run_tool_loop
from app.rag.tools.registry import ToolRegistry, build_registry
from tests.fixtures.fakes import FakeRetriever, RecordingLLM

LABEL = "Data akademik SADS"


def _spec(handler: Any) -> ToolSpec:
    return ToolSpec(
        name="get_mk_diampu_dosen",
        description="Dosen pengampu mata kuliah.",
        parameters={
            "type": "object",
            "properties": {"matkul": {"type": "string"}},
            "required": ["matkul"],
        },
        handler=handler,
        triggers=("dosen", "mata kuliah", "mengampu"),
        citation_label=LABEL,
    )


# --- Registry & kelayakan ------------------------------------------------


def _registry() -> ToolRegistry:
    s = Settings(
        _env_file=None,
        tools_enabled=True,
        sads_base_url="https://sads.instiki.ac.id",
        sads_api_secret="rahasia",
        api_key="k",
    )
    return build_registry(s)


class TestRegistry:
    def test_specs_terdaftar(self):
        assert {s.name for s in _registry().specs} == {
            "get_daftar_dosen",
            "get_mk_diampu_dosen",
        }

    def test_eligible_cocok_kata_kunci(self):
        reg = _registry()
        names = {s.name for s in reg.eligible("siapa dosen pengampu mata kuliah Programming?")}
        assert "get_mk_diampu_dosen" in names

    def test_tidak_eligible_di_luar_topik(self):
        assert _registry().eligible("resep rendang yang enak") == []

    def test_schema_openai_berbentuk_benar(self):
        fn = _registry().get("get_mk_diampu_dosen").openai_schema()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "get_mk_diampu_dosen"
        assert fn["function"]["parameters"]["required"] == ["matkul"]


# --- Validasi argumen (keamanan) -----------------------------------------


class TestValidasiArgumen:
    def test_buang_karakter_kontrol_dan_potong(self):
        spec = _spec(None)
        out = validasi_argumen(spec, {"matkul": "  Program\x00ming  "})
        assert out == {"matkul": "Programming"}

    def test_argumen_tak_dikenal_dibuang(self):
        out = validasi_argumen(_spec(None), {"matkul": "X", "url": "http://jahat"})
        assert out == {"matkul": "X"}

    def test_wajib_kosong_ditolak(self):
        with pytest.raises(ToolArgumentError):
            validasi_argumen(_spec(None), {})
        with pytest.raises(ToolArgumentError):
            validasi_argumen(_spec(None), {"matkul": "   "})


# --- Handler SADS (normalisasi) ------------------------------------------


class FakeSads:
    def __init__(self, data: Any) -> None:
        self.data = data
        self.calls: list[tuple[str, dict | None]] = []

    async def get_json(self, path: str, params: dict | None = None) -> Any:
        self.calls.append((path, params))
        return self.data


class TestHandlerSads:
    async def test_daftar_dosen_strip_dan_dedup(self):
        client = FakeSads(
            [
                {"nmdosen": "Budi, S.Kom   "},
                {"nmdosen": "Ani,"},
                {"nmdosen": ""},
                {"nmdosen": "Budi, S.Kom"},
            ]
        )
        r = await sads._daftar_dosen(client)
        assert r.ok and r.label == LABEL
        assert r.text.count("Budi, S.Kom") == 1  # dedup + trailing space dibuang
        assert "- Ani" in r.text  # koma di ujung dibuang

    async def test_mk_diampu_meratakan_dan_meneruskan_matkul(self):
        client = FakeSads(
            [
                {
                    "nmdosen": "Budi ",
                    "matkul": [{"matkul": "Web Programming"}, {"matkul": " Mobile "}],
                },
                {"nmdosen": "", "matkul": []},
            ]
        )
        r = await sads._mk_diampu_dosen(client, matkul="Programming")
        assert client.calls == [
            ("/service/tp/chatbot/mk-diampu-dosen", {"matkul": "Programming"})
        ]
        assert "- Budi: Web Programming, Mobile" in r.text and r.ok

    async def test_hasil_kosong_tidak_ok(self):
        r = await sads._mk_diampu_dosen(FakeSads([]), matkul="xyz")
        assert not r.ok


# --- Kartu sitasi sintetis ----------------------------------------------


def test_hasil_tool_ke_dokumen_satu_kartu_per_label():
    docs = hasil_tool_ke_dokumen(
        [
            ToolResult("a", LABEL, "isi A", True),
            ToolResult("b", LABEL, "isi B", True),
            ToolResult("c", LABEL, "", False),  # gagal: diabaikan
        ]
    )
    assert len(docs) == 1
    assert docs[0].metadata["judul"] == LABEL
    assert docs[0].metadata["jenis"].value == "tanya_jawab"
    assert docs[0].metadata["file_path"] == ""


def test_pesan_tool_membingkai_sebagai_data():
    ok = ToolResult("a", LABEL, "daftar dosen", True).pesan_untuk_model()
    assert f"[{LABEL}]" in ok and "daftar dosen" in ok
    gagal = ToolResult("a", LABEL, "", False).pesan_untuk_model()
    assert "DATA_TIDAK_TERSEDIA" in gagal


def test_jejak_tool_masuk_messages_meta():
    """Fase 4: jejak tool mengalir ke `messages.meta`; None bila tool tak dipakai."""
    from app.observability.chatlog import ChatLogEntry, build_meta
    from app.rag.chain import OutcomeKind, PipelineOutcome

    jejak = [
        {"name": "get_mk_diampu_dosen", "args": {"matkul": "X"}, "ok": True, "latency_ms": 12}
    ]
    outcome = PipelineOutcome(kind=OutcomeKind.ANSWER, text="t", llm_called=True)
    meta = build_meta(
        ChatLogEntry(
            session_id="s",
            question="q",
            outcome=outcome,
            latency_ms=1,
            tool_calls=jejak,
        )
    )
    assert meta["tool_calls"] == jejak
    meta_tanpa = build_meta(
        ChatLogEntry(session_id="s", question="q", outcome=outcome, latency_ms=1)
    )
    assert meta_tanpa["tool_calls"] is None


# --- Loop agentik --------------------------------------------------------


class FakeChat:
    """Streaming LLM yang dapat di-skrip per giliran. Mengabaikan `tools`."""

    def __init__(self, turns: list[list[AIMessageChunk]]) -> None:
        self.turns = turns
        self.i = 0
        self.seen_messages: list[Any] = []

    def bind(self, **_: Any) -> FakeChat:
        return self

    async def astream(self, messages: Any, config: Any = None):
        self.seen_messages = list(messages)
        turn = self.turns[self.i]
        self.i += 1
        for chunk in turn:
            yield chunk


def _tool_turn(name: str, args_json: str, call_id: str = "c1") -> list[AIMessageChunk]:
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": name, "args": args_json, "id": call_id, "index": 0}
            ],
        )
    ]


def _answer_turn(*pieces: str, usage: dict | None = None) -> list[AIMessageChunk]:
    chunks = [AIMessageChunk(content=p) for p in pieces]
    if usage is not None:
        penuh = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, **usage}
        chunks.append(AIMessageChunk(content="", usage_metadata=penuh))
    return chunks


class TestLoop:
    async def test_panggil_tool_lalu_jawab_dan_stream_hanya_final(self):
        dipanggil: list[str] = []

        async def handler(*, matkul: str) -> ToolResult:
            dipanggil.append(matkul)
            return ToolResult("get_mk_diampu_dosen", LABEL, "- Budi: Web Programming", True)

        spec = _spec(handler)
        llm = FakeChat(
            [
                _tool_turn("get_mk_diampu_dosen", '{"matkul":"Programming"}'),
                _answer_turn(
                    "Pengampunya ",
                    f"Budi [{LABEL}].",
                    usage={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
                ),
            ]
        )
        token: list[str] = []

        async def on_token(t: str) -> None:
            token.append(t)

        res = await run_tool_loop(
            llm_tools=llm,
            llm_plain=llm,
            wrapped_question="<pertanyaan_mahasiswa>Programming</pertanyaan_mahasiswa>",
            documents=[],
            specs=[spec],
            on_token=on_token,
            max_rounds=2,
        )
        assert dipanggil == ["Programming"]
        assert f"[{LABEL}]" in res.text and "Budi" in res.text
        # Hanya giliran jawaban final yang mengalir; giliran tool tak berkonten.
        assert "".join(token) == res.text
        assert len(res.documents) == 1
        assert res.usage["total_tokens"] == 7
        # Jejak untuk observability (Fase 4): nama, argumen, ok, latensi.
        assert len(res.tool_calls) == 1
        j = res.tool_calls[0]
        assert j["name"] == "get_mk_diampu_dosen"
        assert j["args"] == {"matkul": "Programming"}
        assert j["ok"] is True
        assert isinstance(j["latency_ms"], int) and j["latency_ms"] >= 0

    async def test_argumen_dihalusinasi_tetap_divalidasi(self):
        async def handler(*, matkul: str) -> ToolResult:
            return ToolResult("get_mk_diampu_dosen", LABEL, f"ok:{matkul}", True)

        llm = FakeChat(
            [
                _tool_turn("get_mk_diampu_dosen", '{"matkul":"A\\u0000B","lain":"x"}'),
                _answer_turn(f"Hasil [{LABEL}].", usage={"total_tokens": 1}),
            ]
        )
        res = await run_tool_loop(
            llm_tools=llm,
            llm_plain=llm,
            wrapped_question="q",
            documents=[],
            specs=[_spec(handler)],
            max_rounds=2,
        )
        # Karakter kontrol dibuang, argumen asing dibuang -> handler aman.
        assert "ok:AB" in res.documents[0].page_content

    async def test_tool_gagal_tidak_menjatuhkan_giliran(self):
        async def handler(*, matkul: str) -> ToolResult:
            raise RuntimeError("layanan mati")

        llm = FakeChat(
            [
                _tool_turn("get_mk_diampu_dosen", '{"matkul":"X"}'),
                _answer_turn("Maaf, [TIDAK_DITEMUKAN]"),
            ]
        )
        res = await run_tool_loop(
            llm_tools=llm,
            llm_plain=llm,
            wrapped_question="q",
            documents=[],
            specs=[_spec(handler)],
            max_rounds=2,
        )
        assert res.documents == []  # tidak ada kartu dari tool yang gagal
        assert "TIDAK_DITEMUKAN" in res.text

    async def test_batas_putaran_memaksa_jawaban_tanpa_tool(self):
        async def handler(*, matkul: str) -> ToolResult:
            return ToolResult("get_mk_diampu_dosen", LABEL, "- Budi: X", True)

        # max_rounds=1: satu giliran tool, lalu dipaksa jawab lewat llm_plain.
        llm = FakeChat(
            [
                _tool_turn("get_mk_diampu_dosen", '{"matkul":"X"}'),
                _answer_turn(f"Jawaban akhir [{LABEL}]."),
            ]
        )
        res = await run_tool_loop(
            llm_tools=llm,
            llm_plain=llm,
            wrapped_question="q",
            documents=[],
            specs=[_spec(handler)],
            max_rounds=1,
        )
        assert "Jawaban akhir" in res.text
        assert llm.i == 2  # dua giliran dijalankan


# --- Integrasi pipeline --------------------------------------------------


class ToolLLM(RecordingLLM):
    """llm_call dengan `run_tools`; `__call__`/`stream` warisan tak dipakai jalur tool."""

    def __init__(self, answer: str, docs: list[Any]) -> None:
        super().__init__()
        self._answer = answer
        self._docs = docs
        self.usage: dict | None = None
        self.model = "m"
        self.tool_calls_made = 0

    async def run_tools(
        self, wrapped, documents, *, specs, on_token=None, on_stage=None, max_rounds=2
    ):
        self.tool_calls_made += 1
        self.usage = {"total_tokens": 5}
        return self._answer, list(self._docs)


class TestIntegrasiPipeline:
    async def test_tool_eligible_konteks_kosong_tetap_menjawab(self):
        docs = hasil_tool_ke_dokumen([ToolResult("a", LABEL, "- Budi: Web Programming", True)])
        llm = ToolLLM(f"Pengampunya Budi [{LABEL}].", docs)
        registry = ToolRegistry([_spec(lambda **_: None)])

        outcome = await run_pipeline(
            "siapa dosen pengampu mata kuliah Web Programming?",
            retriever=FakeRetriever([]),  # konteks kosong -> FR-3 REFUSE tanpa tool
            llm_call=llm,
            tool_registry=registry,
        )
        assert outcome.kind is OutcomeKind.ANSWER
        assert llm.tool_calls_made == 1
        assert any(d.metadata["judul"] == LABEL for d in outcome.documents)

    async def test_tanpa_registry_konteks_kosong_tetap_menolak(self):
        llm = RecordingLLM()
        outcome = await run_pipeline(
            "siapa dosen pengampu mata kuliah Web Programming?",
            retriever=FakeRetriever([]),
            llm_call=llm,
            tool_registry=None,
        )
        assert outcome.kind is OutcomeKind.REFUSAL
        assert not llm.called  # LLM tidak dipanggil pada penolakan FR-3

    async def test_pertanyaan_di_luar_topik_tidak_eligible(self):
        llm = ToolLLM("x", [])
        registry = ToolRegistry([_spec(lambda **_: None)])
        outcome = await run_pipeline(
            "bagaimana cuaca besok di Denpasar?",
            retriever=FakeRetriever([]),
            llm_call=llm,
            tool_registry=registry,
        )
        # Tidak cocok pemicu -> tetap jalur FR-3, tool tidak dipanggil.
        assert outcome.kind is OutcomeKind.REFUSAL
        assert llm.tool_calls_made == 0
