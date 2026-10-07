"""Tool-calling: registry & kelayakan, validasi argumen, handler SADS, loop
agentik, dan integrasi pipeline (docs/tool-call.md).

Tidak ada test yang menyentuh jaringan: LLM dan klien SADS dipalsukan, sama
seperti invarian "LLM tidak dipanggil" pada test pipeline lain.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from app.config import Settings
from app.rag.chain import OutcomeKind, run_pipeline
from app.rag.prompts import NOT_FOUND_MARKER
from app.rag.providers import GalatGateway
from app.rag.rewriter import Turn
from app.rag.tools import sads
from app.rag.tools.base import (
    ToolArgumentError,
    ToolResult,
    ToolSpec,
    hasil_tool_ke_dokumen,
    validasi_argumen,
)
from app.rag.tools.client import SadsClient
from app.rag.tools.loop import MAKS_TOOL_PER_GILIRAN, PESAN_BATAS, run_tool_loop
from app.rag.tools.registry import ToolRegistry, build_registry
from tests.fixtures.fakes import FakeRetriever, RecordingLLM, RecordingRewriter

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

    def test_pertanyaan_inggris_pendek_eligible_tanpa_rewrite(self):
        """Regresi live: "who teaches X?" tidak di-rewrite (`looks_english` butuh
        >= 2 kata tugas), jadi pemicunya harus mengenal kata Inggris sendiri."""
        reg = _registry()
        assert reg.eligible("who teaches Web Programming?")
        assert reg.eligible("Who is the lecturer for Database?")
        assert reg.eligible("how do I register for courses (KRS)?") == []

    def test_eligible_membuka_semua_tool_bukan_hanya_yang_cocok(self):
        """Regresi: pemicu asimetris pernah menyembunyikan tool yang benar.

        "ada berapa dosen?" hanya cocok pemicu `get_mk_diampu_dosen` (yang
        mewajibkan `matkul`). `get_daftar_dosen` -- satu-satunya tool yang bisa
        menjawabnya -- harus tetap ter-bind supaya model dapat memilihnya."""
        names = {s.name for s in _registry().eligible("ada berapa dosen?")}
        assert names == {"get_daftar_dosen", "get_mk_diampu_dosen"}

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

    def test_args_yang_dicatat_ke_meta_dipotong(self):
        """Argumen raksasa dari model tidak boleh menggelembungkan kolom `meta`."""
        from app.rag.tools.base import MAKS_PANJANG_ARGUMEN
        from app.rag.tools.loop import _args_untuk_log

        out = _args_untuk_log({"matkul": "x" * 5000, "n": 3})
        assert len(out["matkul"]) == MAKS_PANJANG_ARGUMEN
        assert out["n"] == 3


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
        # Jumlah dihitung kode, bukan model (dulu model menjawab 223 dari 220 nama).
        assert "Jumlah dosen yang mengajar di INSTIKI: 2 orang." in r.text

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
        self.last_wrapped: str | None = None
        self.last_specs: list[Any] = []

    async def run_tools(
        self, wrapped, documents, *, specs, on_token=None, on_stage=None, max_rounds=2
    ):
        self.tool_calls_made += 1
        self.last_wrapped = wrapped
        self.last_specs = list(specs)
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

    async def test_tool_eligible_tanpa_run_tools_tetap_menolak_tanpa_llm(self):
        """Invarian FR-3 tetap berlaku bila tool ternyata tak dapat dijalankan.

        Regresi: `llm_call` tanpa `run_tools` (test, atau LLM pengganti) pernah
        membuat pertanyaan tool-eligible berkonteks lemah lolos ke LLM dengan
        KONTEKS kosong -- satu panggilan berbayar yang hanya menghasilkan
        [TIDAK_DITEMUKAN], dan tercatat `refusal_source = llm`."""
        from app.rag.chain import refusal_source

        llm = RecordingLLM()  # tidak punya run_tools
        registry = ToolRegistry([_spec(lambda **_: None)])
        outcome = await run_pipeline(
            "siapa dosen pengampu mata kuliah Web Programming?",
            retriever=FakeRetriever([]),
            llm_call=llm,
            tool_registry=registry,
        )
        assert outcome.kind is OutcomeKind.REFUSAL
        assert not llm.called
        assert outcome.llm_called is False
        assert refusal_source(outcome) == "threshold"

    async def test_konteks_kuat_tetap_dijawab_walau_tool_tak_tersedia(self, strong_documents):
        """Penjagaan di atas tidak boleh ikut menolak konteks yang kuat."""
        llm = RecordingLLM()
        registry = ToolRegistry([_spec(lambda **_: None)])
        outcome = await run_pipeline(
            "siapa dosen pengampu mata kuliah Web Programming?",
            retriever=FakeRetriever(strong_documents),
            llm_call=llm,
            tool_registry=registry,
        )
        assert outcome.kind is OutcomeKind.ANSWER
        assert llm.called

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


def test_jawaban_parsial_tool_tidak_dibuang_jadi_penolakan():
    """Regresi: kutipan `[Data akademik SADS]` tak terbaca sebagai sitasi, sehingga
    jawaban parsial yang menyebut "tidak ditemukan" berubah menjadi refusal."""
    from app.rag.chain import is_not_found

    jawab = f"Basis Data diampu oleh Budi [{LABEL}]. Untuk Kalkulus tidak ditemukan."
    assert not is_not_found(jawab, {LABEL: 1})
    assert is_not_found("[TIDAK_DITEMUKAN]", {LABEL: 1})


# --- Tinjauan kedua: klaim dokumentasi yang dulu tidak dipenuhi kode ----------


async def _jalankan(llm: FakeChat, handler: Any, **lain: Any):
    return await run_tool_loop(
        llm_tools=llm,
        llm_plain=llm,
        wrapped_question="q",
        documents=[],
        specs=[_spec(handler)],
        **lain,
    )


def _banyak_panggilan(*matkul: str) -> list[AIMessageChunk]:
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": "get_mk_diampu_dosen",
                    "args": f'{{"matkul":"{m}"}}',
                    "id": f"c{i}",
                    "index": i,
                }
                for i, m in enumerate(matkul)
            ],
        )
    ]


class TestSadsClient:
    async def test_mengirim_header_secret_bukan_bearer(self):
        diterima: dict[str, Any] = {}

        def layani(request: httpx.Request) -> httpx.Response:
            diterima["secret"] = request.headers.get("secret")
            diterima["authorization"] = request.headers.get("authorization")
            diterima["path"] = request.url.path
            diterima["matkul"] = request.url.params.get("matkul")
            return httpx.Response(200, json=[{"nmdosen": "Budi"}])

        client = SadsClient(
            base_url="https://sads.contoh/",
            secret="rahasia",
            timeout=5,
            transport=httpx.MockTransport(layani),
        )
        data = await client.get_json(
            "/service/tp/chatbot/mk-diampu-dosen", params={"matkul": "Basis Data"}
        )
        assert data == [{"nmdosen": "Budi"}]
        assert diterima["secret"] == "rahasia"
        assert diterima["authorization"] is None
        assert diterima["path"] == "/service/tp/chatbot/mk-diampu-dosen"
        assert diterima["matkul"] == "Basis Data"

    async def test_status_galat_menjadi_http_error(self):
        client = SadsClient(
            base_url="https://sads.contoh",
            secret="salah",
            timeout=5,
            transport=httpx.MockTransport(lambda _r: httpx.Response(401)),
        )
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_json("/service/tp/chatbot/dosen-mengajar")


class TestLoopTinjauanKedua:
    async def test_beberapa_tool_satu_giliran_berjalan_bersamaan(self):
        """Bila dijalankan berurutan, A menunggu B yang belum mulai, lalu timeout."""
        b_mulai = asyncio.Event()

        async def handler(*, matkul: str) -> ToolResult:
            if matkul == "A":
                await asyncio.wait_for(b_mulai.wait(), timeout=2)
            else:
                b_mulai.set()
            return ToolResult("get_mk_diampu_dosen", LABEL, f"- dosen {matkul}", True)

        llm = FakeChat([_banyak_panggilan("A", "B"), _answer_turn(f"A dan B [{LABEL}].")])
        res = await _jalankan(llm, handler)
        assert [j["args"]["matkul"] for j in res.tool_calls] == ["A", "B"]
        assert all(j["ok"] for j in res.tool_calls)
        ids = [m.tool_call_id for m in llm.seen_messages if isinstance(m, ToolMessage)]
        assert ids == ["c0", "c1"]

    async def test_panggilan_di_atas_batas_dibalas_tanpa_dijalankan(self):
        dipanggil: list[str] = []

        async def handler(*, matkul: str) -> ToolResult:
            dipanggil.append(matkul)
            return ToolResult("get_mk_diampu_dosen", LABEL, f"- {matkul}", True)

        semua = [f"M{i}" for i in range(MAKS_TOOL_PER_GILIRAN + 2)]
        llm = FakeChat([_banyak_panggilan(*semua), _answer_turn(f"ok [{LABEL}].")])
        res = await _jalankan(llm, handler)
        assert sorted(dipanggil) == sorted(semua[:MAKS_TOOL_PER_GILIRAN])
        balasan = [m for m in llm.seen_messages if isinstance(m, ToolMessage)]
        # Setiap tool_call_id tetap dibalas; kalau tidak, API menolak giliran berikutnya.
        assert len(balasan) == len(semua)
        assert sum(m.content == PESAN_BATAS for m in balasan) == 2
        assert [j.get("error") for j in res.tool_calls].count("batas_per_giliran") == 2

    async def test_argumen_bukan_json_tidak_dikira_jawaban_kosong(self):
        async def handler(*, matkul: str) -> ToolResult:  # pragma: no cover
            raise AssertionError("argumen rusak tidak boleh sampai ke handler")

        llm = FakeChat(
            [
                _tool_turn("get_mk_diampu_dosen", "{matkul: A}"),
                _answer_turn(f"Jawab [{LABEL}]."),
            ]
        )
        res = await _jalankan(llm, handler)
        assert llm.i == 2  # lanjut ke giliran berikutnya, bukan berhenti dengan teks kosong
        assert res.tool_calls[0]["error"] == "argumen_rusak"
        assert res.text == f"Jawab [{LABEL}]."

    async def test_giliran_final_kosong_menjadi_penanda_tidak_ditemukan(self):
        res = await _jalankan(FakeChat([_answer_turn("")]), None)
        assert res.text == NOT_FOUND_MARKER

    async def test_galat_gateway_di_jawaban_final_dilempar(self):
        async def handler(*, matkul: str) -> ToolResult:
            return ToolResult("get_mk_diampu_dosen", LABEL, "- Budi", True)

        llm = FakeChat(
            [
                _tool_turn("get_mk_diampu_dosen", '{"matkul":"X"}'),
                _answer_turn("Untuk ", "[Error] Our servers are currently overloaded."),
            ]
        )
        with pytest.raises(GalatGateway):
            await _jalankan(llm, handler)

    async def test_stage_berganti_selama_tool_berjalan(self):
        async def handler(*, matkul: str) -> ToolResult:
            return ToolResult("get_mk_diampu_dosen", LABEL, "- Budi", True)

        stage: list[str] = []

        async def on_stage(s: str) -> None:
            stage.append(s)

        llm = FakeChat(
            [
                _tool_turn("get_mk_diampu_dosen", '{"matkul":"X"}'),
                _answer_turn(f"ok [{LABEL}]."),
            ]
        )
        await _jalankan(llm, handler, on_stage=on_stage)
        assert stage == ["mengambil data akademik", "menyusun jawaban"]


class TestPertanyaanLanjutan:
    async def test_lanjutan_eligible_lewat_hasil_rewrite(self):
        """'kalau Basis Data?' tidak memuat pemicu; versi mandirinya memuat."""
        docs = hasil_tool_ke_dokumen([ToolResult("a", LABEL, "- Budi: Basis Data", True)])
        llm = ToolLLM(f"Basis Data diampu Budi [{LABEL}].", docs)
        rewriter = RecordingRewriter("Siapa dosen pengampu mata kuliah Basis Data?")
        outcome = await run_pipeline(
            "kalau Basis Data?",
            retriever=FakeRetriever([]),
            llm_call=llm,
            rewrite_call=rewriter,
            history=[
                Turn("user", "siapa dosen pengampu mata kuliah Programming?"),
                Turn("assistant", f"Budi [{LABEL}]."),
            ],
            tool_registry=ToolRegistry([_spec(lambda **_: None)]),
        )
        assert outcome.kind is OutcomeKind.ANSWER
        assert llm.tool_calls_made == 1
        wrapped = llm.last_wrapped
        # Loop tool menerima versi mandirinya, dan keduanya di dalam tag (FR-5).
        buka = wrapped.index("<pertanyaan_mahasiswa>")
        tutup = wrapped.index("</pertanyaan_mahasiswa>")
        assert buka < wrapped.index("kalau Basis Data?") < tutup
        assert buka < wrapped.index("Siapa dosen pengampu mata kuliah Basis Data?") < tutup

    async def test_tanpa_rewrite_lanjutan_tidak_eligible(self):
        llm = ToolLLM("x", [])
        outcome = await run_pipeline(
            "kalau Basis Data?",
            retriever=FakeRetriever([]),
            llm_call=llm,
            tool_registry=ToolRegistry([_spec(lambda **_: None)]),
        )
        assert outcome.kind is OutcomeKind.REFUSAL
        assert llm.tool_calls_made == 0

    async def test_seluruh_registry_di_bind(self):
        reg = _registry()
        llm = ToolLLM(f"x [{LABEL}].", [])
        await run_pipeline(
            "ada berapa dosen?", retriever=FakeRetriever([]), llm_call=llm, tool_registry=reg
        )
        assert {s.name for s in llm.last_specs} == {s.name for s in reg.specs}


async def test_setiap_giliran_llm_mendapat_config_sendiri():
    """Regresi: satu config (satu run_id) dipakai ulang untuk semua giliran loop,
    sehingga LangSmith menolak giliran kedua -- giliran yang berisi jawabannya."""

    async def handler(*, matkul: str) -> ToolResult:
        return ToolResult("get_mk_diampu_dosen", LABEL, "- Budi", True)

    dibuat: list[dict] = []

    def buat_config() -> dict:
        dibuat.append({"run_id": f"run-{len(dibuat)}"})
        return dibuat[-1]

    llm = FakeChat(
        [
            _tool_turn("get_mk_diampu_dosen", '{"matkul":"X"}'),
            _answer_turn(f"ok [{LABEL}]."),
        ]
    )
    await _jalankan(llm, handler, buat_config=buat_config)
    assert [c["run_id"] for c in dibuat] == ["run-0", "run-1"]
