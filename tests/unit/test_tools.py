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
            "get_mk_dosen",
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
        assert names == {"get_daftar_dosen", "get_mk_diampu_dosen", "get_mk_dosen"}

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
        # Varian huruf ganda ikut ditanyakan (T47); hasil yang sama tidak berlipat.
        assert client.calls == [
            ("/service/tp/chatbot/mk-diampu-dosen", {"matkul": "Programming"}),
            ("/service/tp/chatbot/mk-diampu-dosen", {"matkul": "Programing"}),
        ]
        assert r.ok
        assert 'yang namanya memuat "Programming": 2 mata kuliah.' in r.text
        assert r.text.count("Web Programming (jumlah dosen pengampu: 1 orang):\n- Budi") == 1
        assert r.text.count("Mobile (jumlah dosen pengampu: 1 orang):\n- Budi") == 1

    async def test_hasil_kosong_bukan_kegagalan(self):
        """T43/T47: tidak ada yang cocok adalah jawaban sah SADS. Dulu dibalas
        DATA_TIDAK_TERSEDIA ("tidak dapat diambil saat ini"), sehingga model
        mengira layanannya gangguan dan tidak mencoba nama Inggrisnya."""
        r = await sads._mk_diampu_dosen(FakeSads([]), matkul="Kecerdasan Buatan")
        assert r.ok and r.attachments == ()
        assert '"Kecerdasan Buatan": 0 dosen pengampu' in r.text
        assert "DATA_TIDAK_TERSEDIA" not in r.pesan_untuk_model()

    async def test_varian_ejaan_digabung_jadi_satu_mata_kuliah(self):
        """T47: SADS menulis "Artificial Intelligence" dan "Artificial Inteligence";
        kata kunci yang benar ejaannya melewatkan dosen yang hanya ada di varian."""
        client = FakeSadsPerKataKunci(
            {
                "Artificial Intelligence": [
                    {"nmdosen": "Ani", "matkul": [{"matkul": "Artificial Intelligence"}]},
                    {"nmdosen": "Budi", "matkul": [{"matkul": "Artificial Intelligence"}]},
                ],
                "Artificial Inteligence": [
                    {"nmdosen": "Budi", "matkul": [{"matkul": "Artificial Inteligence"}]},
                    {"nmdosen": "Citra", "matkul": [{"matkul": "Artificial Inteligence"}]},
                ],
            }
        )
        r = await sads._mk_diampu_dosen(client, matkul="Artificial Intelligence")
        assert ": 1 mata kuliah." in r.text
        # Seri (2 lawan 2 dosen): penulisan yang muncul lebih dulu menjadi nama.
        assert (
            'Artificial Intelligence (juga tertulis "Artificial Inteligence"; '
            "jumlah dosen pengampu: 3 orang):\n- Ani\n- Budi\n- Citra"
        ) in r.text
        [lampiran] = r.attachments
        assert lampiran.items == ("Ani", "Budi", "Citra")
        assert lampiran.disebut == "Artificial Intelligence"

    async def test_setiap_mata_kuliah_dihitung_dan_berlampiran_sendiri(self):
        """T49: hasil yang diratakan per dosen membuat model menyaring "Lanjut",
        menghitung, lalu menyalin 35 nama -- dan kadang kehilangan satu nama."""
        client = FakeSads(
            [
                _mk("Budi", "Basis Data Lanjut"),
                _mk("Ani ", "Basis Data", "Basis Data Lanjut"),
                _mk("Citra", "basis data"),
                _mk("Dewi", "Basis Data Lanjut"),
            ]
        )
        r = await sads._mk_diampu_dosen(client, matkul="Basis Data")
        assert 'yang namanya memuat "Basis Data": 2 mata kuliah.' in r.text
        # Terbanyak dosennya lebih dulu; huruf besar-kecil tidak memisahkan.
        lanjut = "Basis Data Lanjut (jumlah dosen pengampu: 3 orang):"
        dasar = 'Basis Data (juga tertulis "basis data"; jumlah dosen pengampu: 2 orang):'
        assert r.text.index(lanjut) < r.text.index(dasar)
        assert [(a.title, a.items, a.disebut) for a in r.attachments] == [
            ("Dosen pengampu Basis Data Lanjut", ("Ani", "Budi", "Dewi"), "Basis Data Lanjut"),
            ("Dosen pengampu Basis Data", ("Ani", "Citra"), "Basis Data"),
        ]
        assert all(a.source == LABEL for a in r.attachments)

    async def test_beberapa_mata_kuliah_minta_model_menulis_namanya(self):
        from app.rag.prompts import TOOL_SYSTEM_PROMPT
        from app.rag.tools.base import CATATAN_LAMPIRAN, CATATAN_SEBUT

        dua = await sads._mk_diampu_dosen(
            FakeSads([_mk("Ani", "Basis Data", "Basis Data Lanjut")]), matkul="Basis Data"
        )
        assert CATATAN_LAMPIRAN in dua.pesan_untuk_model()
        assert CATATAN_SEBUT in dua.pesan_untuk_model()
        satu = await sads._mk_diampu_dosen(
            FakeSads([_mk("Ani", "Web Programming")]), matkul="Web Programming"
        )
        assert CATATAN_LAMPIRAN in satu.pesan_untuk_model()
        assert CATATAN_SEBUT not in satu.pesan_untuk_model()
        # Aturan T5 mengesahkan penanda yang sama.
        assert CATATAN_SEBUT.startswith("(DAFTAR_DITAMPILKAN:")
        assert "DAFTAR_DITAMPILKAN" in TOOL_SYSTEM_PROMPT

    async def test_tanpa_huruf_ganda_satu_panggilan(self):
        client = FakeSads([])
        await sads._mk_diampu_dosen(client, matkul="Basis Data")
        assert client.calls == [
            ("/service/tp/chatbot/mk-diampu-dosen", {"matkul": "Basis Data"})
        ]

    def test_deskripsi_mengarahkan_ke_nama_inggris_dan_get_mk_dosen(self):
        """T47/T48: petunjuknya di `description` (dipercaya), bukan di hasil tool
        (data, aturan T2)."""
        desc = _registry().get("get_mk_diampu_dosen").description
        assert "bukan nama dosen" in desc and "get_mk_dosen" in desc
        assert "Inggris" in desc and "0 dosen" in desc


class FakeSadsPerKataKunci:
    """SADS palsu yang menjawab menurut `matkul`; tanpa `matkul` = semua data."""

    def __init__(self, per_kata: dict[str, Any], semua: Any = None) -> None:
        self.per_kata = per_kata
        self.semua = semua
        self.calls: list[tuple[str, dict | None]] = []

    async def get_json(self, path: str, params: dict | None = None) -> Any:
        self.calls.append((path, params))
        if not params:
            return self.semua
        return self.per_kata.get(params["matkul"], [])


def _mk(nama: str, *matkul: str) -> dict[str, Any]:
    return {"nmdosen": nama, "matkul": [{"matkul": m} for m in matkul]}


_MK_SEMUA = [
    _mk("Ahmad Asroni, S.Kom., M.Kom  ", "Web Programming", "Database", "Algorithms"),
    _mk("Dr. I Gede Totok Suryawan, S.Kom., M.T.", "Basis Data"),
    _mk("I Wayan Satu, S.Kom", "A"),
    _mk("I Wayan Dua, S.Kom", "B"),
    _mk("I Wayan Tiga, S.Kom", "C"),
    _mk("I Wayan Empat, S.Kom", "D"),
    _mk("I Wayan Lima, S.Kom", "E"),
    _mk("I Wayan Enam, S.Kom", "F"),
    _mk("Tanpa MK"),
]


class TestMkDosen:
    """T48: "mata kuliah apa yang diajar dosen X?" -- dulu model mengisi
    `get_mk_diampu_dosen(matkul="Ahmad Asroni")` lalu menolak."""

    async def test_satu_dosen_dirinci_dihitung_dan_berlampiran(self):
        client = FakeSadsPerKataKunci({}, semua=_MK_SEMUA)
        r = await sads._mk_dosen(client, nama="Ahmad Asroni")
        # Tanpa `matkul`: SADS mengembalikan semua dosen beserta seluruh MK-nya.
        assert client.calls == [("/service/tp/chatbot/mk-diampu-dosen", None)]
        assert r.ok and r.label == LABEL
        assert "Ahmad Asroni, S.Kom., M.Kom (jumlah: 3 mata kuliah)" in r.text
        assert len(r.attachments) == 1
        assert r.attachments[0].items == ("Algorithms", "Database", "Web Programming")
        assert r.attachments[0].title == "Mata kuliah yang diampu Ahmad Asroni, S.Kom., M.Kom"
        assert r.attachments[0].source == LABEL

    @pytest.mark.parametrize(
        "nama", ["totok", "Pak Totok", "bapak totok suryawan", "Bu. Totok"]
    )
    async def test_nama_tanpa_sapaan_dan_gelar(self, nama):
        r = await sads._mk_dosen(FakeSadsPerKataKunci({}, semua=_MK_SEMUA), nama=nama)
        assert len(r.attachments) == 1
        assert r.attachments[0].items == ("Basis Data",)

    async def test_nama_umum_hanya_nama_dosen(self):
        r = await sads._mk_dosen(FakeSadsPerKataKunci({}, semua=_MK_SEMUA), nama="Wayan")
        assert r.ok and r.attachments == ()
        assert 'Ada 6 dosen dengan nama memuat "Wayan"' in r.text
        assert "- I Wayan Enam, S.Kom" in r.text
        assert "Mata kuliah yang diampu" not in r.text

    async def test_beberapa_dosen_dirinci_tanpa_lampiran(self):
        semua = [_mk("Budi Satu", "A"), _mk("Budi Dua", "B", "C")]
        r = await sads._mk_dosen(FakeSadsPerKataKunci({}, semua=semua), nama="budi")
        assert r.attachments == ()
        assert "Mata kuliah yang diampu Budi Dua (jumlah: 2 mata kuliah)" in r.text
        assert "Mata kuliah yang diampu Budi Satu (jumlah: 1 mata kuliah)" in r.text

    async def test_tidak_ada_dosen_bukan_kegagalan(self):
        r = await sads._mk_dosen(FakeSadsPerKataKunci({}, semua=_MK_SEMUA), nama="Zaenal")
        assert r.ok and r.attachments == ()
        assert '"Zaenal": 0 orang (dari 8 dosen yang mengampu mata kuliah)' in r.text

    async def test_sads_kosong_tetap_kegagalan(self):
        r = await sads._mk_dosen(FakeSadsPerKataKunci({}, semua=[]), nama="Asroni")
        assert not r.ok

    def test_skema_nama_wajib(self):
        params = _registry().get("get_mk_dosen").parameters
        assert params["required"] == ["nama"]
        assert set(params["properties"]) == {"nama"}

    def test_pertanyaan_mk_per_dosen_eligible(self):
        assert _registry().eligible("Pak Asroni ngajar apa aja?")
        assert _registry().eligible("Matkul yang diampu Bu Ayu apa saja?")
        assert _registry().eligible("What courses does Asroni teach?")


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
    assert docs[0].metadata["dari_tool"] is True


def test_kartu_tool_bertipe_data_bukan_tanya_jawab():
    """T44: kartu "Data akademik SADS" dulu bertipe `tanya_jawab`, sehingga widget
    melabelinya "Tanya jawab resmi". Entri tanya jawab admin tetap `tanya_jawab`,
    dan keduanya tetap dikutip tanpa halaman."""
    from app.rag.chain import PipelineOutcome
    from app.routers.chat import citations_for
    from app.schemas.chat import CitationType
    from tests.fixtures.fakes import make_document

    faq = make_document("f1", judul="Berapa biaya TOEIC?", halaman=1)
    faq.metadata["jenis"] = "tanya_jawab"
    tool = hasil_tool_ke_dokumen([ToolResult("a", LABEL, "Budi", True)])
    outcome = PipelineOutcome(
        kind=OutcomeKind.ANSWER,
        text=f"Diampu Budi [{LABEL}]. TOEIC Rp675.000 [Berapa biaya TOEIC?].",
        documents=(*tool, faq),
    )
    assert [(c.title, c.type) for c in citations_for(outcome)] == [
        (LABEL, CitationType.DATA),
        ("Berapa biaya TOEIC?", CitationType.TANYA_JAWAB),
    ]


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


# --- Saringan get_daftar_dosen dan lampiran (T46, docs/tool-call.md §10a) ---


_DOSEN = [
    {"nmdosen": "Dr. Ani Wijaya, S.Kom., M.T."},
    {"nmdosen": "Drs. Budi Santoso,M.Ag"},
    {"nmdosen": "Dr.Ir. Citra Dewi, S.Kom.,M.Kom."},
    {"nmdosen": "Ni  Wayan Dita, Ph.D."},
    {"nmdosen": "I Wayan Komang, S.Kom"},
]


def _lampiran(*items: str):
    from app.rag.tools.base import Lampiran

    return Lampiran(title="Dosen bergelar Dr.", source=LABEL, items=tuple(items))


class TestSaringDaftarDosen:
    async def test_gelar_disaring_dan_dihitung_handler(self):
        """T46: "berapa dosen bergelar Dr." dijawab model 38 lalu 39 dari 25 nama.
        Hitungannya kini dari handler, dan "Dr." tidak ikut mencocokkan "Drs."."""
        r = await sads._daftar_dosen(FakeSads(_DOSEN), gelar="Dr.")
        assert r.ok
        assert "Jumlah dosen yang mengajar di INSTIKI bergelar Dr.: 2 orang." in r.text
        assert "(Seluruh dosen yang mengajar: 5 orang.)" in r.text
        assert len(r.attachments) == 1
        assert r.attachments[0].items == (
            "Dr. Ani Wijaya, S.Kom., M.T.",
            "Dr.Ir. Citra Dewi, S.Kom.,M.Kom.",
        )
        assert r.attachments[0].title == "Dosen bergelar Dr."
        assert r.attachments[0].source == LABEL

    @pytest.mark.parametrize(
        ("gelar", "jumlah"),
        [
            ("dr", 2),
            ("DR.", 2),
            ("doktor", 3),
            ("Ph.D", 1),
            ("Drs.", 1),
            ("M.Kom.", 1),
            ("Prof.", 0),
        ],
    )
    async def test_penulisan_gelar_disamakan(self, gelar, jumlah):
        r = await sads._daftar_dosen(FakeSads(_DOSEN), gelar=gelar)
        assert len(r.attachments[0].items if r.attachments else ()) == jumlah

    async def test_nama_dicocokkan_ke_nama_inti_bukan_gelar(self):
        """"kom" mencocokkan "Komang", bukan gelar "S.Kom" milik hampir semua orang."""
        r = await sads._daftar_dosen(FakeSads(_DOSEN), nama="kom")
        assert r.attachments[0].items == ("I Wayan Komang, S.Kom",)

    async def test_nama_tanpa_beda_huruf_dan_spasi(self):
        r = await sads._daftar_dosen(FakeSads(_DOSEN), nama="ni wayan")
        assert r.attachments[0].items == ("Ni  Wayan Dita, Ph.D.",)
        semua_wayan = await sads._daftar_dosen(FakeSads(_DOSEN), nama="WAYAN")
        assert len(semua_wayan.attachments[0].items) == 2

    async def test_nama_dan_gelar_digabung(self):
        r = await sads._daftar_dosen(FakeSads(_DOSEN), nama="wayan", gelar="doktor")
        assert r.attachments[0].items == ("Ni  Wayan Dita, Ph.D.",)
        assert 'bergelar doktor dengan nama memuat "wayan"' in r.text

    async def test_saringan_tanpa_hasil_bukan_kegagalan(self):
        """Tidak ada yang cocok = jawaban sah SADS, bukan DATA_TIDAK_TERSEDIA (T43):
        yang terakhir membuat model menolak dan menyuruh bertanya ke FO."""
        r = await sads._daftar_dosen(FakeSads(_DOSEN), gelar="Prof.")
        assert r.ok and r.attachments == ()
        assert "Tidak ada dosen yang mengajar di INSTIKI bergelar Prof.: 0 orang" in r.text
        assert "DATA_TIDAK_TERSEDIA" not in r.pesan_untuk_model()

    async def test_sads_kosong_tetap_kegagalan(self):
        r = await sads._daftar_dosen(FakeSads([]), gelar="Dr.")
        assert not r.ok and r.attachments == ()

    async def test_tanpa_saringan_seluruh_dosen_berlampiran(self):
        r = await sads._daftar_dosen(FakeSads(_DOSEN))
        assert r.attachments[0].title == "Dosen yang mengajar di INSTIKI"
        assert len(r.attachments[0].items) == 5
        assert "Seluruh dosen" not in r.text  # baris pembanding hanya saat disaring

    def test_skema_saringan_opsional(self):
        params = _registry().get("get_daftar_dosen").parameters
        assert set(params["properties"]) == {"nama", "gelar"}
        assert "required" not in params

    def test_argumen_opsional_null_dianggap_tidak_disaring(self):
        """Model kadang mengirim `"nama": null`; jangan jadi saringan "None"."""
        spec = _registry().get("get_daftar_dosen")
        assert validasi_argumen(spec, {"nama": None, "gelar": " Dr. "}) == {"gelar": "Dr."}


class TestLampiran:
    def test_pesan_untuk_model_memberi_tahu_daftar_sudah_tampil(self):
        from app.rag.prompts import TOOL_SYSTEM_PROMPT
        from app.rag.tools.base import CATATAN_LAMPIRAN

        berlampiran = ToolResult("a", LABEL, "- Ani", True, attachments=(_lampiran("Ani"),))
        pesan = berlampiran.pesan_untuk_model()
        assert CATATAN_LAMPIRAN in pesan and "- Ani" in pesan  # data tetap dikirim
        tanpa = ToolResult("a", LABEL, "- Ani", True).pesan_untuk_model()
        assert CATATAN_LAMPIRAN not in tanpa
        # Aturan T5 merujuk penanda yang sama persis.
        assert "DAFTAR_DITAMPILKAN" in CATATAN_LAMPIRAN
        assert "DAFTAR_DITAMPILKAN" in TOOL_SYSTEM_PROMPT

    async def test_loop_mengumpulkan_lampiran_tool_yang_berhasil(self):
        lampiran = _lampiran("Ani", "Citra")

        async def handler(*, matkul: str) -> ToolResult:
            return ToolResult(
                "get_mk_diampu_dosen", LABEL, "- Ani", True, attachments=(lampiran,)
            )

        llm = FakeChat(
            [
                _tool_turn("get_mk_diampu_dosen", '{"matkul":"X"}'),
                _answer_turn(f"Ada 2 dosen [{LABEL}]."),
            ]
        )
        res = await _jalankan(llm, handler)
        assert res.attachments == [lampiran]

    async def test_loop_hanya_meneruskan_mata_kuliah_yang_ditulis_jawaban(self):
        bd, bdl = _mk_lampiran("Basis Data"), _mk_lampiran("Basis Data Lanjut")

        async def handler(*, matkul: str) -> ToolResult:
            return ToolResult("get_mk_diampu_dosen", LABEL, "-", True, attachments=(bdl, bd))

        llm = FakeChat(
            [
                _tool_turn("get_mk_diampu_dosen", '{"matkul":"Basis Data"}'),
                _answer_turn(f"Basis Data diampu 2 dosen [{LABEL}]."),
            ]
        )
        assert (await _jalankan(llm, handler)).attachments == [bd]

    async def test_pipeline_meneruskan_lampiran_yang_sumbernya_dikutip(self):
        lampiran = _lampiran("Ani", "Citra")
        docs = hasil_tool_ke_dokumen([ToolResult("a", LABEL, "- Ani\n- Citra", True)])
        llm = ToolLLMBerlampiran(f"Ada 2 dosen bergelar Dr. [{LABEL}].", docs, [lampiran])
        outcome = await run_pipeline(
            "berapa dosen bergelar Dr.?",
            retriever=FakeRetriever([]),
            llm_call=llm,
            tool_registry=ToolRegistry([_spec(lambda **_: None)]),
        )
        assert outcome.kind is OutcomeKind.ANSWER
        assert outcome.attachments == (lampiran,)

    async def test_lampiran_tidak_ditempel_bila_sumbernya_tidak_dikutip(self):
        """Model memanggil tool lalu menjawab tanpa memakainya: daftar 220 dosen
        tidak boleh menempel di bawah jawaban yang tidak bersumber darinya."""
        docs = hasil_tool_ke_dokumen([ToolResult("a", LABEL, "- Ani", True)])
        llm = ToolLLMBerlampiran("Jawaban tanpa penanda sumber.", docs, [_lampiran("Ani")])
        outcome = await run_pipeline(
            "berapa dosen bergelar Dr.?",
            retriever=FakeRetriever([]),
            llm_call=llm,
            tool_registry=ToolRegistry([_spec(lambda **_: None)]),
        )
        assert outcome.kind is OutcomeKind.ANSWER
        assert outcome.attachments == ()

    async def test_penolakan_tidak_membawa_lampiran(self):
        docs = hasil_tool_ke_dokumen([ToolResult("a", LABEL, "- Ani", True)])
        llm = ToolLLMBerlampiran(NOT_FOUND_MARKER, docs, [_lampiran("Ani")])
        outcome = await run_pipeline(
            "berapa dosen bergelar Dr.?",
            retriever=FakeRetriever([]),
            llm_call=llm,
            tool_registry=ToolRegistry([_spec(lambda **_: None)]),
        )
        assert outcome.kind is OutcomeKind.REFUSAL
        assert outcome.attachments == ()

    def test_respons_dan_meta_membawa_lampiran(self):
        from app.observability.chatlog import ChatLogEntry, build_meta
        from app.rag.chain import PipelineOutcome
        from app.routers.chat import to_response

        outcome = PipelineOutcome(
            kind=OutcomeKind.ANSWER,
            text=f"Ada 2 dosen [{LABEL}].",
            documents=tuple(hasil_tool_ke_dokumen([ToolResult("a", LABEL, "- Ani", True)])),
            llm_called=True,
            attachments=(_lampiran("Ani", "Citra"),),
        )
        diharapkan = [
            {"title": "Dosen bergelar Dr.", "source": LABEL, "items": ["Ani", "Citra"]}
        ]
        assert [a.model_dump() for a in to_response(outcome).attachments] == diharapkan
        meta = build_meta(
            ChatLogEntry(session_id="s", question="q", outcome=outcome, latency_ms=1)
        )
        assert meta["attachments"] == diharapkan

        tanpa = PipelineOutcome(kind=OutcomeKind.ANSWER, text="t", llm_called=True)
        assert to_response(tanpa).attachments == []
        meta_tanpa = build_meta(
            ChatLogEntry(session_id="s", question="q", outcome=tanpa, latency_ms=1)
        )
        assert meta_tanpa["attachments"] is None


def _mk_lampiran(nama: str):
    from app.rag.tools.base import Lampiran

    return Lampiran(title=f"Dosen pengampu {nama}", source=LABEL, items=("Ani",), disebut=nama)


class TestPilihLampiran:
    """T49: satu pencarian `get_mk_diampu_dosen` berlampiran per mata kuliah; yang
    tampil hanya mata kuliah yang ditulis jawaban."""

    def _pilih(self, jawaban: str, *nama: str) -> list[str]:
        from app.rag.tools.base import pilih_lampiran

        return [a.disebut for a in pilih_lampiran([_mk_lampiran(n) for n in nama], jawaban)]

    def test_hanya_yang_ditulis(self):
        jawaban = "Basis Data diampu 10 dosen [Data akademik SADS]."
        assert self._pilih(jawaban, "Basis Data Lanjut", "Basis Data") == ["Basis Data"]

    def test_nama_panjang_tidak_dihitung_sebagai_nama_pendek(self):
        jawaban = "Basis Data Lanjut diampu 13 dosen."
        assert self._pilih(jawaban, "Basis Data", "Basis Data Lanjut") == ["Basis Data Lanjut"]
        keduanya = "Basis Data diampu 10 dosen, Basis Data Lanjut 13 dosen."
        assert self._pilih(keduanya, "Basis Data", "Basis Data Lanjut") == [
            "Basis Data",
            "Basis Data Lanjut",
        ]

    def test_ejaan_dan_huruf_besar_disamakan(self):
        jawaban = "Kecerdasan Buatan (tercatat sebagai **artificial  inteligence**)."
        nama = ("Artificial Intelligence", "Advanced Artificial Intelligence")
        assert self._pilih(jawaban, *nama) == ["Artificial Intelligence"]

    def test_tidak_ada_yang_ditulis_semuanya_tampil(self):
        """Model hanya menulis "Kecerdasan Buatan": lebih baik dua daftar daripada
        "diampu 35 dosen" tanpa daftar yang dijanjikan `CATATAN_LAMPIRAN`."""
        nama = ("Artificial Intelligence", "Advanced Artificial Intelligence")
        assert self._pilih("Kecerdasan Buatan diampu 35 dosen.", *nama) == list(nama)

    def test_bukan_potongan_kata(self):
        assert self._pilih("Ada Databases dan Data.", "Database", "Data") == ["Data"]

    def test_satu_lampiran_atau_tanpa_sebutan_selalu_tampil(self):
        from app.rag.tools.base import pilih_lampiran

        assert self._pilih("tidak menyebut apa pun", "Basis Data") == ["Basis Data"]
        tanpa = _lampiran("Ani")
        assert pilih_lampiran([tanpa], "x") == [tanpa]


class ToolLLMBerlampiran(ToolLLM):
    """`ToolLLM` yang, seperti `LLMCall.run_tools`, mencatat lampiran tool."""

    def __init__(self, answer: str, docs: list[Any], attachments: list[Any]) -> None:
        super().__init__(answer, docs)
        self._lampiran = attachments
        self.attachments: list[Any] = []

    async def run_tools(self, *args: Any, **kwargs: Any):
        hasil = await super().run_tools(*args, **kwargs)
        self.attachments = list(self._lampiran)
        return hasil
