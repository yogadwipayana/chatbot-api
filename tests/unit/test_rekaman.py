"""Rekaman input/output giliran untuk tab Graf (`app/observability/rekaman.py`)."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from enum import StrEnum

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from app.observability.rekaman import (
    MAKS_BUTIR,
    MAKS_TEKS,
    buka,
    catat_panggilan,
    jadikan_json,
    kemas,
    samarkan,
)


class Warna(StrEnum):
    MERAH = "merah"


@dataclass(frozen=True)
class Kotak:
    warna: Warna
    isi: tuple[str, ...]


class Model(BaseModel):
    nama: str


class TestJadikanJson:
    def test_dataclass_enum_dan_tuple(self):
        assert jadikan_json(Kotak(Warna.MERAH, ("a", "b"))) == {
            "warna": "merah",
            "isi": ["a", "b"],
        }

    def test_document_dan_pydantic(self):
        doc = Document(page_content="isi", metadata={"judul": "Panduan", "halaman": 3})
        assert jadikan_json(doc) == {
            "page_content": "isi",
            "metadata": {"judul": "Panduan", "halaman": 3},
        }
        assert jadikan_json(Model(nama="x")) == {"nama": "x"}

    def test_pesan_langchain_beserta_tool_call(self):
        pesan = [
            HumanMessage(content="siapa dosen BD?"),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_mk", "args": {"matkul": "BD"}, "id": "c1"}],
            ),
            ToolMessage(content="10 dosen", tool_call_id="c1"),
        ]
        assert jadikan_json(pesan) == [
            {"role": "human", "content": "siapa dosen BD?"},
            {
                "role": "ai",
                "content": "",
                "tool_calls": [{"name": "get_mk", "args": {"matkul": "BD"}, "id": "c1"}],
            },
            {"role": "tool", "content": "10 dosen", "tool_call_id": "c1"},
        ]

    def test_string_dan_daftar_dipotong(self):
        panjang = jadikan_json("x" * (MAKS_TEKS + 10))
        assert panjang.startswith("x" * MAKS_TEKS)
        assert "10 karakter lagi" in panjang
        daftar = jadikan_json(list(range(MAKS_BUTIR + 5)))
        assert len(daftar) == MAKS_BUTIR + 1
        assert daftar[-1] == "… 5 butir lagi"

    def test_objek_asing_menjadi_repr(self):
        class Asing:
            def __repr__(self) -> str:
                return "<asing>"

        assert jadikan_json({"a": Asing()}) == {"a": "<asing>"}


class TestKemas:
    def test_bolak_balik(self):
        isi = {"nodes": [{"input": {"q": "kapan KRS?"}, "output": None}], "angka": 1.5}
        assert buka(kemas(isi)) == isi

    def test_subpohon_panjang_disimpan_sekali(self):
        """State kumulatif: daftar dokumen yang sama muncul di beberapa node."""
        dokumen = [{"page_content": f"DOK-{i} " + "isi " * 100} for i in range(5)]
        isi = {"nodes": [{"input": {"documents": dokumen}} for _ in range(4)]}
        mentah = zlib.decompress(kemas(isi)).decode()
        # Isi setiap dokumen hanya sekali di blob, walau dirujuk empat node.
        assert mentah.count("DOK-0") == 1
        assert buka(kemas(isi)) == isi

    def test_samaran_diterapkan_sebelum_disimpan(self):
        blob = kemas(
            {"q": "saya ingin bunuh diri", "prompt": "Pertanyaan: saya ingin bunuh diri."},
            sensitif=["saya ingin bunuh diri"],
            pengganti="[disembunyikan]",
        )
        assert b"bunuh" not in zlib.decompress(blob)
        assert buka(blob) == {"q": "[disembunyikan]", "prompt": "Pertanyaan: [disembunyikan]."}


class TestSamarkan:
    def test_yang_panjang_diganti_lebih_dulu(self):
        """Versi bersih (lebih pendek) tidak boleh menyisakan ekor versi mentah."""
        hasil = samarkan(
            {"a": "saya stres  sekali!!", "b": ["saya stres sekali"]},
            ["saya stres sekali", "saya stres  sekali!!"],
            "[X]",
        )
        assert hasil == {"a": "[X]", "b": ["[X]"]}

    def test_tanpa_teks_tidak_berubah(self):
        assert samarkan({"a": "b"}, ["", "  "], "[X]") == {"a": "b"}


async def test_catat_panggilan_di_luar_graf_tidak_melempar():
    """Skrip dan test fungsi tunggal memanggil JEV/tool tanpa run induk."""
    import time

    await catat_panggilan(jenis="tool", nama="x", masukan={}, mulai=time.perf_counter())
