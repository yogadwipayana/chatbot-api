"""Galat yang dikirim gateway sebagai isi jawaban (`PENANDA_GALAT_GATEWAY`).

Gateway proyek ini membalas 200 lalu menyisipkan "[Error] Our servers are
currently overloaded..." ke aliran, bahkan setelah beberapa potongan jawaban.
Sebelum pemeriksaan ini ada, teks itu tersimpan sebagai jawaban biasa
(`kind=answer`) dan tampil di widget lengkap dengan tombol penilaian.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.config import Settings
from app.deps import LLMCall, RewriteCall
from app.rag import providers
from app.rag.providers import (
    GalatGateway,
    PenyaringGalatGateway,
    periksa_galat_gateway,
)

pytestmark = pytest.mark.langchain

GALAT = "[Error] Our servers are currently overloaded. Please try again later."
JAWABAN = "Harga TOEIC Rp675.000 [HARGA SERTIFIKASI, hal. 1]."


def alirkan(potongan: list[str]) -> str:
    penyaring = PenyaringGalatGateway()
    keluar = [penyaring.terima(p) for p in potongan]
    return "".join(keluar) + penyaring.sisa()


class TestPeriksa:
    def test_jawaban_biasa_diteruskan(self):
        assert periksa_galat_gateway(JAWABAN) == JAWABAN

    def test_penanda_di_tengah_jawaban(self):
        with pytest.raises(GalatGateway) as galat:
            periksa_galat_gateway("Untuk membayar" + GALAT)
        # Pesannya untuk log: galat gateway saja, tanpa potongan jawaban.
        assert str(galat.value) == GALAT


class TestPenyaring:
    def test_jawaban_biasa_utuh(self):
        potongan = ["Harga TOEIC ", "Rp675.000 [", "HARGA SERTIFIKASI, hal. 1", "]."]
        assert alirkan(potongan) == JAWABAN

    def test_kurung_di_akhir_aliran_dilepas(self):
        penyaring = PenyaringGalatGateway()
        assert penyaring.terima("lihat [") == "lihat "
        assert penyaring.sisa() == "["

    def test_penanda_di_potongan_pertama(self):
        with pytest.raises(GalatGateway):
            PenyaringGalatGateway().terima(GALAT)

    @pytest.mark.parametrize("belah", range(1, len("Untuk membayar" + GALAT)))
    def test_penanda_terbelah_tidak_pernah_diteruskan(self, belah):
        aliran = "Untuk membayar" + GALAT
        penyaring = PenyaringGalatGateway()
        keluar = ""
        with pytest.raises(GalatGateway):
            for potongan in (aliran[:belah], aliran[belah:]):
                keluar += penyaring.terima(potongan)
        assert "Untuk membayar".startswith(keluar)


def model_palsu(monkeypatch, teks: str) -> None:
    """`build_llm` diganti model yang mengalirkan `teks` per kata, tanpa jaringan."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content=teks)]))
    monkeypatch.setattr(providers, "build_llm", lambda *a, **kw: model)


@pytest.fixture
def settings():
    return Settings(_env_file=None, api_key="sk-uji")


class TestLLMCall:
    async def test_aliran_berhenti_sebelum_penanda(self, monkeypatch, settings):
        model_palsu(monkeypatch, "Untuk membayar" + GALAT)
        keluar = []
        with pytest.raises(GalatGateway):
            async for potongan in LLMCall(settings).stream("pertanyaan", []):
                keluar.append(potongan)
        assert "[Error]" not in "".join(keluar)

    async def test_aliran_biasa_utuh(self, monkeypatch, settings):
        model_palsu(monkeypatch, JAWABAN)
        keluar = [p async for p in LLMCall(settings).stream("pertanyaan", [])]
        assert "".join(keluar) == JAWABAN

    async def test_tanpa_aliran(self, monkeypatch, settings):
        model_palsu(monkeypatch, GALAT)
        with pytest.raises(GalatGateway):
            await LLMCall(settings)("pertanyaan", [])


class TestRewriteCall:
    async def test_galat_tidak_menjadi_query(self, monkeypatch, settings):
        model_palsu(monkeypatch, GALAT)
        with pytest.raises(GalatGateway):
            await RewriteCall(settings)("bagaimana caranya?", "user: biaya TOEIC")
