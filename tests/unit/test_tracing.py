"""Penyalaan tracing LangSmith (FR-8).

Tidak satu pun test di sini menghubungi LangSmith. Yang diperiksa adalah
keputusan yang diambil sebelum trace pertama dikirim -- variabel lingkungan mana
yang ditulis, ID mana yang layak dicatat -- dan justru di situlah kerusakannya
tidak terlihat: tracing yang diam-diam mati, atau ID yang menunjuk trace yang
tidak pernah ada.
"""

from __future__ import annotations

import os
from uuid import UUID

import pytest

from app.config import Settings
from app.observability import tracing

VARIABEL_TRACING = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
)


@pytest.fixture
def lingkungan_bersih(monkeypatch):
    """Kosongkan variabel tracing, dan pastikan nilainya pulih setelah test.

    `monkeypatch.delenv` di sini bukan sekadar membersihkan: ia mendaftarkan
    setiap nama supaya nilai aslinya dikembalikan. Tanpa pendaftaran itu,
    variabel yang ditulis `configure_tracing` akan bertahan dan menyalakan
    tracing untuk seluruh test yang berjalan sesudahnya.
    """
    for nama in VARIABEL_TRACING:
        monkeypatch.delenv(nama, raising=False)
    tracing._bersihkan_cache_env()
    yield
    tracing._bersihkan_cache_env()


def settings(**overrides) -> Settings:
    """Settings tanpa membaca .env, supaya hasil test tidak ikut kunci di mesin."""
    return Settings(_env_file=None, **overrides)


class TestConfigureTracing:
    def test_menyala_saat_kunci_ada(self, lingkungan_bersih):
        aktif = tracing.configure_tracing(
            settings(langsmith_api_key="ls-uji", langsmith_project="chatbot-uji")
        )
        assert aktif is True
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == "ls-uji"
        assert os.environ["LANGSMITH_PROJECT"] == "chatbot-uji"

    def test_mati_tanpa_kunci(self, lingkungan_bersih):
        """LANGSMITH_TRACING=true tanpa kunci bukan berarti tracing bisa jalan."""
        aktif = tracing.configure_tracing(settings(langsmith_tracing=True))
        assert aktif is False
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert "LANGSMITH_API_KEY" not in os.environ

    def test_mati_saat_dimatikan_walau_kunci_ada(self, lingkungan_bersih):
        aktif = tracing.configure_tracing(
            settings(langsmith_tracing=False, langsmith_api_key="ls-uji")
        )
        assert aktif is False
        assert os.environ["LANGSMITH_TRACING"] == "false"

    @pytest.mark.parametrize("nilai_lama", ["true", "false"])
    @pytest.mark.parametrize("aktif", [True, False])
    def test_variabel_langchain_lama_dibuang(self, lingkungan_bersih, nilai_lama, aktif):
        """`LANGCHAIN_TRACING_V2` yang tertinggal tidak boleh ikut menentukan.

        langsmith membacanya LEBIH DULU daripada `LANGSMITH_TRACING`, jadi sisa
        variabel dari proyek lain di mesin yang sama bisa menyalakan tracing yang
        sudah dimatikan -- atau, arah sebaliknya, memadamkan tracing yang
        dinyalakan tanpa satu pun pesan.
        """
        os.environ["LANGCHAIN_TRACING_V2"] = nilai_lama
        os.environ["LANGCHAIN_TRACING"] = nilai_lama

        tracing.configure_tracing(
            settings(langsmith_api_key="ls-uji" if aktif else None)
        )

        assert "LANGCHAIN_TRACING_V2" not in os.environ
        assert "LANGCHAIN_TRACING" not in os.environ
        assert os.environ["LANGSMITH_TRACING_V2"] == ("true" if aktif else "false")

    def test_endpoint_diteruskan_ke_sdk(self, lingkungan_bersih):
        """Region non-default hanya sampai ke SDK lewat os.environ.

        `langsmith.Client` membacanya dengan `os.getenv`; nilai yang berhenti di
        objek Settings tidak pernah ia lihat, dan trace mendarat di region bawaan
        tanpa galat apa pun.
        """
        tracing.configure_tracing(
            settings(
                langsmith_api_key="ls-uji",
                langsmith_endpoint="https://eu.api.smith.langchain.com",
            )
        )
        assert os.environ["LANGSMITH_ENDPOINT"] == "https://eu.api.smith.langchain.com"

    def test_endpoint_kosong_tidak_ditulis(self, lingkungan_bersih):
        """Bawaan SDK dibiarkan berlaku, bukan ditimpa string kosong."""
        tracing.configure_tracing(settings(langsmith_api_key="ls-uji"))
        assert "LANGSMITH_ENDPOINT" not in os.environ


class TestIdGiliran:
    def test_none_saat_tracing_mati(self, lingkungan_bersih):
        tracing.configure_tracing(settings())
        assert tracing.sedang_menjejak() is False
        assert tracing.id_giliran() is None

    def test_uuid_v7_saat_tracing_aktif(self, lingkungan_bersih):
        """v7 menyimpan timestamp, sehingga run terurut benar di dalam trace."""
        tracing.configure_tracing(settings(langsmith_api_key="ls-uji"))
        assert tracing.sedang_menjejak() is True
        assert UUID(tracing.id_giliran()).version == 7

    def test_setiap_giliran_dapat_id_sendiri(self, lingkungan_bersih):
        tracing.configure_tracing(settings(langsmith_api_key="ls-uji"))
        assert tracing.id_giliran() != tracing.id_giliran()


class TestJejakGiliran:
    async def test_tidak_membuka_apa_pun_saat_tracing_mati(self, lingkungan_bersih):
        async with tracing.jejak_giliran(run_id=None) as akar:
            assert akar is None

    async def test_langkah_di_dalamnya_menempel_ke_akar(
        self, lingkungan_bersih, monkeypatch
    ):
        """Inilah yang menyatukan satu giliran menjadi satu trace.

        langchain-core menentukan induk sebuah run dari run tree yang sedang
        aktif di konteks langsmith (`callbacks/manager.py` membacanya sebagai
        `tracing_context["parent"]`). Jadi begitu akar ini terpasang di konteks,
        penulisan ulang query, retrieval, dan panggilan LLM menempel padanya
        sendiri tanpa satu pun dari mereka perlu diubah.

        `post`/`patch` dimatikan supaya tidak ada yang terkirim ke LangSmith;
        yang diuji adalah pemasangan konteksnya, bukan pengirimannya.
        """
        from langsmith import run_trees
        from langsmith.run_helpers import get_current_run_tree

        monkeypatch.setattr(run_trees.RunTree, "post", lambda self, *a, **k: None)
        monkeypatch.setattr(run_trees.RunTree, "patch", lambda self, *a, **k: None)
        tracing.configure_tracing(settings(langsmith_api_key="ls-uji"))
        run_id = tracing.id_giliran()

        async with tracing.jejak_giliran(run_id=run_id, session_id="sesi-1"):
            induk = get_current_run_tree()

        assert induk is not None, "tidak ada akar yang terpasang di konteks"
        assert str(induk.id) == run_id
        assert induk.metadata["session_id"] == "sesi-1"

    async def test_tidak_meninggalkan_konteks_setelah_selesai(
        self, lingkungan_bersih, monkeypatch
    ):
        """Giliran berikutnya tidak boleh tersambung ke akar giliran sebelumnya."""
        from langsmith import run_trees
        from langsmith.run_helpers import get_current_run_tree

        monkeypatch.setattr(run_trees.RunTree, "post", lambda self, *a, **k: None)
        monkeypatch.setattr(run_trees.RunTree, "patch", lambda self, *a, **k: None)
        tracing.configure_tracing(settings(langsmith_api_key="ls-uji"))

        async with tracing.jejak_giliran(run_id=tracing.id_giliran()):
            pass

        assert get_current_run_tree() is None


class TestKonfigurasiRun:
    def test_membawa_nama_dan_session(self):
        config = tracing.konfigurasi_run("generate_answer", session_id="sesi-1")
        assert config["run_name"] == "generate_answer"
        assert config["metadata"] == {"session_id": "sesi-1"}

    def test_tanpa_session_tidak_menitipkan_metadata_kosong(self):
        """Metadata kosong hanya menambah derau pada setiap run di LangSmith."""
        assert "metadata" not in tracing.konfigurasi_run("rewrite_query")

    def test_run_id_opsional(self):
        assert "run_id" not in tracing.konfigurasi_run("rewrite_query")
        assert tracing.konfigurasi_run("x", run_id="abc")["run_id"] == "abc"


class TestTandaiSesi:
    def test_menempelkan_session_id(self):
        class Panggilan:
            pass

        satu, dua = Panggilan(), Panggilan()
        tracing.tandai_sesi("sesi-1", satu, dua)
        assert satu.session_id == dua.session_id == "sesi-1"

    def test_objek_yang_menolak_atribut_tidak_menggagalkan_jawaban(self):
        """Pengelompokan trace tidak layak menjatuhkan jawaban yang sudah benar."""

        class Tertutup:
            __slots__ = ()

        tracing.tandai_sesi("sesi-1", Tertutup())  # tidak boleh melempar


class TestAkhiriJejak:
    def test_aman_saat_tracing_mati(self):
        tracing.akhiri_jejak(None, kind="answer")  # tidak boleh melempar

    def test_menutup_akar_dengan_keluaran(self):
        class AkarPalsu:
            def __init__(self) -> None:
                self.outputs = None

            def end(self, outputs) -> None:
                self.outputs = outputs

        akar = AkarPalsu()
        tracing.akhiri_jejak(akar, kind="answer", text="jawaban")
        assert akar.outputs == {"kind": "answer", "text": "jawaban"}
