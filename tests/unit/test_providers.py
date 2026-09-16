"""Pabrik model chat dan embedding -- BASE_URL, API_KEY, CHAT_MODEL, EMBED_MODEL.

Model dirakit tanpa satu pun permintaan jaringan: konstruktor LangChain hanya
menyimpan konfigurasi. Yang diuji adalah bahwa nilai dari .env benar-benar
sampai ke klien -- kesalahan di lapisan ini tidak menghasilkan galat, melainkan
permintaan yang diam-diam terkirim ke endpoint atau model yang salah.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.db.models import EMBEDDING_DIM
from app.rag.providers import build_embeddings, build_llm

pytestmark = pytest.mark.langchain

VARIABEL_LINGKUNGAN = (
    "ENVIRONMENT",
    "BASE_URL",
    "API_KEY",
    "CHAT_MODEL",
    "EMBED_MODEL",
    # Dibaca SDK OpenAI sendiri bila nilainya tidak diteruskan.
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
)

ENDPOINT = "https://penyedia.contoh/v1"
SECRET_PROD = "x" * 48


@pytest.fixture(autouse=True)
def lingkungan_bersih(monkeypatch):
    """Nilai yang kebetulan ada di mesin pengembang tidak boleh memengaruhi test."""
    for nama in VARIABEL_LINGKUNGAN:
        monkeypatch.delenv(nama, raising=False)


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


class TestChat:
    def test_model_dan_kunci_diteruskan(self):
        llm = build_llm(settings(api_key="sk-uji", chat_model="gpt-5.5"))
        assert llm.model_name == "gpt-5.5"
        assert llm.openai_api_key.get_secret_value() == "sk-uji"

    def test_base_url_diteruskan(self):
        llm = build_llm(settings(api_key="sk-uji", base_url=ENDPOINT))
        assert llm.openai_api_base == ENDPOINT

    def test_base_url_kosong_memakai_endpoint_resmi(self):
        assert build_llm(settings(api_key="sk-uji")).openai_api_base is None

    def test_temperature_nol(self):
        """Jawaban administrasi harus konsisten; variasi kreatif tidak diinginkan."""
        assert build_llm(settings(api_key="sk-uji")).temperature == 0

    def test_flag_streaming_diteruskan(self):
        s = settings(api_key="sk-uji")
        assert build_llm(s, streaming=True).streaming is True
        assert build_llm(s, streaming=False).streaming is False


class TestEmbedding:
    def test_model_dan_kunci_diteruskan(self):
        s = settings(api_key="sk-uji", embed_model="text-embedding-3-large")
        emb = build_embeddings(s)
        assert emb.model == "text-embedding-3-large"
        assert emb.openai_api_key.get_secret_value() == "sk-uji"

    def test_dimensi_mengikuti_kolom_database(self):
        """Beda dimensi baru ketahuan saat INSERT ke kolom vector(1024)."""
        assert build_embeddings(settings(api_key="sk-uji")).dimensions == EMBEDDING_DIM == 1024

    def test_endpoint_resmi_tetap_memakai_tokenisasi_bawaan(self):
        emb = build_embeddings(settings(api_key="sk-uji"))
        assert emb.openai_api_base is None
        assert emb.check_embedding_ctx_length is True

    def test_base_url_mengirim_teks_mentah_bukan_token_id(self):
        """Default langchain-openai men-tokenisasi dengan tiktoken lalu mengirim
        token ID. Banyak endpoint non-OpenAI hanya menerima teks."""
        emb = build_embeddings(settings(api_key="sk-uji", base_url=ENDPOINT))
        assert emb.openai_api_base == ENDPOINT
        assert emb.check_embedding_ctx_length is False

    def test_chat_dan_embedding_memakai_endpoint_dan_kunci_yang_sama(self):
        s = settings(api_key="sk-uji", base_url=ENDPOINT)
        llm, emb = build_llm(s), build_embeddings(s)
        assert llm.openai_api_base == emb.openai_api_base == ENDPOINT
        assert llm.openai_api_key.get_secret_value() == emb.openai_api_key.get_secret_value()


class TestApiKey:
    @pytest.mark.parametrize("pembangun", [build_llm, build_embeddings])
    def test_kunci_hilang_ditolak_dengan_nama_variabel(self, pembangun):
        with pytest.raises(ValueError, match="API_KEY"):
            pembangun(settings())

    @pytest.mark.parametrize("pembangun", [build_llm, build_embeddings])
    def test_kunci_kosong_dianggap_tidak_diisi(self, pembangun):
        """Mengirim string kosong ke API hanya menghasilkan 401 yang membingungkan."""
        with pytest.raises(ValueError, match="API_KEY"):
            pembangun(settings(api_key="   "))


class TestValidasiKonfigurasi:
    def test_base_url_tanpa_skema_ditolak(self):
        with pytest.raises(ValidationError, match="BASE_URL"):
            settings(base_url="penyedia.contoh/v1")

    def test_lokal_boleh_start_tanpa_kunci(self):
        """Pengembangan dan test tidak boleh terhalang ketiadaan kunci."""
        assert settings().kunci_api() is None

    def test_produksi_tanpa_kunci_ditolak(self):
        with pytest.raises(ValidationError, match="API_KEY"):
            settings(environment="production", admin_jwt_secret=SECRET_PROD)

    def test_produksi_kunci_kosong_tetap_ditolak(self):
        with pytest.raises(ValidationError, match="API_KEY"):
            settings(environment="production", admin_jwt_secret=SECRET_PROD, api_key="")

    def test_produksi_dengan_kunci_diterima(self):
        s = settings(environment="production", admin_jwt_secret=SECRET_PROD, api_key="sk")
        assert s.environment == "production"


class TestUserAgent:
    """UA bawaan SDK `OpenAI/Python x.y.z` diblokir firewall Cloudflare di depan
    gateway proyek ini (403 "Your request was blocked."), sementara UA lain lolos.
    Kalau header ini hilang, chat dan embedding gagal total -- dengan pesan yang
    tampak seperti masalah kunci padahal bukan."""

    @pytest.mark.parametrize("pembangun", [build_llm, build_embeddings])
    def test_memakai_user_agent_aplikasi(self, pembangun):
        from app.rag.providers import USER_AGENT

        klien = pembangun(settings(api_key="sk-uji", base_url=ENDPOINT))
        assert klien.default_headers["User-Agent"] == USER_AGENT

    @pytest.mark.parametrize("pembangun", [build_llm, build_embeddings])
    def test_berlaku_juga_tanpa_base_url(self, pembangun):
        from app.rag.providers import USER_AGENT

        klien = pembangun(settings(api_key="sk-uji"))
        assert klien.default_headers["User-Agent"] == USER_AGENT

    def test_bukan_user_agent_bawaan_sdk_openai(self):
        from app.rag.providers import USER_AGENT

        assert not USER_AGENT.startswith("OpenAI/Python")
