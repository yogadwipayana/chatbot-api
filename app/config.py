"""Konfigurasi aplikasi.

Semua nilai yang membedakan environment ada di sini, dibaca dari variabel
lingkungan / `.env`. Tidak ada kunci API yang boleh muncul sebagai default.

Ambang penolakan (FR-3) sengaja tidak diberi default "aman": biarkan
`ThresholdPolicy` yang memegang placeholder, dan catat di README bahwa nilai
produksi berasal dari kalibrasi empiris.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER_JWT_SECRET = "GANTI-NILAI-INI-SEBELUM-DEPLOY-KE-PRODUKSI-0000"
"""Nilai sengaja panjang agar lolos syarat 32 byte saat pengembangan lokal,
tetapi ditolak mentah-mentah di produksi oleh validator di bawah."""

ORIGIN_LOKAL = (
    "http://localhost:3000",  # admin/
    "http://localhost:3001",  # client/
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
)
"""Asal dev yang selalu diizinkan saat `ENVIRONMENT=local`. `CORS_ORIGINS` di
.env berisi domain produksi; tanpa daftar ini mengisinya akan memblokir
`npm run dev` di mesin sendiri."""


def _terisi(kunci: SecretStr | None) -> SecretStr | None:
    """None bila kunci tidak ada atau hanya berisi spasi.

    Lapisan kedua di samping `env_ignore_empty`: nilai kosong bisa juga masuk
    lewat argumen konstruktor, bukan hanya dari berkas .env.
    """
    if kunci is None or not kunci.get_secret_value().strip():
        return None
    return kunci


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `KUNCI=` yang dibiarkan kosong (salinan .env.example) berarti "tidak
        # diisi" dan memakai default. Tanpa ini ia menjadi string kosong:
        # ADMIN_JWT_SECRET= menggagalkan start dan VECTOR_THRESHOLD= gagal di-parse.
        env_ignore_empty=True,
    )

    environment: Literal["local", "staging", "production"] = "local"
    debug: bool = False

    # --- Database -----------------------------------------------------
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://chatbot:chatbot@localhost:5432/chatbot",
    )
    db_pool_size: int = 10

    # --- Model AI (satu endpoint OpenAI-compatible) -------------------
    base_url: str | None = None
    """Kosong = endpoint resmi OpenAI. Isi untuk gateway atau penyedia lain."""
    api_key: SecretStr | None = None
    chat_model: str = "gpt-4o-mini"
    llm_stream_usage: bool = True
    """Minta jumlah token ikut dikirim saat menjawab secara streaming (FE-1).

    langchain-openai hanya menyalakan ini sendiri untuk endpoint resmi OpenAI;
    dengan `BASE_URL` terisi defaultnya mati, dan tanpa itu `usage_metadata`
    kosong sehingga seluruh jawaban streaming kehilangan estimasi biaya AD-5.
    Matikan hanya bila gateway menolak `stream_options` -- jawabannya tetap
    utuh, yang hilang hanya angka biayanya."""
    embed_model: str = "text-embedding-3-large"
    """Harus menghasilkan 1024 dimensi, sama dengan kolom `chunks.embedding`
    (`app.db.models.EMBEDDING_DIM`). Mengganti model = re-index seluruh dokumen."""

    # --- Retrieval (FR-2, FR-3) --------------------------------------
    retrieval_candidates: int = 20
    """Top-N per sumber sebelum fusi. PRD FR-2: 20 vector + 20 fulltext."""
    retrieval_top_n: int = 5
    """Jumlah chunk yang masuk konteks LLM setelah RRF."""
    rrf_k: int = 60
    rrf_weight_vector: float = 1.0
    rrf_weight_fulltext: float = 1.0
    vector_threshold: float = 0.35
    lexical_threshold: float = 0.05

    # --- Ingestion (FR-1) --------------------------------------------
    chunk_size: int = 900
    """Pagar atas untuk bagian yang kepanjangan. Batas chunk yang sebenarnya
    struktural: judul bagian di dalam dokumen (lihat `ingestion.chunker`)."""
    chunk_overlap: int = 105
    """~12% dari chunk_size. Hanya berlaku saat satu bagian harus dipecah."""
    storage_dir: str = "storage/documents"
    """Dipakai hanya bila `storage_backend` bernilai `local`."""

    # --- Penyimpanan objek -------------------------------------------
    storage_backend: Literal["local", "s3"] = "local"
    """`local` = disk server (default, tanpa kredensial). `s3` = S3 / R2 /
    MinIO / Backblaze B2 -- semuanya lewat protokol yang sama."""

    s3_bucket: str = ""
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None

    s3_endpoint_url: str | None = None
    """Kosongkan untuk AWS S3 asli. Untuk Cloudflare R2:
    `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`"""

    s3_region: str = "auto"
    """R2 tidak punya region tetapi SDK menuntut nilai; `auto` yang benar.
    Untuk AWS S3 isi region sungguhan, mis. `ap-southeast-1`."""

    s3_addressing_style: Literal["auto", "path", "virtual"] = "auto"
    """`auto` cocok untuk S3 dan R2. MinIO umumnya butuh `path`."""

    s3_checksum_compat: bool = True
    """Turunkan `request_checksum_calculation` ke `when_required`.

    Sejak botocore 1.36 defaultnya `when_supported`, yang menempelkan checksum
    CRC32 FULL_OBJECT pada setiap unggahan. R2 hanya mendukung CRC32 sebagai
    COMPOSITE, sehingga unggahan dapat ditolak. Biarkan True kecuali memakai
    AWS S3 asli dan memang menginginkan checksum penuh."""

    s3_public_base_url: str | None = None
    """Domain publik bucket, bila ada -- mis. `https://dokumen.instiki.ac.id`
    atau URL r2.dev. Diisi berarti PDF disajikan lewat URL publik yang stabil
    dan dapat di-cache. Dikosongkan berarti memakai presigned URL."""

    s3_presign_ttl_seconds: int = 900
    """Masa berlaku presigned URL. Cukup untuk membuka PDF, cukup pendek
    supaya tautan yang bocor tidak berlaku lama."""

    # --- Observability (FR-8) ----------------------------------------
    langsmith_tracing: bool = True
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "chatbot-administrasi"

    langsmith_endpoint: str | None = None
    """Alamat API LangSmith. Kosong berarti memakai bawaan SDK
    (`https://api.smith.langchain.com`).

    Wajib diisi untuk region Eropa (`https://eu.api.smith.langchain.com`) atau
    instans self-hosted. Field ini harus ada meski nilainya jarang diubah:
    `extra="ignore"` membuat LANGSMITH_ENDPOINT di .env dibuang diam-diam bila
    tidak dideklarasikan, dan `configure_tracing` tidak akan pernah
    meneruskannya ke SDK -- trace mendarat di region yang salah tanpa satu pun
    pesan galat."""

    # --- Keamanan (FR-9) ---------------------------------------------
    admin_jwt_secret: SecretStr = SecretStr(PLACEHOLDER_JWT_SECRET)
    admin_token_ttl_minutes: int = 480
    rate_limit_per_session: str = "20/minute"
    rate_limit_per_ip: str = "60/minute"
    kill_switch_enabled: bool = False
    """True = layanan chat dimatikan sejak proses mulai, endpoint mengembalikan 503.
    Jalur cadangan bila dashboard admin tidak bisa diakses; jalur utamanya
    `POST /api/admin/kill-switch`."""

    # --- Dashboard admin (AD-1..AD-6) --------------------------------
    cors_origins: str = ""
    """Asal peramban yang boleh memanggil API, dipisah koma, mis.
    `https://admin.instiki.ac.id,https://portal.instiki.ac.id`.

    Wajib diisi begitu dashboard/portal memakai domain yang berbeda dari API
    (mis. admin.* dan sads.* memanggil api.*): peramban menolak respons yang
    tidak menyebut asalnya, dan permintaan gagal sebelum mencapai handler.
    Kosong saat `ENVIRONMENT=local` berarti semua asal diizinkan; asal localhost
    tetap ditambahkan saat local meski daftar produksi sudah diisi, supaya
    `admin/` dan `client/` versi dev tidak ikut terkunci."""

    max_upload_mb: int = 50
    """Batas ukuran PDF yang diunggah admin (AD-3). Batas keras ukuran body
    tetap perlu dipasang di Caddy; nilai ini untuk pesan galat yang jelas."""

    timezone: str = "Asia/Jakarta"
    """Zona waktu IANA untuk pengelompokan harian statistik (AD-5)."""

    def cors_origin_list(self) -> list[str]:
        """Daftar asal untuk `CORSMiddleware`, tanpa duplikat.

        Daftar kosong di luar `local` berarti tidak ada pemanggil lintas-asal
        yang dilayani -- benar hanya bila front-end berbagi domain dengan API.
        """
        asal = [a.strip().rstrip("/") for a in self.cors_origins.split(",") if a.strip()]
        if self.environment != "local":
            return list(dict.fromkeys(asal))
        if not asal:
            return ["*"]
        return list(dict.fromkeys([*asal, *ORIGIN_LOKAL]))

    @model_validator(mode="after")
    def _model_ai_valid(self) -> Settings:
        """BASE_URL harus berskema; API_KEY wajib di produksi.

        Di lokal dan saat test aplikasi harus bisa start tanpa kunci;
        kekurangannya dilaporkan saat model dipanggil (`app/rag/providers.py`).
        """
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"BASE_URL harus diawali http:// atau https://, diberi {self.base_url!r}"
            )
        if self.environment == "production" and self.kunci_api() is None:
            raise ValueError("ENVIRONMENT=production tetapi API_KEY belum diisi")
        return self

    def kunci_api(self) -> SecretStr | None:
        """API_KEY, atau None bila kosong."""
        return _terisi(self.api_key)

    @model_validator(mode="after")
    def _penyimpanan_objek_lengkap(self) -> Settings:
        """Gagalkan konfigurasi S3 yang tidak lengkap saat startup.

        Tanpa ini, kekurangan kredensial baru ketahuan saat admin pertama kali
        mengunggah dokumen -- setelah PDF selesai diproses dan di-embed, yang
        berarti biaya API sudah terlanjur keluar.
        """
        if self.storage_backend != "s3":
            return self

        kurang = [
            nama
            for nama, nilai in (
                ("S3_BUCKET", self.s3_bucket),
                ("S3_ACCESS_KEY_ID", self.s3_access_key_id),
                ("S3_SECRET_ACCESS_KEY", self.s3_secret_access_key),
            )
            if not nilai
        ]
        if kurang:
            raise ValueError(
                f"STORAGE_BACKEND=s3 tetapi belum diisi: {', '.join(kurang)}"
            )

        if self.s3_endpoint_url and not self.s3_endpoint_url.startswith(("http://", "https://")):
            raise ValueError(
                f"S3_ENDPOINT_URL harus diawali http:// atau https://, diberi "
                f"{self.s3_endpoint_url!r}"
            )

        if self.s3_public_base_url and not self.s3_public_base_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("S3_PUBLIC_BASE_URL harus diawali http:// atau https://")

        # Batas keras SigV4; presigned URL berumur lebih panjang ditolak penyedia.
        if not 1 <= self.s3_presign_ttl_seconds <= 604_800:
            raise ValueError(
                f"S3_PRESIGN_TTL_SECONDS harus 1..604800 (7 hari), diberi "
                f"{self.s3_presign_ttl_seconds}"
            )

        if self.is_r2 and self.s3_region not in {"auto", "us-east-1", ""}:
            raise ValueError(
                f"R2 tidak punya region. S3_REGION harus 'auto' (atau 'us-east-1'/kosong "
                f"yang di-alias ke auto), diberi {self.s3_region!r}"
            )

        return self

    @property
    def is_r2(self) -> bool:
        """True bila endpoint menunjuk ke Cloudflare R2.

        Dipakai untuk memberi pesan galat yang tepat sasaran dan untuk
        peringatan di README; bukan untuk mengubah perilaku diam-diam.
        """
        if not self.s3_endpoint_url:
            return False
        return "r2.cloudflarestorage.com" in self.s3_endpoint_url

    @field_validator("database_url", mode="before")
    @classmethod
    def _paksa_driver_asyncpg(cls, v):
        """`postgresql://` dan `postgres://` diubah menjadi `postgresql+asyncpg://`.

        Aplikasi sepenuhnya async dan hanya memasang asyncpg. URL tanpa driver --
        bentuk yang diberikan kebanyakan penyedia Postgres -- membuat SQLAlchemy
        memilih psycopg2, yang tidak terpasang, sehingga aplikasi gagal start.
        Driver yang ditulis eksplisit (mis. `+psycopg`) dibiarkan apa adanya.
        """
        if isinstance(v, str):
            for awalan in ("postgresql://", "postgres://"):
                if v.startswith(awalan):
                    return "postgresql+asyncpg://" + v[len(awalan) :]
        return v

    @field_validator("s3_public_base_url")
    @classmethod
    def _rapikan_base_url(cls, v: str | None) -> str | None:
        return v.rstrip("/") if v else v

    @field_validator("admin_jwt_secret")
    @classmethod
    def _secret_cukup_kuat(cls, v: SecretStr, info) -> SecretStr:
        rahasia = v.get_secret_value()
        if len(rahasia.encode("utf-8")) < 32:
            raise ValueError(
                "admin_jwt_secret minimal 32 byte (RFC 7518 §3.2). Bangkitkan dengan: "
                'python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        if info.data.get("environment") == "production" and rahasia == PLACEHOLDER_JWT_SECRET:
            raise ValueError("admin_jwt_secret masih memakai nilai placeholder")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lebih_kecil_dari_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) harus lebih kecil dari chunk_size ({chunk_size})"
            )
        return v

    @field_validator("max_upload_mb")
    @classmethod
    def _batas_unggah_positif(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_upload_mb harus >= 1, diberi {v}")
        return v

    @field_validator("retrieval_top_n")
    @classmethod
    def _top_n_tidak_melebihi_kandidat(cls, v: int, info) -> int:
        candidates = info.data.get("retrieval_candidates")
        if candidates is not None and v > candidates:
            raise ValueError(
                f"retrieval_top_n ({v}) melebihi retrieval_candidates ({candidates})"
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """Instance tunggal; di-cache agar `.env` hanya dibaca sekali."""
    return Settings()
