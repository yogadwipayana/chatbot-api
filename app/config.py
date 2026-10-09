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

from app.security.ratelimit import BatasLaju, parse_batas

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
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    """Batas tunggu satu permintaan LLM, termasuk jeda antar potongan saat
    streaming. Tanpa nilai ini klien OpenAI menunggu sampai 600 detik per
    percobaan: gateway yang macet membuat widget mahasiswa tertahan di
    "Menyusun jawaban..." selama itu, dan pertanyaan berikutnya tidak dapat
    dikirim karena giliran sebelumnya belum selesai."""
    llm_max_retries: int = Field(default=1, ge=0)
    """Percobaan ulang setelah galat atau timeout. Bawaan klien OpenAI 2;
    setiap percobaan menambah waktu tunggu mahasiswa sebesar timeout di atas."""
    embed_model: str = "text-embedding-3-large"
    """Harus menghasilkan 1024 dimensi, sama dengan kolom `chunks.embedding`
    (`app.db.models.EMBEDDING_DIM`). Mengganti model = re-index seluruh dokumen
    (`python -m scripts.reindex_embeddings`)."""

    embed_provider: Literal["api", "local"] = "api"
    """`api` = EMBED_MODEL lewat BASE_URL. `local` = sentence-transformers di
    server ini, mis. `EMBED_MODEL=intfloat/multilingual-e5-small`; butuh
    `uv sync --extra local`. Awalan `query: `/`passage: ` model e5 dipasang
    otomatis, dan vektor yang lebih pendek dari 1024 diisi nol di ekornya --
    cosine similarity tidak berubah, jadi skema database tidak perlu diubah."""

    # --- Tool-calling (data layanan akademik; docs/tool-call.md) ------
    tools_enabled: bool = False
    """Sakelar utama tool-calling, seperti JEV_ENABLED/RERANK_ENABLED. Mati =
    LLM tidak pernah diberi tool dan jalur chat persis seperti sebelumnya."""
    tools_max_rounds: int = Field(default=2, ge=1, le=5)
    """Batas putaran loop agentik: berapa kali LLM boleh memanggil tool sebelum
    dipaksa menjawab tanpa tool (docs/tool-call.md §5)."""
    sads_base_url: str | None = None
    """Host layanan akademik SADS, mis. `https://sads.instiki.ac.id`."""
    sads_api_secret: SecretStr | None = None
    """Nilai header `secret` SADS. Hanya di env; tidak pernah dikirim ke LLM."""
    sads_timeout_seconds: float = Field(default=10.0, gt=0)

    # --- Reranker (setelah RRF, sebelum threshold) -------------------
    rerank_enabled: bool = False
    """Sakelar reranker, seperti JEV_ENABLED. Mati = urutan RRF langsung dipakai
    dan RERANK_* lainnya diabaikan. Mengganti model cukup dengan mengganti
    RERANK_BASE_URL, RERANK_API_KEY, dan RERANK_MODEL."""
    rerank_provider: Literal["tei", "api", "local"] = "tei"
    """Bentuk endpoint, bukan modelnya. `tei` = Text Embeddings Inference
    (`/rerank` dengan `texts`), `api` = gaya Cohere/Jina (`documents` ->
    `results`), `local` = cross-encoder sentence-transformers di proses API.
    Diganti hanya bila jenis servernya berganti."""
    rerank_model: str = ""
    """Mis. `Alibaba-NLP/gte-multilingual-reranker-base`. TEI mengabaikannya
    (satu server satu model), jadi di sana nama ini hanya untuk log; `api` dan
    `local` memakainya untuk memilih model."""
    rerank_base_url: str | None = None
    """Alamat server reranker, mis. `http://localhost:8081`; `/rerank`
    ditambahkan di belakangnya. Wajib kecuali `local`. Tidak jatuh ke BASE_URL:
    gateway tidak menyediakan `/rerank`."""
    rerank_api_key: SecretStr | None = None
    """Kosong = tanpa header Authorization. Tidak jatuh ke API_KEY, supaya kunci
    gateway tidak terkirim ke server lain."""
    rerank_candidates: int = 20
    """Jumlah hasil RRF yang dinilai ulang; yang lolos tetap RETRIEVAL_TOP_N."""
    rerank_threshold: float | None = None
    """Skor reranker minimum (0..1). Terisi = FR-3 memakai skor reranker, yang
    bersifat absolut, alih-alih skor mentah vector/fulltext. Kosong = ambang lama.
    Kalibrasi dulu; skor tiap model reranker tersebar berbeda."""
    rerank_timeout_seconds: float = 10.0

    # --- Gerbang JEV (Decisions API lewat gateway) ------------------
    jev_enabled: bool = False
    """Klasifikasi pesan sebelum retrieval: academic / smalltalk / out_of_scope /
    nonsense / malicious. Gagal atau lewat batas waktu = pesan diteruskan.
    Mati pun aman: saringan aturan (`app.rag.rule_gate`) dan penanda
    `[DI_LUAR_TOPIK]` LLM penjawab menggantikannya tanpa biaya per pesan."""
    jev_url: str | None = None
    """Kosong = `<BASE_URL>/systemone`, endpoint gateway yang meneruskan JEV.
    Diisi hanya bila JEV dilayani alamat lain dengan badan permintaan yang sama."""
    jev_api_key: SecretStr | None = None
    """Kosong = API_KEY, kunci yang sama dengan gateway."""
    jev_model: str = "openrouter/typesafe/jev-1.13"
    """Nama model JEV menurut gateway."""
    jev_timeout_seconds: float = Field(default=10.0, gt=0)
    """Batas keras satu panggilan JEV. Biasanya bukan ini yang memutusnya:
    JEV berjalan paralel dengan pencarian dan diputus `jev_grace_seconds`
    setelah pencarian selesai (`app.rag.graph.jev_gate`). Batas ini hanya
    berlaku bila pencarian sendiri juga lambat."""
    jev_grace_seconds: float = Field(default=1.5, ge=0)
    """Waktu tambahan bagi JEV setelah pencarian paralel selesai -- satu-
    satunya waktu tunggu yang dapat ditambahkan JEV ke giliran mahasiswa.
    Latensi JEV berayun dari ~1 dtk sampai ~29 dtk (2026-09-28); batas tetap
    3 dtk dulu memutusnya walau pencarian masih berjalan 5-10 dtk."""
    jev_block_threshold: float = 0.7
    """Keyakinan minimum untuk menghentikan pesan nonsense/malicious/smalltalk.
    Kalibrasi 2026-09-29: pertanyaan akademik tidak pernah di atas 0,03 untuk
    ketiga label ini (lihat `app.rag.gate.GatePolicy`)."""
    jev_out_of_scope_threshold: float = 0.9
    """Lebih ketat: pertanyaan di luar topik yang lolos gerbang masih ditandai
    LLM penjawab (`[DI_LUAR_TOPIK]`), sedangkan salah blokir menelan pertanyaan
    akademik tanpa jejak. Akademik paling tinggi 0,57, di luar topik sungguhan
    paling rendah 0,94 (2026-09-29)."""
    jev_nonsense_threshold: float | None = None
    """Ambang khusus label nonsense; kosong = `jev_block_threshold` (JEV tidak
    berubah). Untuk Laya hasil latih (laya.md): pertanyaan akademik yang sangat
    pendek ("ukm", "toeic brp") mendapat nonsense sampai 0,81, sedangkan pesan
    acak sungguhan sudah dihentikan `app.rag.rule_gate` lebih dulu."""

    # --- Retrieval (FR-2, FR-3) --------------------------------------
    retrieval_candidates: int = 20
    """Top-N per sumber sebelum fusi. PRD FR-2: 20 vector + 20 fulltext."""
    retrieval_top_n: int = 5
    """Jumlah chunk yang masuk konteks LLM setelah RRF."""
    retrieval_max_per_document: int = Field(default=4, ge=0)
    """Paling banyak berapa chunk satu dokumen di antara `retrieval_top_n` (T59).
    0 = tanpa batas. Kursi yang tak terisi dokumen lain tetap diisi dokumen yang
    sama, jadi unit berdokumen tunggal tidak kehilangan konteks.

    4 = satu kursi top 5 selalu untuk dokumen lain bila ada. "sertifikasi dasar
    DKV apa dan berapa?" mengisi kelima kursi dengan TRANSKRIP UPS, dan harganya
    (HARGA SERTIFIKASI, peringkat 6) tidak pernah sampai ke LLM. Simulasi
    2026-10-09 atas 27 pertanyaan: 4 mengubah top 5 di 6 pertanyaan, 2 dan 3
    memasukkan banyak potongan yang tidak relevan."""
    retrieval_neighbors: int = Field(default=5, ge=0)
    """Berapa chunk teratas yang diberi potongan sesudahnya dari dokumen yang
    sama. Konteks LLM paling banyak `retrieval_top_n + retrieval_neighbors`
    chunk. 0 = mati. Lihat `app.rag.retriever.NEIGHBOR_SQL`.

    5 = setiap chunk konteks. Dengan 2, daftar larangan Pasal 10 Kode Etik
    (peringkat 3) terpotong di butir 6; dengan 5 lengkap 10 butir, dengan
    tambahan sekitar 12% token input (uji 2026-09-29)."""
    retrieval_outline: int = Field(default=2, ge=0)
    """Paling banyak berapa dokumen yang daftar babnya ikut ke konteks LLM (T9).
    0 = mati. Hanya dokumen yang potongan teratasnya menyentuh dua bab atau
    lebih, jadi dengan `retrieval_top_n` 5 paling banyak dua. Lihat
    `app.rag.outline`.

    Tanpa daftar bab, "jenis beasiswa apa saja?" dijawab empat dari enam jenis
    (10/10 uji 2026-10-09), walau bab Berprestasi dan Talenta ikut terambil."""
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

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    """Level minimum logger `app.*`, untuk konsol maupun SQLite."""
    log_db_path: str = "log/app.db"
    """Berkas SQLite log aplikasi (`logs.md`), relatif terhadap direktori kerja.
    Di Docker, direktorinya harus di-mount sebagai volume: filesystem container
    hilang setiap redeploy."""
    log_retention_days: int = Field(default=7, ge=1)
    """Baris log yang lebih tua dari ini dihapus saat start dan setiap jam."""
    log_node_io: bool = True
    """Simpan juga teks pertanyaan, jawaban, NIM, dan input/output setiap node
    serta panggilan LLM/tool di dalamnya (tab Graf halaman Log, `logs.md` tahap
    4). Pertanyaan sensitif (FR-7) tetap disamarkan. False: hanya metrik dan log,
    seperti sebelum tahap 4."""

    # --- Keamanan (FR-9) ---------------------------------------------
    admin_jwt_secret: SecretStr = SecretStr(PLACEHOLDER_JWT_SECRET)
    admin_token_ttl_minutes: int = 480
    rate_limit_per_session: str = "20/minute"
    """Per `X-Session-Id` (satu tab). Bentuk `N/second|minute|hour|day`; `0` = mati."""
    rate_limit_per_ip: str = "60/minute"
    """Per IP -- yang menahan skrip dan Postman. IP kampus dipakai bersama
    ratusan mahasiswa: naikkan bila mahasiswa di jaringan kampus kena 429."""
    rate_limit_per_embed_site: str = "60/minute"
    """Per kunci sematan: seluruh pengunjung satu situs penyemat bersama-sama."""
    client_ip_header: str = "X-Forwarded-For"
    """Dari mana IP pengunjung dibaca untuk batas per IP (`ratelimit.client_ip`).

    `X-Forwarded-For` (entri terakhir) untuk API di balik satu reverse proxy
    seperti Caddy; `CF-Connecting-IP` di balik Cloudflare; kosong bila API
    langsung menghadap internet. Salah pilih berakibat salah satu dari dua:
    semua pengunjung tampak ber-IP sama dan berbagi satu jatah, atau IP dapat
    dipalsukan dan batasnya tidak berguna. Header hanya dapat dipercaya bila
    port API tidak terjangkau tanpa melewati proxy."""
    chat_daily_limit: int = Field(default=3000, ge=0)
    """Pertanyaan per hari (zona `TIMEZONE`) dari semua sumber. Terlampaui =
    kill switch menyala otomatis sampai superadmin menyalakan layanan lagi.
    `0` = tanpa batas. Bawaan 2x beban puncak musim KRS di PRD §11 (500/hari,
    3x lipat). Dapat diubah dari halaman Konfigurasi."""
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

    portal_url: str = ""
    """Alamat portal mahasiswa (`client/`) sebagaimana dibuka peramban, mis.
    `https://sads.instiki.ac.id`. Dipakai menyusun kode sematan yang ditampilkan
    di halaman Sematan dashboard. Kosong saat `ENVIRONMENT=local` berarti
    `http://localhost:3001`; kosong di lingkungan lain berarti dashboard tidak
    menampilkan kode sematan siap tempel."""

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

    def url_portal(self) -> str | None:
        """PORTAL_URL tanpa garis miring akhir, atau None bila belum diketahui."""
        if self.portal_url.strip():
            return self.portal_url.strip().rstrip("/")
        return "http://localhost:3001" if self.environment == "local" else None

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

    def kunci_sads(self) -> SecretStr | None:
        """SADS_API_SECRET, atau None bila kosong -- sengaja tidak jatuh ke API_KEY."""
        return _terisi(self.sads_api_secret)

    def kunci_rerank(self) -> SecretStr | None:
        """RERANK_API_KEY, atau None bila kosong -- sengaja tidak jatuh ke API_KEY."""
        return _terisi(self.rerank_api_key)

    def kunci_jev(self) -> SecretStr | None:
        """JEV_API_KEY, atau API_KEY bila kosong."""
        return _terisi(self.jev_api_key) or self.kunci_api()

    def url_jev(self) -> str | None:
        """JEV_URL, atau `<BASE_URL>/systemone` bila kosong."""
        if self.jev_url:
            return self.jev_url
        return f"{self.base_url.rstrip('/')}/systemone" if self.base_url else None

    @model_validator(mode="after")
    def _reranker_dan_jev_valid(self) -> Settings:
        """Setelan yang setengah terisi gagal saat start, bukan saat mahasiswa bertanya."""
        if self.rerank_enabled:
            if not self.rerank_model.strip():
                raise ValueError("RERANK_ENABLED=true tetapi RERANK_MODEL kosong")
            if self.rerank_provider != "local":
                if not self.rerank_base_url:
                    raise ValueError(
                        f"RERANK_ENABLED=true dengan RERANK_PROVIDER={self.rerank_provider} "
                        "butuh RERANK_BASE_URL"
                    )
                if not self.rerank_base_url.startswith(("http://", "https://")):
                    raise ValueError(
                        "RERANK_BASE_URL harus diawali http:// atau https://, "
                        f"diberi {self.rerank_base_url!r}"
                    )
        # rerank_candidates < retrieval_top_n tidak ditolak di sini: RETRIEVAL_TOP_N
        # dapat dinaikkan dari dashboard, dan retriever memakai yang lebih besar.
        if self.rerank_candidates < 1:
            raise ValueError(f"rerank_candidates harus >= 1, diberi {self.rerank_candidates}")
        if self.rerank_threshold is not None and not 0.0 <= self.rerank_threshold <= 1.0:
            raise ValueError(f"rerank_threshold di luar 0..1: {self.rerank_threshold}")

        if self.tools_enabled:
            if not self.sads_base_url:
                raise ValueError("TOOLS_ENABLED=true butuh SADS_BASE_URL")
            if not self.sads_base_url.startswith(("http://", "https://")):
                raise ValueError(
                    f"SADS_BASE_URL harus diawali http:// atau https://, "
                    f"diberi {self.sads_base_url!r}"
                )
            if self.kunci_sads() is None:
                raise ValueError("TOOLS_ENABLED=true butuh SADS_API_SECRET")

        if self.jev_enabled:
            if self.kunci_jev() is None:
                raise ValueError(
                    "JEV_ENABLED=true tetapi JEV_API_KEY maupun API_KEY belum diisi"
                )
            if not self.url_jev():
                raise ValueError("JEV_ENABLED=true butuh BASE_URL (atau JEV_URL)")
            # Tanpa skema, httpx menolak setiap panggilan dan gerbang diam-diam
            # fail-open: semua pesan lolos tanpa diperiksa (2026-10-02, laya.md).
            if self.jev_url and not self.jev_url.startswith(("http://", "https://")):
                raise ValueError(
                    f"JEV_URL harus diawali http:// atau https://, diberi {self.jev_url!r}"
                )
        for nama in (
            "jev_block_threshold",
            "jev_out_of_scope_threshold",
            "jev_nonsense_threshold",
        ):
            nilai = getattr(self, nama)
            if nilai is not None and not 0.0 < nilai <= 1.0:
                raise ValueError(f"{nama} harus di (0, 1], diberi {nilai}")
        return self

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
            raise ValueError(f"STORAGE_BACKEND=s3 tetapi belum diisi: {', '.join(kurang)}")

        if self.s3_endpoint_url and not self.s3_endpoint_url.startswith(
            ("http://", "https://")
        ):
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

    @field_validator("rerank_provider", mode="before")
    @classmethod
    def _provider_none_sudah_diganti(cls, v):
        """`.env` lama berisi RERANK_PROVIDER=none; arahkan ke sakelar yang baru."""
        if isinstance(v, str) and v.strip().lower() == "none":
            raise ValueError(
                "RERANK_PROVIDER=none tidak dipakai lagi: matikan reranker dengan "
                "RERANK_ENABLED=false, dan isi RERANK_PROVIDER dengan tei, api, atau local"
            )
        return v

    @field_validator(
        "rate_limit_per_session", "rate_limit_per_ip", "rate_limit_per_embed_site"
    )
    @classmethod
    def _batas_laju_sah(cls, v: str) -> str:
        """Salah ketik ditolak saat start, bukan diam-diam menjadi tanpa batas."""
        parse_batas(v)
        return v

    def batas_laju(self) -> dict[str, BatasLaju | None]:
        """Batas per sesi, IP, dan kunci sematan; None = tidak dibatasi."""
        return {
            "sesi": parse_batas(self.rate_limit_per_session),
            "ip": parse_batas(self.rate_limit_per_ip),
            "kunci": parse_batas(self.rate_limit_per_embed_site),
        }

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

    @field_validator("rrf_weight_vector", "rrf_weight_fulltext")
    @classmethod
    def _bobot_tidak_negatif(cls, v: float, info) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} tidak boleh negatif, diberi {v}")
        return v

    @field_validator("rrf_weight_fulltext")
    @classmethod
    def _minimal_satu_sumber_aktif(cls, v: float, info) -> float:
        """Bobot 0 mematikan sumbernya (`app.rag.fusion`). Keduanya 0 berarti
        tidak ada pencarian sama sekali, dan setiap pertanyaan ditolak."""
        if v == 0 and info.data.get("rrf_weight_vector") == 0:
            raise ValueError(
                "Bobot pencarian makna dan bobot pencarian kata tidak boleh keduanya 0: "
                "minimal satu sumber pencarian harus aktif"
            )
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
