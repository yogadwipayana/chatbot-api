"""Konfigurasi aplikasi -- validasi nilai yang mudah salah diisi."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def settings(**overrides) -> Settings:
    """Bangun Settings tanpa membaca .env, agar test tidak tergantung mesin."""
    return Settings(_env_file=None, **overrides)


def settings_produksi(**overrides) -> Settings:
    """Settings produksi lengkap dengan kredensial yang diwajibkan validator."""
    return settings(
        environment="production", admin_jwt_secret="x" * 48, api_key="sk-uji", **overrides
    )


class TestNilaiDefault:
    def test_default_valid(self):
        assert settings().environment == "local"

    def test_default_sesuai_PRD(self):
        s = settings()
        assert s.retrieval_candidates == 20  # FR-2: 20 per sumber
        assert s.retrieval_top_n == 5  # FR-2: top 5 masuk konteks
        # FR-1 menyebut 500-800 *token*; `chunk_size` dihitung dalam KARAKTER,
        # jadi angkanya tidak pernah sebanding langsung (900 karakter kira-kira
        # 250 token untuk teks Indonesia). Yang dijaga di sini hanya bahwa
        # nilainya tetap pada ordo yang benar: cukup besar untuk memuat satu
        # prosedur bernomor utuh, cukup kecil agar satu chunk tidak memborong
        # seluruh halaman.
        assert 500 <= s.chunk_size <= 1000

    def test_overlap_sekitar_15_persen(self):
        """FR-1 menyebut overlap ~15%."""
        s = settings()
        assert 0.10 <= s.chunk_overlap / s.chunk_size <= 0.20


class TestValidasiSilang:
    def test_overlap_lebih_besar_dari_chunk_ditolak(self):
        """Kombinasi ini membuat splitter berputar tanpa henti."""
        with pytest.raises(ValidationError, match="chunk_overlap"):
            settings(chunk_size=500, chunk_overlap=500)

    def test_overlap_sama_dengan_chunk_ditolak(self):
        with pytest.raises(ValidationError):
            settings(chunk_size=700, chunk_overlap=700)

    def test_top_n_melebihi_kandidat_ditolak(self):
        """Meminta 30 chunk dari 20 kandidat diam-diam hanya menghasilkan 20."""
        with pytest.raises(ValidationError, match="retrieval_top_n"):
            settings(retrieval_candidates=20, retrieval_top_n=30)

    def test_top_n_sama_dengan_kandidat_diterima(self):
        assert settings(retrieval_candidates=20, retrieval_top_n=20).retrieval_top_n == 20


class TestRahasia:
    def test_kunci_api_tidak_muncul_saat_dicetak(self):
        """Settings sering ikut ter-log saat startup; kunci tidak boleh bocor."""
        s = settings(api_key="sk-rahasia-sekali")
        assert "rahasia-sekali" not in repr(s)
        assert "rahasia-sekali" not in str(s)

    def test_nilai_asli_tetap_dapat_diambil(self):
        s = settings(api_key="sk-rahasia")
        assert s.api_key.get_secret_value() == "sk-rahasia"

    def test_kunci_api_default_kosong(self):
        """Tidak ada kunci yang boleh punya default; harus dari environment."""
        s = settings()
        assert s.api_key is None


class TestSecretJWT:
    def test_secret_pendek_ditolak(self):
        with pytest.raises(ValidationError, match="32 byte"):
            settings(admin_jwt_secret="pendek")

    def test_placeholder_boleh_dipakai_saat_lokal(self):
        """Pengembangan lokal tidak boleh terhalang; produksi yang dijaga."""
        assert settings(environment="local").admin_jwt_secret is not None

    def test_placeholder_ditolak_di_produksi(self):
        from app.config import PLACEHOLDER_JWT_SECRET

        with pytest.raises(ValidationError, match="placeholder"):
            settings(environment="production", admin_jwt_secret=PLACEHOLDER_JWT_SECRET)

    def test_secret_asli_diterima_di_produksi(self):
        # Produksi juga mewajibkan kunci model AI; lihat test_providers.py.
        s = settings(
            environment="production",
            admin_jwt_secret="x" * 48,
            api_key="sk-uji",
        )
        assert s.environment == "production"


class TestDatabaseUrl:
    @pytest.mark.parametrize("url", ["postgresql://u:p@h:5432/db", "postgres://u:p@h:5432/db"])
    def test_url_tanpa_driver_dipaksa_asyncpg(self, url):
        """Bentuk URL dari kebanyakan penyedia Postgres. Tanpa driver, SQLAlchemy
        memilih psycopg2 yang tidak terpasang dan aplikasi gagal start."""
        assert settings(database_url=url).database_url.scheme == "postgresql+asyncpg"

    def test_driver_asyncpg_tidak_diubah(self):
        url = "postgresql+asyncpg://u:p@h:5432/db"
        assert settings(database_url=url).database_url.scheme == "postgresql+asyncpg"

    def test_driver_eksplisit_lain_dihormati(self):
        s = settings(database_url="postgresql+psycopg://u:p@h:5432/db")
        assert s.database_url.scheme == "postgresql+psycopg"

    def test_kredensial_dan_nama_database_tetap_utuh(self):
        s = settings(database_url="postgresql://app:rahasia@db.contoh:6543/appdb")
        host = s.database_url.hosts()[0]
        assert (host["username"], host["password"], host["host"], host["port"]) == (
            "app",
            "rahasia",
            "db.contoh",
            6543,
        )
        assert s.database_url.path == "/appdb"


class TestPenyimpananObjek:
    KREDENSIAL = {
        "storage_backend": "s3",
        "s3_bucket": "dokumen-kampus",
        "s3_access_key_id": "akses",
        "s3_secret_access_key": "rahasia",
    }
    R2_ENDPOINT = "https://akun123.r2.cloudflarestorage.com"

    def test_default_lokal_tanpa_kredensial(self):
        """Pengembangan lokal harus jalan tanpa akun cloud apa pun."""
        s = settings()
        assert s.storage_backend == "local"
        assert s.s3_access_key_id is None

    def test_s3_tanpa_kredensial_ditolak(self):
        """Kekurangan kredensial harus ketahuan saat startup, bukan saat admin
        mengunggah dokumen -- pada saat itu biaya embedding sudah keluar."""
        with pytest.raises(ValidationError, match="S3_BUCKET"):
            settings(storage_backend="s3")

    @pytest.mark.parametrize(
        "hilang,pesan",
        [
            ("s3_bucket", "S3_BUCKET"),
            ("s3_access_key_id", "S3_ACCESS_KEY_ID"),
            ("s3_secret_access_key", "S3_SECRET_ACCESS_KEY"),
        ],
    )
    def test_setiap_kredensial_wajib_disebut_saat_kurang(self, hilang, pesan):
        kw = dict(self.KREDENSIAL)
        kw[hilang] = "" if hilang == "s3_bucket" else None
        with pytest.raises(ValidationError, match=pesan):
            settings(**kw)

    def test_konfigurasi_r2_lengkap_diterima(self):
        s = settings(**self.KREDENSIAL, s3_endpoint_url=self.R2_ENDPOINT)
        assert s.is_r2
        assert s.s3_region == "auto"

    def test_aws_s3_asli_bukan_r2(self):
        s = settings(**self.KREDENSIAL, s3_region="ap-southeast-1")
        assert not s.is_r2

    def test_region_salah_untuk_r2_ditolak(self):
        """R2 tidak punya region. Nilai selain auto akan gagal saat menandatangani."""
        with pytest.raises(ValidationError, match="R2 tidak punya region"):
            settings(
                **self.KREDENSIAL,
                s3_endpoint_url=self.R2_ENDPOINT,
                s3_region="ap-southeast-1",
            )

    @pytest.mark.parametrize("region", ["auto", "us-east-1", ""])
    def test_region_yang_di_alias_r2_diterima(self, region):
        """R2 meng-alias nilai kosong dan us-east-1 ke auto."""
        assert settings(**self.KREDENSIAL, s3_endpoint_url=self.R2_ENDPOINT, s3_region=region)

    def test_endpoint_tanpa_skema_ditolak(self):
        """Tanpa http:// boto3 gagal dengan pesan yang membingungkan."""
        with pytest.raises(ValidationError, match="http://"):
            settings(**self.KREDENSIAL, s3_endpoint_url="akun123.r2.cloudflarestorage.com")

    def test_base_url_publik_tanpa_skema_ditolak(self):
        with pytest.raises(ValidationError, match="http://"):
            settings(**self.KREDENSIAL, s3_public_base_url="dokumen.instiki.ac.id")

    def test_garis_miring_akhir_base_url_dirapikan(self):
        s = settings(**self.KREDENSIAL, s3_public_base_url="https://dokumen.instiki.ac.id/")
        assert s.s3_public_base_url == "https://dokumen.instiki.ac.id"

    def test_presign_melebihi_batas_sigv4_ditolak(self):
        """SigV4 membatasi presigned URL maksimal 7 hari."""
        with pytest.raises(ValidationError, match="604800"):
            settings(**self.KREDENSIAL, s3_presign_ttl_seconds=604_801)

    def test_presign_tepat_di_batas_diterima(self):
        assert settings(**self.KREDENSIAL, s3_presign_ttl_seconds=604_800)

    def test_presign_nol_ditolak(self):
        with pytest.raises(ValidationError):
            settings(**self.KREDENSIAL, s3_presign_ttl_seconds=0)

    def test_checksum_compat_menyala_secara_default(self):
        """Default aman untuk R2, MinIO, dan B2 sekaligus; AWS S3 tidak dirugikan."""
        assert settings().s3_checksum_compat is True

    def test_validasi_s3_dilewati_saat_backend_lokal(self):
        """Konfigurasi S3 setengah jadi tidak boleh menghalangi mode lokal."""
        assert settings(storage_backend="local", s3_bucket="").storage_backend == "local"

    def test_factory_membangun_local_storage(self):
        from app.storage import LocalStorage, build_storage

        assert isinstance(build_storage(settings()), LocalStorage)

    def test_factory_membangun_s3_storage(self):
        from app.storage import build_storage
        from app.storage.s3 import S3Storage

        dibangun = build_storage(settings(**self.KREDENSIAL, s3_endpoint_url=self.R2_ENDPOINT))
        assert isinstance(dibangun, S3Storage)
        assert dibangun.bucket == "dokumen-kampus"

    def test_factory_meneruskan_pengaturan_r2(self):
        from app.storage import build_storage

        dibangun = build_storage(
            settings(
                **self.KREDENSIAL,
                s3_endpoint_url=self.R2_ENDPOINT,
                s3_public_base_url="https://dokumen.instiki.ac.id",
                s3_presign_ttl_seconds=300,
            )
        )
        assert dibangun.public_base_url == "https://dokumen.instiki.ac.id"
        assert dibangun.presign_ttl == 300
        assert dibangun.client.meta.config.request_checksum_calculation == "when_required"


class TestKillSwitch:
    def test_default_mati(self):
        assert settings().kill_switch_enabled is False

    def test_dapat_dinyalakan_lewat_konfigurasi(self):
        assert settings(kill_switch_enabled=True).kill_switch_enabled is True


class TestCorsOrigin:
    """Salah isi di sini tidak menggagalkan startup -- peramban yang menolak,
    jauh kemudian, dengan 'No Access-Control-Allow-Origin header'."""

    def test_local_tanpa_daftar_mengizinkan_semua(self):
        assert settings(environment="local", cors_origins="").cors_origin_list() == ["*"]

    def test_produksi_tanpa_daftar_tidak_mengizinkan_apa_pun(self):
        """Tanpa daftar, front-end di domain lain diblokir -- benar hanya bila
        API dan front-end berbagi domain."""
        assert settings_produksi(cors_origins="").cors_origin_list() == []

    def test_beberapa_asal_dipisah_koma(self):
        s = settings_produksi(
            cors_origins="https://admin.dwipa.my.id, https://sads.dwipa.my.id"
        )
        assert s.cors_origin_list() == [
            "https://admin.dwipa.my.id",
            "https://sads.dwipa.my.id",
        ]

    def test_garis_miring_akhir_dibuang(self):
        """Header `Origin` tidak pernah berakhiran '/'; tanpa ini nilai .env yang
        disalin dari address bar tidak akan pernah cocok."""
        s = settings_produksi(cors_origins="https://sads.dwipa.my.id/")
        assert s.cors_origin_list() == ["https://sads.dwipa.my.id"]

    def test_asal_ganda_tidak_diulang(self):
        s = settings_produksi(
            cors_origins="https://sads.dwipa.my.id,https://sads.dwipa.my.id/"
        )
        assert s.cors_origin_list() == ["https://sads.dwipa.my.id"]

    def test_local_tetap_mengizinkan_dev_meski_daftar_produksi_diisi(self):
        """.env pengembang berisi domain produksi; `npm run dev` tetap harus jalan."""
        s = settings(environment="local", cors_origins="https://admin.dwipa.my.id")
        asal = s.cors_origin_list()
        assert "https://admin.dwipa.my.id" in asal
        assert "http://localhost:3000" in asal
        assert "http://localhost:3001" in asal

    def test_produksi_tidak_kebocoran_localhost(self):
        s = settings_produksi(cors_origins="https://admin.dwipa.my.id")
        assert all("localhost" not in a for a in s.cors_origin_list())
