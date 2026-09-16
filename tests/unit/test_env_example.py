"""Menjaga `.env.example` tetap lengkap dan benar.

`.env.example` adalah dokumentasi yang pasti dibaca orang saat memasang
sistem. Ia membusuk dengan cara yang sama-sama diam:

- pengaturan baru di `app/config.py` lupa dicantumkan, sehingga tidak ada yang
  tahu pengaturan itu ada;
- nama variabel salah ketik (`S3_ACESS_KEY_ID`), sehingga nilainya diabaikan
  pydantic-settings tanpa pesan apa pun dan default yang dipakai;
- kunci sungguhan ikut tertulis lalu ter-commit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.config import PLACEHOLDER_JWT_SECRET, Settings

AKAR = Path(__file__).resolve().parents[2]
CONTOH = AKAR / ".env.example"
POLA_VARIABEL = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=(.*)$")
POLA_RAHASIA = re.compile(r"(_KEY|_SECRET|_KEY_ID)$")
NAMA_PENGATURAN = {nama.upper() for nama in Settings.model_fields}


def variabel(*, aktif_saja: bool = False) -> dict[str, list[str]]:
    """Nama variabel -> semua nilai yang muncul (aktif maupun dikomentari)."""
    hasil: dict[str, list[str]] = {}
    for baris in CONTOH.read_text(encoding="utf-8").splitlines():
        baris = baris.strip()
        if aktif_saja and baris.startswith("#"):
            continue
        cocok = POLA_VARIABEL.match(baris)
        if cocok:
            hasil.setdefault(cocok.group(1), []).append(cocok.group(2).strip())
    return hasil


@pytest.fixture
def lingkungan_bersih(monkeypatch):
    for nama in NAMA_PENGATURAN:
        monkeypatch.delenv(nama, raising=False)


class TestKelengkapan:
    def test_berkas_ada(self):
        assert CONTOH.is_file()

    def test_setiap_pengaturan_tercantum(self):
        """Pengaturan baru di Settings wajib didokumentasikan di sini."""
        hilang = NAMA_PENGATURAN - set(variabel())
        assert not hilang, f"belum tercantum di .env.example: {sorted(hilang)}"

    def test_setiap_pengaturan_punya_baris_aktif(self):
        """Contoh yang dikomentari saja tidak cukup; orang menyalin lalu mengisi."""
        hilang = NAMA_PENGATURAN - set(variabel(aktif_saja=True))
        assert not hilang, f"hanya ada sebagai komentar: {sorted(hilang)}"

    def test_tidak_ada_nama_variabel_salah_ketik(self):
        """Variabel yang tidak dikenal Settings diabaikan tanpa peringatan."""
        asing = set(variabel()) - NAMA_PENGATURAN
        assert not asing, f"tidak dikenal app/config.py: {sorted(asing)}"

    def test_baris_aktif_tidak_ganda(self):
        """Nama ganda membuat baris kedua diam-diam menimpa yang pertama."""
        ganda = {n: v for n, v in variabel(aktif_saja=True).items() if len(v) > 1}
        assert not ganda, f"didefinisikan lebih dari sekali: {sorted(ganda)}"


class TestKeamanan:
    def test_tidak_ada_rahasia_terisi(self):
        """Semua kunci harus kosong atau berupa placeholder <...>."""
        for nama, nilai_nilai in variabel().items():
            if not POLA_RAHASIA.search(nama):
                continue
            for nilai in nilai_nilai:
                assert re.fullmatch(r"(<[^>]*>)?", nilai), (
                    f"{nama} berisi nilai yang tampak seperti rahasia sungguhan"
                )

    def test_env_diabaikan_git(self):
        baris = (AKAR / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert ".env" in [b.strip() for b in baris]


class TestDapatDipakai:
    def test_valid_setelah_kredensial_wajib_diisi(self, lingkungan_bersih, monkeypatch):
        """Salinan .env.example cukup diisi kredensialnya -- tanpa mengubah nilai
        lain -- lalu langsung valid. Menangkap nilai contoh yang gagal di-parse."""
        monkeypatch.setenv("S3_BUCKET", "dokumen-kampus")
        monkeypatch.setenv("S3_ACCESS_KEY_ID", "akses")
        monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "rahasia")
        monkeypatch.setenv("S3_ENDPOINT_URL", "https://akun123.r2.cloudflarestorage.com")
        s = Settings(_env_file=CONTOH)
        assert s.environment == "local"

    def test_nilai_kosong_jatuh_ke_default(self, lingkungan_bersih, tmp_path):
        """Baris `KUNCI=` yang dibiarkan kosong berarti 'tidak diisi', bukan
        string kosong -- kalau tidak, ADMIN_JWT_SECRET= menggagalkan start dan
        VECTOR_THRESHOLD= gagal di-parse."""
        env = tmp_path / ".env"
        env.write_text("ADMIN_JWT_SECRET=\nVECTOR_THRESHOLD=\nS3_REGION=\n", encoding="utf-8")
        s = Settings(_env_file=env)
        assert s.admin_jwt_secret.get_secret_value() == PLACEHOLDER_JWT_SECRET
        assert s.vector_threshold == 0.35
        assert s.s3_region == "auto"

    def test_kunci_kosong_dari_berkas_dianggap_tidak_diisi(self, lingkungan_bersih, tmp_path):
        env = tmp_path / ".env"
        env.write_text("API_KEY=\n", encoding="utf-8")
        assert Settings(_env_file=env).kunci_api() is None
