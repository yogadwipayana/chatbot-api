"""Migrasi saat start (`app.db.migrasi`): database lebih baru tidak membuat restart-loop."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from app.db import migrasi
from app.db.migrasi import Rencana, rencana

API = Path(__file__).resolve().parents[2]


@pytest.fixture
def script() -> ScriptDirectory:
    return ScriptDirectory.from_config(Config(str(API / "alembic.ini")))


@pytest.fixture
def script_image_lama(tmp_path) -> ScriptDirectory:
    """Image yang dibangun sebelum revisi terakhir ada: salinan `alembic/` tanpa head."""
    sekarang = ScriptDirectory.from_config(Config(str(API / "alembic.ini")))
    head = sekarang.get_revision(sekarang.get_current_head())
    salinan = tmp_path / "alembic"
    shutil.copytree(API / "alembic", salinan, ignore=shutil.ignore_patterns("__pycache__"))
    (salinan / "versions" / Path(head.path).name).unlink()
    config = Config()
    config.set_main_option("script_location", str(salinan))
    return ScriptDirectory.from_config(config)


class TestRencana:
    def test_database_baru_dinaikkan(self, script):
        assert rencana([], script) is Rencana.NAIKKAN

    def test_revisi_lama_dinaikkan(self, script):
        assert rencana(["0015"], script) is Rencana.NAIKKAN

    def test_sudah_di_head(self, script):
        assert rencana([script.get_current_head()], script) is Rencana.TERBARU

    def test_revisi_tak_dikenal_berarti_database_lebih_baru(self, script):
        assert rencana(["9999"], script) is Rencana.DB_LEBIH_BARU

    def test_kejadian_nyata_database_di_head_image_belum(self, script, script_image_lama):
        """2026-10-08 (0015 vs 0014) dan 2026-10-09 (0016): migrasi dari lokal
        mendahului image server."""
        head = script.get_current_head()
        assert rencana([head], script_image_lama) is Rencana.DB_LEBIH_BARU


class TestMain:
    def jalankan(self, revisi_db: list[str]):
        dipanggil = []

        async def baca():
            return revisi_db

        kode = migrasi.main(baca=baca, naikkan=lambda: dipanggil.append("upgrade"))
        return kode, dipanggil

    def test_database_lebih_baru_dilewati_tanpa_gagal(self, caplog):
        kode, dipanggil = self.jalankan(["9999"])
        assert kode == 0
        assert dipanggil == []
        assert "DILEWATI" in caplog.text
        assert "9999" in caplog.text

    def test_revisi_lama_menjalankan_upgrade(self):
        kode, dipanggil = self.jalankan(["0015"])
        assert kode == 0
        assert dipanggil == ["upgrade"]

    def test_database_kosong_menjalankan_upgrade(self):
        assert self.jalankan([])[1] == ["upgrade"]

    def test_sudah_di_head_tidak_menjalankan_apa_pun(self, script):
        assert self.jalankan([script.get_current_head()])[1] == []
