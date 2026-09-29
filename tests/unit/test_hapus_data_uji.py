"""Skrip penghapus data uji (`scripts.hapus_data_uji`).

Database pengembangan dan produksi saat ini satu database, jadi yang dijaga di
sini adalah hal yang bisa menghapus data mahasiswa sungguhan: pola awalan yang
kebablasan, kombinasi argumen yang berbahaya, dan urutan penghapusan yang
meninggalkan entri AD-4 tanpa induk.
"""

from __future__ import annotations

import pytest

from scripts.hapus_data_uji import (
    KONFIRMASI_SEMUA,
    Target,
    hapus,
    periksa_argumen,
    pola_awalan,
)


class TestPolaAwalan:
    def test_awalan_biasa(self):
        assert pola_awalan("jev-uji-") == "jev-uji-%"

    @pytest.mark.parametrize(
        ("awalan", "pola"),
        [("uji_", "uji\\_%"), ("50%an", "50\\%an%"), ("a\\b", "a\\\\b%")],
    )
    def test_wildcard_diperlakukan_harfiah(self, awalan, pola):
        """Tanpa ini `uji_` juga mencocokkan "uji-", "ujiX", dan seterusnya."""
        assert pola_awalan(awalan) == pola


class TestArgumen:
    def test_daftar_tanpa_target_boleh(self):
        assert periksa_argumen(Target(), hapus=False, konfirmasi=None) is None

    def test_hapus_tanpa_target_ditolak(self):
        assert "butuh target" in periksa_argumen(Target(), hapus=True, konfirmasi=None)

    @pytest.mark.parametrize("awalan", ["", "a", "8", "abc"])
    def test_awalan_pendek_ditolak(self, awalan):
        """Awalan "8" bisa mencocokkan sesi mahasiswa sungguhan."""
        galat = periksa_argumen(Target(awalan=(awalan,)), hapus=False, konfirmasi=None)
        assert "terlalu pendek" in galat

    def test_semua_hapus_wajib_dikonfirmasi(self):
        target = Target(semua=True)
        assert "konfirmasi" in periksa_argumen(target, hapus=True, konfirmasi=None)
        assert "konfirmasi" in periksa_argumen(target, hapus=True, konfirmasi="ya")
        assert periksa_argumen(target, hapus=True, konfirmasi=KONFIRMASI_SEMUA) is None

    def test_semua_tidak_bisa_digabung(self):
        galat = periksa_argumen(
            Target(semua=True, awalan=("jev-uji-",)), hapus=False, konfirmasi=None
        )
        assert "tidak bisa digabung" in galat

    def test_target_sah(self):
        target = Target(sesi=("82360357-f1af-4c88-b49b-9f71395ad4c2",), awalan=("jev-uji-",))
        assert periksa_argumen(target, hapus=True, konfirmasi=None) is None


class TestKondisi:
    def test_sesi_dan_awalan_jadi_parameter_bukan_teks_sql(self):
        kondisi, params = Target(sesi=("s1",), awalan=("jev-uji-",)).kondisi()
        assert "s1" not in kondisi and "jev" not in kondisi
        assert params == {"sesi": ["s1"], "pola": ["jev-uji-%"]}

    def test_semua(self):
        assert Target(semua=True).kondisi() == ("TRUE", {})


class HasilPalsu:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class SesiPalsu:
    def __init__(self) -> None:
        self.sql: list[str] = []
        self.committed = False

    async def execute(self, sql, params=None):
        self.sql.append(str(sql))
        return HasilPalsu(3)

    async def commit(self):
        self.committed = True


class TestHapus:
    async def test_tak_terjawab_dihapus_sebelum_percakapan(self):
        """FK `unanswered_questions.message_id` adalah ON DELETE SET NULL:
        percakapan dulu berarti entri AD-4 tertinggal tanpa induk."""
        sesi = SesiPalsu()
        hasil = await hapus(sesi, Target(awalan=("jev-uji-",)), yatim=False)
        assert sesi.sql[0].startswith("DELETE FROM unanswered_questions")
        assert sesi.sql[1].startswith("DELETE FROM conversations")
        assert len(sesi.sql) == 2
        assert sesi.committed
        assert hasil == {"tak_terjawab": 3, "percakapan": 3, "yatim": 0}

    async def test_yatim_hanya_bila_diminta(self):
        sesi = SesiPalsu()
        hasil = await hapus(sesi, Target(awalan=("jev-uji-",)), yatim=True)
        assert "message_id IS NULL" in sesi.sql[2]
        assert hasil["yatim"] == 3
