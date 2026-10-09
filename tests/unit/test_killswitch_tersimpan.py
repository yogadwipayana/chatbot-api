"""Kill switch yang tersimpan di database dan dipulihkan saat proses mulai."""

from __future__ import annotations

from app.config import Settings
from app.main import apply_initial_kill_switch, pulihkan_kill_switch
from app.security.killswitch import KillSwitch, pulihkan, simpan_status
from tests.fixtures.fakes import FakeKillSwitchStore


def tersimpan(alasan="kunci API bocor", oleh="super@instiki.ac.id") -> KillSwitch:
    switch = KillSwitch()
    switch.engage(alasan, by=oleh)
    return switch


class TestPulihkan:
    async def test_menyalakan_dengan_alasan_pelaku_dan_waktu_awal(self):
        lama = tersimpan()
        baru = KillSwitch()
        assert await pulihkan(baru, FakeKillSwitchStore(lama))
        assert baru.engaged
        assert baru.reason == "kunci API bocor"
        assert baru.engaged_by == "super@instiki.ac.id"
        # Durasi insiden di dashboard tidak mulai dari nol setiap restart.
        assert baru.engaged_at == lama.engaged_at

    async def test_tabel_kosong_tidak_mengubah_apa_pun(self):
        switch = KillSwitch()
        assert not await pulihkan(switch, FakeKillSwitchStore())
        assert not switch.engaged

    async def test_simpan_lalu_restart_tetap_menyala(self):
        store = FakeKillSwitchStore()
        sebelum = tersimpan()
        await simpan_status(sebelum, store)
        sesudah_restart = KillSwitch()
        await pulihkan(sesudah_restart, store)
        assert sesudah_restart.engaged
        assert sesudah_restart.reason == sebelum.reason

    async def test_melepas_mengosongkan_tabel(self):
        store = FakeKillSwitchStore(tersimpan())
        await simpan_status(KillSwitch(), store)
        assert store.tersimpan is None


class TestSimpanStatus:
    async def test_galat_database_dicatat_bukan_dilempar(self, caplog):
        await simpan_status(tersimpan(), FakeKillSwitchStore(gagal=True))
        assert "tidak akan bertahan setelah restart" in caplog.text


class TestSaatStart:
    async def test_galat_membaca_tidak_menghentikan_start(self, caplog):
        """DB mati atau tabel belum dimigrasi: layanan mulai hidup, seperti dulu."""
        switch = KillSwitch()
        await pulihkan_kill_switch(switch, lambda: FakeKillSwitchStore(gagal=True))
        assert not switch.engaged
        assert "tidak dapat dibaca" in caplog.text

    async def test_status_tersimpan_menang_atas_kill_switch_enabled(self):
        """Urutan di lifespan: pulihkan dulu, baru `.env`. Alasan insiden yang
        tersimpan tidak tertimpa alasan generik dari konfigurasi server."""
        switch = KillSwitch()
        await pulihkan_kill_switch(switch, lambda: FakeKillSwitchStore(tersimpan()))
        apply_initial_kill_switch(Settings(_env_file=None, kill_switch_enabled=True), switch)
        assert switch.reason == "kunci API bocor"

    async def test_kill_switch_enabled_tetap_berlaku_tanpa_status_tersimpan(self):
        switch = KillSwitch()
        await pulihkan_kill_switch(switch, lambda: FakeKillSwitchStore())
        apply_initial_kill_switch(Settings(_env_file=None, kill_switch_enabled=True), switch)
        assert switch.engaged
