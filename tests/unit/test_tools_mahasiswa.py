"""Tool SADS data pribadi (`sads_mahasiswa`) dan jalur NIM dari request
(docs/tool-call.md §7, §11). Tanpa jaringan."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.prodi import ProfilMahasiswa, cari_prodi
from app.rag.chain import OutcomeKind, run_pipeline
from app.rag.prompts import TOOL_SYSTEM_PROMPT
from app.rag.tools import sads_mahasiswa as mhs
from app.rag.tools.base import (
    ARGUMEN_TERLARANG,
    Penanya,
    ToolResult,
    ToolSpec,
    hasil_tool_ke_dokumen,
)
from app.rag.tools.loop import run_tool_loop
from app.rag.tools.registry import ToolRegistry, build_registry
from tests.fixtures.fakes import FakeRetriever
from tests.unit.test_tools import FakeChat, ToolLLM, _answer_turn, _tool_turn

NIM = "2401010101"
NIM_LAIN = "2401010999"
PROFIL_IF = ProfilMahasiswa(prodi=cari_prodi("1010"), angkatan=2024)  # type: ignore[arg-type]
LABEL = "Data akademik SADS"
HARI = date(2026, 8, 10)

MAHASISWA = {
    "nim": NIM,
    "nama": "Nama Rahasia",
    "status": "aktif",
    "angkatan": "2024/2025",
    "prodi_kode": "IF",
    "prodi_nama": "IF-Manajemen Data & Informasi",
    "jenis_kelas": "Reguler",
}


@pytest.fixture(autouse=True)
def _hari_tetap(monkeypatch):
    monkeypatch.setattr(mhs, "hari_ini", lambda: HARI)


class FakeSads:
    def __init__(self, data: Any = None, status: int = 200) -> None:
        self.data = data
        self.status = status
        self.paths: list[str] = []

    async def get_json(self, path: str, params: dict | None = None) -> Any:
        self.paths.append(path)
        if self.status != 200:
            url = f"https://sads.palsu{path}"
            raise httpx.HTTPStatusError(
                f"Server error '{self.status}' for url '{url}'",
                request=httpx.Request("GET", url),
                response=httpx.Response(self.status),
            )
        return self.data


def _spec(nama: str, sads: FakeSads) -> ToolSpec:
    [spec] = [s for s in mhs.tool_specs(sads) if s.name == nama]  # type: ignore[arg-type]
    return spec


async def _panggil(nama: str, data: Any, penanya: Penanya | None = None, status: int = 200):
    sads = FakeSads(data, status)
    hasil = await _spec(nama, sads).handler(
        penanya=penanya or Penanya(nim=NIM, profile=PROFIL_IF)
    )
    return hasil, sads


# --- Identitas dari request, bukan dari model ----------------------------------


class TestNimDariRequest:
    def test_tidak_ada_skema_dengan_argumen_nim(self):
        reg = build_registry(
            Settings(
                _env_file=None,
                tools_enabled=True,
                sads_base_url="https://x",
                sads_api_secret="r",
                api_key="k",
            )
        )
        for spec in reg.specs:
            assert not ARGUMEN_TERLARANG & set(spec.parameters.get("properties", {})), (
                spec.name
            )
        pribadi = [s for s in reg.specs if s.name.endswith("_saya")]
        assert len(pribadi) == 7 and all(s.pakai_penanya for s in pribadi)

    def test_toolspec_menolak_argumen_nim(self):
        with pytest.raises(ValueError, match="dilarang"):
            ToolSpec(
                name="x",
                description="",
                parameters={"type": "object", "properties": {"nim": {"type": "string"}}},
                handler=lambda **_: None,  # type: ignore[arg-type, return-value]
            )

    async def test_nim_dari_penanya_masuk_path(self):
        _, sads = await _panggil("get_biaya_krs_saya", {"mahasiswa": MAHASISWA})
        assert sads.paths == [f"/service/tp/chatbot/biaya-krs/{NIM}"]

    async def test_nim_dari_model_diabaikan_loop(self):
        """Model yang mengisi `nim` tetap mendapat data penanya sendiri:
        `validasi_argumen` membuang argumen tak dikenal, dan loop menambahkan
        `penanya` dari request."""
        sads = FakeSads({"mahasiswa": MAHASISWA, "periode": "2026/1", "tagihan_krs": None})
        llm = FakeChat(
            [
                _tool_turn("get_biaya_krs_saya", f'{{"nim":"{NIM_LAIN}","penanya":"x"}}'),
                _answer_turn(f"Belum ada tagihan [{LABEL}]."),
            ]
        )
        hasil = await run_tool_loop(
            llm_tools=llm,
            llm_plain=llm,
            wrapped_question="q",
            documents=[],
            specs=mhs.tool_specs(sads),  # type: ignore[arg-type]
            penanya=Penanya(nim=NIM),
        )
        assert sads.paths == [f"/service/tp/chatbot/biaya-krs/{NIM}"]
        assert hasil.tool_calls[0]["ok"] is True

    async def test_tanpa_nim_tidak_memanggil_sads(self):
        """Uji coba admin tidak membawa NIM."""
        hasil, sads = await _panggil("get_nilai_saya", {}, Penanya())
        assert not hasil.ok and sads.paths == []
        assert hasil.pesan_untuk_model() == f"DATA_TIDAK_TERSEDIA: {mhs.GALAT_TANPA_NIM}"

    async def test_nim_tak_terdaftar_404(self):
        hasil, _ = await _panggil("get_nilai_saya", None, status=404)
        assert not hasil.ok
        assert mhs.GALAT_NIM_TIDAK_TERDAFTAR in hasil.pesan_untuk_model()

    async def test_galat_lain_diteruskan_ke_loop_dan_log_tanpa_nim(self, caplog):
        """SADS membalas 500 untuk jadwal-krs (uji 2026-11-07). Pesan httpx memuat
        URL, dan URL memuat NIM: log hanya boleh mencatat status."""
        sads = FakeSads(status=500)
        llm = FakeChat([_tool_turn("get_jadwal_krs_saya", "{}"), _answer_turn("x")])
        with caplog.at_level(logging.WARNING, logger="app.rag.tools.loop"):
            hasil = await run_tool_loop(
                llm_tools=llm,
                llm_plain=llm,
                wrapped_question="q",
                documents=[],
                specs=mhs.tool_specs(sads),  # type: ignore[arg-type]
                penanya=Penanya(nim=NIM),
            )
        assert hasil.tool_calls[0]["ok"] is False
        assert "HTTP 500" in caplog.text
        assert NIM not in caplog.text

    async def test_teks_untuk_model_tanpa_nim_dan_nama(self):
        hasil, _ = await _panggil(
            "get_nilai_saya",
            {"mahasiswa": MAHASISWA, "nilai": [_nilai("BIFW-001", "Algoritma", "A", 4)]},
        )
        teks = hasil.pesan_untuk_model()
        assert NIM not in teks and "Nama Rahasia" not in teks
        assert (
            "status aktif, IF-Manajemen Data & Informasi, angkatan 2024/2025, kelas Reguler"
            in teks
        )
        assert mhs.KEPALA_PRIBADI in teks

    def test_aturan_t8_ada_di_prompt_tool(self):
        assert "T8." in TOOL_SYSTEM_PROMPT and "_saya" in TOOL_SYSTEM_PROMPT


class TestPipelineMembawaNim:
    async def test_nim_dan_profil_sampai_ke_run_tools(self):
        docs = hasil_tool_ke_dokumen([ToolResult("a", LABEL, "Lunas", True)])
        llm = ToolLLM(f"Tagihan Anda lunas [{LABEL}].", docs)
        reg = ToolRegistry([_spec("get_biaya_krs_saya", FakeSads())])
        outcome = await run_pipeline(
            "berapa tagihan saya?",
            retriever=FakeRetriever([]),
            llm_call=llm,
            tool_registry=reg,
            profile=PROFIL_IF,
            nim=NIM,
        )
        assert outcome.kind is OutcomeKind.ANSWER
        assert llm.last_penanya == Penanya(nim=NIM, profile=PROFIL_IF)
        # NIM tidak ikut ke pesan untuk LLM.
        assert NIM not in (llm.last_wrapped or "")

    async def test_tanpa_nim_penanya_kosong(self):
        llm = ToolLLM(f"x [{LABEL}].", [])
        reg = ToolRegistry([_spec("get_biaya_krs_saya", FakeSads())])
        await run_pipeline(
            "berapa tagihan saya?",
            retriever=FakeRetriever([]),
            llm_call=llm,
            tool_registry=reg,
        )
        assert llm.last_penanya == Penanya()


# --- Nilai ---------------------------------------------------------------------


def _nilai(
    kdmk: str, matkul: str, grade: str, na: int | None, sks: int = 3, **lain: Any
) -> dict[str, Any]:
    return {
        "kdmk": kdmk,
        "matkul": matkul,
        "jnsmk": 1,
        "jenis_matkul": "Wajib",
        "teori": sks,
        "prak": 0,
        "sks": sks,
        "ntot": 81.6,
        "na": na,
        "grade": grade,
        "sumber": "",
        "hide": 0,
        **lain,
    }


REFERENSI = {
    "sks_wajib_lulus": 84,
    "sks_wajib_minimal": 135,
    "sks_pilihan_lulus": 9,
    "sks_pilihan_minimal": 9,
    "total_sks_lulus": 93,
}


class TestNilai:
    async def test_ipk_sebaran_dan_sisa_sks_dihitung_handler(self):
        data = {
            "mahasiswa": MAHASISWA,
            "nilai": [
                _nilai("BIFW-001", "Algoritma", "A", 4, sks=3),
                _nilai("BIFW-002", "Analisa Proses Bisnis", "B", 3, sks=2),
                _nilai("BIFW-003", "Basis Data", "C", 2, sks=3),
            ],
            "referensi_sks": REFERENSI,
        }
        hasil, _ = await _panggil("get_nilai_saya", data)
        # (4*3 + 3*2 + 2*3) / 8 = 3
        assert "dibagi jumlah SKS): 3,00." in hasil.text
        assert "Jumlah mata kuliah bernilai: 3 (8 SKS)." in hasil.text
        assert "Sebaran nilai huruf: A 1, B 1, C 1." in hasil.text
        assert "wajib 84 dari minimal 135 (kurang 51)" in hasil.text
        assert "pilihan 9 dari minimal 9 (terpenuhi)" in hasil.text
        assert "SKS yang masih kurang untuk kelulusan (wajib + pilihan): 51." in hasil.text
        assert "- BIFW-001 Algoritma (Wajib, 3 SKS): A, bobot 4, nilai 81,6" in hasil.text
        [lampiran] = hasil.attachments
        assert lampiran.title == "Daftar nilai Anda" and len(lampiran.items) == 3
        assert lampiran.items[0] == "Algoritma (BIFW-001) · 3 SKS · A"

    async def test_ipk_dua_desimal(self):
        data = {
            "nilai": [_nilai("A-001", "a", "A", 4, sks=3), _nilai("A-002", "b", "B", 3, sks=1)]
        }
        hasil, _ = await _panggil("get_nilai_saya", data)
        assert "jumlah SKS): 3,75." in hasil.text

    async def test_hide_dibuang_dan_ulang_dihitung_terbaik(self):
        data = {
            "nilai": [
                _nilai("BIFW-001", "Algoritma", "E", 0),
                _nilai("BIFW-001", "Algoritma", "B", 3),
                _nilai("BIFW-009", "Tersembunyi", "A", 4, hide=1),
            ]
        }
        hasil, _ = await _panggil("get_nilai_saya", data)
        assert "Jumlah mata kuliah bernilai: 1 (3 SKS)." in hasil.text
        assert "Tersembunyi" not in hasil.text
        assert ": B, bobot 3" in hasil.text

    def test_nilai_e_mengalahkan_belum_ada_nilai(self):
        """Bobot 0 (E) bukan "belum ada nilai" saat memilih percobaan terbaik."""
        baris = mhs._baris_nilai(
            {"nilai": [_nilai("X-001", "x", "", None), _nilai("X-001", "x", "E", 0)]}
        )
        assert [n.grade for n in baris] == ["E"]
        assert mhs.ipk(baris) == 0

    async def test_belum_ada_nilai(self):
        hasil, _ = await _panggil("get_nilai_saya", {"mahasiswa": MAHASISWA, "nilai": []})
        assert hasil.ok and "0 mata kuliah" in hasil.text and hasil.attachments == ()


# --- Jadwal -----------------------------------------------------------------------

JADWAL = {
    "mahasiswa": MAHASISWA,
    "periode": "2026/1",
    "tanggal_buka": "2026-08-01",
    "tanggal_tutup": "2026-08-15",
    "status": "Dibuka",
    "masih_bisa_mengisi": True,
    "kode_status_krs": 1,
}


class TestJadwal:
    async def test_jadwal_krs_tanggal_dan_selisih_dihitung(self):
        hasil, _ = await _panggil("get_jadwal_krs_saya", JADWAL)
        assert "dibuka 1 Agustus 2026 (9 hari yang lalu)" in hasil.text
        assert "ditutup 15 Agustus 2026 (5 hari lagi)" in hasil.text
        assert "Status menurut SADS: Dibuka; masih bisa mengisi: ya." in hasil.text
        assert "Tanggal hari ini: 10 Agustus 2026." in hasil.text
        assert "kode_status_krs" not in hasil.text

    async def test_revisi_membawa_keterangan(self):
        data = {
            **JADWAL,
            "jenis_jadwal": "Jendela pengisian dan revisi KRS",
            "keterangan": "Sama dengan KRS.",
        }
        hasil, sads = await _panggil("get_revisi_krs_saya", data)
        assert "Jadwal revisi KRS penanya (Jendela pengisian dan revisi KRS)." in hasil.text
        assert "Keterangan SADS: Sama dengan KRS." in hasil.text
        assert sads.paths == [f"/service/tp/chatbot/revisi-krs/{NIM}"]

    async def test_semester_aktif(self):
        data = {
            "mahasiswa": MAHASISWA,
            "periode": "2026/1",
            "tahun_akademik": "2026/2027",
            "semester": "Ganjil",
            "semester_ke_mahasiswa": 5,
            "tanggal_mulai_semester": "2026-09-01",
            "tanggal_selesai_semester": "2027-01-31",
            "jadwal_krs": {
                **JADWAL,
                "mahasiswa": {},
                "status": "Ditutup",
                "masih_bisa_mengisi": False,
            },
        }
        hasil, _ = await _panggil("get_semester_aktif_saya", data)
        assert "tahun akademik 2026/2027, semester Ganjil." in hasil.text
        assert "Bagi penanya ini semester ke-5." in hasil.text
        assert "1 September 2026 (22 hari lagi) sampai 31 Januari 2027" in hasil.text
        assert "Status menurut SADS: Ditutup; masih bisa mengisi: tidak." in hasil.text

    async def test_semester_aktif_bentuk_baru_dengan_jadwal_kuliah(self):
        # Respons SADS sejak 2026-11: tanpa tanggal semester dan jadwal_krs,
        # dengan jadwal kuliah penanya. HARI (10 Agustus 2026) = Senin.
        kosong = {"hari2": "", "jam2": " - ", "ruang2": "", "hari3": "", "jam3": " - "}
        data = {
            "mahasiswa": MAHASISWA,
            "periode": "2026/1",
            "tahun_akademik": "2026/2027",
            "semester": "Ganjil",
            "semester_ke_mahasiswa": 5,
            "jadwal_matakuliah": [
                {
                    **kosong,
                    "kdmk": "BIFW-056",
                    "matkul": "Metodologi Penelitian",
                    "nama_kelas": "AD",
                    "sks": 3,
                    "is_mbkm": 0,
                    "nmdosen": "Dosen Satu, M.Kom.",
                    "hari": "Jum'at",
                    "jam": "19:50 - 22:20",
                    "ruang": "ONLINE",
                },
                {
                    **kosong,
                    "kdmk": "BIFW-030",
                    "matkul": "Data Mining",
                    "nama_kelas": "AB",
                    "sks": 3,
                    "is_mbkm": 0,
                    "nmdosen": "",
                    "hari": "Senin",
                    "jam": "08:00 - 10:30",
                    "ruang": "L2.3",
                    "hari2": "Rabu",
                    "jam2": "08:00 - 09:40",
                    "ruang2": "Lab 1",
                },
                {
                    **kosong,
                    "kdmk": "BIFW-070",
                    "matkul": "Magang Industri",
                    "nama_kelas": "MB",
                    "sks": 20,
                    "is_mbkm": 1,
                    "nmdosen": "Dosen Dua",
                    "hari": "",
                    "jam": " - ",
                    "ruang": "",
                },
            ],
        }
        hasil, _ = await _panggil("get_semester_aktif_saya", data)
        teks = hasil.text
        assert "Bagi penanya ini semester ke-5." in teks
        assert "Tanggal mulai dan selesai perkuliahan semester ini tidak tercantum" in teks
        assert "- sampai -" not in teks and "Jadwal KRS semester ini" not in teks
        assert "Jadwal kuliah penanya semester ini: 3 kelas, 26 SKS." in teks
        assert (
            "- Data Mining (BIFW-030, kelas AB, 3 SKS): Senin 08:00 - 10:30 ruang L2.3; "
            "Rabu 08:00 - 09:40 ruang Lab 1"
        ) in teks
        assert (
            "- Magang Industri (BIFW-070, kelas MB, 20 SKS, MBKM): jadwal belum tercantum"
            in teks
        )
        # Urut per hari: Senin dulu, Jumat, lalu yang tanpa hari.
        assert teks.index("Data Mining") < teks.index("Metodologi") < teks.index("Magang")
        assert "Hari ini hari Senin; kuliah hari ini: Data Mining." in teks
        assert "Besok hari Selasa; kuliah besok: tidak ada." in teks
        [lampiran] = hasil.attachments
        assert lampiran.title == "Jadwal kuliah Anda"
        assert lampiran.items[1] == (
            "Metodologi Penelitian (BIFW-056) · Jum'at 19:50 - 22:20 ruang ONLINE · "
            "Dosen Satu, M.Kom."
        )

    async def test_semester_aktif_tanpa_kuliah_hari_ini(self, monkeypatch):
        monkeypatch.setattr(mhs, "hari_ini", lambda: date(2026, 8, 16))  # Minggu
        data = {
            "periode": "2026/1",
            "jadwal_matakuliah": [
                {"kdmk": "BIFW-056", "matkul": "Metodologi", "sks": 3, "hari": "Senin"}
            ],
        }
        hasil, _ = await _panggil("get_semester_aktif_saya", data)
        assert "Hari ini hari Minggu; kuliah hari ini: tidak ada." in hasil.text
        assert "Besok hari Senin; kuliah besok: Metodologi." in hasil.text

    @pytest.mark.parametrize(
        "pertanyaan",
        ["hari ini saya ada kuliah tidak?", "besok kuliah apa?", "jadwal kuliah saya"],
    )
    def test_pemicu_jadwal_kuliah(self, pertanyaan):
        assert _spec("get_semester_aktif_saya", FakeSads()).cocok(pertanyaan)


# --- Skripsi, biaya, pembayaran ----------------------------------------------


class TestSkripsiBiayaPembayaran:
    async def test_syarat_skripsi(self):
        data = {
            "mahasiswa": MAHASISWA,
            "syarat": {
                "minimal_sks_lulus": 100,
                "mata_kuliah_metodologi_penelitian": [
                    {"kdmk": "BIFW-056", "matkul": "Metodologi Penelitian"},
                    {"kdmk": "BIFW-056", "matkul": "Metodologi Penelitian   "},
                    {"kdmk": "CBDW-238", "matkul": "Research methods"},
                ],
            },
            "status_mahasiswa": {
                "sks_lulus": 84,
                "memenuhi_minimal_sks": False,
                "lulus_metodologi_penelitian": False,
                "layak_mengajukan_seminar": False,
                "keterangan": "Maaf, anda belum memenuhi syarat",
            },
        }
        hasil, _ = await _panggil("get_syarat_skripsi_saya", data)
        assert "- SKS lulus: 84 dari minimal 100 (kurang 16)" in hasil.text
        assert "Metodologi Penelitian (BIFW-056); Research methods (CBDW-238)" in hasil.text
        assert "- layak mengajukan seminar proposal: tidak" in hasil.text
        assert "keterangan SADS: Maaf, anda belum memenuhi syarat" in hasil.text

    async def test_biaya_rupiah(self):
        data = {
            "mahasiswa": MAHASISWA,
            "periode": "2026/1",
            "tarif": {"registrasi": 350000, "sks_teori": 145000, "admin_bank": 3500},
            "tagihan_krs": {
                "nokrs": "KR-1",
                "total_biaya": 5185000,
                "terbayar": 3000000,
                "sisa_tagihan": 2185000,
                "status_pembayaran": "Belum Lunas",
            },
        }
        hasil, _ = await _panggil("get_biaya_krs_saya", data)
        assert (
            "registrasi Rp350.000; per SKS teori Rp145.000; admin bank Rp3.500." in hasil.text
        )
        assert (
            "total Rp5.185.000; terbayar Rp3.000.000; sisa Rp2.185.000; status Belum Lunas"
            in hasil.text
        )
        assert "KR-1" not in hasil.text

    async def test_biaya_tanpa_krs(self):
        hasil, _ = await _panggil(
            "get_biaya_krs_saya", {"mahasiswa": MAHASISWA, "tagihan_krs": None}
        )
        assert "belum memiliki KRS pada semester aktif" in hasil.text

    async def test_pembayaran_tunggakan_dijumlahkan(self):
        tunggak = {
            "thnaka": "2025/2026",
            "semester": "Genap",
            "tipe": "N",
            "total_biaya": 5000000,
            "terbayar": 3000000,
        }
        data = {
            "mahasiswa": MAHASISWA,
            "periode": "2026/1",
            "pembayaran_dibuka": True,
            "batas_pembayaran": "2026-08-20 23:59:59",
            "tagihan_aktif": None,
            "memiliki_tunggakan": True,
            "tunggakan": [
                {**tunggak, "sisa_tagihan": 2000000},
                {**tunggak, "semester": "Ganjil", "sisa_tagihan": 500000},
            ],
            "boleh_mengisi_krs": False,
        }
        hasil, _ = await _panggil("get_jadwal_pembayaran_saya", data)
        assert "batas pembayaran 20 Agustus 2026 pukul 23.59 (10 hari lagi)" in hasil.text
        assert "Tunggakan: 2 tagihan, total sisa Rp2.500.000:" in hasil.text
        assert (
            "- 2025/2026 Genap: total Rp5.000.000; terbayar Rp3.000.000; sisa Rp2.000.000"
            in hasil.text
        )
        assert "Tagihan aktif: tidak ada." in hasil.text
        assert "Boleh mengisi KRS menurut SADS: tidak." in hasil.text

    async def test_pembayaran_tanpa_tunggakan(self):
        data = {"mahasiswa": MAHASISWA, "memiliki_tunggakan": False, "tunggakan": []}
        hasil, _ = await _panggil("get_jadwal_pembayaran_saya", data)
        assert "Tunggakan: tidak ada." in hasil.text


class TestPemicu:
    @pytest.mark.parametrize(
        "pertanyaan",
        [
            "berapa IPK saya?",
            "nilai basis data saya berapa?",
            "kapan jadwal KRS dibuka?",
            "sampai kapan revisi KRS?",
            "apakah saya sudah bisa daftar sempro?",
            "berapa tagihan saya semester ini?",
            "saya punya tunggakan?",
            "sekarang semester berapa?",
            "apa prasyarat Basis Data Lanjut?",
            "mata kuliah semester 5 apa saja?",
            "what is my GPA?",
        ],
    )
    def test_pertanyaan_data_pribadi_dan_kurikulum_eligible(self, pertanyaan: str):
        reg = build_registry(
            Settings(
                _env_file=None,
                tools_enabled=True,
                sads_base_url="https://x",
                sads_api_secret="r",
                api_key="k",
            )
        )
        assert reg.eligible(pertanyaan), pertanyaan


def test_format_angka_dan_tanggal():
    assert mhs._rupiah(1234567) == "Rp1.234.567"
    assert mhs._rupiah(None) == "-"
    assert mhs._angka(81.6) == "81,6" and mhs._angka(87.0) == "87"
    assert mhs._tanggal("2026-09-12 21:00:00") == "12 September 2026 pukul 21.00"
    assert mhs._tanggal("2026-09-12") == "12 September 2026"
    assert mhs._tanggal("bukan tanggal") == "bukan tanggal"
