"""Tool SADS kurikulum: `get_prasyarat_matkul` dan `get_matkul_kurikulum`
(docs/tool-call.md §7). Tanpa jaringan: SADS dipalsukan per path."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.prodi import ProfilMahasiswa, cari_prodi
from app.rag.tools import sads_kurikulum as kur
from app.rag.tools.base import Penanya

NIM = "2401010101"
PROFIL_IF = ProfilMahasiswa(prodi=cari_prodi("1010"), angkatan=2024)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _cache_kosong():
    kur.kosongkan_cache()
    yield
    kur.kosongkan_cache()


def _mk(
    kdmk: str,
    matkul: str,
    *,
    smt: int = 1,
    kurikulum_id: int = 1,
    kurikulum: str = "OBE 2025",
    prodi: str = "IF",
    prodi_nama: str = "Informatika",
    sks: int = 3,
    status: str = "Wajib",
) -> dict[str, Any]:
    return {
        "kurikulum_id": kurikulum_id,
        "kurikulum": kurikulum,
        "prodi_id": 4,
        "prodi_kode": prodi,
        "prodi_nama": prodi_nama,
        "kdmk": kdmk,
        "matkul": matkul,
        "smtrke": smt,
        "teori": sks,
        "prak": 0,
        "sks": sks,
        "jnsmk": 1,
        "status_matkul": status,
    }


class FakeSads:
    """SADS palsu: kurikulum dari daftar baris, prasyarat per kode, nilai per NIM."""

    base_url = "https://sads.palsu"

    def __init__(
        self,
        baris: list[dict[str, Any]],
        prasyarat: dict[str, list[dict[str, Any]]] | None = None,
        nilai: list[str] | None = None,
    ) -> None:
        self.baris = baris
        self.prasyarat = prasyarat or {}
        self.nilai = nilai
        self.calls: list[tuple[str, dict | None]] = []

    async def get_json(self, path: str, params: dict | None = None) -> Any:
        self.calls.append((path, params))
        if path.startswith(kur.PATH_KURIKULUM):
            smt = int(path.rsplit("/", 1)[1])
            kid = (params or {}).get("kurikulum_id")
            # Tanpa kurikulum_id = kurikulum aktif; id 2 tidak aktif (seperti SADS).
            return {
                "data": [
                    b
                    for b in self.baris
                    if b["smtrke"] == smt
                    and (
                        b["kurikulum_id"] == kid if kid is not None else b["kurikulum_id"] != 2
                    )
                ]
            }
        if path.startswith(kur.PATH_PRASYARAT):
            return self.prasyarat.get(path.rsplit("/", 1)[1], [])
        if "/krs-mahasiswa/" in path:
            if self.nilai is None:
                raise httpx.HTTPStatusError(
                    "404",
                    request=httpx.Request("GET", "https://x"),
                    response=httpx.Response(404),
                )
            return {
                "mahasiswa": {},
                "nilai": [{"kdmk": k, "matkul": k, "na": 4, "sks": 3} for k in self.nilai],
            }
        raise AssertionError(f"path tak terduga: {path}")

    def jalur(self, awalan: str) -> list[tuple[str, dict | None]]:
        return [c for c in self.calls if c[0].startswith(awalan)]


def _syarat(kdjur: str, kdmk: str, nama: str, kelompok: int = 1) -> dict[str, Any]:
    return {
        "kdjur": kdjur,
        "type": 1,
        "nilai_syarat": kdmk,
        "kelompok": kelompok,
        "kdmk_syarat": kdmk,
        "matkul_syarat": nama + "      ",
        "jenis": "mata_kuliah",
    }


def _sks_lulus(kdjur: str, n: int) -> dict[str, Any]:
    return {
        "kdjur": kdjur,
        "type": 2,
        "nilai_syarat": str(n),
        "kelompok": 1,
        "kdmk_syarat": None,
        "matkul_syarat": None,
        "jenis": "sks_lulus",
    }


async def _prasyarat(sads: FakeSads, matkul: str, penanya: Penanya | None = None):
    return await kur._prasyarat(sads, matkul=matkul, penanya=penanya or Penanya())  # type: ignore[arg-type]


async def _semester(
    sads: FakeSads, semester: Any, prodi: str = "", penanya: Penanya | None = None
):
    return await kur._matkul_kurikulum(
        sads,  # type: ignore[arg-type]
        semester=semester,
        prodi=prodi,
        penanya=penanya or Penanya(),
    )


class TestRumpunDanProdi:
    @pytest.mark.parametrize(
        ("kdmk", "rumpun"),
        [
            ("BIFW-063", "BIF"),
            ("BIFN-004", "BIF"),
            ("CDVG-001", "CDV"),
            ("TIW-063", "TIW"),
            ("MKB-105", "MKB"),
        ],
    )
    def test_rumpun_kode(self, kdmk: str, rumpun: str):
        assert kur.rumpun_kode(kdmk) == rumpun

    @pytest.mark.parametrize(
        ("teks", "kode"),
        [
            ("DKV", "SK"),
            ("Desain Komunikasi Visual", "SK"),
            ("TI", "IF"),
            ("informatika", "IF"),
            ("Magister Informatika", "MI"),
            ("RSK", "RSK"),
            ("bisnis digital", "BD"),
            ("Kedokteran", None),
        ],
    )
    def test_kode_prodi_sads(self, teks: str, kode: str | None):
        """DKV di SADS berkode "SK"; saringan `prodi` SADS sendiri tidak dipakai
        karena "TI" juga mencocokkan Magister Informatika."""
        assert kur.kode_prodi_sads(teks) == kode


class TestMatkulKurikulum:
    async def test_prodi_penanya_dan_jumlah_dihitung_handler(self):
        sads = FakeSads(
            [
                _mk("BIFW-501", "Data Mining", smt=5),
                _mk("BIFP-235", "Computer Vision", smt=5, status="Pilihan", sks=2),
                _mk(
                    "CDVW-501",
                    "Ilustrasi",
                    smt=5,
                    prodi="SK",
                    kurikulum_id=6,
                    kurikulum="DKV 2025",
                ),
            ]
        )
        r = await _semester(sads, 5, penanya=Penanya(profile=PROFIL_IF))
        assert r.ok
        assert (
            "2 mata kuliah, 5 SKS (wajib 1 mata kuliah, 3 SKS; "
            "pilihan atau lainnya 1 mata kuliah, 2 SKS)" in r.text
        )
        assert "Ilustrasi" not in r.text
        [lampiran] = r.attachments
        assert lampiran.title == "Mata kuliah semester 5, OBE 2025"
        assert lampiran.items[0] == "Data Mining (BIFW-501) · 3 SKS · Wajib"

    async def test_prodi_eksplisit_mengalahkan_profil(self):
        sads = FakeSads(
            [
                _mk("BIFW-301", "Algoritma", smt=3),
                _mk(
                    "CDVW-301",
                    "Ilustrasi",
                    smt=3,
                    prodi="SK",
                    kurikulum_id=6,
                    kurikulum="DKV 2025",
                ),
            ]
        )
        r = await _semester(sads, "3", prodi="DKV", penanya=Penanya(profile=PROFIL_IF))
        assert "Ilustrasi" in r.text and "Algoritma" not in r.text

    async def test_prodi_tidak_dikenal_dijawab_dengan_daftar_prodi(self):
        r = await _semester(FakeSads([]), 3, prodi="Kedokteran")
        assert r.ok and "tidak dikenal" in r.text and "Informatika" in r.text

    @pytest.mark.parametrize("semester", [0, 9, "x", True, "5/../../x"])
    async def test_semester_tidak_sah_tanpa_memanggil_sads(self, semester: Any):
        """Semester masuk path URL: hanya angka 1-8 yang dikirim ke SADS."""
        sads = FakeSads([])
        r = await _semester(sads, semester)
        assert not r.ok and "1 sampai 8" in r.pesan_untuk_model()
        assert sads.calls == []

    async def test_kurikulum_tidak_aktif_ikut_ditanyakan(self):
        """Kurikulum 2022 Informatika (id 2) tidak aktif di SADS tetapi dipakai
        mahasiswa IF angkatan 2024 (uji 2026-11-07)."""
        sads = FakeSads(
            [
                _mk("BIFW-501", "Data Mining", smt=5),
                _mk(
                    "BIFW-050",
                    "Cloud Computing",
                    smt=5,
                    kurikulum_id=2,
                    kurikulum="Kurikulum 2022 Informatika",
                ),
            ]
        )
        r = await _semester(sads, 5, penanya=Penanya(profile=PROFIL_IF))
        assert "Cloud Computing" in r.text and "Data Mining" in r.text
        assert len(r.attachments) == 2
        assert (f"{kur.PATH_KURIKULUM}/5", {"kurikulum_id": 2}) in sads.calls

    async def test_disaring_ke_kurikulum_dan_rumpun_penanya(self):
        """Kurikulum 2022 Informatika mencampur rumpun kode (BIF, BTI, TIW, MKB):
        semester 5 berisi 110 mata kuliah, rumpun BIF penanya hanya 16."""
        k22 = {"kurikulum_id": 2, "kurikulum": "Kurikulum 2022 Informatika"}
        sads = FakeSads(
            [
                _mk("BIFW-001", "Algoritma", smt=1, **k22),
                _mk("BIFW-050", "Cloud Computing", smt=5, **k22),
                _mk("BTIW-050", "Cloud Computing", smt=5, **k22),
                _mk("MKB-219", "Decision Support System", smt=5, **k22),
                _mk("BIFW-501", "Data Mining", smt=5),
            ],
            nilai=["BIFW-001"],
        )
        r = await _semester(sads, 5, penanya=Penanya(nim=NIM, profile=PROFIL_IF))
        assert "BIFW-050" in r.text
        assert (
            "BTIW-050" not in r.text and "MKB-219" not in r.text and "BIFW-501" not in r.text
        )
        assert "kurikulum penanya sendiri: Kurikulum 2022 Informatika" in r.text
        assert "berawalan BIF" in r.text
        # NIM hanya di path SADS, tidak di teks untuk model.
        assert NIM not in r.pesan_untuk_model()

    async def test_prodi_penanya_sendiri_tetap_disaring(self):
        """Regresi live: model mengisi `prodi="Informatika"` dari baris profil, dan
        jawabannya menjadi "110 mata kuliah" alih-alih kurikulum penanya."""
        k22 = {"kurikulum_id": 2, "kurikulum": "Kurikulum 2022 Informatika"}
        sads = FakeSads(
            [
                _mk("BIFW-001", "Algoritma", smt=1, **k22),
                _mk("BIFW-050", "Cloud Computing", smt=5, **k22),
                _mk("BTIW-050", "Cloud Computing", smt=5, **k22),
            ],
            nilai=["BIFW-001"],
        )
        penanya = Penanya(nim=NIM, profile=PROFIL_IF)
        r = await _semester(sads, 5, prodi="Informatika", penanya=penanya)
        assert "BIFW-050" in r.text and "BTIW-050" not in r.text

    async def test_nilai_tidak_dapat_diambil_berarti_tanpa_saringan(self):
        sads = FakeSads(
            [
                _mk(
                    "BIFW-050",
                    "Cloud Computing",
                    smt=5,
                    kurikulum_id=2,
                    kurikulum="Kurikulum 2022 Informatika",
                ),
                _mk(
                    "BTIW-050",
                    "Cloud Computing",
                    smt=5,
                    kurikulum_id=2,
                    kurikulum="Kurikulum 2022 Informatika",
                ),
            ],
            nilai=None,  # krs-mahasiswa membalas 404
        )
        r = await _semester(sads, 5, penanya=Penanya(nim=NIM, profile=PROFIL_IF))
        assert r.ok and "BIFW-050" in r.text and "BTIW-050" in r.text
        assert "Disaring" not in r.text

    async def test_respons_kurikulum_di_cache(self):
        sads = FakeSads([_mk("BIFW-501", "Data Mining", smt=5)])
        await _semester(sads, 5)
        sebelum = len(sads.calls)
        await _semester(sads, 5)
        assert len(sads.calls) == sebelum

    async def test_baris_ganda_dibuang_bila_kurikulum_tambahan_kelak_aktif(self, monkeypatch):
        sads = FakeSads([_mk("BIFW-501", "Data Mining", smt=5)])
        monkeypatch.setattr(kur, "KURIKULUM_TAMBAHAN", (1,))
        r = await _semester(sads, 5)
        assert r.text.count("BIFW-501") == 1


class TestPrasyarat:
    async def test_kode_langsung_tanpa_mencari_nama(self):
        sads = FakeSads(
            [_mk("BIFW-242", "Basis Data Lanjut", smt=3)],
            prasyarat={
                "BIFW-242": [
                    _syarat(j, "BIFW-241", "Basis Data") for j in ("IF-AI", "IF-GD", "IF-MDA")
                ]
            },
        )
        r = await _prasyarat(sads, "bifw-242")
        assert r.ok
        assert "Prasyarat Basis Data Lanjut (BIFW-242, OBE 2025):" in r.text
        # Tiga konsentrasi dengan isi sama digabung menjadi satu blok.
        assert r.text.count("memenuhi mata kuliah Basis Data (BIFW-241)") == 1
        assert "IF-AI, IF-GD, IF-MDA" in r.text

    async def test_kode_masuk_path_ter_escape(self):
        """Hanya kode berpola `ABCD-123` yang langsung masuk path; teks lain
        dicari sebagai nama, jadi '/' atau '..' tidak pernah sampai ke URL."""
        sads = FakeSads([])
        r = await _prasyarat(sads, "../../admin")
        assert r.ok and "0 mata kuliah" in r.text
        assert sads.jalur(kur.PATH_PRASYARAT) == []

    async def test_nama_persis_diutamakan(self):
        sads = FakeSads(
            [
                _mk("BIFW-241", "Basis Data", smt=2),
                _mk("BIFW-242", "Basis Data Lanjut", smt=3),
            ],
            prasyarat={"BIFW-241": []},
        )
        r = await _prasyarat(sads, "basis data")
        assert "Basis Data (BIFW-241, OBE 2025): tidak memiliki prasyarat" in r.text
        assert "BIFW-242" not in r.text

    async def test_konsentrasi_berbeda_dirinci_terpisah(self):
        sads = FakeSads(
            [_mk("BIFW-295", "Kerja Praktik", smt=6)],
            prasyarat={
                "BIFW-295": [
                    _syarat("IF-AI", "BIFW-278", "Hukum TI"),
                    _sks_lulus("IF-AI", 100),
                    _syarat("IF-GD", "BIFW-278", "Hukum TI"),
                ]
            },
        )
        r = await _prasyarat(sads, "Kerja Praktik")
        assert "konsentrasi IF-AI:" in r.text and "konsentrasi IF-GD:" in r.text
        assert "SKS lulus minimal 100" in r.text

    async def test_kelompok_dua_ditandai_alternatif(self):
        sads = FakeSads(
            [_mk("BRKW-010", "Aljabar Linear", smt=2, prodi="RSK")],
            prasyarat={
                "BRKW-010": [
                    _syarat("RSK", "BRKW-001", "Mathematics", kelompok=1),
                    _syarat("RSK", "MKK-311", "Matematika 1", kelompok=2),
                ]
            },
        )
        r = await _prasyarat(sads, "Aljabar Linear")
        assert "Kelompok 1:\n- memenuhi mata kuliah Mathematics (BRKW-001)" in r.text
        assert "Kelompok 2 (alternatif kelompok 1" in r.text
        assert "Matematika 1 (MKK-311)" in r.text  # spasi ujung SADS dibuang

    async def test_terlalu_banyak_hanya_nama(self):
        sads = FakeSads(
            [_mk(f"BIFW-{i:03d}", f"Pemrograman Web {i}", smt=3) for i in range(1, 6)]
        )
        r = await _prasyarat(sads, "Pemrograman Web")
        assert r.ok and "Ada 5 mata kuliah" in r.text and "terlalu banyak" in r.text
        assert sads.jalur(kur.PATH_PRASYARAT) == []

    async def test_tidak_ada_yang_cocok_tetap_ok(self):
        r = await _prasyarat(
            FakeSads([_mk("BIFW-049", "Artificial Intelligence", smt=5)]), "Kecerdasan Buatan"
        )
        assert r.ok and "0 mata kuliah" in r.text
        assert "DATA_TIDAK_TERSEDIA" not in r.pesan_untuk_model()

    async def test_prodi_penanya_diutamakan(self):
        sads = FakeSads(
            [
                _mk("BIFW-213", "Statistika", smt=4),
                _mk(
                    "CBDW-213",
                    "Statistika",
                    smt=4,
                    prodi="BD",
                    kurikulum_id=7,
                    kurikulum="BD 2025",
                ),
            ],
            prasyarat={"BIFW-213": []},
        )
        r = await _prasyarat(sads, "Statistika", Penanya(profile=PROFIL_IF))
        assert "BIFW-213" in r.text and "CBDW-213" not in r.text

    async def test_disaring_ke_rumpun_penanya(self):
        """ "Kerja Praktik" tercatat 6 kali di SADS; tanpa saringan, pertanyaan
        paling umum pun berakhir "terlalu banyak untuk dirinci"."""
        k22 = {"kurikulum_id": 2, "kurikulum": "Kurikulum 2022 Informatika"}
        sads = FakeSads(
            [
                _mk("BIFW-001", "Algoritma", **k22),
                *(
                    _mk(k, "Kerja Praktik", smt=6, **k22)
                    for k in ("BIFW-063", "BTIW-063", "TIW-063", "MKB-105")
                ),
                _mk("BIFW-295", "Kerja Praktik", smt=6),
            ],
            prasyarat={"BIFW-063": [_sks_lulus("IF-TIP", 80)]},
            nilai=["BIFW-001"],
        )
        r = await _prasyarat(sads, "Kerja Praktik", Penanya(nim=NIM, profile=PROFIL_IF))
        assert "Prasyarat Kerja Praktik (BIFW-063, Kurikulum 2022 Informatika)" in r.text
        assert "SKS lulus minimal 80" in r.text
        assert "kurikulum atau rumpun kode lain: 4" in r.text
        assert [p for p, _ in sads.jalur(kur.PATH_PRASYARAT)] == [
            f"{kur.PATH_PRASYARAT}/BIFW-063"
        ]

    async def test_satu_calon_tidak_mengambil_nilai(self):
        sads = FakeSads([_mk("BIFW-242", "Basis Data Lanjut", smt=3)], nilai=["BIFW-001"])
        await _prasyarat(sads, "Basis Data Lanjut", Penanya(nim=NIM))
        assert not any("/krs-mahasiswa/" in p for p, _ in sads.calls)


def test_skema_tidak_punya_argumen_nim():
    for spec in kur.tool_specs(FakeSads([])):  # type: ignore[arg-type]
        assert spec.pakai_penanya
        assert not {"nim", "penanya"} & set(spec.parameters["properties"])
