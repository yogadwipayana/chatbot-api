"""Menjaga `api.yaml` tetap jujur terhadap kode.

Spesifikasi API yang ditulis tangan selalu punya satu masalah: ia membusuk
tanpa ada yang tahu, lalu frontend dibangun di atas kontrak yang sudah tidak
berlaku. Test di sini membandingkan `api.yaml` dengan spesifikasi yang benar-benar
dihasilkan FastAPI:

- setiap operasi bertanda `implemented` HARUS ada di aplikasi
- setiap endpoint di aplikasi HARUS ada di api.yaml dan bertanda `implemented`
- bentuk skema untuk endpoint terimplementasi harus cocok dengan model Pydantic

Operasi bertanda `planned` sengaja tidak diperiksa terhadap aplikasi -- itu
kontrak yang disepakati lebih dulu supaya pekerjaan frontend tidak terhambat.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.main import create_app

SPEC_PATH = Path(__file__).resolve().parents[2] / "api.yaml"
METODE_HTTP = {"get", "post", "put", "patch", "delete", "head", "options"}


@pytest.fixture(scope="module")
def spec() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live_spec() -> dict:
    return create_app().openapi()


def operasi(spec: dict):
    """Hasilkan (path, metode, definisi) untuk seluruh operasi di spesifikasi."""
    for path, item in spec["paths"].items():
        for metode, definisi in item.items():
            if metode in METODE_HTTP:
                yield path, metode, definisi


def kumpulkan_ref(node, ditemukan: list[str]) -> list[str]:
    if isinstance(node, dict):
        for kunci, nilai in node.items():
            if kunci == "$ref" and isinstance(nilai, str):
                ditemukan.append(nilai)
            else:
                kumpulkan_ref(nilai, ditemukan)
    elif isinstance(node, list):
        for item in node:
            kumpulkan_ref(item, ditemukan)
    return ditemukan


class TestValiditasFormal:
    def test_sah_menurut_validator_openapi_resmi(self, spec):
        """Diperiksa terhadap meta-schema OpenAPI 3.1, bukan hanya asumsi kita.

        Ini yang menangkap kesalahan yang lolos dari mata: `type` tidak sah,
        `required` menyebut properti yang tidak ada, struktur `responses` keliru.
        Spesifikasi yang tidak sah akan ditolak generator klien dan Swagger UI.
        """
        from openapi_spec_validator import validate

        validate(spec)


class TestBentukSpesifikasi:
    def test_yaml_dapat_diurai(self, spec):
        assert isinstance(spec, dict)

    def test_versi_openapi_sama_dengan_yang_dihasilkan_fastapi(self, spec, live_spec):
        """Beda versi minor membuat perkakas generator klien berperilaku lain."""
        assert spec["openapi"] == live_spec["openapi"]

    def test_ada_info_dan_server(self, spec):
        assert spec["info"]["title"]
        assert spec["info"]["version"]
        assert spec["servers"]

    def test_semua_ref_internal_dapat_diselesaikan(self, spec):
        """Satu `$ref` salah ketik membuat Swagger UI gagal tanpa pesan jelas."""
        for ref in kumpulkan_ref(spec, []):
            assert ref.startswith("#/"), f"ref eksternal tidak didukung: {ref}"
            node = spec
            for bagian in ref.removeprefix("#/").split("/"):
                assert bagian in node, f"{ref} tidak dapat diselesaikan (buntu di {bagian!r})"
                node = node[bagian]

    def test_setiap_operasi_punya_operation_id_unik(self, spec):
        ids = [d["operationId"] for _, _, d in operasi(spec)]
        assert len(ids) == len(set(ids)), "operationId ganda memecah generator klien"

    def test_setiap_operasi_punya_ringkasan_dan_tag(self, spec):
        for path, metode, definisi in operasi(spec):
            assert definisi.get("summary"), f"{metode.upper()} {path} tanpa summary"
            assert definisi.get("tags"), f"{metode.upper()} {path} tanpa tag"

    def test_setiap_tag_yang_dipakai_terdaftar(self, spec):
        terdaftar = {t["name"] for t in spec["tags"]}
        for path, metode, definisi in operasi(spec):
            for tag in definisi["tags"]:
                assert tag in terdaftar, (
                    f"tag '{tag}' pada {metode.upper()} {path} tidak terdaftar"
                )


class TestPenandaStatus:
    def test_setiap_operasi_diberi_x_status(self, spec):
        """Tanpa penanda ini, tidak ada yang tahu mana kontrak yang sudah nyata."""
        for path, metode, definisi in operasi(spec):
            assert "x-status" in definisi, f"{metode.upper()} {path} tanpa x-status"

    def test_operasi_admin_mencatat_level_minimum(self, spec):
        """Tanpa `x-min-role`, frontend tidak tahu menu mana yang boleh tampil
        dan test level akses tidak membangkitkan kasus untuk operasi itu."""
        for path, metode, definisi in operasi(spec):
            if definisi.get("security"):
                assert definisi.get("x-min-role") in {"staf", "admin", "superadmin"}, (
                    f"{metode.upper()} {path} tanpa x-min-role yang sah"
                )

    def test_nilai_x_status_dikenal(self, spec):
        for path, metode, definisi in operasi(spec):
            assert definisi["x-status"] in {"implemented", "planned"}, (
                f"{metode.upper()} {path}: x-status tidak dikenal"
            )


class TestKesesuaianDenganAplikasi:
    def test_operasi_implemented_benar_benar_ada_di_aplikasi(self, spec, live_spec):
        for path, metode, definisi in operasi(spec):
            if definisi["x-status"] != "implemented":
                continue
            assert path in live_spec["paths"], f"{path} ditandai implemented tetapi tidak ada"
            assert metode in live_spec["paths"][path], (
                f"{metode.upper()} {path} ditandai implemented tetapi tidak ada"
            )

    def test_setiap_endpoint_aplikasi_terdokumentasi(self, spec, live_spec):
        """Arah sebaliknya: endpoint baru tidak boleh lolos tanpa masuk kontrak."""
        for path, item in live_spec["paths"].items():
            for metode in item:
                if metode not in METODE_HTTP:
                    continue
                assert path in spec["paths"], (
                    f"{path} ada di aplikasi tetapi tidak di api.yaml"
                )
                assert metode in spec["paths"][path], (
                    f"{metode.upper()} {path} ada di aplikasi tetapi tidak di api.yaml"
                )
                assert spec["paths"][path][metode]["x-status"] == "implemented", (
                    f"{metode.upper()} {path} sudah ada di aplikasi "
                    "tetapi masih ditandai planned"
                )

    def test_jumlah_operasi_implemented_sama_dengan_aplikasi(self, spec, live_spec):
        di_spec = sum(1 for _, _, d in operasi(spec) if d["x-status"] == "implemented")
        di_app = sum(
            1 for item in live_spec["paths"].values() for m in item if m in METODE_HTTP
        )
        assert di_spec == di_app


class TestSkemaCocokDenganModel:
    """Bentuk skema untuk endpoint terimplementasi diperiksa terhadap Pydantic."""

    def test_chat_request_field_wajib_cocok(self, spec, live_spec):
        milik_kita = set(spec["components"]["schemas"]["ChatRequest"]["required"])
        milik_app = set(live_spec["components"]["schemas"]["ChatRequest"]["required"])
        assert milik_kita == milik_app

    def test_chat_response_field_wajib_cocok(self, spec, live_spec):
        milik_kita = set(spec["components"]["schemas"]["ChatResponse"]["required"])
        milik_app = set(live_spec["components"]["schemas"]["ChatResponse"]["required"])
        assert milik_kita == milik_app

    def test_citation_membawa_semua_yang_dibutuhkan_FE2(self, spec):
        """Kartu sitasi harus cukup untuk membuka PDF di halaman yang tepat."""
        wajib = set(spec["components"]["schemas"]["CitationOut"]["required"])
        assert {"judul", "halaman", "document_id"} <= wajib

    def test_citation_out_cocok_dengan_model(self, spec, live_spec):
        milik_kita = set(spec["components"]["schemas"]["CitationOut"]["required"])
        milik_app = set(live_spec["components"]["schemas"]["CitationOut"]["required"])
        assert milik_kita == milik_app

    def test_contact_out_cocok_dengan_model(self, spec, live_spec):
        milik_kita = set(spec["components"]["schemas"]["ContactOut"]["required"])
        milik_app = set(live_spec["components"]["schemas"]["ContactOut"]["required"])
        assert milik_kita == milik_app

    def test_batas_panjang_pertanyaan_cocok(self, spec, live_spec):
        kita = spec["components"]["schemas"]["ChatRequest"]["properties"]["question"]
        app = live_spec["components"]["schemas"]["ChatRequest"]["properties"]["question"]
        assert kita["maxLength"] == app["maxLength"]
        assert kita["minLength"] == app["minLength"]

    def test_enum_outcome_kind_cocok_dengan_kode(self, spec):
        from app.rag.chain import OutcomeKind

        assert set(spec["components"]["schemas"]["OutcomeKind"]["enum"]) == {
            k.value for k in OutcomeKind
        }

    def test_field_wajib_setiap_skema_bernama_sama_cocok(self, spec, live_spec):
        """Generalisasi test per-skema di atas untuk seluruh skema.

        Skema Pydantic dinamai sesuai nama kelasnya, dan nama kelas di
        `app/schemas/` sengaja disamakan dengan api.yaml. Field yang di kode
        wajib tetapi di kontrak opsional (atau sebaliknya) membuat tipe hasil
        generate di frontend berbohong.
        """
        kita = spec["components"]["schemas"]
        app = live_spec["components"]["schemas"]
        bersama = sorted(set(kita) & set(app))
        beda = {
            nama: {
                "api.yaml": sorted(kita[nama].get("required", [])),
                "aplikasi": sorted(app[nama].get("required", [])),
            }
            for nama in bersama
            if set(kita[nama].get("required", [])) != set(app[nama].get("required", []))
        }
        assert not beda
        assert len(bersama) >= 25, f"hanya {len(bersama)} skema yang terbandingkan: {bersama}"

    def test_endpoint_admin_di_aplikasi_menuntut_bearer(self, live_spec):
        """Swagger UI hanya menampilkan tombol Authorize bila skemanya terdaftar."""
        for path, item in live_spec["paths"].items():
            if not path.startswith("/api/admin/") or path == "/api/admin/login":
                continue
            for metode, definisi in item.items():
                assert {"bearerAuth": []} in definisi.get("security", []), (
                    f"{metode.upper()} {path} tanpa bearerAuth"
                )

    def test_enum_keputusan_ambang_cocok_dengan_kode(self, spec):
        """Skema AD-6 memaparkan Decision dan Reason apa adanya."""
        from app.rag.threshold import Decision, Reason

        skema = spec["components"]["schemas"]["TestQueryResponse"]
        keputusan = skema["properties"]["decision"]
        assert set(keputusan["properties"]["decision"]["enum"]) == {d.value for d in Decision}
        assert set(keputusan["properties"]["reason"]["enum"]) == {r.value for r in Reason}


class TestJanjiPerilaku:
    """Kontrak menjanjikan beberapa hal yang juga dijaga test lain; pastikan
    janji itu benar-benar tertulis dan tidak hilang saat spesifikasi dirapikan."""

    def test_endpoint_chat_mendokumentasikan_503_kill_switch(self, spec):
        for path in ("/api/chat", "/api/chat/stream"):
            assert "503" in spec["paths"][path]["post"]["responses"]

    def test_endpoint_chat_mendokumentasikan_429_rate_limit(self, spec):
        for path in ("/api/chat", "/api/chat/stream"):
            assert "429" in spec["paths"][path]["post"]["responses"]

    def test_contoh_penolakan_tidak_membawa_sitasi(self, spec):
        """FE-4: penolakan harus terlihat berbeda dari jawaban. Contoh yang
        memuat sitasi akan menuntun frontend membangunnya dengan keliru."""
        for nama in ("Penolakan", "Dukungan"):
            contoh = spec["components"]["examples"][nama]["value"]
            assert contoh["citations"] == [], f"contoh {nama} tidak boleh punya sitasi"

    def test_contoh_jawaban_membawa_sitasi(self, spec):
        contoh = spec["components"]["examples"]["JawabanNormal"]["value"]
        assert contoh["citations"]

    def test_nilai_kind_pada_contoh_sah(self, spec):
        sah = set(spec["components"]["schemas"]["OutcomeKind"]["enum"])
        for nama, contoh in spec["components"]["examples"].items():
            nilai = contoh["value"]
            if "kind" in nilai:
                assert nilai["kind"] in sah, f"contoh {nama}: kind tidak sah"

    def test_endpoint_admin_memakai_bearer_auth(self, spec):
        for path, metode, definisi in operasi(spec):
            if not any(t.startswith("admin-") for t in definisi["tags"]):
                continue
            if definisi["operationId"] == "admin_login":
                continue
            assert definisi.get("security"), f"{metode.upper()} {path} tanpa security"

    def test_endpoint_mahasiswa_tidak_menuntut_autentikasi(self, spec):
        """PRD §5: tidak ada login mahasiswa di v1."""
        for path, metode, definisi in operasi(spec):
            if "chat" in definisi["tags"]:
                assert not definisi.get("security"), f"{metode.upper()} {path} menuntut auth"
