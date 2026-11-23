"""Kontak PIC per unit (tabel `unit_contacts`, FE-3).

Yang dijaga di sini: teks kontak yang tampil di widget, validasi isian
dashboard, isi awal migrasi 0017 -- salinan `UNIT_CONTACTS` yang dulu ditulis di
app/rag/risk.py -- dan kontak mana yang tampil untuk pertanyaan tertentu (kata
kunci, dosen PA; migrasi 0018).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.rag.risk import (
    DOSEN_PA,
    FRONT_OFFICE,
    KONTAK_FRONT_OFFICE,
    UnitContact,
    butuh_dosen_pa,
    detect,
    menyebut,
    pilih_kontak,
)
from app.schemas.admin import NOMOR_TELEPON, AdminUnitContactCreate, AdminUnitContactUpdate
from app.units import label_bentrok, susun_kontak
from tests.fixtures.fakes import kontak_pic

API = Path(__file__).resolve().parents[2]
BARU_LINE = chr(10)


def _muat_migrasi(nama: str):
    path = API / "alembic" / "versions" / f"{nama}.py"
    spec = importlib.util.spec_from_file_location(f"migrasi_{nama}", path)
    assert spec is not None and spec.loader is not None
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


MIGRASI = _muat_migrasi("0017_kontak_unit")
MIGRASI_KATA_KUNCI = _muat_migrasi("0018_kata_kunci_kontak")


class TestTeksKontak:
    def test_whatsapp_berlabel(self):
        """Tanpa label ini widget membuka nomornya lewat `tel:` (client `kontak.ts`)."""
        assert susun_kontak(None, "+62 813-5346-2032", None) == "WhatsApp +62 813-5346-2032"

    def test_telepon_lalu_whatsapp_seperti_front_office(self):
        assert susun_kontak("(0361) 256995", "+62 813-3982-2383", None) == KONTAK_FRONT_OFFICE

    def test_email_berdiri_sendiri_menjadi_mailto(self):
        """Widget hanya menautkan email bila seluruh teks kontaknya alamat email."""
        assert susun_kontak(None, None, "baak@instiki.ac.id") == "baak@instiki.ac.id"

    def test_bentuk_tampil_untuk_pipeline(self):
        kontak = kontak_pic("AKADEMIK", "PIC BAAK", whatsapp="+62 813-5346-2032")
        assert kontak.untuk_tampil() == UnitContact(
            unit="PIC BAAK", jam_layanan="", kontak="WhatsApp +62 813-5346-2032"
        )

    def test_jam_layanan_ikut(self):
        kontak = kontak_pic(
            "FRONT OFFICE",
            "Front Office INSTIKI",
            phone="(0361) 256995",
            whatsapp="+62 813-3982-2383",
            service_hours="Senin-Sabtu, 08.00-22.00 WITA",
        )
        assert kontak.untuk_tampil() == FRONT_OFFICE

    def test_label_bentrok_tanpa_membedakan_huruf_besar(self):
        """Sama dengan index unik `ix_unit_contacts_unit_label`."""
        kontak = kontak_pic("BAAK", "PIC BAAK", whatsapp="+62 813-5346-2032")
        assert label_bentrok([kontak], "pic baak")
        assert not label_bentrok([kontak], "pic baak", kecuali=kontak.id)
        assert not label_bentrok([kontak], "PIC PLK")


class TestValidasiIsian:
    @pytest.mark.parametrize(
        "nomor",
        [
            "+62 813-5346-2032",
            "+6281353462032",
            "081353462032",
            "0813-5346-2032",
            "0813 5346 2032",
            "(0361) 256995",
            "0361 256995",
        ],
    )
    def test_nomor_sah(self, nomor):
        assert AdminUnitContactCreate(label="PIC", whatsapp=nomor).whatsapp == nomor
        assert AdminUnitContactCreate(label="PIC", phone=nomor).phone == nomor

    @pytest.mark.parametrize(
        "nomor",
        [
            "62813-5346-2032",
            "+1 555 123 4567",
            "12345",
            "wa.me/6281353462032",
            "0813-5346-2032 (Pak Budi)",
            "0813-5346-2032 / 0812-1111-2222",
            "nomor menyusul",
        ],
    )
    def test_nomor_yang_tidak_terbaca_utuh_oleh_widget_ditolak(self, nomor):
        with pytest.raises(ValidationError):
            AdminUnitContactCreate(label="PIC", whatsapp=nomor)

    def test_spasi_dirapikan(self):
        isian = AdminUnitContactCreate(label="  PIC   BAAK ", whatsapp=" 0813  5346 2032 ")
        assert (isian.label, isian.whatsapp) == ("PIC BAAK", "0813 5346 2032")

    def test_isian_kosong_menjadi_null(self):
        isian = AdminUnitContactCreate(
            label="PIC", phone="", whatsapp="0813-5346-2032", email="  ", service_hours=""
        )
        assert (isian.phone, isian.email, isian.service_hours) == (None, None, None)

    def test_minimal_satu_kanal(self):
        with pytest.raises(ValidationError, match="Isi minimal satu"):
            AdminUnitContactCreate(label="PIC", phone="", whatsapp=None, email="")

    def test_email_saja_boleh(self):
        assert AdminUnitContactCreate(label="BAAK", email="baak@instiki.ac.id").email

    def test_email_tidak_sah_ditolak(self):
        with pytest.raises(ValidationError):
            AdminUnitContactCreate(label="BAAK", email="bukan-email")

    @pytest.mark.parametrize(
        "label",
        [
            "PIC" + BARU_LINE + "BAAK",
            "PIC https://contoh.id",
            "Hubungi www.contoh.id",
            "x" * 201,
            "   ",
        ],
    )
    def test_label_tidak_sah(self, label):
        """Label masuk ke teks penolakan yang tautannya aktif di widget."""
        with pytest.raises(ValidationError):
            AdminUnitContactCreate(label=label, whatsapp="0813-5346-2032")

    def test_pesan_menyebut_nama_isian(self):
        """Dashboard menampilkan pesan `value_error` apa adanya, tanpa nama kolom."""
        with pytest.raises(ValidationError, match="WhatsApp harus satu nomor"):
            AdminUnitContactCreate(label="PIC", whatsapp="abc")
        with pytest.raises(ValidationError, match="Nama kontak harus satu baris"):
            AdminUnitContactCreate(label="PIC" + BARU_LINE + "X", phone="0361 256995")

    def test_jam_layanan_tanpa_tautan(self):
        with pytest.raises(ValidationError, match="Jam layanan tidak boleh memuat tautan"):
            AdminUnitContactCreate(
                label="PIC", whatsapp="0813-5346-2032", service_hours="lihat https://x.id"
            )

    def test_field_tak_dikenal_ditolak(self):
        with pytest.raises(ValidationError):
            AdminUnitContactCreate(label="PIC", whatsapp="0813-5346-2032", kontak="bebas")

    @pytest.mark.parametrize("field", ["label", "sort_order"])
    def test_ubah_field_wajib_tidak_boleh_null(self, field):
        with pytest.raises(ValidationError):
            AdminUnitContactUpdate(**{field: None})

    def test_ubah_boleh_mengosongkan_kanal(self):
        """Apakah masih ada kanal tersisa diperiksa router, terhadap baris lama."""
        assert AdminUnitContactUpdate(phone="").model_dump(exclude_unset=True) == {
            "phone": None
        }

    def test_pola_nomor_sama_dengan_widget(self):
        kontak_ts = API.parent / "client" / "src" / "lib" / "kontak.ts"
        if not kontak_ts.exists():
            pytest.skip("repo client tidak ada di sebelah api/")
        awalan = "const NOMOR = /"
        baris = next(
            b
            for b in kontak_ts.read_text(encoding="utf-8").splitlines()
            if b.startswith(awalan)
        )
        assert baris[len(awalan) : baris.rindex("/g")] == NOMOR_TELEPON.pattern


class TestIsiAwalMigrasi:
    """Migrasi 0017 memindahkan `UNIT_CONTACTS` ke database tanpa mengubah isinya."""

    @pytest.mark.parametrize(
        "unit,nomor",
        [
            ("KEMAHASISWAAN", "+62 813-3936-5103"),
            ("AKADEMIK", "+62 813-5346-2032"),
            ("UNIT PELAYANAN SERTIFIKASI", "+62 813-3734-8654"),
            ("PLK", "+62 851-1931-5346"),
            ("KEUANGAN", "+62 851-7426-1765"),
            ("FRONT OFFICE", "+62 813-3982-2383"),
            # Nama sebelum penggantian 2026-11-18 (`UNIT_ALIASES` dulu).
            ("Kemahasiswaan", "+62 813-3936-5103"),
            ("BAAK", "+62 813-5346-2032"),
            ("UPS", "+62 813-3734-8654"),
            ("Keuangan", "+62 851-7426-1765"),
            ("FO", "+62 813-3982-2383"),
            (" baak ", "+62 813-5346-2032"),
        ],
    )
    def test_pic_utama_per_unit(self, unit, nomor):
        assert MIGRASI._isi_untuk(unit)[0][2] == nomor

    def test_akademik_membawa_pic_plk(self):
        """Dokumen PLK dipindah ke AKADEMIK (2026-11-18), jadi PIC-nya ikut."""
        assert [k[0] for k in MIGRASI._isi_untuk("AKADEMIK")] == [
            "PIC BAAK",
            "Admin PLK INSTIKI",
        ]

    def test_kemahasiswaan_membawa_pic_per_bidang(self):
        assert [k[2] for k in MIGRASI._isi_untuk("Kemahasiswaan")[1:]] == [
            "+62 812-3673-3230",
            "+62 819-9843-9707",
            "+62 812-3608-8993",
        ]

    @pytest.mark.parametrize("unit", ["", "Prodi", "Fakultas", "A", "RPL"])
    def test_unit_tanpa_pic_kosong(self, unit):
        """Unit ini tetap memakai Front Office dari kode."""
        assert MIGRASI._isi_untuk(unit) == ()

    def test_alias_menunjuk_unit_yang_ada(self):
        kunci = {nama.casefold() for nama in MIGRASI.ISI_AWAL}
        for lama, baru in MIGRASI.ALIAS.items():
            assert baru.casefold() in kunci, lama
            assert lama.casefold() not in kunci, lama

    def test_label_unik_dalam_satu_unit(self):
        """Widget memakai label sebagai kunci baris dan kunci banner terakhir."""
        for isi in MIGRASI.ISI_AWAL.values():
            assert len({k[0].casefold() for k in isi}) == len(isi)

    def test_setiap_baris_lolos_validasi_dashboard(self):
        """Kalau tidak, admin tidak dapat menyimpan ulang kontak hasil migrasi."""
        for isi in MIGRASI.ISI_AWAL.values():
            for label, phone, whatsapp, jam in isi:
                AdminUnitContactCreate(
                    label=label, phone=phone, whatsapp=whatsapp, service_hours=jam
                )

    def test_front_office_sama_dengan_cadangan_di_kode(self):
        ((label, phone, whatsapp, jam),) = MIGRASI.ISI_AWAL["FRONT OFFICE"]
        assert (
            UnitContact(label, jam or "", susun_kontak(phone, whatsapp, None)) == FRONT_OFFICE
        )


# --- Kontak mana yang tampil untuk pertanyaan ini --------------------------------

BAAK = kontak_pic("AKADEMIK", "PIC BAAK", whatsapp="+62 813-5346-2032").untuk_tampil()
PLK = kontak_pic(
    "AKADEMIK",
    "Admin PLK INSTIKI",
    whatsapp="+62 851-1931-5346",
    sort_order=2,
    keywords=MIGRASI_KATA_KUNCI.KATA_KUNCI_PLK,
).untuk_tampil()
AKADEMIK = (BAAK, PLK)


def _pilih(pertanyaan: str, kontak_unit=AKADEMIK) -> tuple[UnitContact, ...]:
    return pilih_kontak(kontak_unit, detect(pertanyaan), pertanyaan)


class TestKataKunci:
    @pytest.mark.parametrize(
        "pertanyaan",
        [
            "cara membuat surat keterangan aktif kuliah",
            "legalisir ijazah di mana?",
            "syarat cuti kuliah",
            "kaplk",  # bukan kata utuh
        ],
    )
    def test_urusan_lain_hanya_baak(self, pertanyaan):
        """Tangkapan layar 2026-11-23: pertanyaan surat-menyurat di unit Akademik
        ikut menampilkan nomor Admin PLK."""
        assert _pilih(pertanyaan) == (BAAK,)

    @pytest.mark.parametrize(
        "pertanyaan",
        [
            "syarat ikut PLK apa?",
            "plk itu apa",
            "PLK-nya kapan dimulai",
            "info Pembelajaran di Luar Kampus",
            "pembelajaran   luar kampus daftarnya gimana",
        ],
    )
    def test_urusan_plk_membawa_nomor_plk(self, pertanyaan):
        assert _pilih(pertanyaan) == (BAAK, PLK)

    def test_tulisan_ulang_dari_riwayat_ikut_dicocokkan(self):
        """Graf menggabungkan pertanyaan dengan tulisan ulangnya (`_kontak`)."""
        teks = "syaratnya apa?" + BARU_LINE + "Apa syarat mengikuti PLK?"
        assert pilih_kontak(AKADEMIK, detect(teks), teks) == (BAAK, PLK)

    def test_unit_plk_sendiri_tanpa_kata_kunci(self):
        """Di unit PLK nomor itu kontak utamanya, jadi selalu tampil."""
        plk_utama = kontak_pic("PLK", "Admin PLK INSTIKI", whatsapp="+62 851-1931-5346")
        assert _pilih("surat keterangan", (plk_utama.untuk_tampil(),)) == (
            plk_utama.untuk_tampil(),
        )

    def test_semua_kontak_berkata_kunci_tidak_cocok_jatuh_ke_cadangan(self):
        """Kosong = penolakan memakai Front Office; topik berisiko memakai kontak topiknya."""
        assert _pilih("surat keterangan aktif", (PLK,)) == ()
        assert _pilih("kapan deadline pembayaran UKT?", (PLK,)) == (FRONT_OFFICE,)

    def test_kata_kunci_kosong_diabaikan(self):
        assert not menyebut("apa saja", ["", "  "])


class TestDosenPA:
    @pytest.mark.parametrize(
        "pertanyaan",
        [
            "saya mau ambil matkul semester depan",
            "berapa sks yang bisa saya ambil?",
            "mata kuliah pilihan apa saja yang bisa saya ambil",
            "boleh ambil 24 sks?",
            "pengambilan mata kuliah gimana",
            "Matakuliah apa yg boleh diambil kalau IPK 2.5",
            "bisa drop matkul ga?",
            "mau tambah matkul setelah KRS",
            "mengulang mata kuliah yang nilainya D",
            "kapan perwalian?",
            "KRS saya belum di acc dosen",
            "maksimal sks per semester berapa",
        ],
    )
    def test_pengambilan_mata_kuliah(self, pertanyaan):
        assert butuh_dosen_pa(pertanyaan)
        assert _pilih(pertanyaan) == (DOSEN_PA,)

    @pytest.mark.parametrize(
        "pertanyaan",
        [
            "cara membuat surat keterangan aktif kuliah",
            "jadwal mata kuliah semester ini",
            "ganti jadwal mata kuliah",
            "ujian ulang mata kuliah",
            "cara isi KRS di SADS",
            "nilai mata kuliah saya belum keluar",
            "dosen pengampu mata kuliah basis data siapa",
            "saya mau ambil ijazah",
            "ambil formulir di mana",
        ],
    )
    def test_bukan_pengambilan_mata_kuliah(self, pertanyaan):
        assert not butuh_dosen_pa(pertanyaan)

    def test_kontak_berkata_kunci_yang_cocok_tetap_ikut(self):
        """Lebih spesifik daripada BAAK: mengambil mata kuliah PLK."""
        assert _pilih("mata kuliah PLK yang bisa diambil apa saja") == (DOSEN_PA, PLK)

    def test_topik_berisiko_tetap_membawa_kontak_umum(self):
        """Dosen PA tidak memutuskan batas KRS atau KRS yang terkunci karena tagihan."""
        assert _pilih("kapan batas akhir ambil matkul?") == (DOSEN_PA, BAAK)
        assert _pilih("belum bayar UKT, masih bisa ambil sks?") == (DOSEN_PA, BAAK)

    def test_tanpa_kontak_unit_topik_berisiko_memakai_kontak_topik(self):
        assert _pilih("kapan deadline ambil matkul?", ()) == (DOSEN_PA, FRONT_OFFICE)

    def test_tanpa_nomor(self):
        """Dosen PA berbeda per mahasiswa; nomor di sini pasti salah untuk sebagian besar."""
        assert not any(c.isdigit() for c in DOSEN_PA.kontak)


class TestIsianKataKunci:
    def test_dirapikan_dan_kembar_dibuang(self):
        isian = AdminUnitContactCreate(
            label="PIC",
            whatsapp="0813-5346-2032",
            keywords=[" PLK ", "plk", "", "luar   kampus"],
        )
        assert isian.keywords == ["PLK", "luar kampus"]

    def test_bawaan_kosong(self):
        assert AdminUnitContactCreate(label="PIC", whatsapp="0813-5346-2032").keywords == []

    @pytest.mark.parametrize(
        "kata_kunci",
        [["a"], ["PLK" + BARU_LINE + "x"], ["x" * 101], [f"k{i}" for i in range(21)]],
    )
    def test_tidak_sah(self, kata_kunci):
        with pytest.raises(ValidationError):
            AdminUnitContactCreate(label="PIC", whatsapp="0813-5346-2032", keywords=kata_kunci)

    def test_ubah_null_menghapus_semua(self):
        assert AdminUnitContactUpdate(keywords=None).model_dump(exclude_unset=True) == {
            "keywords": []
        }

    def test_ubah_tanpa_kata_kunci_tidak_menyentuhnya(self):
        isian = AdminUnitContactUpdate(label="X")
        assert "keywords" not in isian.model_dump(exclude_unset=True)


class TestMigrasiKataKunci:
    def test_label_sama_dengan_isi_awal_0017(self):
        """0018 mencari Admin PLK lewat label yang dipasang 0017."""
        assert MIGRASI._PIC_PLK[0] == MIGRASI_KATA_KUNCI.LABEL_PLK
        assert MIGRASI_KATA_KUNCI.UNIT_PLK in MIGRASI.ISI_AWAL

    def test_lolos_validasi_dashboard(self):
        isian = AdminUnitContactCreate(
            label="PIC",
            whatsapp="0813-5346-2032",
            keywords=MIGRASI_KATA_KUNCI.KATA_KUNCI_PLK,
        )
        assert isian.keywords == list(MIGRASI_KATA_KUNCI.KATA_KUNCI_PLK)
