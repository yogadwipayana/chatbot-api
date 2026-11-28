"""Penolakan yang diputuskan LLM (`NOT_FOUND_MARKER`).

Dengan embedding e5, pertanyaan di luar dokumen ("harga sertifikasi CCNA")
mendapat skor vektor setara pertanyaan yang terjawab ("harga sertifikasi
TOEIC"), sehingga lolos threshold FR-3. Sebelum penanda ini ada, LLM menulis
"tidak menemukan" sebagai jawaban biasa: membawa kartu sitasi dokumen yang
tidak relevan, tanpa kontak unit, dan tidak pernah masuk AD-4.
"""

from __future__ import annotations

import pytest

from app.observability.chatlog import ChatLogEntry, build_meta
from app.rag.chain import (
    OutcomeKind,
    is_not_found,
    refusal_text,
    run_pipeline,
    strip_markers,
)
from app.rag.prompts import NOT_FOUND_MARKER, SYSTEM_PROMPT
from app.rag.risk import FRONT_OFFICE
from app.routers.chat import to_response
from tests.fixtures.fakes import RecordingLLM

JAWABAN = "Harga TOEIC Rp675.000 [Panduan Akademik 2025, hal. 12]."
PROSA_TOLAK = (
    "Maaf, informasi mengenai tata cara pengajuan cuti akademik tidak tercantum "
    "dalam dokumen yang tersedia. Silakan menghubungi bagian akademik."
)


async def tanya(retriever, reply: str, **kw):
    return await run_pipeline(
        "berapa harga sertifikasi CCNA?",
        retriever=retriever,
        llm_call=RecordingLLM(reply),
        **kw,
    )


class TestPrompt:
    def test_prompt_menyebut_penanda_yang_dideteksi_kode(self):
        """Prompt yang menyebut penanda lain membuat seluruh deteksi ini mati diam-diam."""
        assert NOT_FOUND_MARKER in SYSTEM_PROMPT


class TestDeteksi:
    def test_penanda_saja(self):
        assert is_not_found(NOT_FOUND_MARKER)

    def test_penanda_berspasi(self):
        assert is_not_found(f"\n {NOT_FOUND_MARKER} \n")

    @pytest.mark.parametrize(
        "prosa",
        [
            PROSA_TOLAK,
            "Saya tidak menemukan informasi harga sertifikasi CCNA dalam dokumen.",
            "Informasi tersebut tidak ditemukan di dokumen resmi.",
            "Tidak ada informasi tentang jadwal wisuda di konteks.",
        ],
    )
    def test_prosa_tanpa_sitasi_dianggap_penolakan(self, prosa):
        """Jaring pengaman untuk model yang lupa memakai penanda."""
        assert is_not_found(prosa)

    def test_jawaban_parsial_bersitasi_tetap_jawaban(self):
        teks = (
            "Harga TOEIC Rp675.000 [HARGA SERTIFIKASI, hal. 1]. Harga CCNA tidak "
            "ditemukan di dokumen."
        )
        assert not is_not_found(teks)

    def test_penanda_di_jawaban_bersitasi_tetap_jawaban(self):
        assert not is_not_found(f"{JAWABAN} {NOT_FOUND_MARKER}")

    def test_jawaban_biasa(self):
        assert not is_not_found(JAWABAN)

    def test_penanda_dibuang_dari_jawaban_parsial(self):
        assert strip_markers(f"{JAWABAN}\n{NOT_FOUND_MARKER}") == JAWABAN

    def test_jawaban_tanpa_penanda_tidak_diubah(self):
        assert strip_markers(f"  {JAWABAN}") == f"  {JAWABAN}"


# T72 (2026-11-18): model membalas penanda dengan T, A, K Kiril, dan teks itu
# tampil ke mahasiswa sebagai jawaban. Dibangun dari kode titik supaya sumber
# test tetap ASCII.
HOMOGLIF_KIRIL = "[" + chr(0x0422) + "ID" + chr(0x0410) + chr(0x041A) + "_DITEMUKAN]"
HOMOGLIF_YUNANI = "[" + chr(0x03A4) + "IDAK_DIT" + chr(0x0395) + "MUKAN]"
LEBAR_PENUH = chr(0xFF3B) + "TIDAK_DITEMUKAN" + chr(0xFF3D)
TAK_TERLIHAT = "[TIDAK" + chr(0x200B) + "_DITEMUKAN]"
# Varian yang lolos perbaikan pertama T72 (2026-11-28): huruf kecil, pemisah
# lain, karakter tak terlihat di luar daftar awal, dan homoglif huruf kecil.
HURUF_KECIL = "[tidak_ditemukan]"
BERSPASI = "[TIDAK DITEMUKAN]"
SOFT_HYPHEN = "[TIDAK" + chr(0x00AD) + "_DITEMUKAN]"
FUNGSI_TAK_TERLIHAT = "[TIDAK" + chr(0x2061) + "_DITEMUKAN]"
GRAPHEME_JOINER = "[TIDAK" + chr(0x034F) + "_DITEMUKAN]"
KIRIL_KECIL = "[TID" + chr(0x0430) + "K_DITEMUKAN]"
TEBAL = f"**{NOT_FOUND_MARKER}**"

SEMUA_VARIAN = [
    HOMOGLIF_KIRIL,
    HOMOGLIF_YUNANI,
    LEBAR_PENUH,
    TAK_TERLIHAT,
    HURUF_KECIL,
    BERSPASI,
    SOFT_HYPHEN,
    FUNGSI_TAK_TERLIHAT,
    GRAPHEME_JOINER,
    KIRIL_KECIL,
    TEBAL,
]


class TestPenandaHomoglif:
    @pytest.mark.parametrize("penanda", SEMUA_VARIAN)
    def test_dikenali_sebagai_penolakan(self, penanda):
        assert penanda != NOT_FOUND_MARKER
        assert is_not_found(penanda)
        assert is_not_found(f"  {penanda} ")

    def test_tetap_jawaban_bila_bersitasi(self):
        assert not is_not_found(f"{JAWABAN} {HOMOGLIF_KIRIL}")

    @pytest.mark.parametrize("penanda", SEMUA_VARIAN)
    def test_dibuang_dari_jawaban_parsial(self, penanda):
        assert strip_markers(f"{JAWABAN} {penanda}") == JAWABAN

    def test_penekanan_lain_tidak_disentuh(self):
        teks = f"**Syarat**: {JAWABAN} **[Catatan]** {TEBAL}"
        assert strip_markers(teks) == f"**Syarat**: {JAWABAN} **[Catatan]**"

    def test_teks_berkurung_lain_tidak_disentuh(self):
        teks = f"{JAWABAN} [Catatan] {NOT_FOUND_MARKER}"
        assert strip_markers(teks) == f"{JAWABAN} [Catatan]"

    async def test_pipeline_menjadi_penolakan_resmi(self, strong_retriever):
        hasil = await tanya(strong_retriever, HOMOGLIF_KIRIL)
        assert hasil.kind is OutcomeKind.REFUSAL
        assert chr(0x0422) not in hasil.text
        assert NOT_FOUND_MARKER not in hasil.text

    @pytest.mark.parametrize("penanda", [HOMOGLIF_KIRIL, HURUF_KECIL, BERSPASI, TEBAL])
    async def test_tidak_pernah_dialirkan(self, strong_retriever, penanda):
        potongan: list[str] = []

        async def on_token(teks: str) -> None:
            potongan.append(teks)

        hasil = await run_pipeline(
            "kapan KRS dibuka?",
            retriever=strong_retriever,
            llm_call=RecordingLLM(penanda),
            on_token=on_token,
        )
        assert potongan == []
        assert hasil.kind is OutcomeKind.REFUSAL


class TestPipeline:
    async def test_penanda_menjadi_penolakan(self, strong_retriever):
        hasil = await tanya(strong_retriever, NOT_FOUND_MARKER)
        assert hasil.kind is OutcomeKind.REFUSAL
        assert hasil.llm_called is True

    async def test_teksnya_penolakan_resmi_dengan_kontak(self, strong_retriever):
        hasil = await tanya(strong_retriever, NOT_FOUND_MARKER)
        assert FRONT_OFFICE in hasil.contacts
        assert hasil.text == refusal_text(hasil.contacts)
        assert NOT_FOUND_MARKER not in hasil.text

    async def test_petunjuk_unit_ikut_seperti_penolakan_threshold(self, strong_retriever):
        hasil = await tanya(strong_retriever, NOT_FOUND_MARKER, unit="UPS")
        assert "dokumen unit UPS" in hasil.text

    @pytest.mark.parametrize("unit", [None, "BAAK"])
    def test_penolakan_mengajak_menulis_lebih_rinci(self, unit):
        """Keputusan user 2026-11-23: sebagian penolakan terjadi karena cara
        bertanya ("Matakuliah ..." ditolak, "mata kuliah ..." dijawab, log
        2026-11-12), jadi teksnya tidak boleh menyatakan informasinya tidak ada."""
        teks = refusal_text([FRONT_OFFICE], unit=unit)
        kalimat_awal = teks.split("\n", 1)[0]
        assert kalimat_awal.startswith("Maaf, saya belum menemukan jawabannya di dokumen")
        assert "lebih rinci" in kalimat_awal
        assert "belum membahas" not in teks
        assert f"- {FRONT_OFFICE.unit}" in teks

    async def test_prosa_tolak_juga_menjadi_penolakan(self, strong_retriever):
        hasil = await tanya(strong_retriever, PROSA_TOLAK)
        assert hasil.kind is OutcomeKind.REFUSAL

    async def test_jawaban_bersitasi_tetap_jawaban(self, strong_retriever):
        hasil = await tanya(strong_retriever, JAWABAN)
        assert hasil.kind is OutcomeKind.ANSWER
        assert hasil.text == JAWABAN

    async def test_penolakan_llm_tanpa_kartu_sitasi(self, strong_retriever):
        """Kartu sitasi pada penolakan memberi kesan jawabannya bersumber."""
        respons = to_response(await tanya(strong_retriever, PROSA_TOLAK))
        assert respons.kind == "refusal"
        assert respons.citations == []
        assert respons.contacts

    async def test_dokumen_terambil_tetap_dibawa_untuk_log(self, strong_retriever):
        hasil = await tanya(strong_retriever, NOT_FOUND_MARKER)
        assert len(hasil.documents) == 2


class TestStreaming:
    async def alirkan(self, retriever, reply: str) -> tuple[list[str], object]:
        potongan: list[str] = []

        async def on_token(teks: str) -> None:
            potongan.append(teks)

        hasil = await run_pipeline(
            "kapan KRS dibuka?",
            retriever=retriever,
            llm_call=RecordingLLM(reply),
            on_token=on_token,
        )
        return potongan, hasil

    async def test_penanda_tidak_pernah_dialirkan(self, strong_retriever):
        potongan, hasil = await self.alirkan(strong_retriever, NOT_FOUND_MARKER)
        assert potongan == []
        assert hasil.kind is OutcomeKind.REFUSAL

    async def test_jawaban_biasa_dialirkan_utuh(self, strong_retriever):
        reply = "Pengisian KRS dibuka 1 Agustus [Panduan Akademik 2025, hal. 12]."
        potongan, _ = await self.alirkan(strong_retriever, reply)
        assert "".join(potongan) == reply
        assert len(potongan) > 1, "harus tetap bertahap, bukan sekaligus di akhir"

    async def test_jawaban_berawalan_sitasi_tidak_tertahan(self, strong_retriever):
        """`[` di awal sitasi sempat mirip penanda; ia harus dilepas begitu menyimpang."""
        reply = "[Panduan Akademik 2025, hal. 12] menyebut KRS dibuka 1 Agustus."
        potongan, hasil = await self.alirkan(strong_retriever, reply)
        assert "".join(potongan) == reply
        assert hasil.kind is OutcomeKind.ANSWER

    @pytest.mark.parametrize(
        "reply",
        [
            "**Jadwal** KRS dibuka 1 Agustus [Panduan Akademik 2025, hal. 12].",
            "* KRS dibuka 1 Agustus [Panduan Akademik 2025, hal. 12].",
        ],
    )
    async def test_jawaban_berawalan_penekanan_tidak_tertahan(self, strong_retriever, reply):
        """Penekanan di depan penanda ikut ditahan, tetapi jawaban bertebal atau
        berbutir harus lepas begitu kata pertamanya menyimpang."""
        potongan, hasil = await self.alirkan(strong_retriever, reply)
        assert "".join(potongan) == reply
        assert len(potongan) > 1
        assert hasil.kind is OutcomeKind.ANSWER


class TestLog:
    def meta(self, outcome) -> dict:
        entry = ChatLogEntry(
            session_id="sesi-uji-12345", question="q", outcome=outcome, latency_ms=1
        )
        return build_meta(entry)

    async def test_penolakan_llm_tercatat_sebagai_refusal(self, strong_retriever):
        meta = self.meta(await tanya(strong_retriever, NOT_FOUND_MARKER))
        assert meta["kind"] == "refusal"
        assert meta["refusal_source"] == "llm"
        assert meta["llm_called"] is True

    async def test_penolakan_threshold_bersumber_threshold(self, weak_retriever):
        meta = self.meta(await tanya(weak_retriever, JAWABAN))
        assert meta["refusal_source"] == "threshold"
        assert meta["llm_called"] is False

    async def test_jawaban_tanpa_sumber_penolakan(self, strong_retriever):
        assert self.meta(await tanya(strong_retriever, JAWABAN))["refusal_source"] is None
