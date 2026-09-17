"""Lapisan API endpoint chat -- FE-1..FE-5, FR-3, FR-7, FR-9."""

from __future__ import annotations

import json

import pytest

from app.rag.chain import OutcomeKind


def events(response) -> list[tuple[str, dict]]:
    """Urai aliran SSE menjadi daftar `(nama event, data)` sesuai urutannya."""
    hasil: list[tuple[str, dict]] = []
    for blok in response.text.split("\n\n"):
        nama, data = "message", None
        for baris in blok.splitlines():
            if baris.startswith("event: "):
                nama = baris.removeprefix("event: ")
            elif baris.startswith("data: "):
                data = json.loads(baris.removeprefix("data: "))
        if data is not None:
            hasil.append((nama, data))
    return hasil


def nama_event(response) -> list[str]:
    return [nama for nama, _ in events(response)]


def pesan_akhir(response) -> dict:
    """Payload `ChatResponse` dari event `message`."""
    return next(data for nama, data in events(response) if nama == "message")


def potongan(response) -> list[str]:
    return [data["text"] for nama, data in events(response) if nama == "token"]


class TestJawabanNormal:
    def test_mengembalikan_200(self, client, payload):
        assert client.post("/api/chat", json=payload).status_code == 200

    def test_jenisnya_jawaban(self, client, payload):
        assert client.post("/api/chat", json=payload).json()["kind"] == OutcomeKind.ANSWER

    def test_menyertakan_kartu_sitasi(self, client, payload):
        """FE-2 disebut PRD sebagai komponen paling kritis."""
        sitasi = client.post("/api/chat", json=payload).json()["citations"]
        assert sitasi
        assert {"judul", "halaman", "document_id", "file_path"} <= set(sitasi[0])

    def test_sitasi_cukup_untuk_membuka_pdf_di_halaman_tepat(self, client, payload):
        sitasi = client.post("/api/chat", json=payload).json()["citations"][0]
        assert sitasi["file_path"].endswith(".pdf")
        assert sitasi["halaman"] >= 1

    def test_sitasi_dideduplikasi_per_halaman(self, make_client, payload):
        from tests.fixtures.fakes import make_document

        docs = [make_document("c1", halaman=12), make_document("c2", halaman=12)]
        sitasi = make_client(docs).post("/api/chat", json=payload).json()["citations"]
        assert len(sitasi) == 1

    def test_top_score_dilaporkan(self, client, payload):
        """Dipakai AD-6 untuk menunjukkan chunk terambil beserta skornya."""
        assert client.post("/api/chat", json=payload).json()["top_score"] is not None


class TestSitasiHanyaYangDikutip:
    """FE-2: kartu sumber menunjuk kalimat yang benar-benar dijawab.

    Retrieval mengambil beberapa chunk sekaligus sebagai konteks LLM. Kalau
    semuanya ikut tampil sebagai "Sumber", mahasiswa harus menebak kartu mana
    yang relevan -- verifikasi sekali klik yang dituntut PRD §8 jadi hilang.
    """

    def test_chunk_yang_tidak_dikutip_tidak_jadi_kartu(
        self, make_client, strong_documents, payload, api_llm
    ):
        api_llm.reply = "Pembayaran lewat bank mitra [Panduan Akademik 2025, hal. 13]."
        data = make_client(strong_documents).post("/api/chat", json=payload).json()
        assert [c["halaman"] for c in data["citations"]] == [13]

    def test_urutan_mengikuti_kemunculan_di_jawaban(
        self, make_client, strong_documents, payload, api_llm
    ):
        """Urutan baca mahasiswa, bukan peringkat retrieval."""
        api_llm.reply = (
            "A [Panduan Akademik 2025, hal. 13] lalu B [Panduan Akademik 2025, hal. 12]."
        )
        data = make_client(strong_documents).post("/api/chat", json=payload).json()
        assert [c["halaman"] for c in data["citations"]] == [13, 12]

    def test_sumber_karangan_tidak_pernah_jadi_kartu(
        self, make_client, strong_documents, payload, api_llm
    ):
        """Kartu yang tampak sah tetapi tautannya buntu lebih berbahaya daripada
        tidak ada kartu sama sekali."""
        api_llm.reply = (
            "Menurut [Peraturan Fiktif 2030, hal. 9] dan [Panduan Akademik 2025, hal. 12]."
        )
        data = make_client(strong_documents).post("/api/chat", json=payload).json()
        assert [(c["judul"], c["halaman"]) for c in data["citations"]] == [
            ("Panduan Akademik 2025", 12)
        ]

    def test_jawaban_tanpa_penanda_jatuh_ke_seluruh_chunk(
        self, make_client, strong_documents, payload, api_llm
    ):
        """LLM melanggar FR-5 di sini; mahasiswa tetap harus punya jalan verifikasi."""
        api_llm.reply = "Silakan hubungi bagian akademik."
        data = make_client(strong_documents).post("/api/chat", json=payload).json()
        assert [c["halaman"] for c in data["citations"]] == [12, 13]


class TestPenolakan:
    def test_skor_lemah_menghasilkan_penolakan(self, make_client, weak_documents, payload):
        data = make_client(weak_documents).post("/api/chat", json=payload).json()
        assert data["kind"] == OutcomeKind.REFUSAL

    def test_llm_tidak_dipanggil(self, make_client, weak_documents, payload, api_llm):
        make_client(weak_documents).post("/api/chat", json=payload)
        assert api_llm.calls == []

    def test_tanpa_kartu_sitasi(self, make_client, weak_documents, payload):
        """FE-4: tampilan penolakan harus berbeda dari jawaban. Kartu sitasi di
        layar penolakan memberi kesan keliru bahwa jawabannya bersumber."""
        data = make_client(weak_documents).post("/api/chat", json=payload).json()
        assert data["citations"] == []

    def test_menyertakan_kontak(self, make_client, weak_documents, payload):
        data = make_client(weak_documents).post("/api/chat", json=payload).json()
        assert data["contacts"]
        assert data["contacts"][0]["kontak"] in data["text"]

    def test_retrieval_kosong_juga_ditolak(self, make_client, payload, api_llm):
        data = make_client([]).post("/api/chat", json=payload).json()
        assert data["kind"] == OutcomeKind.REFUSAL
        assert api_llm.calls == []


class TestPertanyaanSensitif:
    def test_diarahkan_ke_konseling(self, client):
        data = client.post(
            "/api/chat",
            json={"question": "saya stres berat mau menyerah", "session_id": "sesi-uji-12345"},
        ).json()
        assert data["kind"] == OutcomeKind.SUPPORT
        assert any("Konseling" in c["unit"] for c in data["contacts"])

    def test_tanpa_sitasi_dan_tanpa_llm(self, client, api_llm):
        data = client.post(
            "/api/chat",
            json={"question": "saya depresi", "session_id": "sesi-uji-12345"},
        ).json()
        assert data["citations"] == []
        assert api_llm.calls == []


class TestSapaan:
    def test_dibalas_singkat_tanpa_llm(self, client, api_llm):
        data = client.post(
            "/api/chat", json={"question": "hai", "session_id": "sesi-uji-12345"}
        ).json()
        assert data["kind"] == OutcomeKind.SMALLTALK
        assert data["text"].strip()
        assert api_llm.calls == []

    def test_tanpa_sitasi_dan_tanpa_kontak(self, client):
        """Sapaan bukan jawaban bersumber dokumen, dan tidak perlu mengarahkan
        siapa pun ke loket biro."""
        data = client.post(
            "/api/chat", json={"question": "halo", "session_id": "sesi-uji-12345"}
        ).json()
        assert data["citations"] == []
        assert data["contacts"] == []
        assert data["escalated"] is False


class TestEskalasi:
    def test_topik_berisiko_menandai_escalated(self, client):
        """FE-3: banner eskalasi muncul berdasarkan flag ini."""
        data = client.post(
            "/api/chat",
            json={
                "question": "kapan deadline pembayaran UKT?",
                "session_id": "sesi-uji-12345",
            },
        ).json()
        assert data["escalated"] is True
        assert data["contacts"]

    def test_pertanyaan_biasa_tidak_dieskalasi(self, client, payload):
        assert client.post("/api/chat", json=payload).json()["escalated"] is False


class TestValidasiRequest:
    def test_pertanyaan_kosong_ditolak(self, client):
        r = client.post("/api/chat", json={"question": "", "session_id": "sesi-uji-12345"})
        assert r.status_code == 422

    def test_pertanyaan_terlalu_panjang_ditolak(self, client):
        r = client.post(
            "/api/chat", json={"question": "a" * 2001, "session_id": "sesi-uji-12345"}
        )
        assert r.status_code == 422

    def test_session_id_wajib(self, client):
        assert client.post("/api/chat", json={"question": "halo"}).status_code == 422

    def test_session_id_terlalu_pendek_ditolak(self, client):
        r = client.post("/api/chat", json={"question": "halo", "session_id": "abc"})
        assert r.status_code == 422

    def test_peran_riwayat_dibatasi(self, client, payload):
        r = client.post(
            "/api/chat", json={**payload, "history": [{"role": "system", "konten": "x"}]}
        )
        assert r.status_code == 422


class TestKillSwitch:
    def test_chat_diblokir_saat_aktif(self, client, payload, kill_switch):
        kill_switch.engage("insiden jawaban salah")
        r = client.post("/api/chat", json=payload)
        assert r.status_code == 503

    def test_pesan_blokir_menyebut_kontak_manusia(self, client, payload, kill_switch):
        kill_switch.engage("insiden")
        assert "Akademik" in client.post("/api/chat", json=payload).json()["detail"]

    def test_llm_tidak_dipanggil_saat_diblokir(self, client, payload, kill_switch, api_llm):
        kill_switch.engage("insiden")
        client.post("/api/chat", json=payload)
        assert api_llm.calls == []

    def test_normal_kembali_setelah_dilepas(self, client, payload, kill_switch):
        kill_switch.engage("insiden")
        kill_switch.release()
        assert client.post("/api/chat", json=payload).status_code == 200

    def test_streaming_juga_diblokir(self, client, payload, kill_switch):
        kill_switch.engage("insiden")
        assert client.post("/api/chat/stream", json=payload).status_code == 503


class TestStreaming:
    def test_content_type_sse(self, client, payload):
        r = client.post("/api/chat/stream", json=payload)
        assert r.headers["content-type"].startswith("text/event-stream")

    def test_mengirim_status_sebelum_jawaban(self, client, payload):
        """FE-1 menampilkan indikator 'mencari dokumen...' sebelum token pertama."""
        teks = client.post("/api/chat/stream", json=payload).text
        assert teks.index("event: status") < teks.index("event: message")

    def test_diakhiri_event_done(self, client, payload):
        assert client.post("/api/chat/stream", json=payload).text.rstrip().endswith("{}")

    def test_payload_message_berisi_jawaban_dan_sitasi(self, client, payload):
        pesan = pesan_akhir(client.post("/api/chat/stream", json=payload))
        assert pesan["kind"] == OutcomeKind.ANSWER
        assert pesan["citations"]

    def test_header_anti_buffering(self, client, payload):
        """Tanpa ini Caddy boleh menahan aliran sampai selesai (FE-1)."""
        r = client.post("/api/chat/stream", json=payload)
        assert r.headers["cache-control"] == "no-cache"
        assert r.headers["x-accel-buffering"] == "no"


class TestPotonganJawaban:
    """FE-1: jawaban tiba sepotong demi sepotong, bukan sekaligus di akhir.

    Tanpa test ini `/api/chat/stream` bisa saja tetap memenuhi kontrak SSE --
    `status`, `message`, `done` -- sambil menahan seluruh jawaban sampai
    pipeline selesai, yang di layar mahasiswa tidak berbeda dari tanpa
    streaming sama sekali.
    """

    def test_dikirim_lebih_dari_satu_potongan(self, client, payload):
        assert len(potongan(client.post("/api/chat/stream", json=payload))) > 1

    def test_potongan_dirangkai_menjadi_jawaban_yang_sama(self, client, payload):
        r = client.post("/api/chat/stream", json=payload)
        assert "".join(potongan(r)) == pesan_akhir(r)["text"]

    def test_potongan_mendahului_message(self, client, payload):
        nama = nama_event(client.post("/api/chat/stream", json=payload))
        assert nama.index("token") < nama.index("message")

    def test_status_berganti_sebelum_potongan_pertama(self, client, payload):
        """Indikator FE-1 tidak boleh tertinggal di 'mencari dokumen' selama LLM
        menyusun kalimat pertamanya."""
        peristiwa = events(client.post("/api/chat/stream", json=payload))
        tahap = [d["stage"] for n, d in peristiwa if n == "status"]
        nama = [n for n, _ in peristiwa]
        assert tahap == ["mencari dokumen", "menyusun jawaban"]
        assert nama.index("status") < nama.index("token")

    def test_penolakan_tidak_mengalirkan_potongan(
        self, make_client, weak_documents, payload
    ):
        """FR-3 menolak tanpa memanggil LLM; tidak ada yang bisa dialirkan."""
        r = make_client(weak_documents).post("/api/chat/stream", json=payload)
        assert potongan(r) == []
        assert pesan_akhir(r)["kind"] == OutcomeKind.REFUSAL

    def test_jawaban_utuh_tetap_dicatat_sekali(self, client, payload, chat_logger):
        """FR-8 mencatat jawaban, bukan potongan terakhirnya."""
        r = client.post("/api/chat/stream", json=payload)
        assert len(chat_logger.entries) == 1
        assert chat_logger.entries[0].outcome.text == pesan_akhir(r)["text"]


class TestPertanyaanKosongSetelahSanitasi:
    @pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
    @pytest.mark.parametrize("pertanyaan", ["   ", "​​", "\n\t"])
    def test_ditolak_422_bukan_500(self, client, path, pertanyaan):
        """Lolos `min_length=1`, tetapi kosong setelah karakter kontrol dibuang."""
        r = client.post(path, json={"question": pertanyaan, "session_id": "sesi-uji-12345"})
        assert r.status_code == 422


class TestPencatatan:
    def test_message_id_dari_log(self, client, payload, chat_logger):
        data = client.post("/api/chat", json=payload).json()
        assert data["message_id"] == chat_logger.message_id

    def test_log_gagal_tidak_menggagalkan_jawaban(self, client, payload, chat_logger):
        chat_logger.fail = True
        r = client.post("/api/chat", json=payload)
        assert r.status_code == 200
        assert r.json()["message_id"] is None

    def test_yang_dicatat_pertanyaan_tersanitasi(self, client, chat_logger):
        client.post(
            "/api/chat",
            json={"question": "  kapan   KRS dibuka?​ ", "session_id": "sesi-uji-12345"},
        )
        entri = chat_logger.entries[0]
        assert entri.question == "kapan KRS dibuka?"
        assert entri.session_id == "sesi-uji-12345"
        assert entri.latency_ms >= 0

    def test_penolakan_ikut_dicatat(self, make_client, weak_documents, payload, chat_logger):
        make_client(weak_documents).post("/api/chat", json=payload)
        assert chat_logger.entries[0].outcome.kind == OutcomeKind.REFUSAL

    def test_streaming_juga_dicatat(self, client, payload, chat_logger):
        pesan = pesan_akhir(client.post("/api/chat/stream", json=payload))
        assert pesan["message_id"] == chat_logger.message_id
        assert len(chat_logger.entries) == 1

    def test_diblokir_kill_switch_tidak_dicatat(
        self, client, payload, kill_switch, chat_logger
    ):
        kill_switch.engage("insiden")
        client.post("/api/chat", json=payload)
        assert chat_logger.entries == []


class TestFeedback:
    def test_message_id_harus_uuid(self, client):
        r = client.post("/api/feedback", json={"message_id": "bukan-uuid", "helpful": True})
        assert r.status_code == 422

    def test_catatan_dibatasi_panjangnya(self, client):
        r = client.post(
            "/api/feedback",
            json={
                "message_id": "9c3e1a44-6b2d-4f51-8a70-2d9b5c1e7f03",
                "helpful": False,
                "catatan": "x" * 1001,
            },
        )
        assert r.status_code == 422
