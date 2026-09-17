"""LangSmith tracing (FR-8).

Aktif di semua environment. PRD §12 menyebut "debugging tersembunyi di balik
abstraksi" sebagai risiko; tracing adalah mitigasinya, jadi ia bukan opsional.

Satu giliran tanya-jawab menghasilkan SATU trace. Penulisan ulang query (FR-4),
retrieval (FR-2), dan penyusunan jawaban (FR-5) menjadi child run di bawah satu
akar yang sama. Tanpa akar itu ketiganya menjadi trace lepas yang tidak dapat
dihubungkan kembali: admin yang membuka satu jawaban buruk hanya melihat
panggilan LLM terakhir, tanpa query yang sebenarnya dicari maupun dokumen yang
terambil -- persis bagian yang biasanya menjadi penyebabnya.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager, suppress
from typing import Any
from uuid import uuid4

from app.config import Settings

VAR_LANGSMITH = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2")
VAR_LANGCHAIN = ("LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2")
"""Empat variabel yang sama-sama menyalakan tracing, dibaca langsmith dengan
urutan tetap: `*_TRACING_V2` mendahului `*_TRACING`, dan namespace `LANGSMITH`
mendahului `LANGCHAIN` (`langsmith.utils.get_env_var`).

Menulis satu saja tidak cukup. `LANGCHAIN_TRACING_V2=true` yang tertinggal di
environment -- sisa proyek lain di mesin yang sama, atau variabel lama yang
belum dibersihkan dari deployment -- dibaca LEBIH DULU daripada
`LANGSMITH_TRACING=false`, sehingga tracing tetap menyala meski konfigurasi
aplikasi mematikannya. Kebalikannya sama buruknya: `LANGCHAIN_TRACING_V2=false`
yang tertinggal membuat tracing tidak pernah menyala tanpa pesan apa pun.

Karena itu pasangan `LANGSMITH_*` selalu ditulis eksplisit dan pasangan
`LANGCHAIN_*` selalu dihapus: satu aturan untuk jalur nyala maupun mati.
Menghapus, bukan mengisi "false", karena `LANGCHAIN_TRACING` yang sekadar ada
dan berisi nilai truthy membangunkan pemeriksaan tracer v1 di langchain-core
(`callbacks/manager.py`) yang sudah tidak didukung.
"""


def tracing_aktif(settings: Settings) -> bool:
    """Apakah konfigurasi MEMINTA tracing: dinyalakan DAN kuncinya ada.

    `LANGSMITH_TRACING=true` tanpa kunci API bukan permintaan yang bisa dipenuhi,
    jadi ia dihitung sebagai mati -- lebih baik ketahuan begitu daripada menyala
    setengah jalan.

    Ini keputusan di sisi konfigurasi, dipakai `configure_tracing` untuk menyetel
    environment. Untuk pertanyaan "apakah trace sedang benar-benar terkirim",
    pakai `sedang_menjejak`, yang bertanya langsung ke langsmith.
    """
    return settings.langsmith_tracing and settings.langsmith_api_key is not None


def configure_tracing(settings: Settings) -> bool:
    """Set variabel lingkungan yang dibaca LangSmith. Return: aktif atau tidak."""
    aktif = tracing_aktif(settings)
    nilai = "true" if aktif else "false"

    for nama in VAR_LANGSMITH:
        os.environ[nama] = nilai
    for nama in VAR_LANGCHAIN:
        os.environ.pop(nama, None)

    # `kunci is not None` sudah dijamin `tracing_aktif`; disebut ulang di sini
    # karena type checker tidak dapat membawa jaminan itu lintas fungsi, dan
    # `assert` bukan penggantinya -- ia hilang saat Python dijalankan dengan -O.
    kunci = settings.langsmith_api_key
    if aktif and kunci is not None:
        os.environ["LANGSMITH_API_KEY"] = kunci.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        if settings.langsmith_endpoint:
            # Dibaca `langsmith.Client` lewat `os.getenv`, bukan lewat Settings.
            # Nilai yang hanya ada di .env tidak pernah sampai ke sana:
            # pydantic-settings membacanya ke objek Settings dan berhenti di situ.
            os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    _bersihkan_cache_env()
    return aktif


def _bersihkan_cache_env() -> None:
    """Buang hasil pembacaan environment yang sudah terlanjur di-cache langsmith.

    `langsmith.utils.get_env_var` memakai `functools.lru_cache`. Nilai yang
    sempat dibaca sebelum fungsi ini berjalan -- oleh impor yang menyentuh
    langsmith lebih awal, atau oleh `configure_tracing` sebelumnya di proses yang
    sama, seperti di test -- akan bertahan dan mengalahkan apa yang baru ditulis.

    Kegagalan diabaikan karena ini menyentuh detail internal langsmith: bila
    suatu saat cache-nya hilang, tracing tetap boleh jalan dengan nilai yang
    dibaca ulang secara normal, bukan menggagalkan start aplikasi.
    """
    try:
        from langsmith.utils import get_env_var
    except ImportError:  # pragma: no cover - langsmith adalah dependensi wajib
        return

    # `getattr`, bukan pemanggilan langsung: `get_env_var` adalah fungsi ber-overload
    # dan atribut yang ditambahkan `lru_cache` tidak terlihat oleh type checker.
    bersihkan = getattr(get_env_var, "cache_clear", None)
    if bersihkan is not None:
        bersihkan()


def sedang_menjejak() -> bool:
    """Apakah langsmith BENAR-BENAR akan mengirim trace saat ini.

    Sengaja bertanya ke langsmith, bukan menyimpulkan sendiri dari Settings.
    Keduanya bisa berbeda, dan yang menentukan ada-tidaknya trace adalah
    langsmith: `.env` yang berisi kunci API tidak membuat tracing menyala kalau
    `configure_tracing` belum pernah berjalan (mis. di test, yang tidak menjalankan
    lifespan). Menyimpulkan dari Settings di situ menghasilkan ID yang tercatat
    rapi ke basis data untuk trace yang tidak pernah dikirim -- persis kerusakan
    yang ingin dicegah.

    `"local"` -- trace ditulis ke sink lokal, bukan ke LangSmith -- dihitung
    sebagai tidak menjejak: tidak ada halaman yang bisa dibuka admin dari sana.
    """
    try:
        from langsmith.utils import tracing_is_enabled

        return tracing_is_enabled() is True
    except Exception:  # pragma: no cover - bergantung versi langsmith
        return False


def id_giliran() -> str | None:
    """ID akar trace untuk satu giliran, atau None bila tracing mati.

    None bukan detail teknis: ID yang disimpan ke `messages.langsmith_run_id`
    adalah tautan yang akan diklik admin. Saat tracing mati tidak ada trace yang
    dibuat, jadi ID apa pun yang dicatat menunjuk ke halaman yang tidak ada.
    """
    if not sedang_menjejak():
        return None
    return str(id_run())


def id_run() -> Any:
    """UUID untuk satu run: v7 bila tersedia, jatuh ke v4 bila tidak.

    LangSmith meminta v7 untuk ID run buatan sendiri: v7 menyimpan timestamp,
    sehingga run terurut benar menurut waktu di dalam satu trace. v4 acak murni
    dan membuat urutannya sembarang.
    """
    try:
        from langsmith import uuid7

        return uuid7()
    except (ImportError, AttributeError):  # pragma: no cover - versi langsmith lama
        return uuid4()


def konfigurasi_run(
    nama: str, *, run_id: Any = None, session_id: str | None = None
) -> dict[str, Any]:
    """`RunnableConfig` untuk satu langkah LangChain di dalam giliran.

    `session_id` diulang di setiap run, bukan hanya di akar. LangSmith
    mengelompokkan trace menjadi thread dari metadata ini, dan child run yang
    tidak membawanya akan terlewat saat trace difilter per percakapan maupun saat
    token dan biaya satu percakapan dijumlahkan.
    """
    config: dict[str, Any] = {"run_name": nama}
    if run_id is not None:
        config["run_id"] = run_id
    if session_id:
        config["metadata"] = {"session_id": session_id}
    return config


def tandai_sesi(session_id: str | None, *panggilan: Any) -> None:
    """Titipkan `session_id` ke dependency LLM sebelum pipeline berjalan.

    Dependency-nya dibangun ulang per permintaan (lihat `LLMCall`), jadi menaruh
    nilai ini padanya tidak bocor antar mahasiswa. Jalur ini dipilih daripada
    menambah parameter pada `run_pipeline`: session_id tidak mengubah satu pun
    keputusan pipeline -- ia hanya label untuk trace -- dan menyelipkannya ke
    tanda tangan pipeline akan memaksa setiap test menyediakannya.

    Kegagalan diabaikan: test menyuntikkan callable polos, dan pengelompokan
    trace tidak layak menggagalkan jawaban yang sudah benar.
    """
    for objek in panggilan:
        with suppress(AttributeError):  # objek tanpa __dict__, mis. ber-__slots__
            objek.session_id = session_id


@asynccontextmanager
async def jejak_giliran(
    *,
    run_id: str | None,
    session_id: str | None = None,
    pertanyaan: str | None = None,
    nama: str = "giliran_chat",
):
    """Buka satu trace akar yang menaungi seluruh giliran tanya-jawab.

    Langkah LangChain yang berjalan di dalamnya menempel sendiri ke akar ini:
    langchain-core membaca run tree yang sedang aktif dari langsmith dan
    memakainya sebagai induk (`callbacks/manager.py`). Retrieval ikut masuk tanpa
    perlu disentuh, karena retriever proyek ini sudah berupa `BaseRetriever`.

    Tanpa `run_id` -- yaitu saat tracing mati -- tidak ada yang dibuka sama
    sekali, sehingga jalur non-tracing tidak membayar ongkos apa pun.
    """
    if run_id is None:
        yield None
        return

    from langsmith import trace

    metadata = {"session_id": session_id} if session_id else None
    inputs = {"question": pertanyaan} if pertanyaan is not None else None
    async with trace(
        name=nama, run_type="chain", run_id=run_id, metadata=metadata, inputs=inputs
    ) as akar:
        yield akar


def akhiri_jejak(akar: Any, **keluaran: Any) -> None:
    """Tutup akar trace dengan hasil giliran.

    Tanpa ini akar hanya memuat pertanyaan dan tidak satu pun jawaban, sehingga
    di LangSmith giliran yang berhasil tampak seperti giliran yang menggantung.
    `akar` bernilai None saat tracing mati, dan pemanggil tidak perlu
    memeriksanya sendiri di setiap tempat.
    """
    if akar is None:
        return
    akar.end(outputs=keluaran)
