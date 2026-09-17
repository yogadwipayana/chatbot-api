"""Dependency injection FastAPI.

Semua ketergantungan berat (DB, LLM, retriever) melewati fungsi di sini,
supaya test dapat menggantinya lewat `app.dependency_overrides` tanpa
menyentuh jaringan.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.permissions import ROLE_LABELS, AdminRole, CurrentAdmin
from app.config import Settings, get_settings
from app.db.session import get_session
from app.observability.tracing import id_run, konfigurasi_run
from app.security.auth import decode_access_token
from app.security.killswitch import KillSwitch, get_kill_switch
from app.security.ratelimit import FailureLimiter, get_login_limiter

BaseSettingsDep = Annotated[Settings, Depends(get_settings)]
"""Isi `.env` apa adanya, tanpa penimpaan dari dashboard.

Dipakai untuk hal yang memang hanya boleh diubah pengelola server -- rahasia
JWT, zona waktu, batas unggah -- dan untuk jalur yang tidak perlu membayar
satu query tambahan (verifikasi token pada setiap permintaan admin).
"""

SessionDep = Annotated[AsyncSession, Depends(get_session)]
KillSwitchDep = Annotated[KillSwitch, Depends(get_kill_switch)]
LoginLimiterDep = Annotated[FailureLimiter, Depends(get_login_limiter)]


def get_runtime_config_store(session: SessionDep) -> Any:
    """Tabel setelan yang dapat diubah dari dashboard. Di-override di test."""
    from app.admin.runtime_config import SqlRuntimeConfigStore

    return SqlRuntimeConfigStore(session)


RuntimeConfigStoreDep = Annotated[Any, Depends(get_runtime_config_store)]


async def get_effective_settings(
    base: BaseSettingsDep, store: RuntimeConfigStoreDep
) -> Settings:
    """`.env` ditimpa setelan dashboard (`app.admin.runtime_config`).

    Dibaca per permintaan, bukan di-cache: perubahan dari halaman Konfigurasi
    harus langsung berlaku di semua worker, termasuk worker yang tidak melayani
    permintaan yang mengubahnya. Tanpa baris penimpaan sama sekali -- keadaan
    normal -- fungsinya mengembalikan objek `.env` yang sama tanpa menyusun
    ulang apa pun.
    """
    from app.admin.runtime_config import terapkan

    return terapkan(base, await store.load())


SettingsDep = Annotated[Settings, Depends(get_effective_settings)]
"""Setelan yang benar-benar berlaku: `.env` + penimpaan dari dashboard."""


def pastikan_unit(admin: CurrentAdmin, unit: str | None, *, apa: str) -> None:
    """403 bila staf/dosen menyentuh isi milik unit lain.

    `apa` adalah nama bendanya dalam kalimat ("dokumen", "entri tanya jawab"),
    supaya admin membaca penolakan yang menyebut hal yang benar-benar ia buka.
    """
    if not admin.can_manage_unit(unit):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"{apa.capitalize()} ini milik unit lain. Akun Anda hanya dapat "
            f"mengelola {apa} unit {admin.unit}.",
        )


def guard_kill_switch(switch: KillSwitchDep) -> None:
    """Blokir endpoint mahasiswa saat layanan dimatikan (FR-9)."""
    if switch.engaged:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=switch.message,
        )


@lru_cache(maxsize=1)
def _storage_singleton() -> Any:
    """Satu instans penyimpanan per proses.

    Klien boto3 relatif mahal dibuat (memuat data model service dan menyiapkan
    connection pool). Membuatnya ulang per permintaan akan terasa saat musim
    KRS -- PRD §11 memperkirakan beban puncak 3x lipat.

    Tidak menerima `Settings` sebagai argumen karena model Pydantic tidak
    hashable sehingga tidak bisa jadi kunci cache; `get_settings()` sendiri
    sudah singleton, jadi membacanya di sini setara.
    """
    from app.storage import build_storage

    return build_storage(get_settings())


def get_storage() -> Any:
    """Dependency penyimpanan objek. Di-override di test dengan penyimpanan lokal."""
    return _storage_singleton()


StorageDep = Annotated[Any, Depends(get_storage)]


class EmbedQuery:
    """`str -> list[float]`, sambil mencatat pemakaian yang dilaporkan penyedia.

    Sama seperti `LLMCall`: FastAPI membangun dependency ulang untuk setiap
    permintaan, jadi satu instans hanya melayani satu permintaan dan angkanya
    tidak tertukar antar permintaan yang berjalan bersamaan.

    Angkanya diakumulasi, bukan ditimpa. Saat ini satu putaran hanya memanggil
    retrieval sekali (`app/rag/chain.py`), tetapi penambahan seperti multi-query
    akan memanggilnya lagi -- dan menimpa berarti diam-diam melaporkan biaya
    panggilan terakhir saja.
    """

    def __init__(self, embeddings: Any, model: str) -> None:
        self._embeddings = embeddings
        self.model = model
        """Model yang DIMINTA, bukan yang dilaporkan penyedia: inilah yang cocok
        dengan `EMBED_MODEL` dan dengan kunci di `costs.PRICES_PER_MTOK`. Gateway
        proyek ini menjawab `text-embedding-3-small` untuk permintaan
        `openrouter/openai/text-embedding-3-small`, dan nama pendek itu tidak
        terdaftar -- memakainya justru membuat tarifnya tidak ketemu."""
        self.model_dilaporkan: str | None = None
        """Diisi hanya bila penyedia menyebut model yang berbeda dari yang
        diminta. Gateway boleh memetakan ulang nama model ke model lain yang
        tarifnya jauh berbeda; itu harus terlihat, bukan tersamar."""
        self.panggilan = 0
        """Berapa kali embedding benar-benar dipanggil. Membedakan "tidak pernah
        dipanggil" (FR-7 dan smalltalk berhenti sebelum retrieval) dari "dipanggil
        tetapi endpoint tidak melaporkan pemakaian" -- yang pertama memang tidak
        berbiaya, yang kedua berbiaya tetapi tidak terhitung."""
        self.tokens: int | None = None
        self.biaya_usd: float | None = None
        self.biaya_sumber: str | None = None
        self.is_byok: bool | None = None

    async def __call__(self, text: str) -> list[float]:
        from app.rag.providers import embed_with_usage

        hasil = await embed_with_usage(self._embeddings, [text])
        self.panggilan += 1
        self._catat(hasil)
        return hasil.vectors[0]

    def _catat(self, hasil: Any) -> None:
        from app.observability.costs import SUMBER_ESTIMASI, biaya_embedding

        if hasil.model and hasil.model != self.model:
            self.model_dilaporkan = hasil.model
        if hasil.is_byok is not None:
            self.is_byok = hasil.is_byok
        if hasil.tokens is None:
            return
        self.tokens = (self.tokens or 0) + hasil.tokens

        biaya, sumber = biaya_embedding(self.model, hasil.tokens, hasil.biaya_usd)
        if biaya is None:
            return
        if self.biaya_sumber is not None and self.biaya_sumber != sumber:
            # Total yang mencampur biaya asli penyedia dengan taksiran kita,
            # secara keseluruhan, tetap sebuah taksiran. Klaim yang lebih lemah
            # yang menang -- melabelinya "provider" akan melebihkan keyakinan.
            sumber = SUMBER_ESTIMASI
        self.biaya_usd = (self.biaya_usd or 0.0) + biaya
        self.biaya_sumber = sumber


def build_retriever(settings: SettingsDep) -> Any:
    """Rakit `PostgresHybridRetriever`. Di-override di test dengan retriever palsu.

    Retriever menerima pabrik sesi, bukan satu sesi: vector search dan fulltext
    search berjalan paralel dan masing-masing butuh koneksinya sendiri.
    """
    from app.db.session import SessionLocal
    from app.rag.providers import build_embeddings
    from app.rag.retriever import PostgresHybridRetriever

    embeddings = build_embeddings(settings)
    return PostgresHybridRetriever(
        session_factory=SessionLocal,
        embed_query=EmbedQuery(embeddings, settings.embed_model),
        candidates=settings.retrieval_candidates,
        top_n=settings.retrieval_top_n,
        rrf_k=settings.rrf_k,
        weight_vector=settings.rrf_weight_vector,
        weight_fulltext=settings.rrf_weight_fulltext,
    )


class LLMCall:
    """(pertanyaan_terbungkus, dokumen) -> teks jawaban, sambil mencatat pemakaian token.

    FastAPI membangun dependency ulang untuk setiap permintaan, jadi satu
    instans hanya melayani satu permintaan dan `usage` tidak tertukar antar
    permintaan yang berjalan bersamaan. Pemakaian token dibutuhkan estimasi
    biaya AD-5 (FR-8).
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.chat_model
        self.usage: dict[str, Any] | None = None
        self.run_id: str | None = None
        self.session_id: str | None = None
        """Diisi router lewat `tandai_sesi` supaya run ini ikut terkelompok ke
        thread percakapannya di LangSmith."""

    def _config(self) -> dict[str, Any]:
        run_id = id_run()
        self.run_id = str(run_id)
        return konfigurasi_run(
            "generate_answer", run_id=run_id, session_id=self.session_id
        )

    async def __call__(self, wrapped_question: str, documents) -> str:
        from app.rag.prompts import answer_prompt, format_context
        from app.rag.providers import build_llm

        chain = answer_prompt() | build_llm(self.settings, streaming=False)
        result = await chain.ainvoke(
            {"context": format_context(documents), "question": wrapped_question},
            config=self._config(),
        )
        self.usage = getattr(result, "usage_metadata", None)
        return result.content

    async def stream(self, wrapped_question: str, documents) -> AsyncIterator[str]:
        """Sama seperti `__call__`, tetapi memancarkan potongan jawaban begitu tiba.

        Dipakai `/api/chat/stream` (FE-1). Potongan tetap dijumlahkan menjadi
        satu pesan utuh karena `usage_metadata` hanya ada pada hasil penjumlahan
        itu -- potongan per potongan tidak membawanya, dan tanpa penjumlahan
        estimasi biaya AD-5 hilang untuk setiap jawaban yang dialirkan.
        """
        from app.rag.prompts import answer_prompt, format_context
        from app.rag.providers import build_llm

        chain = answer_prompt() | build_llm(self.settings, streaming=True)
        utuh: Any = None
        async for potongan in chain.astream(
            {"context": format_context(documents), "question": wrapped_question},
            config=self._config(),
        ):
            utuh = potongan if utuh is None else utuh + potongan
            teks = str(potongan.text)
            if teks:
                yield teks
        self.usage = getattr(utuh, "usage_metadata", None)


def build_llm_call(settings: SettingsDep) -> Any:
    """Kembalikan callable (pertanyaan_terbungkus, dokumen) -> teks jawaban.

    Disuntikkan sebagai dependency, bukan dibuat di dalam handler, supaya test
    API dapat menggantinya dan membuktikan LLM tidak dipanggil pada jalur
    penolakan (FR-3) dan pertanyaan sensitif (FR-7).
    """
    return LLMCall(settings)


class RewriteCall:
    """Tulis pertanyaan lanjutan menjadi query mandiri untuk retrieval (FR-4)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.run_id: str | None = None
        self.session_id: str | None = None

    async def __call__(self, question: str, history: str) -> str:
        from app.rag.prompts import rewrite_prompt
        from app.rag.providers import build_llm

        run_id = id_run()
        self.run_id = str(run_id)
        chain = rewrite_prompt() | build_llm(self.settings, streaming=False)
        result = await chain.ainvoke(
            {"question": question, "history": history},
            config=konfigurasi_run(
                "rewrite_query", run_id=run_id, session_id=self.session_id
            ),
        )
        return str(result.content)


def build_rewrite_call(settings: SettingsDep) -> Any:
    """Dependency terpisah agar rewrite produksi dapat diganti tanpa jaringan di test."""
    return RewriteCall(settings)


def get_embeddings(settings: SettingsDep) -> Any:
    """Model embedding untuk ingestion (AD-3). Di-override di test dengan embedding palsu."""
    from app.rag.providers import build_embeddings

    return build_embeddings(settings)


def get_chat_logger() -> Any:
    """Pencatat percakapan ke Postgres (FR-8). Di-override di test."""
    from app.db.session import SessionLocal
    from app.observability.chatlog import ChatLogger

    return ChatLogger(SessionLocal)


_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="bearerAuth",
    bearerFormat="JWT",
    description="Token dari `POST /api/admin/login`. Hanya untuk endpoint admin.",
)


def get_account_store(session: SessionDep) -> Any:
    """Penyimpanan akun dashboard. Di-override di test dengan penyimpanan di memori."""
    from app.admin.accounts import SqlAccountStore

    return SqlAccountStore(session)


AccountStoreDep = Annotated[Any, Depends(get_account_store)]


async def require_admin(
    settings: BaseSettingsDep,
    store: AccountStoreDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> CurrentAdmin:
    """Verifikasi bearer token admin (AD-1) dan muat akunnya dari database.

    Level dan unit diambil dari database, bukan dari isi token: menurunkan
    level, memindah unit, atau menonaktifkan akun berlaku pada permintaan
    berikutnya. Token yang terbit sebelum kata sandi terakhir diganti ditolak,
    sehingga mengatur ulang kata sandi juga mengeluarkan sesi yang bocor.

    Pesannya ditujukan ke admin non-teknis (PRD §9): dashboard menampilkannya
    apa adanya sebelum mengarahkan kembali ke halaman login.
    """
    if credentials is None or not credentials.credentials:
        raise _tidak_berwenang("Silakan masuk terlebih dahulu.")
    try:
        payload = decode_access_token(
            credentials.credentials, settings.admin_jwt_secret.get_secret_value()
        )
    except Exception as exc:  # jwt.PyJWTError dan turunannya
        raise _tidak_berwenang("Sesi Anda sudah berakhir. Silakan masuk kembali.") from exc

    account = await store.find_by_email(str(payload.get("sub", "")))
    if account is None or not account.is_active:
        raise _tidak_berwenang("Akun Anda sudah tidak aktif. Hubungi superadmin.")

    diganti = account.password_changed_at
    # `iat` dibulatkan ke detik; bandingkan dalam detik juga, supaya token yang
    # terbit pada detik yang sama dengan penggantian (mis. dari endpoint ganti
    # kata sandi itu sendiri) tidak langsung ditolak.
    if diganti is not None and int(payload.get("iat", 0)) < int(diganti.timestamp()):
        raise _tidak_berwenang("Kata sandi akun ini sudah diganti. Silakan masuk kembali.")

    return account.as_current()


def _tidak_berwenang(pesan: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, pesan, headers={"WWW-Authenticate": "Bearer"}
    )


CurrentAdminDep = Annotated[CurrentAdmin, Depends(require_admin)]

_PESAN_TERLARANG = {
    AdminRole.STAF: "Silakan masuk terlebih dahulu.",
    AdminRole.ADMIN: "Fitur ini hanya untuk Admin dan Superadmin.",
    AdminRole.SUPERADMIN: "Fitur ini hanya untuk Superadmin.",
}


def require_role(minimum: AdminRole):
    """Dependency: akun harus minimal berlevel `minimum` (403 bila tidak).

    Setiap operasi admin di `api.yaml` mencatat level minimumnya sebagai
    `x-min-role`; `tests/api/test_admin_roles.py` memastikan kode menegakkannya.
    """

    def check(admin: CurrentAdminDep) -> CurrentAdmin:
        if not admin.at_least(minimum):
            raise HTTPException(status.HTTP_403_FORBIDDEN, _PESAN_TERLARANG[minimum])
        return admin

    check.__name__ = f"require_{minimum.value}"
    check.__doc__ = f"Minimal {ROLE_LABELS[minimum]}."
    return check
